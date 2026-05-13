"""
随机计算技术实现
实现随机计算的基本操作和模块化组件
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union, Dict
import math


class StochasticCore:
    """随机计算核心模块"""

    def __init__(self,
                 stream_length: int = 10,
                 representation: str = 'unipolar',
                 rand_gen: str = 'pseudo'):
        """
        参数:
            stream_length: 随机比特流长度
            representation: 'unipolar'(单极)或'bipolar'(双极)表示
            rand_gen: 随机数生成方式 'pseudo'(伪随机), 'lfsr'(线性反馈移位寄存器)
        """
        self.stream_length = stream_length
        self.representation = representation
        self.rand_gen = rand_gen

        # 预生成随机数表以提高效率
        self._init_random_tables()

    def _init_random_tables(self):
        """初始化随机数表"""
        self.rand_table_size = 1000000  # 1M个随机数
        self.rand_table = torch.rand(self.rand_table_size)
        self.table_ptr = 0

    def _get_random_bits(self, shape: Tuple[int]) -> torch.Tensor:
        """
        获取随机比特
        参数：
        shape: 需要随机比特的形状，比如(10,1000)表示10个样本，每个样本1000个比特

        返回：
        随机比特的多维数组
        """
        if self.rand_gen == 'lfsr':
            return self._lfsr_random(shape)
        else:
            # 使用预生成的随机数表
            total_elements = np.prod(shape)
            if self.table_ptr + total_elements > self.rand_table_size:
                self.table_ptr = 0

            bits = self.rand_table[self.table_ptr:self.table_ptr + total_elements]
            bits = bits.view(shape)
            self.table_ptr += total_elements
            return bits

    def _lfsr_random(self, shape: Tuple[int]) -> torch.Tensor:
        """
        LFSR生成的伪随机数

         参数：
        shape: 需要随机比特的形状

        返回：
        LFSR生成的随机比特
        """
        # 简化的LFSR实现
        size = np.prod(shape)
        bits = []
        state = 0xACE1  # 初始种子

        for _ in range(size):
            # LFSR: x^16 + x^14 + x^13 + x^11 + 1
            bit = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1
            state = (state >> 1) | (bit << 15)
            bits.append(float(bit))

        return torch.tensor(bits).view(shape)

    def to_stochastic(self, x: torch.Tensor) -> torch.Tensor:
        """
        将实数转换为随机比特流

        参数：
        x: 输入的数字（张量形式）

        返回：
        随机比特流
        """
        batch_size = x.shape[0] #一次性处理样本数量
        device = x.device #采用gpu/cpu

        if self.representation == 'unipolar':
            # 单极表示: x ∈ [0,1]
            x_prob = torch.sigmoid(x).clamp(0, 1)
        else:
            # 双极表示: x ∈ [-1,1] -> p ∈ [0,1]
            x_prob = (torch.tanh(x) + 1) / 2

        # 生成随机比特流 [batch, stream_length, ...]
        x_shape = list(x.shape)
        x_shape.insert(1, self.stream_length)  # 插入stream维度

        # 生成随机数
        rand_bits = self._get_random_bits(x_shape).to(device)

        # 比较生成比特流
        x_prob_expanded = x_prob.unsqueeze(1).expand(-1, self.stream_length, *[-1] * (x.dim() - 1))
        stochastic_stream = (rand_bits < x_prob_expanded).float()

        return stochastic_stream

    def from_stochastic(self, stream: torch.Tensor) -> torch.Tensor:
        """
        从比特流恢复实数

        参数：
        stream: 随机比特流

        返回：
        恢复的数字
        """
        if self.representation == 'unipolar':
            prob = stream.mean(dim=1)  # 平均比特流
            return prob
        else:
            prob = stream.mean(dim=1)
            return 2 * prob - 1  # 转换回双极

    def stochastic_multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        随机乘法
        参数：
        a, b: 两个随机比特流

        返回：
        乘积的随机比特流
        """
        if self.representation == 'unipolar':
            # 单极: AND门 两个都是1才得1，概率：P(A AND B) = P(A)*P(B)
            return a * b
        else:
            # 双极: XNOR门 两个相同得1，不同得0 概率：P(A XNOR B) = P(A)*P(B) + (1-P(A))*(1-P(B))
            return 1 - torch.abs(a - b)

    def stochastic_add(self, a: torch.Tensor, b: torch.Tensor,
                       weights: Tuple[float, float] = (0.5, 0.5)) -> torch.Tensor:
        """
        随机加法（使用多路选择器）

        参数：
        a, b: 要相加的两个随机比特流
        weights: 权重，比如(0.3,0.7)表示30%选a，70%选b

        返回：
        加法结果的随机比特流
        """
        batch_size, stream_len = a.shape[:2]
        device = a.device

        # 生成选择信号
        weight_prob = weights[0] / (weights[0] + weights[1])
        select_stream = self.to_stochastic(torch.tensor(weight_prob).expand(batch_size, 1)).squeeze(2)

        # 多路选择器
        result = torch.where(select_stream.unsqueeze(-1) > 0.5, a, b)
        return result

    def stochastic_activation(self, x: torch.Tensor,
                              activation: str = 'relu') -> torch.Tensor:
        """随机激活函数
        激活函数给神经网络添加非线性，让网络能学习复杂模式

        参数：
        x: 输入随机比特流
        activation: 激活函数类型

        返回：
        激活后的随机比特流
        """
        batch_size, stream_len, *dims = x.shape

        if activation == 'relu':
            # ReLU: 如果x>0则输出x，否则输出0
            # 在随机计算中，我们实现为按比特的max(0, x)
            return F.relu(x)
        elif activation == 'sigmoid':
            # Sigmoid: 1/(1+exp(-x))
            # 使用随机比较实现
            x_prob = self.from_stochastic(x)
            sigmoid_prob = torch.sigmoid(x_prob)
            return self.to_stochastic(sigmoid_prob)

        return x


class StochasticModule(nn.Module):
    """随机计算模块基类"""

    def __init__(self, stream_length: int = 10):
        super().__init__() # 调用父类初始化
        self.stochastic_core = StochasticCore(stream_length=stream_length)
        self.stream_length = stream_length
        self.use_stochastic = False # 默认不使用随机计算

    def enable_stochastic(self):
        """启用随机计算模式"""
        self.use_stochastic = True

    def disable_stochastic(self):
        """禁用随机计算模式"""
        self.use_stochastic = False


class StochasticConv2d(StochasticModule):
    """随机计算卷积层"""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, stream_length=10):
        super().__init__(stream_length)

        # 存储普通卷积参数
        # 训练时用普通计算，推理时用随机计算
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        self.in_channels = in_channels # 输入通道数
        self.out_channels = out_channels # 输出通道数
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        self._initialized = False  # 避免在初始化时创建递归依赖

    def forward(self, x):
        """
        前向传播：数据从输入到输出的过程

        参数：
        x: 输入数据

        返回：
        卷积结果
        """

        if not self.use_stochastic or not self.training:
            return self.conv(x)

            # 简化实现，避免复杂的循环结构
        batch_size, _, h, w = x.shape

        # 直接使用普通卷积，但添加噪声模拟随机计算
        output = self.conv(x)

        # 添加随机噪声（模拟随机计算的误差）
        if self.training and self.use_stochastic:
            noise = torch.randn_like(output) * 0.1
            output = output + noise

        return output
        # if not self.use_stochastic or not self.training:
        #     # 训练时或未启用随机计算：使用普通卷积
        #     return self.conv(x)
        #
        # # 随机计算前向传播
        # batch_size, _, h, w = x.shape
        #
        # # 将输入转换为概率
        # x_prob = torch.sigmoid(x)  # 假设输入已归一化
        #
        # # 生成随机比特流
        # x_stream = self.stochastic_core.to_stochastic(x)
        #
        # # 对每个输出通道计算
        # outputs = []
        # for out_ch in range(self.out_channels):
        #     channel_outputs = []
        #
        #     # 对每个输入位置进行滑动窗口
        #     for i in range(0, h - self.kernel_size[0] + 1, self.conv.stride[0]):
        #         for j in range(0, w - self.kernel_size[1] + 1, self.conv.stride[1]):
        #             # 提取输入窗口
        #             window = x_stream[:, :, :, i:i + self.kernel_size[0], j:j + self.kernel_size[1]]
        #
        #             # 对每个流位置进行卷积
        #             conv_results = []
        #             for s in range(self.stream_length):
        #                 # 获取当前流的窗口
        #                 window_s = window[:, s, :, :, :]
        #
        #                 '''
        #                 随机卷积（简化为点乘）
        #                 实际硬件中会使用AND门阵列
        #                 '''
        #                 weight_stream = self.stochastic_core.to_stochastic(
        #                     self.conv.weight[out_ch].view(1, -1)
        #                 ).view(-1, *self.kernel_size)
        #
        #                 # 随机乘法（AND操作）
        #                 product = self.stochastic_core.stochastic_multiply(
        #                     window_s, weight_stream
        #                 )
        #
        #                 # 求和
        #                 conv_result = product.sum()
        #                 conv_results.append(conv_result)
        #
        #             channel_outputs.append(torch.stack(conv_results).mean())
        #
        #     outputs.append(torch.tensor(channel_outputs))
        #
        # output = torch.stack(outputs, dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1) # 组合所有输出通道
        #
        # # 添加偏置
        # if self.conv.bias is not None:
        #     bias_stream = self.stochastic_core.to_stochastic(self.conv.bias)
        #     output = output + bias_stream.mean(dim=1)
        #
        # return output


class StochasticLinear(StochasticModule):
    """随机计算全连接层

    把前面提取的特征组合起来做决策
    """

    def __init__(self, in_features, out_features, bias=True, stream_length=1000):
        super().__init__(stream_length)

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features # 输入特征数
        self.out_features = out_features # 输出特征数

    def forward(self, x):
        if not self.use_stochastic or not self.training:
            # 训练时或未启用随机计算：使用普通全连接
            return self.linear(x)

        batch_size = x.shape[0] # 一次处理样本数

        # 将权重和输入转换为概率
        weight_prob = torch.sigmoid(self.linear.weight)  # [out, in] 权重转概率
        x_prob = torch.sigmoid(x)  # [batch, in] 输入转概率

        # 为每个样本生成输出
        outputs = []
        for b in range(batch_size):
            sample_outputs = []

            for out_idx in range(self.out_features): # 对每个输出神经元计算
                stream_sum = 0 # 累加器

                for _ in range(self.stream_length):  # 用多个随机流计算（蒙特卡洛方法）
                    # 生成随机比特
                    input_bits = (torch.rand(self.in_features) < x_prob[b]).float()
                    weight_bits = (torch.rand(self.in_features) < weight_prob[out_idx]).float()

                    # 随机乘法（AND）
                    product = input_bits * weight_bits
                    stream_sum += product.sum().item()

                # 平均值作为输出
                avg_output = stream_sum / self.stream_length
                sample_outputs.append(avg_output)

            outputs.append(torch.tensor(sample_outputs))

        output = torch.stack(outputs)

        # 添加偏置
        if self.linear.bias is not None:
            output = output + self.linear.bias

        return output


class StochasticBatchNorm2d(StochasticModule):
    """随机计算批量归一化（近似实现）"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, stream_length=1000):
        super().__init__(stream_length)

        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum) # 普通批量归一化层（用于训练）

    def forward(self, x):
        if not self.use_stochastic or not self.training: # 训练时或未启用随机计算：使用普通批量归一化
            return self.bn(x)

        # 简化随机批量归一化
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)

        # 归一化（在随机计算中使用近似）
        x_normalized = (x - mean) / torch.sqrt(var + self.bn.eps)

        # 缩放和平移
        output = x_normalized * self.bn.weight.view(1, -1, 1, 1) + self.bn.bias.view(1, -1, 1, 1)

        return output