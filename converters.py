"""
随机比特流转换器
实现实数到随机比特流的转换
"""
import torch
from typing import Tuple


class StochasticConverter:
    """随机转换器"""

    def __init__(self, core):
        self.core = core

    def real_to_stochastic(self, x: torch.Tensor) -> torch.Tensor:
        """
        实数转换为随机比特流

        Args:
            x: 输入实数张量 [batch, ...]

        Returns:
            随机比特流 [batch, stream_length, ...]
        """
        batch_size = x.shape[0]
        original_shape = x.shape[1:]

        # 转换为概率
        x_prob = self.core.value_to_probability(x)

        # 生成随机比特流
        stream_shape = (batch_size, self.core.stream_length, *original_shape)
        random_bits = self.core._generate_random_bits(stream_shape).to(x.device)

        # 生成随机比特流
        x_prob_expanded = x_prob.unsqueeze(1).expand(-1, self.core.stream_length,
                                                     *[-1] * len(original_shape))
        stochastic_stream = (random_bits < x_prob_expanded).float()

        return stochastic_stream

    def stochastic_to_real(self, stream: torch.Tensor) -> torch.Tensor:
        """
        随机比特流转换为实数

        Args:
            stream: 随机比特流 [batch, stream_length, ...]

        Returns:
            实数张量 [batch, ...]
        """
        # 计算比特流的均值
        prob = stream.mean(dim=1)

        # 转换回值
        value = self.core.probability_to_value(prob)

        return value