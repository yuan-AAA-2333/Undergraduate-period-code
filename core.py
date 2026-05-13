"""
随机计算核心组件
实现随机比特流生成、转换等基础操作
"""
import torch
import numpy as np
from typing import Tuple, Optional, Dict
import math


class StochasticCore:
    """随机计算核心"""

    def __init__(self,
                 stream_length: int = 1000,
                 representation: str = 'unipolar',
                 random_generator: str = 'pseudo',
                 seed: Optional[int] = None):
        """
        初始化随机计算核心

        Args:
            stream_length: 随机比特流长度
            representation: 表示方式 ('unipolar'或'bipolar')
            random_generator: 随机数生成器类型 ('pseudo', 'lfsr', 'sobol')
            seed: 随机种子
        """
        self.stream_length = stream_length
        self.representation = representation
        self.random_generator = random_generator
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self._init_generators()

    def _init_generators(self):
        """初始化随机数生成器"""
        if self.random_generator == 'lfsr':
            self._init_lfsr()
        elif self.random_generator == 'sobol':
            self._init_sobol()

    def _init_lfsr(self):
        """初始化LFSR"""
        # 32-bit LFSR参数
        self.lfsr_state = 0xACE1
        self.lfsr_poly = 0xB4BCD35C  # 多项式

    def _generate_random_bits(self, shape: Tuple[int]) -> torch.Tensor:
        """生成随机比特"""
        if self.random_generator == 'lfsr':
            return self._lfsr_bits(shape)
        elif self.random_generator == 'sobol':
            return self._sobol_bits(shape)
        else:
            # 默认使用伪随机
            return torch.rand(shape)

    def _lfsr_bits(self, shape: Tuple[int]) -> torch.Tensor:
        """LFSR生成随机比特"""
        total_bits = np.prod(shape)
        bits = []

        for _ in range(total_bits):
            bit = self.lfsr_state & 1
            bits.append(bit)

            # 更新LFSR状态
            self.lfsr_state >>= 1
            if bit:
                self.lfsr_state ^= self.lfsr_poly

        return torch.tensor(bits, dtype=torch.float32).view(shape)

    def value_to_probability(self, x: torch.Tensor) -> torch.Tensor:
        """将值转换为概率"""
        if self.representation == 'unipolar':
            return torch.sigmoid(x).clamp(0, 1)
        else:  # bipolar
            return (torch.tanh(x) + 1) / 2

    def probability_to_value(self, p: torch.Tensor) -> torch.Tensor:
        """将概率转换为值"""
        if self.representation == 'unipolar':
            return p
        else:  # bipolar
            return 2 * p - 1

    def estimate_accuracy(self, original: torch.Tensor,
                          stochastic: torch.Tensor) -> Dict:
        """估计随机计算的精度"""
        mse = torch.mean((original - stochastic) ** 2)
        mae = torch.mean(torch.abs(original - stochastic))
        relative_error = torch.mean(torch.abs(original - stochastic) /
                                    (torch.abs(original) + 1e-8))

        return {
            'mse': mse.item(),
            'mae': mae.item(),
            'relative_error': relative_error.item()
        }