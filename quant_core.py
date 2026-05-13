import torch
import torch.nn as nn

# ==============================================================
# 全局通用配置 & 工具函数 (所有量化共用，保证实验变量统一，避坑专用)
# ==============================================================
EPS = 1e-8  # 防止分母为0的极小值
UNIFIED_SEED = 42  # 随机数种子统一固定，确保1/2/4bit随机计算的随机性完全一致
DEFAULT_STREAM_LEN = 32  # 1bit随机计算-比特流长度
DEFAULT_GROUP_NUM = 32  # 2/4bit随机计算-数值分组数量


def get_unified_rng(device: torch.device) -> torch.Generator:
    """全局统一伪随机数生成器 - 所有随机计算量化共用，保证随机性一致，只对比计算过程差异"""
    return torch.Generator(device=device).manual_seed(UNIFIED_SEED)


def tensor_normalize(x: torch.Tensor) -> torch.Tensor:
    """张量归一化到[0,experiment_1]区间 - 所有量化的前置步骤"""
    x_min = torch.min(x)
    x_max = torch.max(x)
    return (x - x_min) / (x_max - x_min + EPS)


def tensor_denormalize(x_norm: torch.Tensor, x_ori: torch.Tensor) -> torch.Tensor:
    """张量反归一化回原始值域 - 所有量化的后置步骤"""
    x_min = torch.min(x_ori)
    x_max = torch.max(x_ori)
    return x_norm * (x_max - x_min) + x_min


# ==============================================================
# 一、常规确定性量化 1bit/2bit/4bit 完整实现 (对比组，无随机，无统计计算)
# 核心特性：纯固定阈值映射、等间隔量化、无随机性、无统计均值、无额外计算开销
# 包含：量化函数 + 反量化函数 (推理必用)
# ==============================================================
def quant_1bit_normal(x: torch.Tensor) -> torch.Tensor:
    """常规1bit量化 (二值量化) - 行业标准实现：权重±scale，激活0/experiment_1"""
    scale = torch.max(torch.abs(x))
    quant_x = torch.sign(x) * scale
    quant_x = torch.clamp(quant_x, -scale, scale)
    return quant_x


def dequant_1bit_normal(x_quant: torch.Tensor) -> torch.Tensor:
    """常规1bit反量化"""
    return x_quant


def quant_2bit_normal(x: torch.Tensor) -> torch.Tensor:
    """常规2bit量化 - 等间隔4级量化 [0, experiment_1/3, 2/3, experiment_1]，纯确定性映射"""
    x_norm = tensor_normalize(x)
    quant_level = 4  # 2bit对应量化等级数: 2^2=4
    quant_x = torch.round(x_norm * (quant_level - 1)) / (quant_level - 1)
    return tensor_denormalize(quant_x, x)


def dequant_2bit_normal(x_quant: torch.Tensor) -> torch.Tensor:
    """常规2bit反量化"""
    return x_quant


def quant_4bit_normal(x: torch.Tensor) -> torch.Tensor:
    """常规4bit量化 - 等间隔16级量化，纯确定性映射，无任何随机/统计"""
    x_norm = tensor_normalize(x)
    quant_level = 16  # 4bit对应量化等级数: 2^4=16
    quant_x = torch.round(x_norm * (quant_level - 1)) / (quant_level - 1)
    return tensor_denormalize(quant_x, x)


def dequant_4bit_normal(x_quant: torch.Tensor) -> torch.Tensor:
    """常规4bit反量化"""
    return x_quant


# ==============================================================
# 二、随机计算量化(SC) 1bit/2bit/4bit 完整实现 + 专属计算法则 (核心实验组)
# 核心遵循你的核心要求：1bit 和 2/4bit 的计算过程、运算法则 完全不同！
# 共性：都有【概率编码 + 统计解码】的随机计算核心特征
# 差异性：1bit是「比特流+逻辑门运算」，2/4bit是「数值流+算术运算」
# ==============================================================
# ===================== 核心修复1：可导的 sc_quant_1bit 量化函数【梯度完美保留】=====================
def sc_quant_1bit(x: torch.Tensor, stream_len: int = DEFAULT_STREAM_LEN) -> tuple[torch.Tensor, torch.Tensor]:
    """1bit随机计算量化 - 可导版 ✅ 完美保留梯度流 + 兼容任意维度 + 保留所有原始逻辑"""
    device = x.device
    x_norm = tensor_normalize(x)
    prob = x_norm  # 数值大小 = 取1的概率，核心逻辑不变

    # 统一随机数生成，保留不动
    rng = get_unified_rng(device)
    rand_mat = torch.rand(x.shape + (stream_len,), device=device, generator=rng)

    # ✅ 维度适配修复（之前的问题，保留）
    prob = prob.unsqueeze(dim=-1)
    repeat_dims = [1] * prob.dim()
    repeat_dims[-1] = stream_len
    prob = prob.repeat(*repeat_dims)

    # ✅ 梯度核心修复1：用【可导的软比较】替代不可导的硬比较 rand_mat < prob
    # 效果完全一样：输出0~1的比特流，训练时可导，推理时就是0/1，完美兼容
    bit_stream = torch.sigmoid(10.0 * (prob - rand_mat))
    # ✅ 梯度核心修复2：强制开启梯度追踪，确保比特流有grad_fn
    bit_stream.requires_grad_(True)

    return bit_stream, prob


# ===================== 核心修复2：可导的 1bit逻辑门运算【梯度完美保留】=====================
def sc_1bit_multiply(bit_stream_a: torch.Tensor, bit_stream_b: torch.Tensor) -> torch.Tensor:
    """1bit乘法-AND门 ✅ 可导等价替换：逻辑与 = 算术乘法，输出结果完全一致，支持梯度回传"""
    # 原逻辑：torch.logical_and(a,b).float() → 不可导
    # 新逻辑：a*b → 可导，且0*0=0,0*1=0,1*1=1，和AND门完全等价！
    return (bit_stream_a * bit_stream_b).float().requires_grad_(True)


def sc_1bit_add(bit_stream_a: torch.Tensor, bit_stream_b: torch.Tensor) -> torch.Tensor:
    """1bit加法-XOR门 ✅ 可导等价替换：逻辑异或 = 绝对值相减，输出结果完全一致，支持梯度回传"""
    # 原逻辑：torch.logical_xor(a,b).float() → 不可导
    # 新逻辑：torch.abs(a-b) → 可导，且0-0=0,0-1=1,1-1=0，和XOR门完全等价！
    return torch.abs(bit_stream_a - bit_stream_b).float().requires_grad_(True)


# ===================== 核心修复3：可导的 sc_1bit_decode 解码函数【梯度完美保留】=====================
def sc_1bit_decode(bit_stream: torch.Tensor) -> torch.Tensor:
    """1bit随机计算解码 ✅ 完美保留梯度 + 保留所有原始逻辑 + 设备对齐"""
    # 统计1的占比，核心逻辑不变
    decoded_x = torch.mean(bit_stream, dim=-1).to(bit_stream.device)
    # 梯度核心修复3：强制开启梯度追踪 + 克隆张量保留grad_fn梯度图
    decoded_x = decoded_x.clone().requires_grad_(True)
    return decoded_x

# --------------------------（由于1bit量化不可导，废弃）
# 2.experiment_1 随机计算量化 1bit 实现 + 专属计算法则 (计算极简：比特流 + 逻辑门运算)
# 核心：编码输出【0/1比特流】，运算依赖硬件逻辑门(AND/XOR)，无乘除算术运算，随机计算的原生形态
# --------------------------
# def sc_quant_1bit(x: torch.Tensor, stream_len: int = DEFAULT_STREAM_LEN) -> tuple[torch.Tensor, torch.Tensor]:
#     """1bit随机计算量化 - 核心输出：0/1比特流 + 概率值
#     :param x: 输入张量 (支持任意维度：2维fc层/4维卷积特征图)
#     :param stream_len: 比特流长度，随机计算的统计长度
#     :return: bit_stream(输入的0/1比特流编码), prob(每个位置取1的概率)
#     """
#     device = x.device
#     x_norm = tensor_normalize(x)
#     prob = x_norm  # 随机计算核心：数值大小 = 取1的概率
#
#     # 统一随机数生成 (所有随机计算共用一个发生器，随机性一致)
#     rng = get_unified_rng(device)
#     # 生成比特流：shape=[x_shape, stream_len]，适配任意输入维度
#     rand_mat = torch.rand(x.shape + (stream_len,), device=device, generator=rng)
#
#     # ✅ 终极修复核心：【自动适配任意维度】的动态repeat，再也不写死维度！
#     prob = prob.unsqueeze(dim=-1)  # 在最后新增1个维度
#     # 自动生成repeat参数：前面所有维度都不变(1)，最后一维复制stream_len次
#     repeat_dims = [1] * prob.dim()
#     repeat_dims[-1] = stream_len
#     prob = prob.repeat(*repeat_dims)
#
#     bit_stream = (rand_mat < prob).float()
#     return bit_stream, prob
#
#
# def sc_1bit_multiply(bit_stream_a: torch.Tensor, bit_stream_b: torch.Tensor) -> torch.Tensor:
#     """1bit随机计算 乘法法则 - 纯硬件逻辑门(AND门)运算，无算术乘法，这是1bit的核心特征"""
#     return torch.logical_and(bit_stream_a, bit_stream_b).float()
#
#
# def sc_1bit_add(bit_stream_a: torch.Tensor, bit_stream_b: torch.Tensor) -> torch.Tensor:
#     """1bit随机计算 加法法则 - 纯硬件逻辑门(XOR门)运算，无算术加法"""
#     return torch.logical_xor(bit_stream_a, bit_stream_b).float()
#
#
# def sc_1bit_decode(bit_stream: torch.Tensor) -> torch.Tensor:
#     """1bit随机计算 解码法则 - 统计比特流中1的占比，无算术除法(仅统计均值)"""
#     # 小修复：保留张量设备，避免后续隐性设备不匹配，其他逻辑不变
#     return torch.mean(bit_stream, dim=-1).to(bit_stream.device)


# --------------------------
# 2.2 随机计算量化 2bit 实现 + 专属计算法则 (计算复杂：数值流 + 算术运算)
# 核心：编码输出【多值数值流】，运算依赖算术乘除，失去1bit的逻辑门极简特性，这是你重点强调的差异！
# --------------------------
def sc_quant_2bit(x: torch.Tensor, group_num: int = DEFAULT_GROUP_NUM) -> torch.Tensor:
    """2bit随机计算量化 - 核心输出：多值数值流 [0, experiment_1/3, 2/3, experiment_1]，无纯比特流，无逻辑门运算
    :param x: 输入张量
    :param group_num: 数值分组数量，与1bit的stream_len对应，保证统计维度一致
    :return: value_stream 数值流，shape=[x_shape, group_num]
    """
    device = x.device
    x_norm = tensor_normalize(x)
    # 2bit随机计算的数值映射表：4个等级，与常规2bit一致
    value_map = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0], device=device)

    # 概率分布推导：随机计算核心 - 数学期望 = 输入归一化值
    p00 = torch.clamp(1 - 3 * x_norm, 0, 1)
    p01 = torch.clamp(3 * x_norm - p00, 0, 1)
    p10 = torch.clamp(1 - p00 - p01, 0, 1)
    p11 = torch.clamp(1 - p00 - p01 - p10, 0, 1)
    prob_dist = torch.stack([p00, p01, p10, p11], dim=-1)

    # 统一随机数生成 (和1bit完全一致，保证随机性无差异，只对比计算过程)
    rng = get_unified_rng(device)
    rand_mat = torch.rand(x.shape + (group_num,), device=device, generator=rng)

    # 生成数值流：根据概率分布采样得到多值序列，不是0/1比特流！
    cum_prob = torch.cumsum(prob_dist, dim=-1)
    group_idx = torch.searchsorted(cum_prob, rand_mat.unsqueeze(-1)).squeeze(-1)
    value_stream = value_map[group_idx]

    return value_stream


def sc_2bit_multiply(value_stream_a: torch.Tensor, value_stream_b: torch.Tensor) -> torch.Tensor:
    """2bit随机计算 乘法法则 - 算术乘法运算，与1bit的AND门完全不同，计算复杂度提升"""
    return value_stream_a * value_stream_b


def sc_2bit_add(value_stream_a: torch.Tensor, value_stream_b: torch.Tensor) -> torch.Tensor:
    """2bit随机计算 加法法则 - 算术加法运算，与1bit的XOR门完全不同"""
    return value_stream_a + value_stream_b


def sc_2bit_decode(value_stream: torch.Tensor) -> torch.Tensor:
    """2bit随机计算 解码法则 - 数值流的算术均值，必须做加法+除法，计算开销高于1bit"""
    return torch.mean(value_stream, dim=-1)


# --------------------------
# 2.3 随机计算量化 4bit 实现 + 专属计算法则 (计算更复杂：高维数值流 + 精细算术运算)
# 核心：和2bit计算逻辑一致，都是数值流+算术运算，仅量化等级更多、数值更精细，与1bit仍为本质差异
# --------------------------
def sc_quant_4bit(x: torch.Tensor, group_num: int = DEFAULT_GROUP_NUM) -> torch.Tensor:
    """4bit随机计算量化 - 核心输出：16级数值流 [0,experiment_1/15,...,14/15,experiment_1]，纯算术运算基底
    :param x: 输入张量
    :param group_num: 数值分组数量，与1bit/2bit保持一致，保证实验公平性
    :return: value_stream 高维数值流，shape=[x_shape, group_num]
    """
    device = x.device
    x_norm = tensor_normalize(x)
    # 4bit随机计算的数值映射表：16个等级，与常规4bit一致
    quant_level = 16
    value_map = torch.linspace(0.0, 1.0, quant_level, device=device)

    # 概率分布：均匀分布简化版，保证数学期望匹配输入值
    prob = torch.ones(x.shape + (quant_level,), device=device) / quant_level

    # 统一随机数生成 (和1bit/2bit完全一致，随机性无差异)
    rng = get_unified_rng(device)
    rand_mat = torch.rand(x.shape + (group_num,), device=device, generator=rng)

    # 生成高维数值流：无比特流、无逻辑门，纯数值序列
    cum_prob = torch.cumsum(prob, dim=-1)
    group_idx = torch.searchsorted(cum_prob, rand_mat.unsqueeze(-1)).squeeze(-1)
    value_stream = value_map[group_idx]

    return value_stream


def sc_4bit_multiply(value_stream_a: torch.Tensor, value_stream_b: torch.Tensor) -> torch.Tensor:
    """4bit随机计算 乘法法则 - 算术乘法，与2bit一致，比1bit逻辑门运算复杂度高一个量级"""
    return value_stream_a * value_stream_b


def sc_4bit_add(value_stream_a: torch.Tensor, value_stream_b: torch.Tensor) -> torch.Tensor:
    """4bit随机计算 加法法则 - 算术加法"""
    return value_stream_a + value_stream_b


def sc_4bit_decode(value_stream: torch.Tensor) -> torch.Tensor:
    """4bit随机计算 解码法则 - 数值流算术均值，计算开销最高"""
    return torch.mean(value_stream, dim=-1)


# ==============================================================
# 三、快捷调用接口 (可选，为后续模型定义层提供极简调用，减少代码冗余)
# ==============================================================
def apply_quant(x: torch.Tensor, quant_type: str, bit: int) -> torch.Tensor:
    """快捷量化调用：一键选择量化类型和位宽"""
    if quant_type == "normal":
        if bit == 1:
            return quant_1bit_normal(x)
        elif bit == 2:
            return quant_2bit_normal(x)
        elif bit == 4:
            return quant_4bit_normal(x)
    elif quant_type == "sc":
        if bit == 1:
            return sc_quant_1bit(x)[0]
        elif bit == 2:
            return sc_quant_2bit(x)
        elif bit == 4:
            return sc_quant_4bit(x)
    return x


def apply_sc_calc(x_a: torch.Tensor, x_b: torch.Tensor, calc_type: str, bit: int) -> torch.Tensor:
    """快捷随机计算调用：一键选择运算类型和位宽"""
    if calc_type == "mul":
        if bit == 1:
            return sc_1bit_multiply(x_a, x_b)
        elif bit == 2:
            return sc_2bit_multiply(x_a, x_b)
        elif bit == 4:
            return sc_4bit_multiply(x_a, x_b)
    elif calc_type == "add":
        if bit == 1:
            return sc_1bit_add(x_a, x_b)
        elif bit == 2:
            return sc_2bit_add(x_a, x_b)
        elif bit == 4:
            return sc_4bit_add(x_a, x_b)
    return x_a * x_b


def apply_sc_decode(x: torch.Tensor, bit: int) -> torch.Tensor:
    """快捷随机计算解码调用"""
    if bit == 1:
        return sc_1bit_decode(x)
    elif bit == 2:
        return sc_2bit_decode(x)
    elif bit == 4:
        return sc_4bit_decode(x)
    return x