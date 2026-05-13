import torch

# SC量化核心配置
SC_STREAM_LEN = 32  # 比特流长度，可调：8/10/16
SC_SEED = 42
torch.manual_seed(SC_SEED)


# ===================== 1. 张量量化（特征/权重通用，推理时实时量化） =====================
# def sc_quantize(x: torch.Tensor) -> torch.Tensor:
#     """推理时实时将FP32张量（特征/权重）转为0/1比特流"""
#     x_min = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0] if x.dim() == 4 else \
#     x.min(dim=-1, keepdim=True)[0]
#     x_max = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] if x.dim() == 4 else \
#     x.max(dim=-1, keepdim=True)[0]
#     prob = (x - x_min) / (x_max - x_min + 1e-8)  # 映射为0~1概率
#
#     rng = torch.Generator(device=x.device)
#     rng.manual_seed(SC_SEED)
#     rand_mat = torch.rand(x.shape + (SC_STREAM_LEN,), device=x.device, generator=rng)
#
#     prob = prob.unsqueeze(-1)
#     repeat_dims = [1] * prob.dim()
#     repeat_dims[-1] = SC_STREAM_LEN
#     prob = prob.repeat(*repeat_dims)
#
#     return (rand_mat < prob).float()  # 纯0/1比特流
def sc_quantize(x: torch.Tensor) -> torch.Tensor:
    """推理实时量化：输出维度严格规范为 [B,C,H,W,SC_STREAM_LEN] (特征) / [C_out,C_in,k,k,SC_STREAM_LEN] (权重)
    保证：所有量化后的张量，最后一维永远是比特流长度，前面是原张量维度，彻底解决维度匹配问题
    """
    # 自适应min-max归一化，适配卷积权重(4D)、特征图(4D)、全连接权重(2D)
    x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # 修复关键：全局求min/max，不用传dim列表，彻底解决TypeError，归一化更稳定
    x_min = torch.min(x)
    x_max = torch.max(x)
    # dims = [i for i in range(x.dim()) if i not in [0]] if x.dim() == 4 else [i for i in range(x.dim())]
    # x_min = x.min(dim=dims, keepdim=True)[0]
    # x_max = x.max(dim=dims, keepdim=True)[0]
    prob = (x - x_min) / (x_max - x_min + 1e-8)  # 映射0~1概率

    # 生成随机矩阵：严格在【原张量维度后拼接比特流维度】
    rng = torch.Generator(device=x.device)
    rng.manual_seed(SC_SEED)
    rand_mat = torch.rand(x.shape + (SC_STREAM_LEN,), device=x.device, generator=rng)

    # 概率矩阵扩维：只扩最后一维，匹配随机矩阵
    prob = prob.unsqueeze(dim=-1).repeat(*[1] * x.dim(), SC_STREAM_LEN)

    # 纯逻辑比较，生成0/1比特流，维度绝对规范
    bit_stream = (rand_mat < prob).float()
    return bit_stream


# ===================== 2. SC逻辑门运算（替代浮点乘加） =====================
def sc_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SC乘法=纯AND门，替代所有浮点乘法，要求a/b可广播（维度规范后必满足）"""
    return torch.logical_and(a, b).float()

def sc_xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SC加法=纯XOR门，替代所有浮点加法"""
    return torch.logical_xor(a, b).float()


# ===================== 3. SC解码+配套算子（推理时还原特征） =====================
def sc_decode(bit_stream: torch.Tensor) -> torch.Tensor:
    """比特流解码 = 统计1的占比，还原特征判别信息"""
    return torch.mean(bit_stream, dim=-1).to(bit_stream.device)


def sc_avg_pool(bit_stream: torch.Tensor) -> torch.Tensor:
    """SC池化 = 空间维度均值统计"""
    return torch.mean(bit_stream, dim=[2, 3]) if bit_stream.dim() == 5 else bit_stream


def sc_relu(x: torch.Tensor) -> torch.Tensor:
    """SC激活 = 逻辑阈值过滤"""
    return torch.where(x > 0, x, torch.zeros_like(x))