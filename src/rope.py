import torch
from typing import Tuple

# 注意：此处的dim应为 dim//n_head，因为我们是对每个head进行旋转嵌入
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    @param dim: head_dim
    @param end: max_seq_len
    @return freqs_cos: cos of freqs, shape of (end, dim // 2)
    @return freqs_sin: sin of freqs, shape of (end, dim // 2)
    """
    # torch.arange(0, dim, 2)[: (dim // 2)].float()生成了一个从0开始，步长为2的序列，长度为dim的一半
    # 然后每个元素除以dim，再得到theta的幂，最后取倒数，得到频率
    # 这一步是为了生成适合旋转嵌入的频率
    freqs = 1.0 / (theta ** torch.arange(0, dim ,2)[:(dim // 2)].float() / dim)
    
    # 生成一个从0到end的序列，长度为end
    # end通常是序列的最大长度
    t = torch.arange(end, device=freqs.device)
    
    # 计算外积，得到一个二维矩阵，每一行是t的元素乘以freqs的元素
    # 每一行是时间序列 t 的元素乘以频率序列 freqs 的元素
    freqs = torch.outer(t, freqs).float()
    
    # 计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs)
    
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs) 
    return freqs_cos, freqs_sin

# 调整 freqs_cis 的形状，使其在进行广播操作时与 x 的维度对齐，从而能够进行正确的张量运算
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    @param freqs_cis: [seq_len, head_dim // 2]
    @param x: usually [batch, seq_len, num_xxx_heads, head_dim // 2] after changing to complex num
              careless about being real/imaginary part because use part of its shape only
    @return freqs_cis: [1, seq_len, 1, head_dim // 2]
    """
    # 获取x的维度数
    ndim = x.ndim

    # 断言，确保1在x的维度范围内
    assert ndim > 1 
        
    # 断言，确保freqs_cis的形状与x的第二维和最后一维相同
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    
    # 构造一个新的形状，除了第二维和最后一维，其他维度都为1，这样做是为了能够将freqs_cis与x进行广播操作
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d, in enumerate(x.shape)]
    
    # 将freqs_cis调整为新的形状，并返回
    freqs_cis = torch.reshape(freqs_cis, shape)
    return freqs_cis

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    @param xq: (bsz, seq_len, num_q_heads, head_dim)
    @param xk: (bsz, seq_len, num_kv_heads, head_dim)
    @param freqs_cos: (seq_len, head_dim // 2)
    @param freqs_sin: (seq_len, head_dim // 2)
    @return xq_out: the same shape as xq
    @return xk_out: the same shape as xk
    """

    # 将查询和键张量转换为浮点数，并重塑形状以分离实部和虚部
    # 为什么要转换成 float：一般训练时用 fp16/bf16 节省显存。但是计算 rope 时还用
    # 就会导致数值溢出，损失精度。这时一般转换成 float 计算完再转回去
    # reshape 的目的是为了让一个张量的 head_dim -> 两个 head_dim // 2，分离实部和虚部
    # TODO：学习为什么这样能分离的数学原理，和相应的数学基础 
    # (bsz, seq_len, num_xxx_heads, head_dim // 2)
    # -1 意味着最后一个维度的总数不会发生变化，自动推导出
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(dim=-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(dim=-1)
    # print(f"shape of xq_r: {xq_r.shape}")  # [1, 50, 6, 24]

    # 重新塑形频率张量以进行广播
    # (1, seq_len, 1, head_dim // 2)
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 应用旋转，分别计算旋转后的实部和虚部
    # 实部计算公式：x * cos - y * sin
    # 虚部计算公式: x * sin + y * cos
    # 维度还是 (bsz, seq_len, num_xxx_heads, head_dim // 2)
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 将最后两个维度合并，并还原为原始张量的形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)  # 转回输入精度，节省显存

# Test rope
xq = torch.randn(1, 50, 6, 48) # bs, seq_len, dim//n_head, n_head_dim
xk = torch.randn(1, 50, 6, 48) # bs, seq_len, dim//n_head, n_head_dim

# 使用 precompute_freqs_cis 函数获取 sin和cos
cos, sin = precompute_freqs_cis(288 // 6, 50)
print(cos.shape, sin.shape)
xq_out, xk_out = apply_rotary_emb(xq, xk, cos, sin)
print(xq_out.shape, xk_out.shape)

# Expected output:
# torch.Size([50, 24]) torch.Size([50, 24])
# torch.Size([1, 50, 6, 48]), torch.Size([1, 50, 6, 48]))
# [NOTE] 26.1.7 pass