"""
注意力机制模块 — 从零实现 Transformer 的核心组件

包含：
- create_causal_mask: 生成因果掩码，确保自回归特性
- scaled_dot_product_attention: 缩放点积注意力
- MultiHeadAttention: 多头注意力
- FeedForward: 前馈神经网络

设计要点：
- 所有组件均从零实现，不使用 nn.MultiheadAttention 等高层 API
- 使用 Pre-LN 架构（LayerNorm 在子层之前而非之后）
- GELU 激活函数（GPT-2 标准）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    创建因果掩码（causal mask），确保位置 i 只能关注位置 ≤ i 的 token

    Args:
        seq_len: 序列长度
        device: 张量所在设备

    Returns:
        shape: (1, 1, seq_len, seq_len) 的上三角掩码，
        需要被遮蔽的位置为 True（值为 -inf 时对应的 mask 位置）
    """
    # 上三角矩阵（不含对角线），即 mask[i][j] = True 当 j > i
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    # 扩展为 (1, 1, seq_len, seq_len) 以适配 (batch, heads, seq, seq) 的注意力权重
    return mask.unsqueeze(0).unsqueeze(0)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None,
    dropout: nn.Dropout = None,
) -> torch.Tensor:
    """
    缩放点积注意力 — Transformer 的核心运算

    公式: Attention(Q, K, V) = softmax(Q·K^T / √d_k + mask) · V

    Args:
        query:  形状 (batch, n_heads, seq_len, d_k)
        key:    形状 (batch, n_heads, seq_len, d_k)
        value:  形状 (batch, n_heads, seq_len, d_v)
        mask:   可选，形状 (1, 1, seq_len, seq_len) 或可广播的布尔掩码
                True 的位置会被填充为 -inf
        dropout: 可选的 dropout 层，作用于注意力权重

    Returns:
        注意力输出，形状 (batch, n_heads, seq_len, d_v)
    """
    d_k = query.size(-1)

    # Q·K^T / √d_k — 缩放点积，防止内积过大导致 softmax 梯度消失
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 应用掩码 — 将需要遮蔽的位置置为 -inf，softmax 后这些位置概率趋近于 0
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # Softmax 归一化得到注意力权重
    attn_weights = F.softmax(scores, dim=-1)

    # Dropout 正则化
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # 加权求和
    output = torch.matmul(attn_weights, value)

    return output


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制

    将 d_model 拆分为 h 个头，每个头独立计算注意力，
    最后拼接并通过线性层投影回 d_model 维度。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型总维度（必须能被 n_heads 整除）
            n_heads: 注意力头数
            dropout: dropout 概率
        """
        super().__init__()
        assert d_model % n_heads == 0, f"d_model({d_model}) 必须能被 n_heads({n_heads}) 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # Q、K、V 的线性投影 — 合并为一个大矩阵以提高效率
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 (batch, seq_len, d_model) 拆分为 (batch, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, n_heads, seq_len, d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 (batch, n_heads, seq_len, d_k) 合并回 (batch, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()  # (batch, seq_len, n_heads, d_k)
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x:     输入，形状 (batch, seq_len, d_model)
            mask:  注意力掩码，形状 (1, 1, seq_len, seq_len)

        Returns:
            输出，形状 (batch, seq_len, d_model)
        """
        # 线性投影并拆分头
        Q = self._split_heads(self.q_proj(x))
        K = self._split_heads(self.k_proj(x))
        V = self._split_heads(self.v_proj(x))

        # 缩放点积注意力
        attn_out = scaled_dot_product_attention(Q, K, V, mask=mask, dropout=self.dropout)

        # 合并头并输出投影
        output = self.out_proj(self._merge_heads(attn_out))

        return output


class FeedForward(nn.Module):
    """
    前馈神经网络 — Transformer Block 中的第二个子层

    结构: Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model)
    使用 GELU 激活函数
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            d_ff:    前馈网络中间层维度（通常为 d_model 的 4 倍）
            dropout: dropout 概率
        """
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入，形状 (batch, seq_len, d_model)

        Returns:
            输出，形状 (batch, seq_len, d_model)
        """
        x = self.fc1(x)
        x = F.gelu(x)  # GELU 激活，比 ReLU 在 Transformer 中效果更好
        x = self.dropout(x)
        x = self.fc2(x)
        return x
