"""
Mini-GPT 模型 — 一个"麻雀虽小，五脏俱全"的自回归语言模型

设计要点：
- Pre-LN Transformer 架构（比 Post-LN 训练更稳定）
- 可学习的 Token Embedding 和 Position Embedding
- Causal Mask 确保自回归生成（不能看到未来 token)
- LM Head 共享 Token Embedding 权重（weight tying，减少参数）

参考：
- GPT-2 论文: "Language Models are Unsupervised Multitask Learners"
- The Annotated Transformer: https://nlp.seas.harvard.edu/annotated-transformer/
"""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention, FeedForward, create_causal_mask


@dataclass
class MiniGPTConfig:
    """Mini-GPT 模型配置"""
    vocab_size: int = 8000        # 词表大小
    d_model: int = 256            # 模型隐藏层维度
    n_heads: int = 8              # 注意力头数
    n_layers: int = 4             # Transformer block 层数
    d_ff: int = 1024              # 前馈网络中间层维度
    max_seq_len: int = 128        # 最大上下文长度
    dropout: float = 0.1          # dropout 概率
    tie_weights: bool = True      # 是否共享 Token Embedding 和 LM Head 权重
    label_smoothing: float = 0.0  # 标签平滑（0.0=不使用，建议 0.1）


class TransformerBlock(nn.Module):
    """
    Transformer Block — Pre-LN 架构

    结构：
        x = x + MHA(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Pre-LN 比 Post-LN 的优势：
    - 训练更稳定，不需要学习率 warmup 也能收敛
    - 梯度传播更平滑，适合深层网络
    - 现代 LLM（GPT-2/3、LLaMA）均采用此架构
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:           输入，形状 (batch, seq_len, d_model)
            causal_mask: 因果掩码

        Returns:
            输出，形状 (batch, seq_len, d_model)
        """
        # 子层 1: Multi-Head Attention + 残差连接
        residual = x
        x = self.ln1(x)
        x = self.mha(x, mask=causal_mask)
        x = self.dropout(x)
        x = x + residual

        # 子层 2: Feed-Forward Network + 残差连接
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + residual

        return x


class MiniGPT(nn.Module):
    """
    Mini-GPT — 小型自回归语言模型

    完整结构：
        Token Embedding + Position Embedding
        ↓
        N × TransformerBlock (Pre-LN)
        ↓
        LayerNorm
        ↓
        LM Head (线性层 → vocab_size)
    """

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config

        # Token 嵌入 — 将 token ID 映射到 d_model 维向量
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # 位置嵌入 — 学习每个位置的位置编码（而非固定正弦编码）
        # GPT 系列使用可学习位置嵌入，让模型自己学习最优的位置表示
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # 最终 LayerNorm
        self.ln_final = nn.LayerNorm(config.d_model)

        # LM Head — 将隐藏向量映射回词表概率分布
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying — 共享 token embedding 和 lm_head 的权重
        # 这是 GPT 系列的标准做法，可以：
        # 1. 减少参数量（约 (vocab_size × d_model) 个参数）
        # 2. 提升泛化能力（输入和输出空间共享表示）
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # 注册 causal mask 作为 buffer（不参与梯度，但会随模型移动设备）
        causal_mask = create_causal_mask(config.max_seq_len)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """权重初始化 — 使用 GPT-2 的初始化策略"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 线性层：正态分布，标准差 = 0.02
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # 嵌入层：正态分布，标准差 = 0.02
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm：权重初始化为 1，偏置初始化为 0
                if module.weight is not None:
                    torch.nn.init.ones_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None) -> dict:
        """
        前向传播

        Args:
            input_ids: token ID 序列，形状 (batch, seq_len)
            targets:   可选，目标 token ID 序列（用于计算 loss），
                       形状 (batch, seq_len)，通常为 input_ids 右移一位

        Returns:
            dict:
                "logits": 形状 (batch, seq_len, vocab_size)
                "loss":   如果提供了 targets，返回交叉熵 loss；否则为 None
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token Embedding + Position Embedding
        # position_ids: [0, 1, 2, ..., seq_len-1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        x = self.dropout(token_emb + pos_emb)

        # 获取当前序列长度对应的 causal mask
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]

        # 通过所有 Transformer Blocks
        for block in self.blocks:
            x = block(x, causal_mask)

        # 最终 LayerNorm
        x = self.ln_final(x)

        # LM Head → logits
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        # 计算 loss
        loss = None
        if targets is not None:
            # 将 (batch, seq_len, vocab_size) 展平为 (batch*seq_len, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,  # 忽略 PAD token 的位置
                label_smoothing=self.config.label_smoothing,  # 标签平滑，防止过拟合
            )

        return {"logits": logits, "loss": loss}

    def get_num_params(self) -> dict:
        """
        统计模型参数数量

        Returns:
            {"total": 总参数, "trainable": 可训练参数}
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        自回归文本生成

        Args:
            input_ids:      初始 token 序列，形状 (1, seq_len)
            max_new_tokens: 最大生成 token 数
            temperature:    温度参数（>1 更随机，<1 更确定）
            top_k:          Top-K 采样参数
            top_p:          Top-P（nucleus）采样参数

        Returns:
            生成的完整 token 序列，形状 (1, seq_len + generated_len)
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # 截取不超过 max_seq_len 的上下文
            if generated.size(1) > self.config.max_seq_len:
                context = generated[:, -self.config.max_seq_len:]
            else:
                context = generated

            # 前向传播获取 logits
            outputs = self.forward(context)
            logits = outputs["logits"][:, -1, :]  # 只取最后一个位置的预测

            # 1. Temperature 缩放
            if temperature != 1.0 and temperature > 0:
                logits = logits / temperature

            # 2. Top-K 过滤 — 只保留概率最高的 K 个 token
            if top_k is not None and top_k > 0:
                top_k = min(top_k, logits.size(-1))
                topk_values, _ = torch.topk(logits, top_k, dim=-1)
                threshold = topk_values[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float("-inf")

            # 3. Top-P (Nucleus) 过滤 — 保留累积概率达到 p 的最小 token 集合
            if top_p is not None and top_p > 0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # 移除累积概率超过 p 的 token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False  # 至少保留一个 token
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # 4. 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接到生成序列
            generated = torch.cat([generated, next_token], dim=-1)

            # 如果生成 EOS，停止
            if next_token.item() == 3:  # EOS token ID
                break

        return generated
