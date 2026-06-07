# Mini-GPT：小型 Transformer 文本生成系统

> 人工智能与大数据技术期末大作业 — 题目二

一个"麻雀虽小，五脏俱全"的小型文本生成模型。从零实现 GPT 风格 Transformer，在中文古诗词数据集上训练，支持命令行交互式文本续写。

## 环境要求

- Python >= 3.12
- PyTorch >= 2.12（CUDA 可选，CPU 也可运行）
- 操作系统：Linux 

## 快速开始

```bash
# 1. 安装依赖
uv sync
source .venv/bin/activate

# 2. 训练模型（快速测试，约 5 分钟）
python main.py --train --epochs 5 --d_model 128 --n_layers 2 --batch_size 32

# 3. 交互式生成
python main.py --generate

# 4. 对比四种采样策略
python main.py --compare "春风"
```

## 数据来源

项目默认使用 HuggingFace 的 [Million/Chinese-Poems](https://huggingface.co/datasets/Million/Chinese-Poems) 数据集（Parquet 格式），包含约 21.7 万首中文古诗词。`datasets` 库自动处理下载和解析。

首次运行训练时自动下载，也可提前缓存：

```python
from datasets import load_dataset
load_dataset("Million/Chinese-Poems")
```

## 运行模式

| 命令 | 说明 |
|------|------|
| `python main.py --train` | 训练模型 |
| `python main.py --generate` | 交互式命令行续写 |
| `python main.py --compare <文本>` | 对比四种采样策略效果 |

### 训练参数（可自定义）

```bash
python main.py --train \
    --epochs 20                \  # 训练轮数
    --batch_size 64            \  # 批次大小
    --d_model 256              \  # 隐藏层维度
    --n_layers 4               \  # Transformer 层数
    --n_heads 8                \  # 注意力头数
    --seq_len 128              \  # 上下文长度
    --lr 3e-4                  \  # 学习率
    --warmup_steps 500           # Warmup 步数
```

## 项目结构

```
AIbasic/
├── main.py                   # 主入口
├── pyproject.toml            # 依赖管理
├── src/
│   ├── tokenizer.py          # 字符级分词器
│   ├── attention.py          # Self-Attention / MHA / Causal Mask（从零实现）
│   ├── model.py              # Mini-GPT 模型（Pre-LN Transformer）
│   ├── trainer.py            # 训练循环 + Cosine Warmup + AdamW
│   ├── generator.py          # 文本生成 + 四种采样策略
│   └── utils.py              # 数据加载、可视化
├── chinese-poetry/           # 古诗词数据集
├── checkpoints/              # 模型保存
└── results/                  # 训练曲线等输出
```

## 模型架构

```
Token Embedding + Position Embedding
        ↓
 N × TransformerBlock (Pre-LN)
   ├── LayerNorm → Multi-Head Attention → +残差
   └── LayerNorm → FeedForward (GELU)  → +残差
        ↓
     LayerNorm → LM Head → 词表概率分布
```

### 默认超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `d_model` | 256 | 隐藏层维度 |
| `n_heads` | 8 | 多头注意力头数 |
| `n_layers` | 4 | Transformer Block 层数 |
| `d_ff` | 1024 | FFN 中间维度 |
| `max_seq_len` | 128 | 最大上下文长度 |
| `dropout` | 0.1 | Dropout 概率 |

参数量约 **5-8M**，可在普通笔记本上训练。

## 核心特性

### 从零实现的模块

- **Scaled Dot-Product Attention**：`Q·K^T / √d_k + mask → softmax → ·V`
- **Causal Mask**：上三角掩码，确保自回归生成时不能"看到未来"
- **Multi-Head Attention**：多个注意力头并行计算，捕捉不同子空间特征
- **Position Embedding**：可学习的位置编码（GPT 风格）
- **Weight Tying**：Token Embedding 与 LM Head 共享权重

### 四种采样策略

| 策略 | 参数 | 特点 |
|------|------|------|
| **Greedy Search** | — | 每步选概率最高的 token，生成确定但易重复 |
| **Temperature** | `temperature` | 缩放 logits 分布，>1 更随机，<1 更保守 |
| **Top-K** | `top_k` | 只从概率最高的 K 个候选中采样 |
| **Top-P (Nucleus)** | `top_p` | 动态选择累积概率达 p 的最小候选集 |

### 训练优化

- **AdamW** 优化器（解耦权重衰减）
- **Cosine Annealing** + **Linear Warmup** 学习率调度
- **梯度裁剪** 防止梯度爆炸
- **定期验证** 并生成样本文本观察训练进展

## 硬件参考

| 配置 | 推荐参数 | 预计训练时间 |
|------|---------|------------|
| CPU | `d_model=128, n_layers=2, batch_size=16` | 1-3 小时 |
| 笔记本 GPU（RTX 4060 等） | `d_model=256, n_layers=4, batch_size=64` | 10-30 分钟 |
| 无 GPU | `d_model=128, n_layers=2, batch_size=8` | 数小时 |

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP
- [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) — 中文古诗词数据库
- [Million/Chinese-Poems](https://huggingface.co/datasets/Million/Chinese-Poems) — HuggingFace 诗词数据集
