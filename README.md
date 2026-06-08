# Mini-GPT：小型 Transformer 文本生成系统

> 人工智能与大数据技术期末大作业 — 题目二

一个"麻雀虽小，五脏俱全"的小型文本生成模型。从零实现 GPT 风格 Transformer（不依赖 `nn.MultiheadAttention`），在中文古诗词数据集上训练，支持命令行交互式续写，并实现了基于分类标签的条件生成（微调模式）。训练后的模型在 checkpoints/tagged_test/best_model.pt

---

## 目录

- [Mini-GPT：小型 Transformer 文本生成系统](#mini-gpt小型-transformer-文本生成系统)
  - [目录](#目录)
  - [环境要求](#环境要求)
    - [环境安装](#环境安装)
  - [快速开始](#快速开始)
  - [项目结构](#项目结构)
  - [数据准备](#数据准备)
    - [步骤一：从零训练（原始数据源）](#步骤一从零训练原始数据源)
    - [步骤二：带标签微调（分类条件生成）](#步骤二带标签微调分类条件生成)
  - [数据清洗方法](#数据清洗方法)
    - [通用清洗（从零训练）](#通用清洗从零训练)
    - [体裁分类与标签重构（微调）](#体裁分类与标签重构微调)
    - [两种方法的关系](#两种方法的关系)
  - [训练命令](#训练命令)
    - [从零训练](#从零训练)
    - [微调（分类条件生成）](#微调分类条件生成)
    - [训练参数说明](#训练参数说明)
  - [推理与生成](#推理与生成)
    - [交互式续写](#交互式续写)
    - [分类条件生成](#分类条件生成)
    - [对比采样策略](#对比采样策略)
  - [模型架构](#模型架构)
    - [关键设计](#关键设计)
    - [从零实现的模块 (`src/attention.py`)](#从零实现的模块-srcattentionpy)
    - [默认模型规格](#默认模型规格)
  - [采样策略](#采样策略)
  - [硬件需求与运行时间](#硬件需求与运行时间)
    - [硬件兼容性](#硬件兼容性)
    - [运行时间参考](#运行时间参考)
    - [训练数据规模](#训练数据规模)
  - [参考资料](#参考资料)

---

## 环境要求

| 依赖 | 版本要求 | 说明 |
|------|---------|------|
| Python | ≥ 3.12 | 推荐 3.12+ |
| PyTorch | ≥ 2.12 | CPU 或 CUDA 均可 |
| datasets | ≥ 5.0 | HuggingFace 数据加载（从零训练模式） |
| matplotlib | ≥ 3.8 | 训练曲线绘制 |
| tqdm | ≥ 4.66 | 训练进度条 |
| opencc | ≥ 1.3 | 繁简转换（可选） |

**操作系统**: Linux（推荐 WSL2 / 原生 Linux）

### 环境安装

```bash
# 1. 克隆或进入项目目录
cd AIbasic

# 2. 使用 uv 安装依赖（推荐）
uv sync
source .venv/bin/activate

# 3. 验证安装
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

> **注意**: 如果无法使用 `uv`，也可用 pip 手动安装：
> ```bash
> pip install torch torchvision datasets matplotlib tqdm opencc
> ```

---

## 快速开始

```bash
# 1. 安装依赖
uv sync && source .venv/bin/activate

# 2. 预处理带标签数据（微调模式必需）
python -m src.preprocess_tagged

# 3. 训练模型（微调模式，快速测试约 10 分钟）
python main.py train --data-source tagged_txt --epochs 5 --d-model 128 --n-layers 2 --batch-size 32

# 4. 交互式条件生成
python main.py generate

# 5. 对比四种采样策略
python main.py compare "<五言绝句>"
```

---

## 项目结构

```
AIbasic/
├── main.py                        # 主入口（训练 / 生成 / 对比）
├── pyproject.toml                 # 依赖管理（uv）
├── train.txt                      # 原始训练集（36.7 万首）
├── test.txt                       # 原始测试集（368 首）
├── train_tagged.txt               # 带标签训练集（预处理生成）
├── test_tagged.txt                # 带标签测试集（预处理生成）
├── README.md                      # 本文件
├── src/
│   ├── tokenizer.py               # 字符级分词器（含分类标签 + 多字符 token 编码）
│   ├── attention.py               # Self-Attention / MHA / Causal Mask（从零实现）
│   ├── model.py                   # Mini-GPT 模型（Pre-LN Transformer + 自回归生成）
│   ├── trainer.py                 # 训练循环 + Cosine Warmup + AdamW
│   ├── generator.py               # 文本生成器 + 四种采样策略
│   ├── utils.py                   # 数据加载 / 清洗 / 可视化 / 两种 DataLoader
│   └── preprocess_tagged.py       # 体裁分类 + 标签数据重构（微调预处理）
├── checkpoints/                   # 模型保存目录
│   ├── vocab.json                 # 字符词表
│   ├── best_model.pt              # 最佳验证模型
│   └── final_model.pt             # 最终模型
└── results/                       # 训练曲线等输出
    └── training_curves.png
```

---

## 数据准备

项目使用了**两步训练方法**，先训练模型再进行微调：

### 步骤一：从零训练（原始数据源）

使用从 HuggingFace、本地 JSON 或目录加载原始古诗词数据，进行通用语言模型训练。

**数据来源**（`--data-source` 参数）：

| 来源 | 参数值 | 说明 |
|------|--------|------|
| HuggingFace | `huggingface` | 自动下载 [Million/Chinese-Poems](https://huggingface.co/datasets/Million/Chinese-Poems)（约 21.7 万首） |
| 本地目录 | `directory` | 遍历 `chinese-poetry/` 下所有 JSON 文件（约 40 万首） |
| 本地 JSON | `json` | 加载单个 JSON 文件 |

**数据加载流程**：

```
HuggingFace datasets 或 JSON 文件
       ↓ extract_poem_texts()     递归提取 paragraphs → 清洗括号注释 → 保留纯汉字+句读
       ↓ CharTokenizer.build_vocab()  构建字符级词表（min_freq 过滤低频字）
       ↓ split_text_into_sequences()  滑动窗口切固定长度序列
       ↓ TextDataset → DataLoader
```

### 步骤二：带标签微调（分类条件生成）

使用 `train.txt` / `test.txt` 的已清洗诗词数据进行体裁分类和条件生成微调。

**预处理步骤**：

```bash
# 运行体裁分类与标签重构（必须先执行）
python -m src.preprocess_tagged
```

**数据格式**：

每首诗重构为一行，格式如下：

```
<BOS><五言绝句>白日依山尽，黄河入海流。欲穷千里目，更上一层楼。<EOS>
<BOS><七言律诗>朝辞白帝彩云间，千里江陵一日还。两岸猿声啼不住，轻舟已过万重山。<EOS>
<BOS><词>绿云高髻，点翠匀红时世。月如眉。浅笑含双靥，低声唱小词。<EOS>
```

**分类标签**（9 种）：

| 标签 | 含义 | 训练集数量 | 占比 |
|------|------|-----------|------|
| `<五言绝句>` | 五言绝句（4句，每句5字） | 62,866 | 17.1% |
| `<五言律诗>` | 五言律诗（8句，每句5字） | 5,122 | 1.4% |
| `<五言古诗>` | 五言古诗（其他行数） | 71,192 | 19.4% |
| `<七言绝句>` | 七言绝句（4句，每句7字） | 70,318 | 19.2% |
| `<七言律诗>` | 七言律诗（8句，每句7字） | 2,446 | 0.7% |
| `<七言古诗>` | 七言古诗（其他行数） | 115,577 | 31.5% |
| `<词>` | 词（长短句） | 24,129 | 6.6% |
| `<曲>` | 曲（元曲等） | 8,730 | 2.4% |
| `<其他>` | 无法归类 | 6,356 | 1.7% |

**体裁分类算法** (`src/preprocess_tagged.py`):

1. 按中文标点（，。！？、；：）将每行拆分为短语
2. 统计所有短语的汉字数分布
3. 短语字数一致（≥85%）→ 五言/七言；再按行数分 绝句(4)/律诗(8)/古诗
4. 短语字数显著混用（第二多字数 ≥20%）→ 词
5. 标题含 `·`（词牌名）→ 优先判定为词
6. 标题含 `・` 或 `【` → 优先判定为曲

---

## 数据清洗方法

### 通用清洗（从零训练）

**位置**: `src/utils.py → clean_chinese_text()`

适用于从 HuggingFace 或 JSON 加载的原始古诗词数据。清洗规则：

1. **移除成对括号及内部内容**：覆盖 `《》「」『』""（）【】〈〉〔〕〖〗〘〙〚〛` 等所有中文括号
2. **保留纯汉字和四大句读符号**（，。！？）：支持 CJK 基本区（U+4E00-U+9FFF）、Extension A（U+3400-U+4DBF）、兼容汉字（U+F900-U+FAFF）以及 Extension B 及以上（U+20000+）的罕见字
3. **去除所有其他字符**：英文字母、数字、特殊符号、空白字符

```python
# 清洗示例
原始: "《静夜思》床前明月光，疑是地上霜。举头望明月，低头思故乡。"
清洗后: "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
```

### 体裁分类与标签重构（微调）

**位置**: `src/preprocess_tagged.py`

适用于已清洗的 `train.txt` / `test.txt`（`<|endoftext|>` 分隔格式）。处理流程：

1. **加载原始诗词**：按 `<|endoftext|>` 分割，提取标题行与正文行
2. **汉字提取**：使用与通用清洗相同的 Unicode 范围（支持 CJK 扩展区罕见字如 㶉𫛶）
3. **短语拆分**：按中文标点拆分为独立短语，统计每短语汉字数
4. **体裁判定**：基于短语字数一致性和标题特征进行九类划分
5. **标签重构**：拼接为 `<BOS><分类标签>正文<EOS>` 格式

### 两种方法的关系

```
原始数据源（HuggingFace/JSON）            已清洗数据（train.txt/test.txt）
        │                                        │
        ▼                                        ▼
clean_chinese_text()                    _extract_chinese_chars()
（移除括号，保留汉字+句读）              （纯汉字提取，支持CJK扩展）
        │                                        │
        ▼                                        ▼
  通用语言模型训练                       _classify_single_poem()
  （从零训练模式）                       （短语字数分析+标题特征）
                                               │
                                               ▼
                                        reconstruct_tagged_poem()
                                        （拼接BOS+标签+正文+EOS）
                                               │
                                               ▼
                                         分类条件生成微调
                                         （微调模式）
```

---

## 训练命令

### 从零训练

使用 HuggingFace 或本地 JSON 数据源，训练通用语言模型。

```bash
# 快速测试（约 5-10 分钟）
python main.py train \
    --data-source huggingface \
    --epochs 5 \
    --d-model 128 \
    --n-layers 2 \
    --batch-size 32

# 正式训练（约 20-30 分钟，需 GPU）
python main.py train \
    --data-source huggingface \
    --epochs 20 \
    --d-model 256 \
    --n-layers 4 \
    --n-heads 8 \
    --d-ff 1024 \
    --seq-len 128 \
    --batch-size 64 \
    --lr 3e-4 \
    --warmup-steps 500

# 使用本地 chinese-poetry 目录
python main.py train \
    --data-source directory \
    --data-path chinese-poetry \
    --epochs 20 \
    --d-model 256 \
    --n-layers 4

# CPU 模式（小模型）
python main.py train \
    --data-source huggingface \
    --epochs 10 \
    --d-model 128 \
    --n-layers 2 \
    --batch-size 8 \
    --seq-len 64
```

### 微调（分类条件生成）

使用带标签的预处理数据，训练体裁条件生成模型。

```bash
# 第一步：生成带标签数据（仅需执行一次）
python -m src.preprocess_tagged

# 第二步：快速测试（约 10 分钟）
python main.py train \
    --data-source tagged_txt \
    --epochs 5 \
    --d-model 128 \
    --n-layers 2 \
    --batch-size 32 \
    --seq-len 80 \
    --sample-prompt "<五言绝句>"

# 正式微调（约 1-2 小时，需 GPU）
python main.py train \
    --data-source tagged_txt \
    --epochs 20 \
    --d-model 256 \
    --n-layers 4 \
    --n-heads 8 \
    --batch-size 64 \
    --seq-len 128 \
    --sample-prompt "<五言绝句>"

# 使用自定义标签数据路径
python main.py train \
    --data-source tagged_txt \
    --train-path /path/to/train_tagged.txt \
    --test-path /path/to/test_tagged.txt \
    --epochs 20
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-source` | `tagged_txt` | 数据源类型：`huggingface` / `json` / `directory` / `tagged_txt` |
| `--train-path` | `train_tagged.txt` | 带标签训练集路径（仅 tagged_txt 模式） |
| `--test-path` | `test_tagged.txt` | 带标签测试集路径（仅 tagged_txt 模式） |
| `--epochs` | 20 | 训练轮数 |
| `--batch-size` | 64 | 批次大小 |
| `--d-model` | 256 | 隐藏层维度 |
| `--n-layers` | 4 | Transformer 层数 |
| `--n-heads` | 8 | 注意力头数 |
| `--d-ff` | 1024 | FFN 中间维度 |
| `--seq-len` | 128 | 最大序列长度 |
| `--lr` | 3e-4 | 学习率 |
| `--dropout` | 0.1 | Dropout 概率 |
| `--label-smoothing` | 0.1 | 标签平滑 |
| `--sample-prompt` | `<五言绝句>` | 训练中生成样本的 prompt |
| `--save-dir` | `checkpoints` | 模型保存目录 |

---

## 推理与生成

### 交互式续写

```bash
# 通用续写模式（适用于从零训练模型）
python main.py generate

# 指定模型和词表路径
python main.py generate \
    --model-path checkpoints/best_model.pt \
    --vocab-path checkpoints/vocab.json

# 单次生成
python main.py generate --prompt "春风" --max-new-tokens 80

# 交互命令
#   你 > 任意文本        → Top-K 40 采样续写
#   你 > compare 文本    → 对比四种采样策略
#   你 > quit / exit     → 退出
```

### 分类条件生成

使用带标签微调的模型，输入分类标签即可控制生成体裁：

```bash
python main.py generate

# 交互示例:
#   你 > <五言绝句>
#   续写 > 山中相送罢，日暮掩柴扉。春草年年绿，王孙归不归。
#
#   你 > <七言律诗>
#   续写 > 剑外忽传收蓟北，初闻涕泪满衣裳。却看妻子愁何在，漫卷诗书喜欲狂。
#
#   你 > <词>
#   续写 > 庭院深深深几许，杨柳堆烟，帘幕无重数。玉勒雕鞍游冶处，楼高不见章台路。
#
#   你 > <曲>
#   续写 > 枯藤老树昏鸦，小桥流水人家，古道西风瘦马。夕阳西下，断肠人在天涯。
```

**可用分类标签**：`<五言绝句>`, `<五言律诗>`, `<五言古诗>`, `<七言绝句>`, `<七言律诗>`, `<七言古诗>`, `<词>`, `<曲>`, `<其他>`

> **原理**：训练时每个序列以 `<BOS><分类标签>` 开头，模型在自回归训练中学会了 `P(正文 | <BOS>, <分类标签>)` 的条件概率分布。推理时只需输入 `<分类标签>` 作为 Prompt，模型便会根据注意力机制中学到的条件概率，生成对应体裁的诗词。

### 对比采样策略

```bash
# 对比四种采样策略
python main.py compare "春风"

# 使用分类标签对比
python main.py compare "<五言绝句>"

# 指定最大生成长度
python main.py compare "<七言律诗>" --max-new-tokens 100
```

四种采样策略对比输出示例：

```
采样策略对比 — Prompt: 「春风」

[Greedy]
春风不逐春归去，花落花开又一春。春去春来春不老，春来春去春长新。

[Temperature (0.8)]
春风不解惜花枝，吹落残红满地飞。日暮无人收不得，随风犹自入罗帏。

[Top-K (40)]
春风何太急，吹我庭前花。花飞不复返，此意良可嗟。

[Top-P (0.9)]
春风吹梦到天涯，梦到天涯路转赊。梦醒不知身是客，梦魂犹自绕京华。
```

---

## 模型架构

```
Token Embedding (vocab_size × d_model)  +  Position Embedding (max_seq_len × d_model)
                         ↓
                  N × TransformerBlock (Pre-LN)
                    ├── LayerNorm → Multi-Head Attention (+ causal mask) → Dropout → 残差连接
                    └── LayerNorm → FeedForward (GELU) → Dropout → 残差连接
                         ↓
                      LayerNorm
                         ↓
                  LM Head (d_model → vocab_size, weight tied with Token Embedding)
```

### 关键设计

| 特性 | 说明 |
|------|------|
| **Pre-LN** | LayerNorm 在 MHA/FFN 之前（GPT-2/LLaMA 标准做法），训练更稳定 |
| **Causal Mask** | 上三角掩码，确保自回归生成时不能"看到未来" |
| **Weight Tying** | Token Embedding 与 LM Head 共享权重，减少参数量 |
| **可学习位置编码** | 非固定正弦编码，让模型学习最优位置表示 |
| **GELU 激活** | FFN 使用 GELU 激活函数 |
| **多字符 Token 支持** | tokenizer 支持 `<BOS>`, `<五言绝句>` 等标签作为单一 token |

### 从零实现的模块 (`src/attention.py`)

- `scaled_dot_product_attention()` — Q·K^T / √d_k + mask → softmax → ·V
- `create_causal_mask()` — 上三角掩码矩阵
- `MultiHeadAttention` — 多头注意力（Q/K/V 线性投影 + 并行注意力 + 输出投影）
- `FeedForward` — 双层 FFN（d_model → d_ff → d_model，GELU 激活）

### 默认模型规格

| 配置 | 参数量 | 适用场景 |
|------|--------|---------|
| d_model=128, n_layers=2 | ~2M | CPU 训练 / 快速测试 |
| d_model=256, n_layers=4 | ~5-8M | GPU 训练 / 正式使用 |
| d_model=512, n_layers=6 | ~20M | 高性能 GPU |

---

## 采样策略

| 策略 | 参数 | 原理 | 特点 |
|------|------|------|------|
| **Greedy Search** | — | 每步选概率最高的 token | 确定性，易产生重复 |
| **Temperature** | `temperature` | 缩放 logits 分布 | >1 更随机，<1 更确定 |
| **Top-K** | `top_k=40` | 只从概率最高的 K 个候选中采样 | 过滤低概率噪声 |
| **Top-P (Nucleus)** | `top_p=0.9` | 动态选择累积概率达 p 的最小候选集 | 自适应候选集大小 |

四种策略链式组合：**Greedy → Temperature 缩放 → Top-K 过滤 → Top-P 过滤 → multinomial 采样**

---

## 硬件需求与运行时间

### 硬件兼容性

| 硬件 | 推荐配置 | 预计训练时间 | 预计显存占用 |
|------|---------|------------|------------|
| **CPU（笔记本）** | d_model=128, n_layers=2, batch_size=8, seq_len=64 | 3-8 小时（从零训练）<br>1-3 小时（微调） | 2-4 GB RAM |
| **笔记本 GPU（RTX 3050/4050 4GB）** | d_model=128, n_layers=4, batch_size=32, seq_len=96 | 1-2 小时（从零训练）<br>30-60 分钟（微调） | ~2 GB VRAM |
| **笔记本 GPU（RTX 4060 8GB）** | d_model=256, n_layers=4, batch_size=64, seq_len=128 | 20-40 分钟（从零训练）<br>15-30 分钟（微调） | ~4 GB VRAM |
| **桌面 GPU（RTX 4070+ 12GB）** | d_model=512, n_layers=6, batch_size=128, seq_len=256 | 10-20 分钟（从零训练）<br>5-15 分钟（微调） | ~8 GB VRAM |

### 运行时间参考

| 操作 | CPU | GPU (RTX 4060) |
|------|-----|----------------|
| 数据预处理（`preprocess_tagged`） | ~30 秒 | ~30 秒 |
| 快速测试训练（5 epochs, d_model=128, n_layers=2） | ~1.5 小时 | ~5 分钟 |
| 正式从零训练（20 epochs, d_model=256, n_layers=4） | ~8 小时 | ~30 分钟 |
| 正式微调（20 epochs, d_model=256, n_layers=4） | ~3 小时 | ~20 分钟 |
| 交互式生成（每次） | ~2-5 秒 | <1 秒 |

> **注意**：以上时间为估算值，实际运行时间受 CPU 频率、内存带宽、散热条件等因素影响。

### 训练数据规模

| 数据集 | 样本数 | 总字符数 |
|--------|--------|---------|
| 从零训练（HuggingFace） | ~21.7 万首 | ~1500 万 |
| 从零训练（本地 JSON 目录） | ~40 万首 | ~2800 万 |
| 微调训练集（train_tagged.txt） | 366,736 首 | ~3000 万 |
| 微调测试集（test_tagged.txt） | 368 首 | ~3 万 |

---

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP
- [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) — 中文古诗词数据库
- [Million/Chinese-Poems](https://huggingface.co/datasets/Million/Chinese-Poems) — HuggingFace 诗词数据集
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
- typer库构建CLI工具
- typst项目撰写论文
