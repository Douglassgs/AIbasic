#!/usr/bin/env python3
"""
Mini-GPT 小型文本生成系统 — 主入口（Typer CLI）

运行模式：
    python main.py train                  # 训练模型
    python main.py generate               # 交互式文本生成（需要已训练模型）
    python main.py compare "<prompt>"     # 对比四种采样策略

示例：
    # 从零训练（HuggingFace 数据）
    python main.py train --data-source huggingface --epochs 20 --batch-size 64

    # 微调训练（带分类标签）
    python main.py train --data-source tagged-txt --epochs 20

    # 交互式续写
    python main.py generate

    # 分类条件生成
    python main.py generate --prompt "<五言绝句>"

    # 对比采样策略
    python main.py compare "春风"

    # 查看帮助
    python main.py --help
    python main.py train --help
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

import typer
import torch

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tokenizer import CharTokenizer
from src.model import MiniGPT, MiniGPTConfig
from src.trainer import Trainer
from src.generator import TextGenerator
from src.utils import (
    extract_poem_texts,
    create_dataloaders,
    create_tagged_dataloaders,
    extract_tagged_texts,
    plot_training_curves,
    format_model_info,
)

# ============================================================
# Typer 应用与数据类
# ============================================================

app = typer.Typer(
    name="mini-gpt",
    help="Mini-GPT 小型 Transformer 文本生成系统",
    add_completion=False,
    rich_markup_mode="rich",
)


class DataSource(str, Enum):
    """数据源类型"""
    huggingface = "huggingface"
    json = "json"
    directory = "directory"
    tagged_txt = "tagged-txt"


# ---- 共享参数 dataclass（兼容现有业务逻辑） ----

@dataclass
class TrainArgs:
    """训练模式参数"""
    # 数据
    data_source: str = "tagged-txt"
    data_path: str = "chinese-poetry"
    train_path: str = ""
    test_path: str = ""
    min_char_freq: int = 2

    # 模型
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    seq_len: int = 128
    dropout: float = 0.1
    label_smoothing: float = 0.1

    # 训练
    epochs: int = 20
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    eval_every: int = 1
    save_every: int = 5
    sample_prompt: str = "<五言绝句>"

    # 路径
    save_dir: str = "checkpoints"
    result_dir: str = "results"


@dataclass
class GenerateArgs:
    """生成模式参数"""
    prompt: str = ""
    max_new_tokens: int = 60
    model_path: str = ""
    vocab_path: str = ""
    save_dir: str = "checkpoints"


@dataclass
class CompareArgs:
    """对比模式参数"""
    prompt: str = ""
    max_new_tokens: int = 50
    model_path: str = ""
    vocab_path: str = ""
    save_dir: str = "checkpoints"


# ============================================================
# 工具函数
# ============================================================

def get_device() -> torch.device:
    """自动选择可用设备（CUDA > MPS > CPU）"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ============================================================
# 数据加载逻辑（从原 main.py 保留，业务逻辑不变）
# ============================================================

def load_data(args: TrainArgs, tokenizer: CharTokenizer):
    """加载并预处理数据（从零训练模式）"""
    print("\n>>> 正在加载数据...")

    texts = None
    source_name = None

    if args.data_source == "huggingface":
        try:
            print("  尝试从 HuggingFace 加载 Million/Chinese-Poems（Parquet 格式）...")
            texts = extract_poem_texts("Million/Chinese-Poems", source_type="huggingface")
            source_name = "HuggingFace Million/Chinese-Poems"
        except Exception as e:
            print(f"  HuggingFace 加载失败: {e}")
            texts = None

    if texts is None and (args.data_source == "directory" or args.data_path):
        data_path = args.data_path
        if os.path.isdir(data_path):
            print(f"  从本地目录加载 chinese-poetry JSON: {data_path}")
            try:
                texts = extract_poem_texts(data_path, source_type="directory")
                source_name = f"本地目录 ({data_path})"
            except Exception as e:
                print(f"  本地目录加载失败: {e}")

    if texts is None and args.data_source == "json":
        data_path = args.data_path
        if os.path.exists(data_path):
            print(f"  从本地 JSON 加载: {data_path}")
            try:
                texts = extract_poem_texts(data_path, source_type="json")
                source_name = f"本地 JSON ({data_path})"
            except Exception as e:
                print(f"  本地 JSON 加载失败: {e}")

    if texts is None or len(texts) == 0:
        print("\n  错误：未能加载任何有效数据！")
        print("  请确保以下之一可用：")
        print("    1. 网络可访问 HuggingFace Million/Chinese-Poems 数据集")
        print("    2. 使用 --data-source directory --data-path chinese-poetry 加载本地数据")
        print("    3. 使用 --data-source json --data-path <文件> 加载单个 JSON")
        sys.exit(1)

    print(f"  数据源: {source_name}")
    print(f"  原始文本数: {len(texts)}")
    print(f"  总字符数: {sum(len(t) for t in texts):,}")

    print(f"\n>>> 构建字符词表...")
    tokenizer.build_vocab(texts, min_freq=args.min_char_freq)
    print(f"  词表大小: {tokenizer.vocab_size}")

    return texts


def load_tagged_data(args: TrainArgs, tokenizer: CharTokenizer):
    """加载带分类标签的训练数据（微调模式）"""
    print("\n>>> 正在加载带标签的训练数据...")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = args.train_path or os.path.join(base_dir, "train_tagged.txt")
    test_path = args.test_path or os.path.join(base_dir, "test_tagged.txt")

    if not os.path.exists(train_path):
        print(f"  错误：训练集标签文件不存在: {train_path}")
        print(f"  请先运行 python -m src.preprocess_tagged 生成带标签数据")
        sys.exit(1)
    if not os.path.exists(test_path):
        print(f"  错误：测试集标签文件不存在: {test_path}")
        print(f"  请先运行 python -m src.preprocess_tagged 生成带标签数据")
        sys.exit(1)

    train_texts = extract_tagged_texts(train_path)
    test_texts = extract_tagged_texts(test_path)
    all_texts = train_texts + test_texts

    print(f"  训练集: {train_path} ({len(train_texts)} 首)")
    print(f"  测试集: {test_path} ({len(test_texts)} 首)")

    print(f"\n>>> 构建字符词表（含预置分类标签）...")
    tokenizer.build_vocab(all_texts, min_freq=args.min_char_freq)
    print(f"  词表大小: {tokenizer.vocab_size}")
    print(f"  分类标签已就绪: {list(tokenizer.category_ids.keys())}")

    print(f"\n>>> 创建数据加载器...")
    train_loader, test_loader = create_tagged_dataloaders(
        train_path=train_path,
        test_path=test_path,
        tokenizer=tokenizer,
        seq_len=args.seq_len + 1,
        batch_size=args.batch_size,
    )

    return train_loader, test_loader


# ============================================================
# 核心业务逻辑
# ============================================================

def run_train(args: TrainArgs):
    """训练模型"""
    device = get_device()
    print(f"\n>>> 使用设备: {device}")

    tokenizer = CharTokenizer()

    # 根据数据源类型加载数据
    if args.data_source == "tagged-txt":
        train_loader, val_loader = load_tagged_data(args, tokenizer)
    else:
        texts = load_data(args, tokenizer)
        vocab_path = os.path.join(args.save_dir, "vocab.json")
        os.makedirs(args.save_dir, exist_ok=True)
        tokenizer.save(vocab_path)
        print(f"  词表已保存: {vocab_path}")

        print(f"\n>>> 创建数据加载器...")
        train_loader, val_loader = create_dataloaders(
            texts,
            tokenizer,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            train_ratio=0.9,
        )

    # 保存词表（带标签模式也需要）
    vocab_path = os.path.join(args.save_dir, "vocab.json")
    os.makedirs(args.save_dir, exist_ok=True)
    tokenizer.save(vocab_path)
    print(f"  词表已保存: {vocab_path}")

    # 创建模型
    print(f"\n>>> 创建模型...")
    config = MiniGPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
    )
    model = MiniGPT(config)
    info = format_model_info(config, model.get_num_params())
    print(info)

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        total_steps=args.epochs * len(train_loader),
        grad_clip=args.grad_clip,
        save_dir=args.save_dir,
    )

    # 开始训练
    print(f"\n>>> 开始训练...")
    trainer.train(
        epochs=args.epochs,
        eval_every=args.eval_every,
        save_every=args.save_every,
        sample_prompt=args.sample_prompt,
    )

    # 绘制训练曲线
    print(f"\n>>> 生成训练曲线...")
    plot_training_curves(
        trainer.train_losses,
        trainer.val_losses,
        save_path=os.path.join(args.result_dir, "training_curves.png"),
    )

    print(f"\n>>> 训练完成！")
    print(f"  模型保存在: {args.save_dir}/")


def run_generate(args: GenerateArgs):
    """加载模型并进入交互式命令行"""
    device = get_device()
    print(f"\n>>> 使用设备: {device}")

    # 加载词表
    vocab_path = args.vocab_path or os.path.join(args.save_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        print(f"  错误：词表文件不存在: {vocab_path}")
        print(f"  请先运行训练，或使用 --vocab-path 指定词表路径")
        sys.exit(1)
    tokenizer = CharTokenizer.load(vocab_path)
    print(f"  词表加载完成，大小: {tokenizer.vocab_size}")

    # 加载模型
    model_path = args.model_path or os.path.join(args.save_dir, "best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(args.save_dir, "final_model.pt")
    if not os.path.exists(model_path):
        print(f"  错误：模型文件不存在: {model_path}")
        print(f"  请先运行训练，或使用 --model-path 指定模型路径")
        sys.exit(1)

    print(f"  正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    info = format_model_info(config, model.get_num_params())
    print(info)

    generator = TextGenerator(model, tokenizer, device)

    if args.prompt:
        result = generator.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            strategy="top_k",
            top_k=40,
            temperature=0.8,
        )
        print(f"\n  Prompt: {args.prompt}")
        print(f"  续写:   {result}")
    else:
        _print_category_hints()
        generator.interactive()


def run_compare(args: CompareArgs):
    """对比四种采样策略"""
    device = get_device()

    # 加载模型和词表
    vocab_path = args.vocab_path or os.path.join(args.save_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        print(f"  错误：词表文件不存在: {vocab_path}")
        sys.exit(1)
    tokenizer = CharTokenizer.load(vocab_path)

    model_path = args.model_path or os.path.join(args.save_dir, "best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(args.save_dir, "final_model.pt")
    if not os.path.exists(model_path):
        print(f"  错误：模型文件不存在: {model_path}")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = MiniGPT(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    generator = TextGenerator(model, tokenizer, device)

    prompt = args.prompt
    if not prompt:
        prompt = typer.prompt("请输入起始文本")

    generator.compare_strategies(prompt, max_new_tokens=args.max_new_tokens)


def _print_category_hints():
    """打印分类标签使用提示"""
    typer.echo("\n  📝 体裁标签提示（微调模型可用）：")
    typer.echo("    typer.style('<五言绝句>', fg='cyan') → 生成五言绝句（四句，每句五字）")
    typer.echo("    typer.style('<七言律诗>', fg='cyan') → 生成七言律诗（八句，每句七字）")
    typer.echo("    typer.style('<词>', fg='cyan')       → 生成词（长短句）")
    typer.echo("    typer.style('<五言古诗>', fg='cyan') → 生成五言古诗")
    typer.echo("    typer.style('<七言古诗>', fg='cyan') → 生成七言古诗")
    typer.echo("    typer.style('<曲>', fg='cyan')       → 生成曲")


# ============================================================
# Typer 命令定义
# ============================================================

@app.command(help="训练 Mini-GPT 模型（支持从零训练和带标签微调两种模式）")
def train(
    # ---- 数据 ----
    data_source: DataSource = typer.Option(
        DataSource.tagged_txt, "--data-source", "-d",
        help="数据源类型。tagged-txt=带标签微调, huggingface=从零训练",
    ),
    data_path: str = typer.Option(
        "chinese-poetry", "--data-path",
        help="本地数据路径（目录或 JSON 文件）",
    ),
    train_path: str = typer.Option(
        "", "--train-path",
        help="带标签训练集路径（默认: train_tagged.txt）",
    ),
    test_path: str = typer.Option(
        "", "--test-path",
        help="带标签测试集路径（默认: test_tagged.txt）",
    ),
    min_char_freq: int = typer.Option(
        2, "--min-char-freq",
        help="词表构建时的最小字符频率",
    ),
    # ---- 模型 ----
    d_model: int = typer.Option(
        256, "--d-model",
        help="隐藏层维度",
    ),
    n_heads: int = typer.Option(
        8, "--n-heads",
        help="注意力头数",
    ),
    n_layers: int = typer.Option(
        4, "--n-layers",
        help="Transformer 层数",
    ),
    d_ff: int = typer.Option(
        1024, "--d-ff",
        help="FFN 中间维度",
    ),
    seq_len: int = typer.Option(
        128, "--seq-len",
        help="最大序列长度",
    ),
    dropout: float = typer.Option(
        0.1, "--dropout",
        help="Dropout 概率",
    ),
    label_smoothing: float = typer.Option(
        0.1, "--label-smoothing",
        help="标签平滑（设为 0 关闭）",
    ),
    # ---- 训练 ----
    epochs: int = typer.Option(
        20, "--epochs", "-e",
        help="训练轮数",
    ),
    batch_size: int = typer.Option(
        64, "--batch-size", "-b",
        help="批次大小",
    ),
    lr: float = typer.Option(
        3e-4, "--lr",
        help="学习率",
    ),
    weight_decay: float = typer.Option(
        0.01, "--weight-decay",
        help="权重衰减",
    ),
    warmup_steps: int = typer.Option(
        500, "--warmup-steps",
        help="Warmup 步数",
    ),
    grad_clip: float = typer.Option(
        1.0, "--grad-clip",
        help="梯度裁剪阈值",
    ),
    eval_every: int = typer.Option(
        1, "--eval-every",
        help="每隔 N 个 epoch 进行验证",
    ),
    save_every: int = typer.Option(
        5, "--save-every",
        help="每隔 N 个 epoch 保存 checkpoint",
    ),
    sample_prompt: str = typer.Option(
        "<五言绝句>", "--sample-prompt",
        help="训练中生成样本文本的 prompt",
    ),
    # ---- 路径 ----
    save_dir: str = typer.Option(
        "checkpoints", "--save-dir",
        help="模型保存目录",
    ),
    result_dir: str = typer.Option(
        "results", "--result-dir",
        help="结果输出目录",
    ),
):
    """
    训练 Mini-GPT 模型。

    [bold]从零训练[/bold]（HuggingFace 数据源）：
        python main.py train --data-source huggingface --epochs 20

    [bold]微调训练[/bold]（带分类标签，需先运行 preprocess_tagged）：
        python main.py train --data-source tagged-txt --epochs 20

    [bold]快速测试[/bold]（小模型，CPU 友好）：
        python main.py train -e 5 --d-model 128 --n-layers 2 -b 32
    """
    # 将 typer 参数转换为 dataclass（兼容现有业务逻辑）
    args = TrainArgs(
        data_source=data_source.value,
        data_path=data_path,
        train_path=train_path,
        test_path=test_path,
        min_char_freq=min_char_freq,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        seq_len=seq_len,
        dropout=dropout,
        label_smoothing=label_smoothing,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        grad_clip=grad_clip,
        eval_every=eval_every,
        save_every=save_every,
        sample_prompt=sample_prompt,
        save_dir=save_dir,
        result_dir=result_dir,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    run_train(args)


@app.command(help="交互式文本生成（加载已训练模型）")
def generate(
    prompt: Optional[str] = typer.Option(
        None, "--prompt", "-p",
        help="起始文本（不指定则进入交互模式）",
    ),
    max_new_tokens: int = typer.Option(
        60, "--max-new-tokens", "-n",
        help="最大生成字符数",
    ),
    model_path: str = typer.Option(
        "", "--model-path", "-m",
        help="模型文件路径（默认: checkpoints/best_model.pt）",
    ),
    vocab_path: str = typer.Option(
        "", "--vocab-path",
        help="词表文件路径（默认: checkpoints/vocab.json）",
    ),
    save_dir: str = typer.Option(
        "checkpoints", "--save-dir",
        help="模型保存目录",
    ),
):
    """
    加载已训练的模型，进行交互式文本生成。

    [bold]交互模式[/bold]（直接运行，输入文本续写）：
        python main.py generate

    [bold]分类条件生成[/bold]（微调模型）：
        python main.py generate --prompt "<五言绝句>"

    [bold]通用续写[/bold]：
        python main.py generate --prompt "春风"

    交互命令：
        [cyan]<任意文本>[/cyan]      - Top-K 40 采样续写
        [cyan]compare <文本>[/cyan]  - 对比四种采样策略
        [cyan]quit / exit[/cyan]     - 退出
    """
    args = GenerateArgs(
        prompt=prompt or "",
        max_new_tokens=max_new_tokens,
        model_path=model_path,
        vocab_path=vocab_path,
        save_dir=save_dir,
    )

    run_generate(args)


@app.command(help="对比四种采样策略（Greedy / Temperature / Top-K / Top-P）")
def compare(
    prompt: str = typer.Argument(
        ...,
        help="起始文本（输入后按回车开始）",
    ),
    max_new_tokens: int = typer.Option(
        50, "--max-new-tokens", "-n",
        help="最大生成字符数",
    ),
    model_path: str = typer.Option(
        "", "--model-path", "-m",
        help="模型文件路径",
    ),
    vocab_path: str = typer.Option(
        "", "--vocab-path",
        help="词表文件路径",
    ),
    save_dir: str = typer.Option(
        "checkpoints", "--save-dir",
        help="模型保存目录",
    ),
):
    """
    使用四种不同采样策略生成文本并排对比。

    [bold]示例[/bold]：
        python main.py compare "春风"
        python main.py compare "<五言绝句>"
        python main.py compare "明月几时有" --max-new-tokens 80
    """
    args = CompareArgs(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        model_path=model_path,
        vocab_path=vocab_path,
        save_dir=save_dir,
    )

    run_compare(args)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    app()
