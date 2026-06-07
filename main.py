#!/usr/bin/env python3
"""
Mini-GPT 小型文本生成系统 — 主入口

运行模式：
    python main.py --train              # 训练模型
    python main.py --generate           # 交互式文本生成（需要已训练模型）
    python main.py --compare <prompt>   # 对比四种采样策略

示例：
    # 训练
    python main.py --train --epochs 20 --batch_size 64

    # 交互式续写
    python main.py --generate

    # 对比采样策略
    python main.py --compare "春花秋月何时了"

数据来源：
    优先使用 HuggingFace Chinese-Poems 数据集（Parquet 格式）
    如遇网络问题，请将 chinese-poetry JSON 文件放入 data/ 目录
"""

import argparse
import os
import sys
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


def get_device() -> torch.device:
    """自动选择可用设备（CUDA > MPS > CPU）"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_data(args, tokenizer: CharTokenizer):
    """
    加载并预处理数据

    加载流程：
    1. 优先尝试 HuggingFace Chinese-Poems（Parquet 格式）
    2. 若失败，尝试本地 data/ 目录下的 JSON 文件
    3. 仍失败则报错提示
    """
    print("\n>>> 正在加载数据...")

    texts = None
    source_name = None

    # 方式一：HuggingFace datasets（Parquet 自动处理）
    if args.data_source == "huggingface":
        try:
            print("  尝试从 HuggingFace 加载 Million/Chinese-Poems（Parquet 格式）...")
            texts = extract_poem_texts("Million/Chinese-Poems", source_type="huggingface")
            source_name = "HuggingFace Million/Chinese-Poems"
        except Exception as e:
            print(f"  HuggingFace 加载失败: {e}")
            texts = None

    # 方式二：本地 chinese-poetry 目录（自动遍历所有 JSON）
    if texts is None and (args.data_source == "directory" or args.data_path):
        data_path = args.data_path
        if os.path.isdir(data_path):
            print(f"  从本地目录加载 chinese-poetry JSON: {data_path}")
            try:
                texts = extract_poem_texts(data_path, source_type="directory")
                source_name = f"本地目录 ({data_path})"
            except Exception as e:
                print(f"  本地目录加载失败: {e}")

    # 方式三：本地单个 JSON 文件
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
        print("    2. 使用 --data_source directory --data_path chinese-poetry 加载本地数据")
        print("    3. 使用 --data_source json --data_path <文件> 加载单个 JSON")
        sys.exit(1)

    print(f"  数据源: {source_name}")
    print(f"  原始文本数: {len(texts)}")
    print(f"  总字符数: {sum(len(t) for t in texts):,}")

    # 构建词表
    print(f"\n>>> 构建字符词表...")
    tokenizer.build_vocab(texts, min_freq=args.min_char_freq)
    print(f"  词表大小: {tokenizer.vocab_size}")

    return texts


def load_tagged_data(args, tokenizer: CharTokenizer):
    """
    加载带分类标签的训练数据（微调模式）

    数据格式：
        train_tagged.txt / test_tagged.txt
        每行：<BOS><分类标签>诗词正文<EOS>

    分类标签已在 tokenizer 初始化时预置，无需通过 build_vocab 学习。
    """
    print("\n>>> 正在加载带标签的训练数据...")

    train_path = args.train_path or os.path.join(os.path.dirname(__file__), "train_tagged.txt")
    test_path = args.test_path or os.path.join(os.path.dirname(__file__), "test_tagged.txt")

    if not os.path.exists(train_path):
        print(f"  错误：训练集标签文件不存在: {train_path}")
        print(f"  请先运行 python -m src.preprocess_tagged 生成带标签数据")
        sys.exit(1)
    if not os.path.exists(test_path):
        print(f"  错误：测试集标签文件不存在: {test_path}")
        print(f"  请先运行 python -m src.preprocess_tagged 生成带标签数据")
        sys.exit(1)

    # 加载带标签文本
    train_texts = extract_tagged_texts(train_path)
    test_texts = extract_tagged_texts(test_path)
    all_texts = train_texts + test_texts

    print(f"  训练集: {train_path} ({len(train_texts)} 首)")
    print(f"  测试集: {test_path} ({len(test_texts)} 首)")

    # 构建词表（分类标签已在 tokenizer 初始化时预置，build_vocab 只添加普通字符）
    print(f"\n>>> 构建字符词表（含预置分类标签）...")
    tokenizer.build_vocab(all_texts, min_freq=args.min_char_freq)
    print(f"  词表大小: {tokenizer.vocab_size}")
    print(f"  分类标签已就绪: {list(tokenizer.category_ids.keys())}")

    # 创建 DataLoader（不需要滑动窗口，每条序列已完整）
    print(f"\n>>> 创建数据加载器...")
    train_loader, test_loader = create_tagged_dataloaders(
        train_path=train_path,
        test_path=test_path,
        tokenizer=tokenizer,
        seq_len=args.seq_len + 1,  # +1 因为 TextDataset 需要 input + target
        batch_size=args.batch_size,
    )

    return train_loader, test_loader


def train_mode(args):
    """训练模式"""
    device = get_device()
    print(f"\n>>> 使用设备: {device}")

    # 1. 初始化分词器
    tokenizer = CharTokenizer()

    # 2. 根据数据源类型加载数据
    if args.data_source == "tagged_txt":
        # 带标签微调模式：直接用预处理好的 train_tagged.txt / test_tagged.txt
        train_loader, val_loader = load_tagged_data(args, tokenizer)
    else:
        # 通用模式：从原始数据源加载
        texts = load_data(args, tokenizer)

        # 3. 保存词表
        vocab_path = os.path.join(args.save_dir, "vocab.json")
        os.makedirs(args.save_dir, exist_ok=True)
        tokenizer.save(vocab_path)
        print(f"  词表已保存: {vocab_path}")

        # 4. 创建 DataLoader
        print(f"\n>>> 创建数据加载器...")
        train_loader, val_loader = create_dataloaders(
            texts,
            tokenizer,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            train_ratio=0.9,
        )

    # 3. 保存词表（带标签模式也需要保存）
    vocab_path = os.path.join(args.save_dir, "vocab.json")
    os.makedirs(args.save_dir, exist_ok=True)
    tokenizer.save(vocab_path)
    print(f"  词表已保存: {vocab_path}")

    # 4/5. 创建模型
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

    # 6. 创建训练器
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

    # 7. 开始训练
    print(f"\n>>> 开始训练...")
    trainer.train(
        epochs=args.epochs,
        eval_every=args.eval_every,
        save_every=args.save_every,
        sample_prompt=args.sample_prompt,
    )

    # 8. 绘制训练曲线
    print(f"\n>>> 生成训练曲线...")
    plot_training_curves(
        trainer.train_losses,
        trainer.val_losses,
        save_path=os.path.join(args.result_dir, "training_curves.png"),
    )

    print(f"\n>>> 训练完成！")
    print(f"  模型保存在: {args.save_dir}/")


def generate_mode(args):
    """生成模式 — 加载模型并进入交互式命令行"""
    device = get_device()
    print(f"\n>>> 使用设备: {device}")

    # 1. 加载词表
    vocab_path = args.vocab_path or os.path.join(args.save_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        print(f"  错误：词表文件不存在: {vocab_path}")
        print(f"  请先运行训练，或使用 --vocab_path 指定词表路径")
        sys.exit(1)
    tokenizer = CharTokenizer.load(vocab_path)
    print(f"  词表加载完成，大小: {tokenizer.vocab_size}")

    # 2. 加载模型
    model_path = args.model_path or os.path.join(args.save_dir, "best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(args.save_dir, "final_model.pt")
    if not os.path.exists(model_path):
        print(f"  错误：模型文件不存在: {model_path}")
        print(f"  请先运行训练，或使用 --model_path 指定模型路径")
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

    # 3. 进入交互式命令循环
    generator = TextGenerator(model, tokenizer, device)

    if args.prompt:
        # 单次生成模式
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
        # 交互模式 — 打印分类标签提示
        print("\n  体裁标签提示：")
        print("    输入 <五言绝句> → 生成五言绝句（四句，每句五字）")
        print("    输入 <七言律诗> → 生成七言律诗（八句，每句七字）")
        print("    输入 <词>       → 生成词（长短句）")
        print("    输入 <五言古诗> → 生成五言古诗")
        print("    输入 <七言古诗> → 生成七言古诗")
        print("    输入 <曲>       → 生成曲")
        generator.interactive()


def compare_mode(args):
    """对比采样策略模式"""
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
        prompt = input("请输入起始文本: ").strip()

    generator.compare_strategies(prompt, max_new_tokens=args.max_new_tokens)


def main():
    parser = argparse.ArgumentParser(
        description="Mini-GPT 小型 Transformer 文本生成系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --train --epochs 20
  python main.py --generate
  python main.py --compare "春风得意马蹄疾"
        """,
    )

    # 运行模式
    parser.add_argument("--train", action="store_true", help="训练模式")
    parser.add_argument("--generate", action="store_true", help="生成模式（交互式续写）")
    parser.add_argument("--compare", action="store_true", help="对比四种采样策略")
    parser.add_argument("--prompt", type=str, default="", help="用于生成/对比的起始文本")

    # 数据参数
    parser.add_argument("--data_source", type=str, default="tagged_txt",
                        choices=["huggingface", "json", "directory", "tagged_txt"],
                        help="数据源类型 (默认: tagged_txt)")
    parser.add_argument("--data_path", type=str, default="chinese-poetry",
                        help="本地数据路径（目录或 JSON 文件）")
    parser.add_argument("--train_path", type=str, default="",
                        help="带标签训练集路径（tagged_txt 模式，默认: train_tagged.txt）")
    parser.add_argument("--test_path", type=str, default="",
                        help="带标签测试集路径（tagged_txt 模式，默认: test_tagged.txt）")
    parser.add_argument("--min_char_freq", type=int, default=2,
                        help="词表构建时的最小字符频率 (默认: 2)")

    # 模型参数
    parser.add_argument("--d_model", type=int, default=256, help="隐藏层维度 (默认: 256)")
    parser.add_argument("--n_heads", type=int, default=8, help="注意力头数 (默认: 8)")
    parser.add_argument("--n_layers", type=int, default=4, help="Transformer 层数 (默认: 4)")
    parser.add_argument("--d_ff", type=int, default=1024, help="FFN 中间维度 (默认: 1024)")
    parser.add_argument("--seq_len", type=int, default=128, help="最大序列长度 (默认: 128)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout 概率 (默认: 0.1)")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="标签平滑 (默认: 0.1，设为 0 关闭)")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数 (默认: 20)")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小 (默认: 64)")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率 (默认: 3e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减 (默认: 0.01)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup 步数 (默认: 500)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值 (默认: 1.0)")
    parser.add_argument("--eval_every", type=int, default=1, help="每隔 N epoch 验证 (默认: 1)")
    parser.add_argument("--save_every", type=int, default=5, help="每隔 N epoch 保存 (默认: 5)")
    parser.add_argument("--sample_prompt", type=str, default="<五言绝句>",
                        help="训练中生成样本的 prompt（带标签模式建议用分类标签如 <五言绝句>）")

    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=60, help="最大生成字符数 (默认: 60)")

    # 路径参数
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="模型保存目录")
    parser.add_argument("--result_dir", type=str, default="results", help="结果输出目录")
    parser.add_argument("--model_path", type=str, default="", help="模型文件路径")
    parser.add_argument("--vocab_path", type=str, default="", help="词表文件路径")

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    # 如果没有任何模式被指定，显示帮助
    if not (args.train or args.generate or args.compare):
        parser.print_help()
        print("\n请指定运行模式: --train, --generate, 或 --compare")
        sys.exit(1)

    if args.train:
        train_mode(args)
    elif args.compare:
        compare_mode(args)
    elif args.generate:
        generate_mode(args)


if __name__ == "__main__":
    main()
