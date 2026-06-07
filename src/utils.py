"""
工具函数模块 — 数据加载、可视化、参数统计

设计要点：
- 支持 HuggingFace datasets（Parquet 格式）和本地 JSON 两种数据源
- 中文文本清洗，保留汉字和中文标点
- 滑动窗口切分长文本，充分利用语料
- Loss 曲线和样本展示的可视化
"""

import json
import os
import re
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免 GUI 依赖
import matplotlib.pyplot as plt

# 设置中文字体 — 按优先级尝试多个常见中文字体
_CN_FONTS = [
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "SimHei",
    "Microsoft YaHei",
    "AR PL UMing CN",
    "AR PL UKai CN",
]
_font_found = False
for _font in _CN_FONTS:
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
        _font_found = True
        break
    except Exception:
        continue

if not _font_found:
    # 如果没有任何中文字体，使用默认字体（中文会显示为方框）
    # 至少避免程序崩溃
    print("  警告：未找到中文字体，图表中的中文可能无法正常显示。")
    print("  如需中文显示，请安装中文字体，例如：")
    print("    sudo apt install fonts-wqy-microhei")

plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# ============================================================
# 文本清洗与预处理
# ============================================================

def clean_chinese_text(text: str) -> str:
    """
    清洗中文文本 — 保留纯汉字和四大句读符号，清除括号及注释

    清洗规则：
    - 保留：中文字符（一-鿿） + 句读符号（，。！？）
    - 移除：所有成对括号及内部内容（书名、注释、引用等）
    - 移除：英文字母、数字、特殊符号、空白字符

    Args:
        text: 原始文本

    Returns:
        清洗后的纯汉字+句读文本
    """
    if not text:
        return ""

    # 第一步：移除成对括号及其内部内容
    # 《》「」『』“”（）【】〈〉〔〕〖〗〘〙〚〛
    bracket_pairs = [
        ("《", "》"),
        ("「", "」"),
        ("『", "』"),
        (""", """),  # 中文双引号
        ("（", "）"),
        ("【", "】"),
        ("〈", "〉"),
        ("〔", "〕"),
        ("〖", "〗"),
        ("〘", "〙"),
        ("〚", "〛"),
    ]
    for left, right in bracket_pairs:
        while left in text and right in text:
            start_pos = text.find(left)
            end_pos = text.find(right, start_pos + len(left))
            if end_pos == -1:
                break
            text = text[:start_pos] + text[end_pos + len(right):]

    # 第二步：保留汉字（含扩展区）和四大句读符号
    # 使用逐字符判断，覆盖 CJK 基本区、Extension A、兼容区及 Extension B+
    result = []
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or      # CJK 基本汉字
            0x3400 <= cp <= 0x4DBF or      # CJK Extension A
            0xF900 <= cp <= 0xFAFF or      # CJK 兼容汉字
            cp >= 0x20000 or               # CJK Extension B 及以上
            ch in "，。！？"):               # 四大句读符号
            result.append(ch)
    text = "".join(result)

    return text.strip()

def extract_poem_texts(data_path: str, source_type: str = "huggingface") -> List[str]:
    """
    从不同数据源提取诗词文本

    Args:
        data_path: 数据路径
        source_type: "huggingface" 或 "json"

    Returns:
        清洗后的诗词文本列表
    """
    texts = []

    if source_type == "huggingface":
        # HuggingFace datasets 加载（Parquet 格式自动处理）
        # Million/Chinese-Poems 列名: ["instruction", "input", "output"]
        # output 格式: "\n《诗题》\n诗句1\n诗句2\n...\n"
        from datasets import load_dataset
        dataset = load_dataset(data_path)
        split = dataset.get("train", None)
        if split is None:
            split = dataset

        # 查找诗词正文所在的列
        text_columns = ["output", "content", "paragraphs", "text", "poem", "verses"]
        col_name = None
        for col in text_columns:
            if col in split.column_names:
                col_name = col
                break

        if col_name is None:
            print(f"可用列名: {list(split.column_names) if hasattr(split, 'column_names') else '无法读取'}")
            raise ValueError(f"未找到诗词正文列，请从以下列中手动指定: {split.column_names}")

        for item in split:
            raw = item.get(col_name, "")
            # 如果正文是列表格式（如 ["床前明月光", "疑是地上霜"]），拼接为字符串
            if isinstance(raw, list):
                raw = "\n".join(raw)
            cleaned = clean_chinese_text(str(raw))
            if cleaned:
                # 清理《诗题》标记 — 虽然清洗函数已保留中文标点，
                # 但诗句正文通常不含书名号
                texts.append(cleaned)

    elif source_type == "json":
        # chinese-poetry GitHub JSON 格式
        # 两种格式：
        #   (a) 目录型: {"title": "...", "content": [{"paragraphs": [...]}, ...]}
        #   (b) 列表型: [{"title": "...", "paragraphs": [...]}, ...]
        #   (c) 单个JSON对象型: {"title": "...", "paragraphs": [...]}
        with open(data_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # JSON Lines 格式
                f.seek(0)
                data = []
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))

        def _extract_poem_item(item: dict):
            """从单个诗词 JSON 对象中提取正文"""
            paragraphs = item.get("paragraphs", [])
            if isinstance(paragraphs, list) and len(paragraphs) > 0:
                raw = "\n".join(paragraphs)
            elif isinstance(paragraphs, str):
                raw = paragraphs
            else:
                return
            cleaned = clean_chinese_text(raw)
            if cleaned:
                texts.append(cleaned)

        if isinstance(data, dict):
            # 格式 (a): 目录型 {"title": "...", "content": [...]}
            content_list = data.get("content", [])
            if content_list:
                for item in content_list:
                    _extract_poem_item(item)
            else:
                # 格式 (c): 单个诗词对象
                _extract_poem_item(data)
        elif isinstance(data, list):
            # 格式 (b): 列表型
            for item in data:
                _extract_poem_item(item)

    elif source_type == "directory":
        # 遍历 chinese-poetry 整个目录，加载所有 JSON 文件
        import glob
        json_files = glob.glob(os.path.join(data_path, "**/*.json"), recursive=True)
        print(f"  在 {data_path} 中找到 {len(json_files)} 个 JSON 文件")
        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            # 递归提取诗词
            def _extract_from_structure(obj):
                if isinstance(obj, dict):
                    paragraphs = obj.get("paragraphs", [])
                    if paragraphs and isinstance(paragraphs, list):
                        # 将列表中的所有元素转为字符串（跳过嵌套 dict）
                        lines = []
                        for p in paragraphs:
                            if isinstance(p, str):
                                lines.append(p)
                            elif isinstance(p, dict):
                                # 有些结构嵌套了 dict，跳过
                                pass
                        if lines:
                            raw = "\n".join(lines)
                            cleaned = clean_chinese_text(raw)
                            if cleaned:
                                texts.append(cleaned)
                    # 如果是目录型，递归处理 content
                    for child in obj.get("content", []):
                        _extract_from_structure(child)
                elif isinstance(obj, list):
                    for item in obj:
                        _extract_from_structure(item)

            _extract_from_structure(data)

    else:
        raise ValueError(f"不支持的数据源类型: {source_type}")

    return texts


def split_text_into_sequences(
    texts: List[str],
    seq_len: int,
    stride: Optional[int] = None,
) -> List[str]:
    """
    使用滑动窗口将长文本切分为固定长度的训练序列

    Args:
        texts:   文本列表
        seq_len: 目标序列长度（字符数）
        stride:  滑动步长，默认为 seq_len（无重叠）

    Returns:
        切分后的序列列表
    """
    if stride is None:
        stride = seq_len

    sequences = []
    for text in texts:
        # 对每条文本按滑动窗口切分
        for i in range(0, len(text) - seq_len, stride):
            seq = text[i : i + seq_len]
            sequences.append(seq)

    return sequences


# ============================================================
# 带标签数据加载（微调模式）
# ============================================================

def extract_tagged_texts(filepath: str) -> List[str]:
    """
    从带分类标签的 txt 文件加载诗词

    每行格式：<BOS><五言绝句>白日依山尽，黄河入海流。...<EOS>
    标签已嵌入在文本中，直接作为训练序列使用。

    Args:
        filepath: 带标签的 txt 文件路径

    Returns:
        带标签的文本行列表
    """
    texts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    return texts


def create_tagged_dataloaders(
    train_path: str,
    test_path: str,
    tokenizer,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    为带标签数据创建训练集和测试集的 DataLoader

    与 create_dataloaders 的区别：
    - 每条文本已包含 <BOS>/<EOS>/分类标签，不进行滑动窗口切分
    - 训练集/测试集直接由文件决定（train_tagged.txt / test_tagged.txt）

    Args:
        train_path:  训练集标签文件路径
        test_path:   测试集标签文件路径
        tokenizer:   分词器
        seq_len:     序列长度
        batch_size:  批次大小
        num_workers: 数据加载线程数

    Returns:
        (train_loader, test_loader)
    """
    train_texts = extract_tagged_texts(train_path)
    test_texts = extract_tagged_texts(test_path)

    print(f"  训练序列数: {len(train_texts)}, 测试序列数: {len(test_texts)}")

    train_dataset = TextDataset(train_texts, tokenizer, seq_len)
    test_dataset = TextDataset(test_texts, tokenizer, seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


# ============================================================
# PyTorch Dataset
# ============================================================

class TextDataset(Dataset):
    """
    文本数据集 — 将字符串序列转换为模型输入

    输入 (input_ids) 和 目标 (targets)：
    - input_ids: [t_0, t_1, ..., t_{n-1}]  — 模型看到的 token
    - targets:   [t_1, t_2, ..., t_n]      — 模型需要预测的下一个 token
    （即 targets 为 input_ids 右移一位，保持相同长度）
    """

    def __init__(self, sequences: List[str], tokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = []

        for seq in sequences:
            # 编码并截断/填充到固定长度
            ids = tokenizer.encode(seq)
            if len(ids) < seq_len:
                # 短序列用 PAD 填充
                ids = ids + [tokenizer.pad_id] * (seq_len - len(ids))
            else:
                ids = ids[:seq_len]
            self.data.append(ids)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = torch.tensor(self.data[idx], dtype=torch.long)
        # input_ids: 前 n-1 个 token
        # targets:   后 n-1 个 token（预测下一个词）
        input_ids = ids[:-1]
        targets = ids[1:]
        return input_ids, targets


def create_dataloaders(
    texts: List[str],
    tokenizer,
    seq_len: int,
    batch_size: int,
    train_ratio: float = 0.9,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练集和验证集的 DataLoader

    Args:
        texts:        原始文本列表
        tokenizer:    分词器
        seq_len:      序列长度
        batch_size:   批次大小
        train_ratio:  训练集比例
        num_workers:  数据加载线程数

    Returns:
        (train_loader, val_loader)
    """
    # 滑动窗口切分
    sequences = split_text_into_sequences(texts, seq_len + 1)  # +1 因为需要 input 和 target

    if len(sequences) == 0:
        raise ValueError(
            f"没有生成任何训练序列！请检查 seq_len({seq_len}) 是否小于文本长度。"
        )

    # 划分训练集/验证集
    split_idx = int(len(sequences) * train_ratio)
    train_seqs = sequences[:split_idx]
    val_seqs = sequences[split_idx:]

    print(f"训练序列数: {len(train_seqs)}, 验证序列数: {len(val_seqs)}")

    train_dataset = TextDataset(train_seqs, tokenizer, seq_len + 1)
    val_dataset = TextDataset(val_seqs, tokenizer, seq_len + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# ============================================================
# 可视化
# ============================================================

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: str = "results/training_curves.png",
):
    """
    绘制训练过程中的 loss 和 perplexity 曲线

    Args:
        train_losses: 训练 loss 列表
        val_losses:   验证 loss 列表
        save_path:    保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    steps = list(range(1, len(train_losses) + 1))
    val_steps = list(range(1, len(val_losses) + 1))

    # 将 val_losses 的步数映射到训练步数（假设每个 epoch 结束验证一次）
    val_step_positions = [int(len(train_losses) / len(val_losses) * (i + 1)) for i in range(len(val_losses))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    ax1.plot(steps, train_losses, label="训练 Loss", color="#2c7bb6", linewidth=1.0)
    if val_losses:
        ax1.plot(val_step_positions, val_losses, label="验证 Loss", color="#d7191c",
                 marker="o", markersize=4, linewidth=1.5)
    ax1.set_xlabel("训练步数")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("训练与验证 Loss 曲线")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Perplexity 曲线
    train_ppl = [2.71828 ** l for l in train_losses]
    ax2.plot(steps, train_ppl, label="训练 Perplexity", color="#2c7bb6", linewidth=1.0)
    if val_losses:
        val_ppl = [2.71828 ** l for l in val_losses]
        ax2.plot(val_step_positions, val_ppl, label="验证 Perplexity", color="#d7191c",
                 marker="o", markersize=4, linewidth=1.5)
    ax2.set_xlabel("训练步数")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Perplexity 曲线（越低越好）")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"训练曲线已保存到：{save_path}")


def format_model_info(config, num_params: dict) -> str:
    """
    格式化模型信息为可读字符串

    Args:
        config:      MiniGPTConfig 实例
        num_params:  get_num_params() 返回的参数字典

    Returns:
        格式化的模型信息字符串
    """
    lines = [
        "=" * 50,
        "  Mini-GPT 模型信息",
        "=" * 50,
        f"  词表大小:      {config.vocab_size}",
        f"  隐藏维度:      {config.d_model}",
        f"  注意力头数:    {config.n_heads}",
        f"  Transformer层:  {config.n_layers}",
        f"  FFN 维度:       {config.d_ff}",
        f"  最大序列长度:  {config.max_seq_len}",
        f"  Dropout:       {config.dropout}",
        f"  Weight Tying:  {config.tie_weights}",
        "-" * 50,
        f"  总参数量:      {num_params['total']:,}",
        f"  可训练参数量:  {num_params['trainable']:,}",
        "=" * 50,
    ]
    return "\n".join(lines)
