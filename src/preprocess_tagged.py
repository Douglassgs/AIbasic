"""
诗歌体裁分类与带标签数据重构脚本

功能：
1. 读取 train.txt / test.txt（<|endoftext|> 分隔的原始诗词数据）
2. 自动识别每首诗的体裁（五言/七言/词/曲/其他）
3. 重构为带分类标签的训练格式：
   <BOS> <五言绝句> 白日依山尽，黄河入海流。... [EOS]

分类策略：
- 提取诗词正文行（跳过标题行），去除标点后统计每行汉字数
- 五言：大部分行含 5 或 10 个汉字（单句/双句）
- 七言：大部分行含 7 或 14 个汉字
- 词：行间字数差异大（长短句），且标题含词牌名标识（· 分隔）
- 曲：类似词但标题含曲牌标识（・ 或 【】）
- 其他：无法归类的文本

用法：
    python -m src.preprocess_tagged
    # 或单独运行
    python src/preprocess_tagged.py
"""

import os
import re
import sys
from typing import List, Tuple, Optional
from collections import Counter


# ============================================================
# 诗歌体裁分类器
# ============================================================

def _extract_chinese_chars(text: str) -> str:
    """
    从文本中提取纯汉字（去掉标点符号和空白）

    使用 Unicode 块判断，覆盖：
    - CJK Unified Ideographs (U+4E00 - U+9FFF)
    - CJK Unified Ideographs Extension A (U+3400 - U+4DBF)
    - CJK Compatibility Ideographs (U+F900 - U+FAFF)
    - CJK Unified Ideographs Extension B+ (U+20000+)
    对非 BMP 字符（如 𫛶 U+2B6F6），使用 ord() 范围判断
    """
    result = []
    for ch in text:
        cp = ord(ch)
        # 基本区 + Extension A + 兼容区
        if (0x4E00 <= cp <= 0x9FFF or
            0x3400 <= cp <= 0x4DBF or
            0xF900 <= cp <= 0xFAFF or
            cp >= 0x20000):  # Extension B 及以上
            result.append(ch)
    return "".join(result)


def _split_to_phrases(line: str) -> List[int]:
    """
    将一行诗按中文标点拆分为短语，返回每个短语的汉字数

    例如：
        "白日依山尽，黄河入海流。" → [5, 5]
        "绿云高髻，点翠匀红时世。月如眉。" → [4, 6, 3]
        "居贫得田不百亩，天赐时雨苗氤氲。" → [7, 7]

    Args:
        line: 一行诗（可能包含标点）

    Returns:
        各短语汉字数的列表
    """
    # 按中文标点拆分
    segments = re.split(r"[，。！？、；：]", line)
    phrase_lengths = []
    for seg in segments:
        chars = _extract_chinese_chars(seg)
        if chars:
            phrase_lengths.append(len(chars))
    return phrase_lengths


def _classify_single_poem(lines: List[str]) -> str:
    """
    对单首诗的体裁进行分类

    分类逻辑（改进版）：
    1. 提取正文行，按标点拆分为短语
    2. 统计各短语的汉字数，找出模式
    3. 短语字数一致的 → 诗（五言/七言）；字数混杂的 → 词/曲
    4. 再根据行数细分绝句/律诗/古诗

    Args:
        lines: 诗词的所有行（第一行为标题）

    Returns:
        分类标签（如 "<五言绝句>"）
    """
    if len(lines) < 2:
        return "<其他>"

    title = lines[0].strip()
    body_lines = [ln.strip() for ln in lines[1:] if ln.strip()]

    if not body_lines:
        return "<其他>"

    # 将每行拆为短语，收集所有短语的汉字数
    all_phrases = []  # 所有短语的字数
    for line in body_lines:
        phrase_lens = _split_to_phrases(line)
        all_phrases.extend(phrase_lens)

    if not all_phrases:
        return "<其他>"

    # 统计短语字数分布
    len_counter = Counter(all_phrases)
    unique_lengths = set(all_phrases)
    most_common_len, most_common_freq = len_counter.most_common(1)[0]
    total_phrases = len(all_phrases)

    # 标题格式判断
    is_ci_title = "·" in title
    is_qu_title = ("・" in title) or ("【" in title)

    # ---- 核心分类逻辑 ----

    # 优先级 0：标题明确标注词牌名或曲牌名 → 直接判定
    # 这类诗词即使短语字数整齐（如菩萨蛮 7-7-5-5），也应属于词/曲
    if is_ci_title:
        return "<词>"
    if is_qu_title:
        return "<曲>"

    # 优先级 1：检测短语字数显著混用 → 词的典型特征（长短句）
    # 如菩萨蛮 7-7-5-5-5-5，两种字数各占显著比例，不是纯粹的齐言诗
    purity = most_common_freq / total_phrases if total_phrases > 0 else 0
    if len(len_counter) >= 2:
        second_most_common_len, second_freq = len_counter.most_common(2)[1]
        second_ratio = second_freq / total_phrases
        # 第二多的字数占比 >= 20%，说明是刻意混用（非偶发变异）
        if second_ratio >= 0.20:
            # 混用的字数在词常见范围内（3-7）→ 词
            if {most_common_len, second_most_common_len}.issubset({3, 4, 5, 6, 7}):
                return "<词>"
            # 否则 → 其他杂言
            return "<其他>"

    # 情况 2：短语字数近乎全部一致 → 诗（五言或七言）
    # 允许少量（<10%）变异（如首句押韵可能会多一字等）
    if purity >= 0.85:
        line_count = len(body_lines)

        if most_common_len == 5:
            if line_count == 4:
                return "<五言绝句>"
            elif line_count == 8:
                return "<五言律诗>"
            else:
                return "<五言古诗>"

        if most_common_len == 7:
            if line_count == 4:
                return "<七言绝句>"
            elif line_count == 8:
                return "<七言律诗>"
            else:
                return "<七言古诗>"

    # 情况 2：短语字数混杂 → 词或曲或杂言
    # 检查是否以 3,5,7 为主（词的典型特征）
    if unique_lengths.issubset({3, 4, 5, 6, 7}):
        if is_qu_title:
            return "<曲>"
        if is_ci_title:
            return "<词>"
        # 无明确标题标识，但字数混杂 — 判定为词
        return "<词>"

    # 情况 3：短语字数混杂且不完全在 {3,4,5,6,7} 范围内
    if is_qu_title:
        return "<曲>"
    if is_ci_title:
        return "<词>"

    # 情况 4：主体字数一致（>=70% 但 < 85%）— 放宽阈值再判断一次
    if purity >= 0.70:
        line_count = len(body_lines)
        if most_common_len == 5:
            if line_count == 4:
                return "<五言绝句>"
            elif line_count == 8:
                return "<五言律诗>"
            else:
                return "<五言古诗>"
        if most_common_len == 7:
            if line_count == 4:
                return "<七言绝句>"
            elif line_count == 8:
                return "<七言律诗>"
            else:
                return "<七言古诗>"

    return "<其他>"


# ============================================================
# 数据加载与重构
# ============================================================

def load_raw_poems(filepath: str) -> List[List[str]]:
    """
    从原始 txt 文件加载诗词

    原始格式：
        诗题
        正文行1
        正文行2
        ...
        <|endoftext|>

    Args:
        filepath: 原始 txt 文件路径

    Returns:
        每首诗为一个列表 [标题, 正文行1, 正文行2, ...]
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 按 <|endoftext|> 分割
    blocks = content.split("<|endoftext|>")

    poems = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        # 过滤完全空白的行
        lines = [ln.strip() for ln in lines if ln.strip()]
        if lines:
            poems.append(lines)

    return poems


def reconstruct_tagged_poem(lines: List[str], category: str) -> str:
    """
    将分类后的诗词重构为带标签的训练格式

    格式：<BOS> <category> 正文行1 正文行2 ... <EOS>

    Args:
        lines: 诗词行列表（第一行为标题）
        category: 分类标签

    Returns:
        重构后的单行训练文本
    """
    # 只取正文行（跳过标题），用空字符串连接保持连续性
    body_lines = lines[1:] if len(lines) > 1 else lines
    body_text = "".join(body_lines)

    # 构建带标签的完整训练序列：
    # <BOS> <分类标签> 正文 <EOS>
    tagged_text = f"<BOS>{category}{body_text}<EOS>"
    return tagged_text


def process_file(
    input_path: str,
    output_path: str,
    verbose: bool = True,
) -> dict:
    """
    处理单个原始诗词文件：分类 → 重构 → 输出

    Args:
        input_path:  原始 txt 文件路径
        output_path: 输出路径
        verbose:     是否打印统计信息

    Returns:
        分类统计字典
    """
    poems = load_raw_poems(input_path)
    if verbose:
        print(f"\n  从 {input_path} 加载了 {len(poems)} 首诗词")

    stats = Counter()
    tagged_texts = []

    for lines in poems:
        category = _classify_single_poem(lines)
        stats[category] += 1
        tagged = reconstruct_tagged_poem(lines, category)
        tagged_texts.append(tagged)

    # 写入输出文件（每行一首诗，用 <|endoftext|> 分隔）
    with open(output_path, "w", encoding="utf-8") as f:
        for tagged in tagged_texts:
            f.write(tagged + "\n")

    if verbose:
        total = sum(stats.values())
        print(f"  分类统计 ({total} 首):")
        # 按预定义顺序显示
        order = [
            "<五言绝句>", "<五言律诗>", "<五言古诗>",
            "<七言绝句>", "<七言律诗>", "<七言古诗>",
            "<词>", "<曲>", "<其他>",
        ]
        for tag in order:
            if tag in stats:
                count = stats[tag]
                print(f"    {tag}: {count:>6} ({count/total*100:5.1f}%)")
        # 显示任何不在预定义顺序中的标签
        for tag in sorted(stats.keys()):
            if tag not in order:
                count = stats[tag]
                print(f"    {tag}: {count:>6} ({count/total*100:5.1f}%)")
        print(f"  输出文件: {output_path}")

    return dict(stats)


def main():
    """主入口：处理 train.txt 和 test.txt"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_input = os.path.join(base_dir, "train.txt")
    test_input = os.path.join(base_dir, "test.txt")
    train_output = os.path.join(base_dir, "train_tagged.txt")
    test_output = os.path.join(base_dir, "test_tagged.txt")

    print("=" * 60)
    print("  诗歌体裁分类与带标签数据重构")
    print("=" * 60)

    # 处理训练集
    print("\n>>> 处理训练集...")
    train_stats = process_file(train_input, train_output)

    # 处理测试集
    print("\n>>> 处理测试集...")
    test_stats = process_file(test_input, test_output)

    # 汇总
    print("\n" + "=" * 60)
    print("  处理完成！")
    print(f"  训练集标签数据: {train_output}")
    print(f"  测试集标签数据: {test_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
