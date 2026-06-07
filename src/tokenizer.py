"""
字符级分词器 — 将中文文本映射为整数 ID 序列

设计要点：
- 字符级编码，词表小（~5000-8000），无需依赖外部分词工具
- 特殊 token：<PAD>=0, <UNK>=1, <BOS>=2, <EOS>=3
- 支持从语料构建词表，也支持保存/加载词表文件
"""

import json
from typing import List, Optional


class CharTokenizer:
    """字符级分词器，用于中文文本的编解码"""

    # 基础特殊 token 定义
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    # 诗歌体裁分类标签 — 用于条件生成（推理时输入标签即可控制生成体裁）
    CATEGORY_TOKENS = [
        "<五言绝句>",   # 五言绝句（4句，每句5字）
        "<五言律诗>",   # 五言律诗（8句，每句5字）
        "<五言古诗>",   # 五言古诗（其他行数的五言诗）
        "<七言绝句>",   # 七言绝句（4句，每句7字）
        "<七言律诗>",   # 七言律诗（8句，每句7字）
        "<七言古诗>",   # 七言古诗（其他行数的七言诗）
        "<词>",         # 词（长短句，句式不规整）
        "<曲>",         # 曲（元曲等）
        "<其他>",       # 无法归类的其他体裁
    ]

    def __init__(self, vocab: Optional[dict] = None):
        """
        初始化分词器

        Args:
            vocab: 可选，已有词表 {"char": id}，若为 None 则需调用 build_vocab 构建
        """
        if vocab is not None:
            self.vocab = vocab  # char -> id
            self.inv_vocab = {v: k for k, v in vocab.items()}  # id -> char
        else:
            # 默认包含基础特殊 token + 体裁分类标签
            self.vocab = {
                self.PAD_TOKEN: 0,
                self.UNK_TOKEN: 1,
                self.BOS_TOKEN: 2,
                self.EOS_TOKEN: 3,
            }
            # 分类标签紧接基础特殊 token 之后
            for tag in self.CATEGORY_TOKENS:
                self.vocab[tag] = len(self.vocab)
            self.inv_vocab = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        """返回词表大小"""
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        """PAD token 的 ID"""
        return self.vocab[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        """UNK token 的 ID"""
        return self.vocab[self.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        """BOS token 的 ID"""
        return self.vocab[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        """EOS token 的 ID"""
        return self.vocab[self.EOS_TOKEN]

    def build_vocab(self, texts: List[str], min_freq: int = 1):
        """
        从文本列表中构建词表

        Args:
            texts: 文本列表，每条为一个字符串
            min_freq: 最小字符频率，低于此频率的字符不会被加入词表
        """
        # 统计字符频率
        char_freq = {}
        for text in texts:
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1

        # 按频率排序，高频字符获得更小的 ID（紧接特殊 token 之后）
        sorted_chars = sorted(
            char_freq.items(), key=lambda x: x[1], reverse=True
        )

        # 构建词表（特殊 token 已在 __init__ 中加入）
        for char, freq in sorted_chars:
            if freq >= min_freq and char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx

        # 更新反向词表
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def _build_special_token_map(self) -> dict:
        """
        构建多字符特殊 token 的匹配映射（按长度降序排列，用于贪婪匹配）

        返回 {token_str: token_id}，按 token 长度从长到短排序
        """
        special_tokens = [
            self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN
        ] + self.CATEGORY_TOKENS
        # 只保留长度 > 1 的 token（单字符 token 走逐字符编码路径）
        multi_char = {t: self.vocab[t] for t in special_tokens if len(t) > 1 and t in self.vocab}
        # 按长度降序排列，保证贪婪匹配时优先匹配长 token
        return dict(sorted(multi_char.items(), key=lambda x: len(x[0]), reverse=True))

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        将文本编码为 token ID 序列

        支持多字符特殊 token（如 <五言绝句>）的正确编码——使用贪婪匹配
        优先匹配长特殊 token，剩余字符按字符级编码。

        Args:
            text: 输入文本字符串
            add_bos: 是否在开头添加 BOS token
            add_eos: 是否在末尾添加 EOS token

        Returns:
            token ID 列表
        """
        ids = []
        if add_bos:
            ids.append(self.bos_id)

        # 多字符特殊 token 匹配表
        multi_char_map = self._build_special_token_map()

        i = 0
        while i < len(text):
            matched = False
            # 尝试匹配多字符特殊 token（已按长度降序排列）
            for token_str, token_id in multi_char_map.items():
                if text[i:].startswith(token_str):
                    ids.append(token_id)
                    i += len(token_str)
                    matched = True
                    break
            if not matched:
                # 单字符编码
                char = text[i]
                ids.append(self.vocab.get(char, self.unk_id))
                i += 1

        if add_eos:
            ids.append(self.eos_id)

        return ids

    @property
    def special_ids(self) -> set:
        """返回所有特殊 token 的 ID 集合（基础 + 分类标签）"""
        ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        for tag in self.CATEGORY_TOKENS:
            if tag in self.vocab:
                ids.add(self.vocab[tag])
        return ids

    @property
    def category_ids(self) -> dict:
        """返回分类标签名 -> ID 的映射"""
        return {tag: self.vocab[tag] for tag in self.CATEGORY_TOKENS if tag in self.vocab}

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        将 token ID 序列解码为文本

        Args:
            ids: token ID 列表
            skip_special: 是否跳过特殊 token（包括分类标签）

        Returns:
            解码后的文本字符串
        """
        result = []
        for idx in ids:
            idx = int(idx)
            if skip_special and idx in self.special_ids:
                continue
            result.append(self.inv_vocab.get(idx, self.UNK_TOKEN))
        return "".join(result)

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[List[int]]:
        """
        批量编码文本

        Args:
            texts: 文本列表
            add_bos: 是否添加 BOS
            add_eos: 是否添加 EOS

        Returns:
            嵌套的 token ID 列表
        """
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def save(self, path: str):
        """
        保存词表到 JSON 文件

        Args:
            path: 保存路径
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """
        从 JSON 文件加载词表

        Args:
            path: 词表文件路径

        Returns:
            CharTokenizer 实例
        """
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)

    def __len__(self) -> int:
        """返回词表大小"""
        return self.vocab_size

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
