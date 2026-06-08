"""
文本生成器模块 — 自回归文本生成与多种采样策略

采样策略说明：
- Greedy Search:    每步选概率最高的 token，确定性生成，容易产生重复
- Temperature:      调整 logits 分布锐度，>1 更随机，<1 更确定
- Top-K Sampling:   只从概率最高的 K 个 token 中采样，避免低概率的无意义 token
- Top-P (Nucleus):  从累积概率达到 p 的最小 token 集合中采样，动态调整候选集大小

设计要点：
- 四种采样策略可自由组合（先 temperature → top-k → top-p，最后采样）
- 支持批量生成并排对比输出
- 可与 CLI 界面无缝对接
"""

from typing import Optional, List, Dict
import torch


class TextGenerator:
    """
    文本生成器 — 封装模型推理和多种采样策略

    用法示例:
        gen = TextGenerator(model, tokenizer, device)
        result = gen.generate("春风", strategy="top_k", top_k=40, max_new_tokens=50)
        gen.print_comparison("明月")  # 对比四种策略
    """

    def __init__(self, model, tokenizer, device: torch.device):
        """
        Args:
            model:     MiniGPT 模型实例
            tokenizer: CharTokenizer 实例
            device:    torch 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 500,
        strategy: str = "greedy",
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        根据指定策略生成文本续写

        Args:
            prompt:          起始文本
            max_new_tokens:  最大生成字符数
            strategy:        采样策略 ("greedy" | "temperature" | "top_k" | "top_p" | "combined")
            temperature:     温度参数（仅 temperature 和 combined 策略）
            top_k:           Top-K 参数（仅 top_k 和 combined 策略）
            top_p:           Top-P 参数（仅 top_p 和 combined 策略）

        Returns:
            生成的完整文本（prompt + 续写）
        """
        # 编码 prompt
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], device=self.device)

        # 根据策略设置参数
        gen_temperature = temperature if strategy in ("temperature", "combined") else 1.0
        gen_top_k = top_k if strategy in ("top_k", "combined") else None
        gen_top_p = top_p if strategy in ("top_p", "combined") else None

        # 如果 strategy 是 "temperature" 但没有指定温度，使用默认值
        if strategy == "temperature" and temperature == 1.0:
            gen_temperature = 0.8

        # 如果 strategy 是 "top_k" 但没有指定 k，使用默认值
        if strategy == "top_k" and top_k is None:
            gen_top_k = 40

        # 如果 strategy 是 "top_p" 但没有指定 p，使用默认值
        if strategy == "top_p" and top_p is None:
            gen_top_p = 0.9

        # 调用模型生成
        generated = self.model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=gen_temperature,
            top_k=gen_top_k,
            top_p=gen_top_p,
        )

        # 解码
        result = self.tokenizer.decode(generated[0].tolist())
        return result

    def compare_strategies(
        self,
        prompt: str,
        max_new_tokens: int = 500,
    ) -> Dict[str, str]:
        """
        对比四种采样策略的生成效果

        Args:
            prompt:          起始文本
            max_new_tokens:  最大生成字符数

        Returns:
            {"greedy": str, "temperature": str, "top_k": str, "top_p": str}
        """
        strategies = {
            "Greedy": {
                "strategy": "greedy",
                "temperature": 1.0,
                "top_k": None,
                "top_p": None,
            },
            "Temperature (0.8)": {
                "strategy": "temperature",
                "temperature": 0.8,
                "top_k": None,
                "top_p": None,
            },
            "Top-K (40)": {
                "strategy": "top_k",
                "temperature": 1.0,
                "top_k": 40,
                "top_p": None,
            },
            "Top-P (0.9)": {
                "strategy": "top_p",
                "temperature": 1.0,
                "top_k": None,
                "top_p": 0.9,
            },
        }

        results = {}
        print(f"\n{'='*70}")
        print(f"  采样策略对比 — Prompt: 「{prompt}」")
        print(f"{'='*70}")

        for name, params in strategies.items():
            result = self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                **params,
            )
            results[name] = result
            print(f"\n  [{name}]")
            print(f"  {result}")

        print(f"\n{'='*70}\n")
        return results

    def interactive(self):
        """
        命令行交互式文本生成

        用法：
            输入文本 → 模型续写 → 显示结果
            输入 "quit" 或 "exit" 退出
            输入 "compare <文本>" 对比四种策略
        """
        print("\n" + "=" * 60)
        print("  Mini-GPT 交互式文本生成")
        print("=" * 60)
        print("  输入文本 → 模型续写")
        print("  命令:")
        print("    <任意文本>      - 续写生成（Top-K 40）")
        print("    compare <文本>  - 对比四种采样策略")
        print("    quit / exit     - 退出")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("  你 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  再见！")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("  再见！")
                break

            if user_input.lower().startswith("compare "):
                prompt = user_input[8:].strip()
                if prompt:
                    self.compare_strategies(prompt)
                else:
                    print("  请输入要比较的起始文本，如: compare 春风")
                continue

            # 默认生成
            result = self.generate(
                user_input,
                max_new_tokens=500,
                strategy="top_k",
                top_k=40,
                temperature=0.8,
            )
            print(f"  续写 > {result}")
            print()
