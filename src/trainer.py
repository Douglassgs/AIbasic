"""
训练器模块 — 训练循环、学习率调度、模型保存

设计要点：
- AdamW 优化器（带权重衰减的 Adam）
- Cosine Annealing 学习率调度 + Linear Warmup
- 梯度裁剪防止梯度爆炸
- 定期在验证集评估并生成样本观测训练效果
- 支持中断后恢复训练
"""

import math
import os
import time
from typing import List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class CosineWarmupScheduler:
    """
    带 Linear Warmup 的 Cosine Annealing 学习率调度器

    学习率变化：
    1. Warmup 阶段：从 0 线性增加到 lr_max
    2. Cosine 衰减阶段：从 lr_max 余弦衰减到接近 0
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        lr_max: float = 3e-4,
        lr_min: float = 1e-5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.current_step = 0

    def get_lr(self) -> float:
        """根据当前步数计算学习率"""
        step = self.current_step

        # Warmup 阶段 — 线性增加
        if step < self.warmup_steps:
            return self.lr_max * (step + 1) / self.warmup_steps

        # Cosine Annealing 阶段
        if step >= self.total_steps:
            return self.lr_min

        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))

    def step(self):
        """更新学习率"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_step += 1


class Trainer:
    """
    Mini-GPT 训练器

    训练流程：
    1. 每个 step：前向传播 → 计算 loss → 反向传播 → 梯度裁剪 → 参数更新
    2. 每个 epoch 结束：验证集评估 → 生成样本 → 保存 checkpoint
    3. 使用 tqdm 显示训练进度
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer,
        device: torch.device,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        total_steps: int = 5000,
        grad_clip: float = 1.0,
        save_dir: str = "checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.grad_clip = grad_clip
        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        # AdamW 优化器 — Adam + 解耦的权重衰减
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),  # GPT-2 使用的 beta 参数
        )

        # 学习率调度器
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            lr_max=lr,
        )

        # 训练状态记录
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train_epoch(self, epoch: int) -> float:
        """
        训练一个 epoch

        Args:
            epoch: 当前 epoch 编号

        Returns:
            该 epoch 的平均训练 loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [训练]", ncols=100)
        for batch_idx, (input_ids, targets) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            # 前向传播
            outputs = self.model(input_ids, targets=targets)
            loss = outputs["loss"]

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪 — 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # 参数更新
            self.optimizer.step()
            self.scheduler.step()

            # 记录
            total_loss += loss.item()
            self.train_losses.append(loss.item())
            self.global_step += 1

            # 更新进度条
            current_lr = self.scheduler.get_lr()
            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "lr": f"{current_lr:.2e}",
            })

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> float:
        """
        在验证集上评估

        Returns:
            验证集的平均 loss
        """
        self.model.eval()
        total_loss = 0.0

        for input_ids, targets in tqdm(self.val_loader, desc="验证", ncols=100, leave=False):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(input_ids, targets=targets)
            total_loss += outputs["loss"].item()

        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss

    def save_checkpoint(self, filename: str, extra: dict = None):
        """保存模型 checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.config,
            "global_step": self.global_step,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        if extra:
            checkpoint.update(extra)
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)

    def load_checkpoint(self, filename: str):
        """加载模型 checkpoint"""
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint.get("val_losses", [])
        return checkpoint

    def train(
        self,
        epochs: int,
        eval_every: int = 1,
        save_every: int = 5,
        sample_prompt: str = "春",
    ):
        """
        完整训练流程

        Args:
            epochs:        训练 epoch 数
            eval_every:    每隔几个 epoch 进行验证
            save_every:    每隔几个 epoch 保存 checkpoint
            sample_prompt: 在验证时用于生成样本的起始文本
        """
        print(f"\n{'='*50}")
        print(f"  开始训练")
        print(f"  设备: {self.device}")
        print(f"  批次大小: {self.train_loader.batch_size}")
        print(f"  训练批次数: {len(self.train_loader)}")
        print(f"  总训练步数: {epochs * len(self.train_loader)}")
        print(f"  每个 epoch 验证: {'是' if eval_every > 0 else '否'}")
        print(f"{'='*50}\n")

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # 训练
            train_loss = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start

            # 打印 epoch 总结
            summary = f"Epoch {epoch + 1}/{epochs} | 训练 Loss: {train_loss:.4f} | 耗时: {epoch_time:.1f}s"

            # 验证
            if eval_every > 0 and (epoch + 1) % eval_every == 0:
                val_loss = self.validate()
                self.model.train()  # 恢复训练模式
                summary += f" | 验证 Loss: {val_loss:.4f} | 验证 PPL: {math.exp(val_loss):.2f}"

                # 生成样本观察训练效果
                if sample_prompt:
                    self._generate_sample(sample_prompt, epoch + 1)

                # 跟踪最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")
                    summary += " ✓ 最佳模型已保存"

            print(summary)

            # 保存 checkpoint
            if save_every > 0 and (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        total_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {total_time:.1f}s ({total_time / 60:.1f}分钟)")
        self.save_checkpoint("final_model.pt")

    def _generate_sample(self, prompt: str, epoch: int):
        """生成样本文本用于观察训练进展"""
        self.model.eval()

        # 编码 prompt
        input_ids = self.tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], device=self.device)

        # 生成
        generated = self.model.generate(
            input_tensor,
            max_new_tokens=40,
            temperature=0.8,
            top_k=40,
        )

        # 解码
        generated_text = self.tokenizer.decode(generated[0].tolist())
        print(f"  [样本 Epoch {epoch}] >>> {generated_text}")

        self.model.train()
