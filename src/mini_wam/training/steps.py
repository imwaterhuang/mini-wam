"""Action-only 的单步优化、学习率调度与离线验证。"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from mini_wam.models.action_only import ActionOnlyPolicy, masked_smooth_l1_loss

from .reproducibility import DeterministicBatchSampler


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    train_steps: int,
) -> LambdaLR:
    """创建线性 warmup（预热）后接余弦衰减的逐步学习率调度器。"""

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(train_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=scale)


def train_step(
    *,
    model: ActionOnlyPolicy,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    sampler: DeterministicBatchSampler,
    device: torch.device,
    gradient_clip_norm: float,
    next_step: int,
) -> tuple[float, float]:
    """执行一次完整参数更新，并在成功后推进 sampler 消费位置。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    mask = batch["action_valid_mask"].to(device)
    prediction = model(
        batch["observation_history"].to(device),
        batch["agent_position"].to(device),
    )
    loss = masked_smooth_l1_loss(
        prediction,
        batch["action_chunk"].to(device),
        mask,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError(f"第 {next_step} 步出现非有限 loss")
    loss.backward()
    gradient_norm = float(clip_grad_norm_(model.parameters(), gradient_clip_norm))
    optimizer.step()
    scheduler.step()
    sampler.mark_consumed()
    return float(loss.item()), gradient_norm


@torch.inference_mode()
def evaluate(
    model: ActionOnlyPolicy,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> float:
    """在验证集上计算按有效动作步加权的平均损失。"""
    model.eval()
    weighted_loss = 0.0
    valid_steps = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        mask = batch["action_valid_mask"].to(device)
        prediction = model(
            batch["observation_history"].to(device),
            batch["agent_position"].to(device),
        )
        loss = masked_smooth_l1_loss(
            prediction,
            batch["action_chunk"].to(device),
            mask,
        )
        count = int(mask.sum().item())
        weighted_loss += float(loss.item()) * count
        valid_steps += count
    if not valid_steps:
        raise RuntimeError("验证集没有有效动作")
    return weighted_loss / valid_steps
