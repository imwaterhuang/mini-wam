"""FutureHeadPolicy 的损失、单步更新与离线验证。"""

from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from mini_wam.models.future_head import (
    FutureHeadPolicy,
    masked_cosine_loss,
    masked_smooth_l1_loss,
)

from .reproducibility import DeterministicBatchSampler
from .runtime import move_tensors_to_device


FUTURE_BATCH_KEYS = (
    "observation_history",
    "agent_position",
    "action_chunk",
    "action_valid_mask",
    "future_observations",
    "future_valid_mask",
)


def compute_losses(
    model: FutureHeadPolicy,
    batch: dict[str, Any],
    device: torch.device,
    lambda_future: float,
) -> dict[str, torch.Tensor]:
    """计算动作损失、未来损失及其加权总和。"""
    tensors = move_tensors_to_device(batch, FUTURE_BATCH_KEYS, device)
    true_action_prefix = tensors["action_chunk"][:, :4]
    action_prediction, future_prediction, future_target = model(
        observation_history=tensors["observation_history"],
        position_history=tensors["agent_position"],
        true_action_prefix=true_action_prefix,
        future_images=tensors["future_observations"],
        compute_future=True,
    )
    action_loss = masked_smooth_l1_loss(
        action_prediction,
        tensors["action_chunk"],
        tensors["action_valid_mask"],
    )
    future_loss = masked_cosine_loss(
        future_prediction,
        future_target,
        tensors["future_valid_mask"],
    )
    total_loss = action_loss + lambda_future * future_loss
    return {
        "total_loss": total_loss,
        "action_loss": action_loss,
        "future_loss": future_loss,
    }

def train_step(
    *,
    model: FutureHeadPolicy,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    sampler: DeterministicBatchSampler,
    device: torch.device,
    lambda_future: float,
    gradient_clip_norm: float,
    next_step: int,
) -> dict[str, float]:
    """执行一次联合损失参数更新，成功后才推进 sampler。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = compute_losses(model, batch, device, lambda_future)
    for name, loss in losses.items():
        if not torch.isfinite(loss):
            raise FloatingPointError(f"第 {next_step} 步出现非有限 {name}")
    losses["total_loss"].backward()
    gradient_norm = float(clip_grad_norm_(model.parameters(), gradient_clip_norm))
    optimizer.step()
    scheduler.step()
    sampler.mark_consumed()
    return {
        "total_loss": float(losses["total_loss"].item()),
        "action_loss": float(losses["action_loss"].item()),
        "future_loss": float(losses["future_loss"].item()),
        "gradient_norm": gradient_norm,
    }


@torch.inference_mode()
def evaluate(
    model: FutureHeadPolicy,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
    lambda_future: float,
) -> dict[str, float]:
    """在 eval 模式下分别按有效动作步和未来步汇总损失。"""
    model.eval()
    action_weighted_loss = 0.0
    future_weighted_loss = 0.0
    valid_action_steps = 0
    valid_future_steps = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        losses = compute_losses(model, batch, device, lambda_future)
        action_count = int(batch["action_valid_mask"].sum().item())
        future_count = int(batch["future_valid_mask"].sum().item())
        action_weighted_loss += float(losses["action_loss"].item()) * action_count
        future_weighted_loss += float(losses["future_loss"].item()) * future_count
        valid_action_steps += action_count
        valid_future_steps += future_count
    if not valid_action_steps or not valid_future_steps:
        raise RuntimeError("验证集没有有效动作或未来监督")
    action_loss = action_weighted_loss / valid_action_steps
    future_loss = future_weighted_loss / valid_future_steps
    return {
        "total_loss": action_loss + lambda_future * future_loss,
        "action_loss": action_loss,
        "future_loss": future_loss,
    }
