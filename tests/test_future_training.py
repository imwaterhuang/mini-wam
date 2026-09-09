from __future__ import annotations

import torch

from mini_wam.models.future_head import FutureHeadPolicy
from mini_wam.training.config import load_training_config
from mini_wam.training.future_head import compute_losses, train_step
from mini_wam.training.reproducibility import DeterministicBatchSampler
from mini_wam.training.runtime import select_device
from mini_wam.training.steps import make_scheduler


def _batch(batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        "observation_history": torch.randn(batch_size, 2, 3, 96, 96),
        "agent_position": torch.randn(batch_size, 2, 2),
        "action_chunk": torch.randn(batch_size, 16, 2),
        "action_valid_mask": torch.ones(batch_size, 16, dtype=torch.bool),
        "future_observations": torch.randn(batch_size, 4, 3, 96, 96),
        "future_valid_mask": torch.ones(batch_size, 4, dtype=torch.bool),
    }


def test_future_smoke_config_and_cpu_device() -> None:
    config = load_training_config(
        "configs/future_aware_smoke.yaml",
        expected_model_name="future_aware",
    )
    assert config["training"]["lambda_future"] == 0.1
    assert select_device("cpu") == torch.device("cpu")


def test_compute_losses_and_train_step_gradient_boundaries() -> None:
    torch.manual_seed(7)
    model = FutureHeadPolicy(pretrained=False, target_pretrained=False)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )
    scheduler = make_scheduler(optimizer, warmup_steps=1, train_steps=2)
    sampler = DeterministicBatchSampler(dataset_size=1, batch_size=1, seed=3)
    next(iter(sampler))
    batch = _batch()

    losses = compute_losses(model, batch, torch.device("cpu"), lambda_future=0.1)
    assert torch.allclose(
        losses["total_loss"],
        losses["action_loss"] + 0.1 * losses["future_loss"],
    )

    target_before = {
        key: value.clone()
        for key, value in model.frozen_target_encoder.state_dict().items()
    }
    metrics = train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        device=torch.device("cpu"),
        lambda_future=0.1,
        gradient_clip_norm=1.0,
        next_step=1,
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any(parameter.grad is not None for parameter in model.action_head.parameters())
    assert any(parameter.grad is not None for parameter in model.future_head.parameters())
    assert any(parameter.grad is not None for parameter in model.history_fusion.parameters())
    assert any(
        parameter.grad is not None
        for parameter in model.action_prefix_encoder.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.frozen_target_encoder.parameters()
    )
    assert all(
        torch.equal(value, target_before[key])
        for key, value in model.frozen_target_encoder.state_dict().items()
    )
    assert sampler.state_dict()["epoch"] == 1
    assert sampler.state_dict()["batch_in_epoch"] == 0
