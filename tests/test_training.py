from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from mini_wam.training.artifacts import sync_run_directory
from mini_wam.training.config import load_training_config
from mini_wam.training.reproducibility import (
    DeterministicBatchSampler,
    capture_random_states,
    restore_random_states,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_MULTIWORKER_TESTS = (
    sys.platform != "darwin" or os.environ.get("MINI_WAM_RUN_MULTIWORKER_TESTS") == "1"
)
MULTIWORKER_TEST = pytest.mark.skipif(
    not RUN_MULTIWORKER_TESTS,
    reason=(
        "Codex 的 macOS 沙箱禁止 torch_shm_manager；"
        "设置 MINI_WAM_RUN_MULTIWORKER_TESTS=1 可在非沙箱终端显式运行"
    ),
)


def test_smoke_and_formal_configs_are_valid() -> None:
    smoke = load_training_config(PROJECT_ROOT / "configs" / "action_only_smoke.yaml")
    formal = load_training_config(PROJECT_ROOT / "configs" / "action_only_seed0.yaml")
    assert smoke["training"]["train_steps"] == 100
    assert smoke["model"]["pretrained"] is False
    assert formal["training"]["train_steps"] == 50_000
    assert formal["model"]["pretrained"] is True
    assert formal["checkpoint"]["frequency"] == 1_000
    assert formal["checkpoint"]["archive_frequency"] == 5_000


def test_run_directory_mirror_updates_changed_files(tmp_path: Path) -> None:
    source = tmp_path / "local"
    destination = tmp_path / "persistent"
    (source / "checkpoints").mkdir(parents=True)
    (source / "train_metrics.csv").write_text("step,loss\n1,0.5\n", encoding="utf-8")
    (source / "checkpoints" / "last.pt").write_bytes(b"first")
    sync_run_directory(source, destination)
    assert (
        (destination / "train_metrics.csv")
        .read_text(encoding="utf-8")
        .endswith("1,0.5\n")
    )
    assert (destination / "checkpoints" / "last.pt").read_bytes() == b"first"

    (source / "train_metrics.csv").write_text(
        "step,loss\n1,0.5\n2,0.4\n", encoding="utf-8"
    )
    (source / "checkpoints" / "last.pt").write_bytes(b"second-version")
    sync_run_directory(source, destination)
    assert (
        (destination / "train_metrics.csv")
        .read_text(encoding="utf-8")
        .endswith("2,0.4\n")
    )
    assert (destination / "checkpoints" / "last.pt").read_bytes() == b"second-version"
    assert not list(destination.rglob("*.sync-tmp"))


def test_sampler_resumes_at_the_exact_next_batch() -> None:
    uninterrupted = DeterministicBatchSampler(dataset_size=11, batch_size=4, seed=7)
    uninterrupted_iterator = iter(uninterrupted)
    first = next(uninterrupted_iterator)
    uninterrupted.mark_consumed()
    state = uninterrupted.state_dict()
    expected_next = next(uninterrupted_iterator)

    restored = DeterministicBatchSampler(dataset_size=11, batch_size=4, seed=7)
    restored.load_state_dict(state)
    actual_next = next(iter(restored))

    assert len(first) == 4
    assert actual_next == expected_next


def test_legacy_sampler_state_still_loads() -> None:
    legacy_state = {
        "dataset_size": 11,
        "batch_size": 4,
        "seed": 7,
        "epoch": 0,
        "batch_in_epoch": 1,
    }
    expected_order = torch.randperm(
        11, generator=torch.Generator().manual_seed(7)
    ).tolist()
    restored = DeterministicBatchSampler(dataset_size=11, batch_size=4, seed=7)
    restored.load_state_dict(legacy_state)

    assert next(iter(restored)) == expected_order[4:8]


@pytest.mark.parametrize("num_workers", [2, 4])
@MULTIWORKER_TEST
def test_worker_prefetch_does_not_advance_checkpoint_position(num_workers: int) -> None:
    dataset = TensorDataset(torch.arange(24))
    sampler = DeterministicBatchSampler(dataset_size=24, batch_size=4, seed=7)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(101),
    )
    iterator = iter(loader)
    first = next(iterator)[0]

    # Workers have normally prefetched several later batches here, but the
    # checkpoint position must still describe the batch being trained now.
    assert sampler.state_dict()["batch_in_epoch"] == 0
    sampler.mark_consumed()
    state = sampler.state_dict()
    expected_next = next(iterator)[0]

    restored = DeterministicBatchSampler(dataset_size=24, batch_size=4, seed=7)
    restored.load_state_dict(state)
    restored_next = next(
        iter(DataLoader(dataset, batch_sampler=restored, num_workers=0))
    )[0]

    assert first.shape == (4,)
    assert torch.equal(restored_next, expected_next)


def test_sampler_rejects_an_incompatible_resume() -> None:
    sampler = DeterministicBatchSampler(dataset_size=11, batch_size=4, seed=7)
    state = sampler.state_dict()
    different = DeterministicBatchSampler(dataset_size=12, batch_size=4, seed=7)
    with pytest.raises(ValueError, match="dataset_size 不兼容"):
        different.load_state_dict(state)


def test_sampler_covers_each_epoch_without_duplicates() -> None:
    sampler = DeterministicBatchSampler(dataset_size=11, batch_size=4, seed=7)
    iterator = iter(sampler)
    epoch = next(iterator) + next(iterator) + next(iterator)
    assert sorted(epoch) == list(range(11))
    assert len(torch.unique(torch.tensor(epoch))) == 11


class _ToyTrainingDataset(Dataset[dict[str, torch.Tensor]]):
    def __len__(self) -> int:
        return 17

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        x = torch.tensor([index / 17.0, (index % 5) / 5.0], dtype=torch.float32)
        y = torch.tensor([(index % 7) / 7.0], dtype=torch.float32)
        return {"x": x, "y": y, "index": torch.tensor(index)}


def _make_toy_training() -> tuple[
    nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Dropout(0.25), nn.Linear(8, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    return model, optimizer, scheduler


def _run_toy_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler: DeterministicBatchSampler,
    *,
    steps: int,
    num_workers: int,
) -> tuple[list[float], list[list[int]]]:
    loader = DataLoader(
        _ToyTrainingDataset(),
        batch_sampler=sampler,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(404),
    )
    iterator = iter(loader)
    losses: list[float] = []
    sample_indices: list[list[int]] = []
    for _ in range(steps):
        batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(batch["x"]), batch["y"])
        loss.backward()
        optimizer.step()
        scheduler.step()
        sampler.mark_consumed()
        losses.append(float(loss.item()))
        sample_indices.append(batch["index"].tolist())
    return losses, sample_indices


def _assert_nested_equal(left: Any, right: Any) -> None:
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


@pytest.mark.parametrize("num_workers", [2, 4])
@MULTIWORKER_TEST
def test_multiworker_interrupted_resume_matches_uninterrupted(num_workers: int) -> None:
    torch.manual_seed(29)
    full_model, full_optimizer, full_scheduler = _make_toy_training()
    full_sampler = DeterministicBatchSampler(dataset_size=17, batch_size=4, seed=31)
    full_losses, full_indices = _run_toy_steps(
        full_model,
        full_optimizer,
        full_scheduler,
        full_sampler,
        steps=9,
        num_workers=num_workers,
    )
    full_rng = torch.get_rng_state().clone()

    torch.manual_seed(29)
    first_model, first_optimizer, first_scheduler = _make_toy_training()
    first_sampler = DeterministicBatchSampler(dataset_size=17, batch_size=4, seed=31)
    first_losses, first_indices = _run_toy_steps(
        first_model,
        first_optimizer,
        first_scheduler,
        first_sampler,
        steps=4,
        num_workers=num_workers,
    )
    checkpoint = {
        "model": copy.deepcopy(first_model.state_dict()),
        "optimizer": copy.deepcopy(first_optimizer.state_dict()),
        "scheduler": copy.deepcopy(first_scheduler.state_dict()),
        "sampler": copy.deepcopy(first_sampler.state_dict()),
        "torch_rng": torch.get_rng_state().clone(),
    }

    resumed_model, resumed_optimizer, resumed_scheduler = _make_toy_training()
    resumed_model.load_state_dict(checkpoint["model"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_scheduler.load_state_dict(checkpoint["scheduler"])
    resumed_sampler = DeterministicBatchSampler(dataset_size=17, batch_size=4, seed=31)
    resumed_sampler.load_state_dict(checkpoint["sampler"])
    torch.set_rng_state(checkpoint["torch_rng"])
    resumed_losses, resumed_indices = _run_toy_steps(
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        resumed_sampler,
        steps=5,
        num_workers=num_workers,
    )

    assert first_indices + resumed_indices == full_indices
    assert first_losses + resumed_losses == full_losses
    _assert_nested_equal(resumed_model.state_dict(), full_model.state_dict())
    _assert_nested_equal(resumed_optimizer.state_dict(), full_optimizer.state_dict())
    _assert_nested_equal(resumed_scheduler.state_dict(), full_scheduler.state_dict())
    assert torch.equal(torch.get_rng_state(), full_rng)


def test_random_states_are_safe_to_load_and_restore(tmp_path: Path) -> None:
    checkpoint = tmp_path / "random_states.pt"
    torch.manual_seed(13)
    states = capture_random_states()
    torch.save({"random_states": states}, checkpoint)

    loaded = torch.load(checkpoint, weights_only=True)
    restore_random_states(loaded["random_states"])
    expected = torch.rand(4)
    restore_random_states(loaded["random_states"])
    actual = torch.rand(4)

    assert torch.equal(actual, expected)
