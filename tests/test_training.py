from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mini_wam.training.action_only import (
    DeterministicBatchSampler,
    _capture_random_states,
    _restore_random_states,
    _sync_run_directory,
    load_training_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    _sync_run_directory(source, destination)
    assert (destination / "train_metrics.csv").read_text(encoding="utf-8").endswith("1,0.5\n")
    assert (destination / "checkpoints" / "last.pt").read_bytes() == b"first"

    (source / "train_metrics.csv").write_text("step,loss\n1,0.5\n2,0.4\n", encoding="utf-8")
    (source / "checkpoints" / "last.pt").write_bytes(b"second-version")
    _sync_run_directory(source, destination)
    assert (destination / "train_metrics.csv").read_text(encoding="utf-8").endswith("2,0.4\n")
    assert (destination / "checkpoints" / "last.pt").read_bytes() == b"second-version"
    assert not list(destination.rglob("*.sync-tmp"))


def test_sampler_resumes_at_the_exact_next_batch() -> None:
    uninterrupted = DeterministicBatchSampler(dataset_size=11, batch_size=4, seed=7)
    uninterrupted_iterator = iter(uninterrupted)
    first = next(uninterrupted_iterator)
    state = uninterrupted.state_dict()
    expected_next = next(uninterrupted_iterator)

    restored = DeterministicBatchSampler(dataset_size=11, batch_size=4, seed=7)
    restored.load_state_dict(state)
    actual_next = next(iter(restored))

    assert len(first) == 4
    assert actual_next == expected_next


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


def test_random_states_are_safe_to_load_and_restore(tmp_path: Path) -> None:
    checkpoint = tmp_path / "random_states.pt"
    torch.manual_seed(13)
    states = _capture_random_states()
    torch.save({"random_states": states}, checkpoint)

    loaded = torch.load(checkpoint, weights_only=True)
    _restore_random_states(loaded["random_states"])
    expected = torch.rand(4)
    _restore_random_states(loaded["random_states"])
    actual = torch.rand(4)

    assert torch.equal(actual, expected)
