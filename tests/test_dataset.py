from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mini_wam.data import MiniWAMDataset, NormalizationStats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "lerobot" / "pusht_image"
AUDIT_PATH = PROJECT_ROOT / "artifacts" / "data_audit.json"
SPLIT_PATH = PROJECT_ROOT / "splits" / "episodes_seed42.json"


@pytest.fixture(scope="module")
def dataset() -> MiniWAMDataset:
    if not DATASET_ROOT.exists():
        pytest.skip("Local Push-T dataset is not downloaded")
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    stats = NormalizationStats.from_audit_file(AUDIT_PATH)
    return MiniWAMDataset(DATASET_ROOT, split["train"][:2], stats)


def test_sample_contract(dataset: MiniWAMDataset) -> None:
    sample = dataset[0]
    assert sample["observation_history"].shape == (2, 3, 96, 96)
    assert sample["agent_position"].shape == (2, 2)
    assert sample["action_chunk"].shape == (16, 2)
    assert sample["action_valid_mask"].shape == (16,)
    assert sample["future_observations"].shape == (4, 3, 96, 96)
    assert sample["future_valid_mask"].shape == (4,)
    assert sample["action_valid_mask"].dtype == torch.bool
    assert sample["future_valid_mask"].dtype == torch.bool


def test_first_window_time_alignment(dataset: MiniWAMDataset) -> None:
    sample = dataset[0]
    episode_id, local_t, global_t, _ = dataset.source_indices(0)
    assert local_t == 1
    assert sample["episode_id"].item() == episode_id
    assert sample["start_step"].item() == 1

    raw_action = dataset.source[global_t]["action"].float()
    recovered_action = dataset.normalization.denormalize_action(sample["action_chunk"][0])
    assert torch.allclose(recovered_action, raw_action, atol=1e-4)

    raw_future = dataset.source[global_t + 1]["observation.image"].float()
    recovered_future = sample["future_observations"][0] * IMAGENET_STD + IMAGENET_MEAN
    assert torch.allclose(recovered_future, raw_future, atol=1e-6)


def test_tail_padding_stays_inside_episode(dataset: MiniWAMDataset) -> None:
    first_episode = dataset.source_indices(0)[0]
    tail_index = max(
        index for index in range(len(dataset)) if dataset.source_indices(index)[0] == first_episode
    )
    sample = dataset[tail_index]
    _, local_t, global_t, episode_end = dataset.source_indices(tail_index)

    assert global_t == episode_end - 2
    assert local_t >= 1
    assert sample["action_valid_mask"].sum().item() == 1
    assert sample["future_valid_mask"].sum().item() == 1
    assert torch.count_nonzero(sample["action_chunk"][1:]).item() == 0
    assert torch.equal(sample["future_observations"][1], sample["future_observations"][0])
    assert torch.equal(sample["future_observations"][3], sample["future_observations"][0])


def test_normalization_round_trip(dataset: MiniWAMDataset) -> None:
    positions = torch.tensor([[10.0, 20.0], [300.0, 400.0]])
    actions = torch.tensor([[30.0, 40.0], [500.0, 250.0]])
    stats = dataset.normalization
    assert torch.allclose(stats.denormalize_position(stats.normalize_position(positions)), positions)
    assert torch.allclose(stats.denormalize_action(stats.normalize_action(actions)), actions)


# Imported explicitly to make the image round-trip expectation readable.
from mini_wam.data.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

