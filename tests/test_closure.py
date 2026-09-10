"""Evidence integrity, pairing, and model-backed counterfactual contracts."""
import copy
import shutil

import pytest
import torch

from mini_wam.evaluation.closure import (
    EVIDENCE_ROOT, PROJECT_ROOT, audit_archive, load_closure_scenes,
    paired_statistics, read_json,
)
from mini_wam.evaluation.closure_runner import (
    counterfactual_batch, load_frozen_policy, reflected_action_prefix,
)
from mini_wam.data import MiniWAMDataset, NormalizationStats, load_episode_split
from mini_wam.models.future_head import FutureHeadPolicy


def test_archive_recomputes_known_paired_outcomes():
    audit = audit_archive()
    assert audit["original_files_verified"] == 14
    assert audit["summary"]["action_only"]["success_count"] == 2
    assert audit["summary"]["future_aware"]["success_count"] == 5
    assert audit["mcnemar_exact_two_sided_p"] == 0.375


def test_corrupted_archive_is_rejected(tmp_path):
    destination = tmp_path / "archive"
    shutil.copytree(EVIDENCE_ROOT, destination)
    with (destination / "results.json").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="checksum"):
        audit_archive(destination)


@pytest.mark.parametrize("mutation", ["duplicate", "seed", "success"])
def test_bad_pairs_are_rejected(mutation):
    episodes = copy.deepcopy(read_json(EVIDENCE_ROOT / "results.json")["episodes"])
    if mutation == "duplicate":
        episodes["future_aware"][-1] = episodes["future_aware"][0]
    elif mutation == "seed":
        episodes["future_aware"][0]["seed"] += 1
    else:
        episodes["future_aware"][0]["is_success"] = False
    with pytest.raises(ValueError):
        paired_statistics(episodes, load_closure_scenes())


def test_closure_rejects_sealed_test_and_wrong_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="frozen closure development"):
        load_closure_scenes(PROJECT_ROOT / "splits/final_test_scenes.json")
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"not the frozen checkpoint")
    with pytest.raises(ValueError, match="checkpoint hash"):
        load_frozen_policy(wrong, "future_aware", torch.device("cpu"))


def test_absolute_action_reflection_uses_current_position_and_clips():
    norm = NormalizationStats(torch.tensor([10., 20.]), torch.tensor([2., 3.]),
                              torch.tensor([50., 60.]), torch.tensor([4., 5.]))
    positions = norm.normalize_position(torch.tensor([[[25., 30.], [100., 200.]]]))
    raw_actions = torch.tensor([[[150., 100.], [400., -200.], [100., 200.], [50., 250.]]])
    prefix = norm.normalize_action(raw_actions)
    actual = norm.denormalize_action(reflected_action_prefix(prefix, positions, norm))
    torch.testing.assert_close(actual, torch.tensor([[[50., 300.], [0., 512.], [100., 200.], [150., 150.]]]))


def test_real_data_counterfactual_batch_is_finite_without_parameter_updates():
    root = PROJECT_ROOT / "data/lerobot/pusht_image"
    if not root.exists():
        pytest.skip("Local Push-T dataset is not downloaded")
    norm = NormalizationStats.from_audit_file(PROJECT_ROOT / "artifacts/data_audit.json")
    dataset = MiniWAMDataset(root, load_episode_split(PROJECT_ROOT / "splits/episodes_seed42.json", "validation"), norm)
    batch = torch.utils.data.default_collate([dataset[0], dataset[100]])
    model = FutureHeadPolicy(pretrained=False, target_pretrained=False).eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    losses, valid = counterfactual_batch(model, batch, norm, torch.device("cpu"))
    assert set(losses) == {"correct", "shuffled", "reversed"}
    assert valid == int(batch["future_valid_mask"].sum())
    assert all(0 <= x <= 2 for x in losses.values())
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])
