from __future__ import annotations

import json
from pathlib import Path

import pytest

from mini_wam.evaluation.scenes import (
    evaluate_random_policy,
    generate_scene_splits,
    load_scene_split,
)
from mini_wam.evaluation.policy import (
    evaluate_action_only_policy,
    summarize_policy_episodes,
)


def test_scene_generation_is_deterministic_and_disjoint() -> None:
    first_dev, first_final = generate_scene_splits(
        generation_seed=9, development_count=2, final_test_count=3
    )
    second_dev, second_final = generate_scene_splits(
        generation_seed=9, development_count=2, final_test_count=3
    )
    assert first_dev == second_dev
    assert first_final == second_final
    dev_states = {tuple(item["state"].values()) for item in first_dev["scenes"]}
    final_states = {tuple(item["state"].values()) for item in first_final["scenes"]}
    assert dev_states.isdisjoint(final_states)


def test_scene_loader_detects_tampering(tmp_path: Path) -> None:
    development, _ = generate_scene_splits(
        generation_seed=10, development_count=1, final_test_count=1
    )
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(development), encoding="utf-8")
    load_scene_split(path, expected_split="development")
    development["scenes"][0]["state"]["agent_x"] += 1
    path.write_text(json.dumps(development), encoding="utf-8")
    with pytest.raises(ValueError, match="哈希不一致"):
        load_scene_split(path, expected_split="development")


def test_random_evaluator_rejects_final_test_split(tmp_path: Path) -> None:
    _, final_test = generate_scene_splits(
        generation_seed=11, development_count=1, final_test_count=1
    )
    path = tmp_path / "final.json"
    path.write_text(json.dumps(final_test), encoding="utf-8")
    with pytest.raises(ValueError, match="期望 development"):
        evaluate_random_policy(path, tmp_path / "result.json", tmp_path / "result.csv")


def test_action_only_evaluator_rejects_final_test_before_loading_checkpoint(tmp_path: Path) -> None:
    _, final_test = generate_scene_splits(
        generation_seed=12, development_count=1, final_test_count=1
    )
    path = tmp_path / "final.json"
    path.write_text(json.dumps(final_test), encoding="utf-8")
    with pytest.raises(ValueError, match="期望 development"):
        evaluate_action_only_policy(
            tmp_path / "missing.pt",
            path,
            tmp_path / "missing-data",
            tmp_path / "missing-normalization.json",
            tmp_path / "results",
        )


def test_policy_summary_aggregates_episode_metrics() -> None:
    episodes = [
        {
            "is_success": False,
            "final_coverage": 0.2,
            "max_coverage": 0.4,
            "reward": 1.0,
            "steps": 300,
            "mean_inference_ms": 10.0,
        },
        {
            "is_success": True,
            "final_coverage": 0.9,
            "max_coverage": 0.95,
            "reward": 5.0,
            "steps": 120,
            "mean_inference_ms": 14.0,
        },
    ]
    summary = summarize_policy_episodes(episodes)
    assert summary["success_count"] == 1
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["mean_reward"] == pytest.approx(3.0)
    assert summary["mean_inference_ms"] == pytest.approx(12.0)
