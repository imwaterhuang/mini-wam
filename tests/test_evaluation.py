from __future__ import annotations

import json
from pathlib import Path

import pytest

from mini_wam.evaluation.scenes import (
    evaluate_random_policy,
    generate_scene_splits,
    load_scene_split,
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

