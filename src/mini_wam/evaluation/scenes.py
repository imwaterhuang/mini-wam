"""Freeze Push-T evaluation scenes and measure a deterministic random baseline."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from mini_wam.studio.pusht import (
    SceneState,
    make_pusht_env,
    random_scene,
    validate_scene,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = 1
DEFAULT_GENERATION_SEED = 42
DEFAULT_RANDOM_POLICY_SEED = 20260904


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _freeze_json(path: Path, payload: dict[str, Any]) -> None:
    """Write once, or verify that an existing frozen file is identical."""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"拒绝覆盖已冻结且内容不同的文件：{path}")
        return
    _atomic_json_write(path, payload)


def _scene_payload(split: str, generation_seed: int, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "pusht",
        "split": split,
        "generation_seed": generation_seed,
        "generator": "mini_wam.studio.pusht.random_scene",
        "count": len(records),
        "scenes_hash": _content_hash(records),
        "scenes": records,
    }


def generate_scene_splits(
    *,
    generation_seed: int = DEFAULT_GENERATION_SEED,
    development_count: int = 20,
    final_test_count: int = 50,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate disjoint valid scenes without executing evaluation episodes."""
    if development_count < 1 or final_test_count < 1:
        raise ValueError("开发和最终测试场景数量必须为正数")

    rng = np.random.default_rng(generation_seed)
    required = development_count + final_test_count
    records: list[dict[str, Any]] = []
    used_environment_seeds: set[int] = set()
    used_states: set[tuple[float, ...]] = set()
    attempts = 0
    while len(records) < required:
        attempts += 1
        if attempts > required * 100:
            raise RuntimeError("无法生成足够多的合法且不重复场景")
        environment_seed = int(rng.integers(0, 2**31 - 1))
        if environment_seed in used_environment_seeds:
            continue
        scene = random_scene(environment_seed)
        try:
            validate_scene(scene)
        except ValueError:
            continue
        state_key = tuple(float(value) for value in scene.as_array())
        if state_key in used_states:
            continue
        used_environment_seeds.add(environment_seed)
        used_states.add(state_key)
        records.append(
            {
                "environment_seed": environment_seed,
                "state": asdict(scene),
            }
        )

    development_records = [
        {"scene_id": f"dev-{index:03d}", **record}
        for index, record in enumerate(records[:development_count])
    ]
    final_records = [
        {"scene_id": f"final-{index:03d}", **record}
        for index, record in enumerate(records[development_count:])
    ]
    return (
        _scene_payload("development", generation_seed, development_records),
        _scene_payload("final_test", generation_seed, final_records),
    )


def freeze_evaluation_scenes(
    development_path: str | Path,
    final_test_path: str | Path,
    manifest_path: str | Path,
    *,
    generation_seed: int = DEFAULT_GENERATION_SEED,
) -> dict[str, Any]:
    """Freeze 20 development and 50 sealed final-test Push-T scenes."""
    development_path = Path(development_path).expanduser().resolve()
    final_test_path = Path(final_test_path).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    development, final_test = generate_scene_splits(generation_seed=generation_seed)
    _freeze_json(development_path, development)
    _freeze_json(final_test_path, final_test)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generation_seed": generation_seed,
        "development": {
            "path": str(development_path.relative_to(PROJECT_ROOT)),
            "count": development["count"],
            "file_sha256": _file_hash(development_path),
            "scenes_hash": development["scenes_hash"],
        },
        "final_test": {
            "path": str(final_test_path.relative_to(PROJECT_ROOT)),
            "count": final_test["count"],
            "file_sha256": _file_hash(final_test_path),
            "scenes_hash": final_test["scenes_hash"],
            "sealed": True,
            "evaluation_allowed_from_stage": 5,
        },
    }
    _freeze_json(manifest_path, manifest)
    return manifest


def load_scene_split(path: str | Path, expected_split: str | None = None) -> dict[str, Any]:
    """Validate schema, hash, identifiers, uniqueness, and scene physics."""
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("task") != "pusht":
        raise ValueError("未知的评测场景文件格式")
    if expected_split and payload.get("split") != expected_split:
        raise ValueError(f"期望 {expected_split} 场景，实际为 {payload.get('split')}")
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or payload.get("count") != len(scenes):
        raise ValueError("场景数量与 count 不一致")
    if payload.get("scenes_hash") != _content_hash(scenes):
        raise ValueError("场景内容哈希不一致，文件可能已被修改")

    ids: set[str] = set()
    environment_seeds: set[int] = set()
    states: set[tuple[float, ...]] = set()
    for record in scenes:
        scene_id = str(record["scene_id"])
        environment_seed = int(record["environment_seed"])
        scene = SceneState(**{key: float(value) for key, value in record["state"].items()})
        validate_scene(scene)
        state_key = tuple(float(value) for value in scene.as_array())
        if scene_id in ids or environment_seed in environment_seeds or state_key in states:
            raise ValueError("场景 ID、环境种子或状态存在重复")
        ids.add(scene_id)
        environment_seeds.add(environment_seed)
        states.add(state_key)
    return payload


def _run_random_episode(
    scene: SceneState,
    *,
    action_seed: int,
    max_steps: int,
) -> dict[str, Any]:
    env = make_pusht_env(max_steps=max_steps)
    env.action_space.seed(action_seed)
    reward_sum = 0.0
    final_coverage = 0.0
    max_coverage = 0.0
    success = False
    steps = 0
    try:
        _, info = env.reset(options={"reset_to_state": scene.as_array()})
        terminated = truncated = False
        while steps < max_steps and not terminated and not truncated:
            _, reward, terminated, truncated, info = env.step(env.action_space.sample())
            steps += 1
            reward_sum += float(reward)
            final_coverage = float(info.get("coverage", 0.0))
            max_coverage = max(max_coverage, final_coverage)
            success = bool(info.get("is_success", False))
    finally:
        env.close()
    return {
        "is_success": success,
        # Quantize far below meaningful reporting precision so harmless
        # platform-level floating noise does not change the result artifact.
        "final_coverage": round(final_coverage, 12),
        "max_coverage": round(max_coverage, 12),
        "reward": round(reward_sum, 12),
        "steps": steps,
    }


def evaluate_random_policy(
    development_path: str | Path,
    output_json: str | Path,
    output_csv: str | Path,
    *,
    policy_seed: int = DEFAULT_RANDOM_POLICY_SEED,
    max_steps: int = 300,
) -> dict[str, Any]:
    """Evaluate only the development split; final-test files are rejected."""
    development_path = Path(development_path).expanduser().resolve()
    output_json = Path(output_json).expanduser().resolve()
    output_csv = Path(output_csv).expanduser().resolve()
    payload = load_scene_split(development_path, expected_split="development")

    episodes: list[dict[str, Any]] = []
    for index, record in enumerate(payload["scenes"]):
        action_seed = policy_seed + index
        scene = SceneState(**{key: float(value) for key, value in record["state"].items()})
        metrics = _run_random_episode(scene, action_seed=action_seed, max_steps=max_steps)
        episodes.append(
            {
                "scene_id": record["scene_id"],
                "environment_seed": record["environment_seed"],
                "action_seed": action_seed,
                **metrics,
            }
        )

    final_coverages = np.asarray([item["final_coverage"] for item in episodes], dtype=np.float64)
    max_coverages = np.asarray([item["max_coverage"] for item in episodes], dtype=np.float64)
    rewards = np.asarray([item["reward"] for item in episodes], dtype=np.float64)
    steps = np.asarray([item["steps"] for item in episodes], dtype=np.float64)
    result = {
        "schema_version": SCHEMA_VERSION,
        "policy": "uniform_random_absolute_target_each_environment_step",
        "policy_seed": policy_seed,
        "scene_file": str(development_path.relative_to(PROJECT_ROOT)),
        "scenes_hash": payload["scenes_hash"],
        "max_steps": max_steps,
        "episode_count": len(episodes),
        "summary": {
            "success_count": sum(int(item["is_success"]) for item in episodes),
            "success_rate": float(np.mean([item["is_success"] for item in episodes])),
            "mean_final_coverage": float(final_coverages.mean()),
            "median_final_coverage": float(np.median(final_coverages)),
            "mean_max_coverage": float(max_coverages.mean()),
            "median_max_coverage": float(np.median(max_coverages)),
            "mean_reward": float(rewards.mean()),
            "mean_steps": float(steps.mean()),
        },
        "episodes": episodes,
    }
    _atomic_json_write(output_json, result)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fields = list(episodes[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(episodes)
    return result
