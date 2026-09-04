"""Reproducible closed-loop evaluation for trained action-only policies."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mini_wam.data import NormalizationStats
from mini_wam.evaluation.scenes import load_scene_split
from mini_wam.studio.checkpoints import checkpoint_normalization, load_checkpoint
from mini_wam.studio.datasets import validate_dataset
from mini_wam.studio.pusht import SceneState, run_rollout


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前运行时不可用")
    return device


def summarize_policy_episodes(episodes: list[dict[str, Any]]) -> dict[str, float | int]:
    if not episodes:
        raise ValueError("至少需要一个评测回合")
    success = np.asarray([item["is_success"] for item in episodes], dtype=np.float64)
    final_coverage = np.asarray([item["final_coverage"] for item in episodes], dtype=np.float64)
    max_coverage = np.asarray([item["max_coverage"] for item in episodes], dtype=np.float64)
    rewards = np.asarray([item["reward"] for item in episodes], dtype=np.float64)
    steps = np.asarray([item["steps"] for item in episodes], dtype=np.float64)
    inference_ms = np.asarray([item["mean_inference_ms"] for item in episodes], dtype=np.float64)
    return {
        "success_count": int(success.sum()),
        "success_rate": float(success.mean()),
        "mean_final_coverage": float(final_coverage.mean()),
        "median_final_coverage": float(np.median(final_coverage)),
        "mean_max_coverage": float(max_coverage.mean()),
        "median_max_coverage": float(np.median(max_coverage)),
        "mean_reward": float(rewards.mean()),
        "mean_steps": float(steps.mean()),
        "mean_inference_ms": float(inference_ms.mean()),
    }


def evaluate_action_only_policy(
    checkpoint_path: str | Path,
    development_path: str | Path,
    dataset_root: str | Path,
    normalization_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    max_steps: int = 300,
    execute_steps: int = 4,
    checkpoint_selection: str = "best_validation_loss",
) -> dict[str, Any]:
    """Evaluate one checkpoint on the sealed development split.

    This stage-3 evaluator deliberately rejects the final-test split. Final
    evaluation gets a separate, explicitly authorized stage-5 entry point.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    development_path = Path(development_path).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    normalization_path = Path(normalization_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    scenes = load_scene_split(development_path, expected_split="development")
    if max_steps < 1:
        raise ValueError("max_steps 必须为正数")
    if execute_steps != 4:
        raise ValueError("正式评估固定每次执行动作块前 4 步")

    dataset = validate_dataset(dataset_root, "lerobot/pusht_image")
    fallback = NormalizationStats.from_audit_file(normalization_path)
    device = _resolve_device(device_name)
    model, checkpoint_info = load_checkpoint(checkpoint_path, device=device)
    normalization = checkpoint_normalization(
        checkpoint_path,
        fallback,
        active_dataset_fingerprint=dataset.fingerprint,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    episodes: list[dict[str, Any]] = []
    for record in scenes["scenes"]:
        scene = SceneState(**{key: float(value) for key, value in record["state"].items()})
        video_path = video_dir / f"{record['scene_id']}.mp4"
        updates = list(
            run_rollout(
                model,
                normalization,
                scene,
                video_path,
                max_steps=max_steps,
                execute_steps=execute_steps,
            )
        )
        metrics = updates[-1].metrics.to_dict()
        episodes.append(
            {
                "scene_id": record["scene_id"],
                "environment_seed": record["environment_seed"],
                **metrics,
                "video": str(video_path.relative_to(output_dir)),
            }
        )

    elapsed_seconds = time.perf_counter() - started
    summary = summarize_policy_episodes(episodes)
    mean_inference_ms = float(summary["mean_inference_ms"])
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": "action_only_v1",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
            "step": checkpoint_info.step,
            "validation_loss": checkpoint_info.validation_loss,
            "selection_policy": checkpoint_selection,
        },
        "scene_file": str(development_path),
        "scenes_hash": scenes["scenes_hash"],
        "dataset_fingerprint": dataset.fingerprint,
        "protocol": {
            "split": "development",
            "episode_count": len(episodes),
            "max_steps": max_steps,
            "action_horizon": 16,
            "execute_steps": execute_steps,
        },
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "peak_vram_mb": (
                float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None
            ),
            "elapsed_seconds": elapsed_seconds,
            "model_calls_per_second": 1000.0 / mean_inference_ms if mean_inference_ms > 0 else None,
        },
        "summary": summary,
        "episodes": episodes,
    }
    _atomic_json_write(output_dir / "results.json", result)
    with (output_dir / "episodes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes[0].keys()))
        writer.writeheader()
        writer.writerows(episodes)
    return result
