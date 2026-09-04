"""Audit the local LeRobot Push-T dataset before creating training windows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "lerobot" / "pusht_image"
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "data_audit.json"
DEFAULT_MONTAGE_PATH = PROJECT_ROOT / "artifacts" / "data_samples.png"
DEFAULT_SPLIT_PATH = PROJECT_ROOT / "splits" / "episodes_seed42.json"

# Keep library caches inside the project instead of writing to a user-global path.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "data" / ".cache" / "hf_datasets"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_scalar_table(dataset_root: Path) -> dict[str, np.ndarray]:
    parquet_paths = sorted((dataset_root / "data").glob("**/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No data parquet files found under {dataset_root}")

    columns = [
        "observation.state",
        "action",
        "episode_index",
        "frame_index",
        "timestamp",
        "next.reward",
        "next.done",
        "next.success",
        "index",
    ]
    table = pa.concat_tables([pq.read_table(path, columns=columns) for path in parquet_paths])
    order = np.argsort(table["index"].to_numpy())

    return {
        "state": np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[order],
        "action": np.asarray(table["action"].to_pylist(), dtype=np.float32)[order],
        "episode": table["episode_index"].to_numpy()[order].astype(np.int64),
        "frame": table["frame_index"].to_numpy()[order].astype(np.int64),
        "timestamp": table["timestamp"].to_numpy()[order].astype(np.float64),
        "reward": table["next.reward"].to_numpy()[order].astype(np.float32),
        "done": table["next.done"].to_numpy()[order].astype(bool),
        "success": table["next.success"].to_numpy()[order].astype(bool),
        "index": table["index"].to_numpy()[order].astype(np.int64),
    }


def create_episode_splits(episode_ids: np.ndarray, seed: int) -> dict[str, Any]:
    unique_ids = np.unique(episode_ids)
    if len(unique_ids) < 10:
        raise ValueError("At least 10 episodes are required for the validation split")

    shuffled = np.random.default_rng(seed).permutation(unique_ids)
    validation_count = max(10, round(0.10 * len(unique_ids)))

    return {
        "seed": seed,
        "ratios": {"train": 0.90, "validation": 0.10},
        "train": sorted(shuffled[:-validation_count].astype(int).tolist()),
        "validation": sorted(shuffled[-validation_count:].astype(int).tolist()),
    }


def vector_stats(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
    }


def audit_structure(rows: dict[str, np.ndarray], fps: int) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    episode_to_rows: dict[int, np.ndarray] = {}
    frame_errors: list[int] = []
    timestamp_errors: list[int] = []
    terminal_errors: list[int] = []

    for episode_id in np.unique(rows["episode"]):
        indices = np.flatnonzero(rows["episode"] == episode_id)
        episode_to_rows[int(episode_id)] = indices
        length = len(indices)

        if not np.array_equal(rows["frame"][indices], np.arange(length)):
            frame_errors.append(int(episode_id))
        expected_timestamps = np.arange(length, dtype=np.float64) / fps
        if not np.allclose(rows["timestamp"][indices], expected_timestamps, atol=1e-4):
            timestamp_errors.append(int(episode_id))
        if not rows["done"][indices[-1]]:
            terminal_errors.append(int(episode_id))

    expected_indices = np.arange(len(rows["index"]), dtype=np.int64)
    lengths = np.asarray([len(indices) for indices in episode_to_rows.values()])

    # Heuristic semantic check: after action_t, the next agent position should
    # usually be closer to the absolute action target than the current position.
    current_rows: list[int] = []
    next_rows: list[int] = []
    for indices in episode_to_rows.values():
        current_rows.extend(indices[:-1])
        next_rows.extend(indices[1:])
    current = np.asarray(current_rows)
    following = np.asarray(next_rows)
    distance_before = np.linalg.norm(rows["action"][current] - rows["state"][current], axis=1)
    distance_after = np.linalg.norm(rows["action"][current] - rows["state"][following], axis=1)
    closer_fraction = float(np.mean(distance_after <= distance_before + 1e-6))

    summary = {
        "total_episodes": len(episode_to_rows),
        "total_frames": len(rows["index"]),
        "episode_length": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
        },
        "global_index_is_contiguous": bool(np.array_equal(rows["index"], expected_indices)),
        "duplicate_global_indices": int(len(rows["index"]) - len(np.unique(rows["index"]))),
        "frame_index_error_episodes": frame_errors,
        "timestamp_error_episodes": timestamp_errors,
        "terminal_flag_error_episodes": terminal_errors,
        "action_semantics_heuristic": {
            "description": "fraction where next agent position is no farther from action_t target",
            "fraction": closer_fraction,
            "is_structural_proof": False,
        },
    }
    return summary, episode_to_rows


def tensor_to_image(tensor: Any) -> Image.Image:
    array = tensor.detach().cpu().numpy()
    if array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * 255.0, 0, 255)
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


def create_montage(dataset_root: Path, rows: dict[str, np.ndarray], episode_to_rows: dict[int, np.ndarray]) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("lerobot/pusht_image", root=dataset_root, return_uint8=True)
    candidates: list[tuple[int, int, int]] = []
    for episode_id, indices in episode_to_rows.items():
        # Require two history frames and four complete future frames for this visualization.
        for local_t in range(1, len(indices) - 4):
            candidates.append((episode_id, local_t, int(indices[local_t])))

    chosen = np.random.default_rng(42).choice(len(candidates), size=16, replace=False)
    labels = ["history -1", "current", "future +1", "future +2", "future +3", "future +4"]
    cell_width, image_height, text_height = 6 * 96, 96, 42
    canvas = Image.new("RGB", (2 * cell_width, 8 * (image_height + text_height)), "white")
    draw = ImageDraw.Draw(canvas)

    for slot, candidate_index in enumerate(chosen):
        episode_id, local_t, global_t = candidates[int(candidate_index)]
        indices = episode_to_rows[episode_id]
        frame_indices = [indices[local_t - 1], indices[local_t], *indices[local_t + 1 : local_t + 5]]
        panel_x = (slot % 2) * cell_width
        panel_y = (slot // 2) * (image_height + text_height)

        for column, frame_index in enumerate(frame_indices):
            image = tensor_to_image(dataset[int(frame_index)]["observation.image"])
            canvas.paste(image, (panel_x + column * 96, panel_y))
            if slot < 2:
                draw.text((panel_x + column * 96 + 3, panel_y + 3), labels[column], fill="black")

        action_prefix = rows["action"][indices[local_t : local_t + 4]]
        action_text = " ".join(f"({x:.0f},{y:.0f})" for x, y in action_prefix)
        draw.text((panel_x + 3, panel_y + image_height + 3), f"ep={episode_id} t={local_t} a[t:t+4]={action_text}", fill="black")

    DEFAULT_MONTAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(DEFAULT_MONTAGE_PATH)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = int(info["fps"])
    rows = load_scalar_table(dataset_root)
    structure, episode_to_rows = audit_structure(rows, fps=fps)
    splits = create_episode_splits(rows["episode"], seed=args.seed)

    split_sets = {name: set(splits[name]) for name in ("train", "validation")}
    split_overlap = {
        "train_validation": sorted(split_sets["train"] & split_sets["validation"]),
    }
    train_mask = np.isin(rows["episode"], splits["train"])

    audit = {
        "dataset": {
            "repo_id": "lerobot/pusht_image",
            "root": str(dataset_root),
            "fps": fps,
            "image_shape_hwc": info["features"]["observation.image"]["shape"],
            "license": "MIT",
        },
        "structure": structure,
        "ranges": {
            "agent_position": vector_stats(rows["state"]),
            "action": vector_stats(rows["action"]),
            "reward": {
                "min": float(rows["reward"].min()),
                "max": float(rows["reward"].max()),
            },
        },
        "splits": {
            "counts": {name: len(ids) for name, ids in split_sets.items()},
            "overlap": split_overlap,
        },
        "training_normalization": {
            "agent_position": vector_stats(rows["state"][train_mask]),
            "action": vector_stats(rows["action"][train_mask]),
            "std_floor": 1e-6,
        },
        "candidate_window_count": int(sum(max(len(indices) - 2, 0) for indices in episode_to_rows.values())),
    }

    DEFAULT_SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SPLIT_PATH.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    DEFAULT_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_ARTIFACT_PATH.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    create_montage(dataset_root, rows, episode_to_rows)

    print(json.dumps(audit, indent=2))
    print(f"\nSaved audit: {DEFAULT_ARTIFACT_PATH}")
    print(f"Saved splits: {DEFAULT_SPLIT_PATH}")
    print(f"Saved samples: {DEFAULT_MONTAGE_PATH}")


if __name__ == "__main__":
    main()
