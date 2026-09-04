"""Push-T dataset import, validation, statistics and expert replay."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from mini_wam.data.dataset import NormalizationStats


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BUILTIN_DATASET_ROOT = PROJECT_ROOT / "data" / "lerobot" / "pusht_image"
STUDIO_DATASET_ROOT = PROJECT_ROOT / "artifacts" / "studio" / "datasets"
AUDIT_PATH = PROJECT_ROOT / "artifacts" / "data_audit.json"


class DatasetValidationError(ValueError):
    """Raised when a dataset is not compatible with the Push-T workbench."""


@dataclass(frozen=True)
class DatasetSummary:
    repo_id: str
    root: Path
    fps: int
    episode_count: int
    frame_count: int
    image_shape: tuple[int, int, int]
    state_dim: int
    action_dim: int
    action_min: tuple[float, float]
    action_max: tuple[float, float]
    position_mean: tuple[float, float]
    position_std: tuple[float, float]
    action_mean: tuple[float, float]
    action_std: tuple[float, float]
    episode_ids: tuple[int, ...]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["root"] = str(self.root)
        return result

    def normalization(self) -> NormalizationStats:
        import torch

        return NormalizationStats(
            position_mean=torch.tensor(self.position_mean, dtype=torch.float32),
            position_std=torch.tensor(self.position_std, dtype=torch.float32),
            action_mean=torch.tensor(self.action_mean, dtype=torch.float32),
            action_std=torch.tensor(self.action_std, dtype=torch.float32),
        )


def _feature_shape(info: dict[str, Any], name: str) -> tuple[int, ...]:
    feature = info.get("features", {}).get(name)
    if not isinstance(feature, dict) or "shape" not in feature:
        raise DatasetValidationError(f"缺少特征 {name!r}")
    return tuple(int(value) for value in feature["shape"])


def _read_tables(paths: list[Path], columns: list[str] | None = None) -> pa.Table:
    if not paths:
        raise DatasetValidationError("数据目录中没有 Parquet 文件")
    try:
        return pa.concat_tables([pq.read_table(path, columns=columns) for path in paths])
    except Exception as exc:
        raise DatasetValidationError(f"无法读取 Parquet 数据：{exc}") from exc


def _task_names(root: Path) -> list[str]:
    task_path = root / "meta" / "tasks.parquet"
    if not task_path.exists():
        raise DatasetValidationError("缺少 meta/tasks.parquet")
    table = pq.read_table(task_path)
    if "task" in table.column_names:
        return [str(value) for value in table["task"].to_pylist()]
    # LeRobot v3 stores task text as the pandas index in some published datasets.
    if "__index_level_0__" in table.column_names:
        return [str(value) for value in table["__index_level_0__"].to_pylist()]
    raise DatasetValidationError("任务元数据缺少 task 文本")


def _episode_ranges(root: Path) -> pa.Table:
    paths = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    return _read_tables(paths, ["episode_index", "dataset_from_index", "dataset_to_index"])


def validate_dataset(root: str | Path, repo_id: str = "local/pusht") -> DatasetSummary:
    """Validate the exact data contract consumed by ActionOnlyPolicy."""
    root = Path(root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise DatasetValidationError(f"找不到 {info_path}")
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DatasetValidationError(f"info.json 无法解析：{exc}") from exc

    image_shape = _feature_shape(info, "observation.image")
    state_shape = _feature_shape(info, "observation.state")
    action_shape = _feature_shape(info, "action")
    if image_shape != (96, 96, 3):
        raise DatasetValidationError(f"图像必须为 96×96×3，实际为 {image_shape}")
    if state_shape != (2,):
        raise DatasetValidationError(f"state 必须为 2 维，实际为 {state_shape}")
    if action_shape != (2,):
        raise DatasetValidationError(f"action 必须为 2 维，实际为 {action_shape}")

    tasks = _task_names(root)
    if not tasks or any("push" not in task.lower() or "t" not in task.lower() for task in tasks):
        raise DatasetValidationError(f"只支持 Push-T，数据中的任务为 {tasks or '空'}")

    data = _read_tables(
        sorted((root / "data").glob("**/*.parquet")),
        ["observation.state", "action", "episode_index", "frame_index", "index"],
    )
    indices = np.asarray(data["index"].to_numpy(), dtype=np.int64)
    order = np.argsort(indices)
    indices = indices[order]
    if len(indices) == 0 or not np.array_equal(indices, np.arange(len(indices), dtype=np.int64)):
        raise DatasetValidationError("全局 frame index 必须从 0 连续递增")

    episode_indices = np.asarray(data["episode_index"].to_numpy(), dtype=np.int64)[order]
    frame_indices = np.asarray(data["frame_index"].to_numpy(), dtype=np.int64)[order]
    ranges = _episode_ranges(root)
    episode_ids = tuple(int(value) for value in ranges["episode_index"].to_pylist())
    if episode_ids != tuple(range(len(episode_ids))):
        raise DatasetValidationError("episode_index 必须从 0 连续递增")
    if tuple(int(value) for value in np.unique(episode_indices)) != episode_ids:
        raise DatasetValidationError("数据帧与 episode 元数据不一致")
    for episode_id in episode_ids:
        local_frames = frame_indices[episode_indices == episode_id]
        if not np.array_equal(local_frames, np.arange(len(local_frames), dtype=np.int64)):
            raise DatasetValidationError(f"episode {episode_id} 的 frame_index 不连续")

    states = np.asarray(data["observation.state"].to_pylist(), dtype=np.float32)[order]
    actions = np.asarray(data["action"].to_pylist(), dtype=np.float32)[order]
    if states.shape != (len(indices), 2) or actions.shape != (len(indices), 2):
        raise DatasetValidationError("Parquet 中 state/action 的实际数组形状不匹配")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise DatasetValidationError("state/action 含 NaN 或无穷值")

    position_mean = states.mean(axis=0)
    position_std = states.std(axis=0)
    action_mean = actions.mean(axis=0)
    action_std = actions.std(axis=0)
    # The two legacy project checkpoints were trained with train-split-only
    # statistics saved by audit_dataset.py. Reuse those exact values; using
    # validation episodes here would introduce inference-time data leakage and
    # slightly change every predicted action.
    if root == BUILTIN_DATASET_ROOT.resolve() and AUDIT_PATH.exists():
        audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))["training_normalization"]
        position_mean = np.asarray(audit["agent_position"]["mean"], dtype=np.float32)
        position_std = np.asarray(audit["agent_position"]["std"], dtype=np.float32)
        action_mean = np.asarray(audit["action"]["mean"], dtype=np.float32)
        action_std = np.asarray(audit["action"]["std"], dtype=np.float32)
    fingerprint_payload = {
        "repo_id": repo_id,
        "fps": int(info.get("fps", 10)),
        "episodes": list(episode_ids),
        "frames": len(indices),
        "features": {"image": image_shape, "state": state_shape, "action": action_shape},
        "position_mean": position_mean.tolist(),
        "position_std": position_std.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return DatasetSummary(
        repo_id=repo_id,
        root=root,
        fps=int(info.get("fps", 10)),
        episode_count=len(episode_ids),
        frame_count=len(indices),
        image_shape=(96, 96, 3),
        state_dim=2,
        action_dim=2,
        action_min=tuple(float(value) for value in actions.min(axis=0)),
        action_max=tuple(float(value) for value in actions.max(axis=0)),
        position_mean=tuple(float(value) for value in position_mean),
        position_std=tuple(float(value) for value in position_std),
        action_mean=tuple(float(value) for value in action_mean),
        action_std=tuple(float(value) for value in action_std),
        episode_ids=episode_ids,
        fingerprint=fingerprint,
    )


def import_dataset(source_kind: str, source: str = "") -> DatasetSummary:
    """Resolve built-in, local or Hugging Face data, then validate it."""
    if source_kind == "内置数据":
        return validate_dataset(BUILTIN_DATASET_ROOT, "lerobot/pusht_image")
    if source_kind == "本地目录":
        if not source.strip():
            raise DatasetValidationError("请输入本地 LeRobot 数据目录")
        return validate_dataset(source, Path(source).name)
    if source_kind != "Hugging Face 仓库":
        raise DatasetValidationError(f"未知数据来源：{source_kind}")
    if not source.strip():
        raise DatasetValidationError("请输入 Hugging Face repo ID（仓库标识）")

    from huggingface_hub import snapshot_download

    destination = STUDIO_DATASET_ROOT / source.strip().replace("/", "__")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=source.strip(),
            repo_type="dataset",
            local_dir=destination,
            allow_patterns=["meta/**", "data/**"],
        )
    except Exception as exc:
        if not any(destination.iterdir()):
            shutil.rmtree(destination)
        raise DatasetValidationError(f"下载数据失败：{exc}") from exc
    return validate_dataset(destination, source.strip())


def export_expert_video(summary: DatasetSummary, episode_id: int, output_path: str | Path) -> Path:
    """Decode one expert episode and export a clear 384×384 MP4 replay."""
    if episode_id not in summary.episode_ids:
        raise DatasetValidationError(f"不存在 episode {episode_id}")

    cache_root = PROJECT_ROOT / "data" / ".cache"
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "hf_datasets"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(summary.repo_id, root=summary.root, return_uint8=True)
    episodes = dataset.meta.episodes
    row = list(int(value) for value in episodes["episode_index"]).index(int(episode_id))
    start = int(episodes["dataset_from_index"][row])
    end = int(episodes["dataset_to_index"][row])
    frames: list[np.ndarray] = []
    for index in range(start, end):
        image = dataset[index]["observation.image"].permute(1, 2, 0).cpu().numpy()
        image = np.asarray(image, dtype=np.uint8)
        frames.append(np.asarray(Image.fromarray(image).resize((384, 384), Image.Resampling.NEAREST)))

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, np.stack(frames), fps=summary.fps, codec="libx264", out_pixel_format="yuv420p")
    return output
