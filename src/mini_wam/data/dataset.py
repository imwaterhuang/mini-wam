"""Convert LeRobot Push-T episodes into Mini-WAM training windows."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class NormalizationStats:
    """Training-only statistics for agent positions and absolute actions."""

    position_mean: Tensor
    position_std: Tensor
    action_mean: Tensor
    action_std: Tensor
    std_floor: float = 1e-6

    @classmethod
    def from_audit_file(cls, path: str | Path) -> "NormalizationStats":
        audit = json.loads(Path(path).read_text(encoding="utf-8"))
        stats = audit["training_normalization"]
        return cls(
            position_mean=torch.tensor(
                stats["agent_position"]["mean"], dtype=torch.float32
            ),
            position_std=torch.tensor(
                stats["agent_position"]["std"], dtype=torch.float32
            ),
            action_mean=torch.tensor(stats["action"]["mean"], dtype=torch.float32),
            action_std=torch.tensor(stats["action"]["std"], dtype=torch.float32),
            std_floor=float(stats["std_floor"]),
        )

    def _safe_std(self, std: Tensor) -> Tensor:
        return std.clamp_min(self.std_floor)

    def normalize_position(self, value: Tensor) -> Tensor:
        return (value - self.position_mean) / self._safe_std(self.position_std)

    def denormalize_position(self, value: Tensor) -> Tensor:
        return value * self._safe_std(self.position_std) + self.position_mean

    def normalize_action(self, value: Tensor) -> Tensor:
        return (value - self.action_mean) / self._safe_std(self.action_std)

    def denormalize_action(self, value: Tensor) -> Tensor:
        return value * self._safe_std(self.action_std) + self.action_mean


def load_episode_split(path: str | Path, split: str) -> list[int]:
    split_data = json.loads(Path(path).read_text(encoding="utf-8"))
    if split not in ("train", "validation"):
        raise ValueError(f"Unknown split {split!r}; expected 'train' or 'validation'")
    return [int(episode_id) for episode_id in split_data[split]]


class MiniWAMDataset(Dataset[dict[str, Tensor]]):
    """A time-window view over complete LeRobot Push-T episodes.

    A valid sample starts at local step ``t >= 1`` and requires at least
    ``observation_t``, ``action_t`` and ``observation_{t+1}``. The final row of
    an episode is therefore never treated as a valid action target.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        episode_ids: list[int],
        normalization: NormalizationStats,
        action_horizon: int = 16,
        future_horizon: int = 4,
        include_future_observations: bool = True,
        image_cache_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if action_horizon < 1 or future_horizon < 1:
            raise ValueError("Horizons must be positive")

        self.dataset_root = Path(dataset_root).resolve()
        self.normalization = normalization
        self.action_horizon = action_horizon
        self.future_horizon = future_horizon
        self.include_future_observations = include_future_observations

        project_root = self.dataset_root.parents[2]
        os.environ.setdefault(
            "HF_HOME", str(project_root / "data" / ".cache" / "huggingface")
        )
        os.environ.setdefault(
            "HF_DATASETS_CACHE", str(project_root / "data" / ".cache" / "hf_datasets")
        )

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.source = LeRobotDataset(
            "lerobot/pusht_image",
            root=self.dataset_root,
            return_uint8=True,
        )
        self._states, self._actions = self._load_scalar_tensors()
        self._windows = self._build_window_index(set(episode_ids))
        if not self._windows:
            raise ValueError("The selected episodes produced no valid training windows")
        self._image_cache = None
        if image_cache_path is not None:
            from .image_cache import NormalizedImageCache
            self._image_cache = NormalizedImageCache(
                image_cache_path, self.dataset_root, len(self._states)
            )

    def _load_scalar_tensors(self) -> tuple[Tensor, Tensor]:
        """Load small numeric columns once, avoiding unnecessary image decoding."""
        parquet_paths = sorted((self.dataset_root / "data").glob("**/*.parquet"))
        table = pa.concat_tables(
            [
                pq.read_table(path, columns=["observation.state", "action", "index"])
                for path in parquet_paths
            ]
        )
        order = np.argsort(table["index"].to_numpy())
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[
            order
        ]
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)[order]
        return torch.from_numpy(states), torch.from_numpy(actions)

    def _build_window_index(
        self, selected_episodes: set[int]
    ) -> list[tuple[int, int, int, int]]:
        metadata = self.source.meta.episodes
        known_episodes = set(int(value) for value in metadata["episode_index"])
        unknown = selected_episodes - known_episodes
        if unknown:
            raise ValueError(f"Unknown episode IDs: {sorted(unknown)}")

        windows: list[tuple[int, int, int, int]] = []
        for episode_id, start, end in zip(
            metadata["episode_index"],
            metadata["dataset_from_index"],
            metadata["dataset_to_index"],
            strict=True,
        ):
            episode_id = int(episode_id)
            start = int(start)
            end = int(end)  # Exclusive.
            if episode_id not in selected_episodes:
                continue

            # local_t ranges from 1 through length - 2, inclusive.
            for global_t in range(start + 1, end - 1):
                local_t = global_t - start
                windows.append((episode_id, local_t, global_t, end))
        return windows

    def __len__(self) -> int:
        return len(self._windows)

    @staticmethod
    def _normalize_image(image: Tensor) -> Tensor:
        image = image.to(dtype=torch.float32)
        if image.max() > 1.0:
            image = image / 255.0
        return (image - IMAGENET_MEAN) / IMAGENET_STD

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        episode_id, local_t, global_t, episode_end = self._windows[index]

        observation_history = torch.stack(
            [self._image(global_t - 1), self._image(global_t)]
        )
        positions = self._states[global_t - 1 : global_t + 1]
        agent_position = self.normalization.normalize_position(positions)

        # The final episode row has no within-episode next observation, so it
        # cannot supervise an action transition.
        valid_action_end = min(global_t + self.action_horizon, episode_end - 1)
        action_count = valid_action_end - global_t
        action_chunk = torch.zeros((self.action_horizon, 2), dtype=torch.float32)
        action_valid_mask = torch.zeros(self.action_horizon, dtype=torch.bool)
        if action_count:
            raw_actions = self._actions[global_t:valid_action_end]
            action_chunk[:action_count] = self.normalization.normalize_action(
                raw_actions
            )
            action_valid_mask[:action_count] = True

        sample = {
            "observation_history": observation_history,
            "agent_position": agent_position,
            "action_chunk": action_chunk,
            "action_valid_mask": action_valid_mask,
            "episode_id": torch.tensor(episode_id, dtype=torch.int64),
            "start_step": torch.tensor(local_t, dtype=torch.int64),
        }
        if not self.include_future_observations:
            return sample

        valid_future_end = min(global_t + 1 + self.future_horizon, episode_end)
        future_rows = list(range(global_t + 1, valid_future_end))
        future_count = len(future_rows)
        future_valid_mask = torch.zeros(self.future_horizon, dtype=torch.bool)
        future_valid_mask[:future_count] = True
        future_images = [
            self._image(row)
            for row in future_rows
        ]
        if not future_images:
            raise RuntimeError(
                "A window was created without a valid future observation"
            )
        future_images.extend(
            [
                future_images[-1].clone()
                for _ in range(self.future_horizon - future_count)
            ]
        )
        sample["future_observations"] = torch.stack(future_images)
        sample["future_valid_mask"] = future_valid_mask
        return sample

    def _image(self, index: int) -> Tensor:
        if self._image_cache is not None:
            return self._image_cache[index]
        return self._normalize_image(self.source[index]["observation.image"])

    def source_indices(self, index: int) -> tuple[int, int, int, int]:
        """Expose window metadata for correctness tests and debugging."""
        return self._windows[index]
