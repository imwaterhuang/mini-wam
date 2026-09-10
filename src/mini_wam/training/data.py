"""共享的训练/验证数据装配。"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from mini_wam.data import MiniWAMDataset, NormalizationStats, load_episode_split

from .config import resolve_project_path
from .reproducibility import DeterministicBatchSampler


@dataclass(frozen=True)
class TrainingDataBundle:
    stats: NormalizationStats
    dataset_root: Path
    split_path: Path
    sampler: DeterministicBatchSampler
    train_loader: DataLoader
    validation_loader: DataLoader


def build_training_data(
    *,
    config: dict[str, Any],
    seed: int,
    num_workers: int,
    include_future_observations: bool,
) -> TrainingDataBundle:
    """用相同窗口、划分和 worker 规则构建两种模型的数据。"""
    training = config["training"]
    data_config = config["data"]
    dataset_root = resolve_project_path(data_config["dataset_root"])
    split_path = resolve_project_path(data_config["split_path"])
    normalization_path = resolve_project_path(data_config["normalization_path"])
    stats = NormalizationStats.from_audit_file(normalization_path)
    dataset_options = {
        "normalization": stats,
        "action_horizon": int(data_config["action_horizon"]),
        "future_horizon": int(data_config["future_horizon"]),
        "include_future_observations": include_future_observations,
        "image_cache_path": os.environ.get("MINI_WAM_IMAGE_CACHE") or None,
    }
    train_dataset = MiniWAMDataset(
        dataset_root,
        load_episode_split(split_path, "train"),
        **dataset_options,
    )
    validation_dataset = MiniWAMDataset(
        dataset_root,
        load_episode_split(split_path, "validation"),
        **dataset_options,
    )
    batch_size = int(training["batch_size"])
    sampler = DeterministicBatchSampler(len(train_dataset), batch_size, seed)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        generator=torch.Generator().manual_seed(seed + 100_000),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + 200_000),
        num_workers=0,
    )
    return TrainingDataBundle(
        stats=stats,
        dataset_root=dataset_root,
        split_path=split_path,
        sampler=sampler,
        train_loader=train_loader,
        validation_loader=validation_loader,
    )
