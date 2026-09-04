"""Print the shapes and masks of one Mini-WAM training sample."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "data" / ".cache" / "hf_datasets"))

from mini_wam.data import MiniWAMDataset, NormalizationStats, load_episode_split


def main() -> None:
    split_path = PROJECT_ROOT / "splits" / "episodes_seed42.json"
    stats = NormalizationStats.from_audit_file(PROJECT_ROOT / "artifacts" / "data_audit.json")
    episode_ids = load_episode_split(split_path, "train")
    dataset = MiniWAMDataset(
        dataset_root=PROJECT_ROOT / "data" / "lerobot" / "pusht_image",
        episode_ids=episode_ids,
        normalization=stats,
    )

    sample = dataset[0]
    print(f"training windows: {len(dataset)}")
    for key, value in sample.items():
        if key.endswith("mask"):
            print(f"{key:24s} shape={tuple(value.shape)!s:18s} valid={int(value.sum())}")
        else:
            print(f"{key:24s} shape={tuple(value.shape)} dtype={value.dtype}")


if __name__ == "__main__":
    main()

