"""Download and verify the exact Push-T dataset used by Mini-WAM."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "data" / ".cache" / "hf_datasets"))

from mini_wam.studio.datasets import validate_dataset


DEFAULT_ROOT = PROJECT_ROOT / "data" / "lerobot" / "pusht_image"
EXPECTED_EPISODES = 206
EXPECTED_FRAMES = 25_650


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只验证已有数据，不访问网络",
    )
    return parser.parse_args()


def download_dataset(dataset_root: Path, *, verify_only: bool = False) -> None:
    dataset_root = dataset_root.expanduser().resolve()
    if not verify_only:
        from huggingface_hub import snapshot_download

        dataset_root.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="lerobot/pusht_image",
            repo_type="dataset",
            local_dir=dataset_root,
            allow_patterns=["meta/**", "data/**", "README.md", ".gitattributes"],
        )

    summary = validate_dataset(dataset_root, "lerobot/pusht_image")
    if summary.episode_count != EXPECTED_EPISODES or summary.frame_count != EXPECTED_FRAMES:
        raise RuntimeError(
            "数据版本不匹配："
            f"期望 {EXPECTED_EPISODES} episodes/{EXPECTED_FRAMES} frames，"
            f"实际 {summary.episode_count}/{summary.frame_count}"
        )
    print(f"dataset_root={summary.root}")
    print(f"episodes={summary.episode_count} frames={summary.frame_count}")
    print(f"fingerprint={summary.fingerprint}")


def main() -> None:
    args = parse_args()
    download_dataset(args.dataset_root, verify_only=args.verify_only)


if __name__ == "__main__":
    main()
