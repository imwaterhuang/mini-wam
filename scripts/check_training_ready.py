"""Fail-fast preflight for a formal Mini-WAM training run."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mini_wam.studio.datasets import validate_dataset
from mini_wam.training.config import load_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def check_training_ready(
    config_path: Path,
    run_root: Path,
    *,
    require_cuda: bool = False,
) -> dict[str, object]:
    config = load_training_config(config_path)
    dataset_root = _resolve_project_path(config["data"]["dataset_root"])
    split_path = _resolve_project_path(config["data"]["split_path"])
    normalization_path = _resolve_project_path(config["data"]["normalization_path"])

    missing = [str(path) for path in (split_path, normalization_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少训练输入文件：{missing}")

    dataset = validate_dataset(dataset_root, "lerobot/pusht_image")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = {int(value) for value in split["train"]}
    validation_ids = {int(value) for value in split["validation"]}
    if train_ids & validation_ids:
        raise RuntimeError("训练与验证 episode 存在重叠")
    if train_ids | validation_ids != set(dataset.episode_ids):
        raise RuntimeError("训练/验证划分没有精确覆盖当前数据集")

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA；请在 Colab 中选择 GPU 运行时后重试")

    run_root = run_root.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="mini-wam-write-test-", dir=run_root, delete=True):
        pass
    free_bytes = shutil.disk_usage(run_root).free
    if free_bytes < 5 * 1024**3:
        raise RuntimeError(f"训练输出位置可用空间不足 5 GiB：{free_bytes / 1024**3:.2f} GiB")

    result: dict[str, object] = {
        "ready": True,
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset_root": str(dataset.root),
        "dataset_episodes": dataset.episode_count,
        "dataset_frames": dataset.frame_count,
        "dataset_fingerprint": dataset.fingerprint,
        "train_episodes": len(train_ids),
        "validation_episodes": len(validation_ids),
        "run_root": str(run_root),
        "run_root_free_gib": round(free_bytes / 1024**3, 2),
        "train_steps": config["training"]["train_steps"],
        "batch_size": config["training"]["batch_size"],
    }
    return result


def main() -> None:
    args = parse_args()
    result = check_training_ready(
        args.config,
        args.run_root,
        require_cuda=args.require_cuda,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
