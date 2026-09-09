"""训练指标、实验身份、checkpoint 与运行目录镜像。"""

from __future__ import annotations

import csv
import hashlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.optim.lr_scheduler import LambdaLR

from mini_wam.data import NormalizationStats
from torch import nn

from .config import PROJECT_ROOT
from .reproducibility import DeterministicBatchSampler, capture_random_states


CHECKPOINT_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    paths = sorted((PROJECT_ROOT / "src").glob("**/*.py"))
    paths += sorted((PROJECT_ROOT / "scripts").glob("*.py"))
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def normalization_dict(stats: NormalizationStats) -> dict[str, Any]:
    return {
        "position_mean": stats.position_mean.tolist(),
        "position_std": stats.position_std.tolist(),
        "action_mean": stats.action_mean.tolist(),
        "action_std": stats.action_std.tolist(),
        "std_floor": stats.std_floor,
    }


def environment_summary(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "git_commit": git_commit(),
        "source_fingerprint": source_fingerprint(),
    }


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    sampler: DeterministicBatchSampler,
    step: int,
    config: dict[str, Any],
    best_validation_loss: float,
    validation_loss: float | None,
    stats: NormalizationStats,
    split_hash: str,
    dataset_fingerprint: str,
    environment: dict[str, Any],
    architecture: str = "action_only_v1",
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """组装与重构前字段完全相同的可恢复 checkpoint。"""
    metadata = {
        "architecture": architecture,
        "task": "pusht",
        "history_frames": 2,
        "image_shape": [3, 96, 96],
        "state_dim": 2,
        "action_horizon": 16,
        "action_dim": 2,
        "action_range": [0.0, 512.0],
        "dataset_fingerprint": dataset_fingerprint,
        "normalization": normalization_dict(stats),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "sampler": sampler.state_dict(),
        "step": step,
        "config": config,
        "best_validation_loss": best_validation_loss,
        "validation_loss": validation_loss,
        "random_states": capture_random_states(),
        "split_hash": split_hash,
        "source_commit": environment["git_commit"],
        "source_fingerprint": environment["source_fingerprint"],
        "metadata": metadata,
    }


def append_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def default_run_dir(seed: int, model_name: str = "action_only") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "runs" / model_name / str(seed) / timestamp


def prepare_run_directory(
    *,
    seed: int,
    model_name: str,
    resume_path: str | Path | None,
    run_dir: str | Path | None,
    mirror_dir: str | Path | None,
) -> tuple[Path | None, Path, Path | None]:
    """解析恢复、本地运行和持久化镜像目录。"""
    resume = Path(resume_path).expanduser().resolve() if resume_path else None
    mirror_path = Path(mirror_dir).expanduser().resolve() if mirror_dir else None
    if resume and run_dir is None:
        run_path = resume.parent.parent
    else:
        run_path = (
            Path(run_dir).expanduser().resolve()
            if run_dir
            else default_run_dir(seed, model_name)
        )
    if (
        resume
        and mirror_path
        and run_path != mirror_path
        and mirror_path.exists()
        and not run_path.exists()
    ):
        shutil.copytree(mirror_path, run_path)
        if resume.is_relative_to(mirror_path):
            mirrored_resume = run_path / resume.relative_to(mirror_path)
            if mirrored_resume.exists():
                resume = mirrored_resume
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "checkpoints").mkdir(exist_ok=True)
    return resume, run_path, mirror_path


def log(run_dir: Path, message: str) -> None:
    print(message, flush=True)
    with (run_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def sync_run_directory(source: Path, destination: Path) -> None:
    """把变化过的运行产物增量、原子地镜像到持久化目录。"""
    if source.resolve() == destination.resolve():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        should_copy = not destination_path.exists()
        if not should_copy:
            source_stat = source_path.stat()
            destination_stat = destination_path.stat()
            should_copy = (
                source_stat.st_size != destination_stat.st_size
                or source_stat.st_mtime_ns != destination_stat.st_mtime_ns
            )
        if should_copy:
            temporary = destination_path.with_suffix(
                destination_path.suffix + ".sync-tmp"
            )
            shutil.copy2(source_path, temporary)
            temporary.replace(destination_path)
