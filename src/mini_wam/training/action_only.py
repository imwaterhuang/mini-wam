"""Reproducible and resumable training for :class:`ActionOnlyPolicy`."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler

from mini_wam.data import MiniWAMDataset, NormalizationStats, load_episode_split
from mini_wam.models.action_only import ActionOnlyPolicy, masked_smooth_l1_loss
from mini_wam.studio.datasets import validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_VERSION = 1


def _require(mapping: dict[str, Any], path: str, expected_type: type) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"配置缺少 {path}")
        value = value[part]
    if expected_type is float and isinstance(value, (int, float)):
        return float(value)
    if expected_type is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if expected_type is bool and isinstance(value, bool):
        return value
    if expected_type is str and isinstance(value, str):
        return value
    raise TypeError(f"配置 {path} 必须是 {expected_type.__name__}，实际为 {type(value).__name__}")


def load_training_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the action-only training configuration."""
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("训练配置顶层必须是字典")

    if _require(raw, "model.name", str) != "action_only":
        raise ValueError("当前训练器只支持 model.name=action_only")
    _require(raw, "model.pretrained", bool)
    _require(raw, "data.dataset_root", str)
    _require(raw, "data.split_path", str)
    _require(raw, "data.normalization_path", str)
    if _require(raw, "data.history_length", int) != 2:
        raise ValueError("ActionOnlyPolicy 固定使用两帧历史")
    if _require(raw, "data.action_horizon", int) != 16:
        raise ValueError("ActionOnlyPolicy 固定输出 16 步动作")
    if _require(raw, "data.future_horizon", int) < 1:
        raise ValueError("future_horizon 必须为正数")

    positive_ints = (
        "training.batch_size",
        "training.train_steps",
        "validation.frequency",
        "checkpoint.frequency",
    )
    for key in positive_ints:
        if _require(raw, key, int) < 1:
            raise ValueError(f"配置 {key} 必须为正数")
    _require(raw, "training.seed", int)
    for key in ("training.learning_rate", "training.gradient_clip_norm"):
        if _require(raw, key, float) <= 0:
            raise ValueError(f"配置 {key} 必须大于 0")
    if _require(raw, "training.weight_decay", float) < 0:
        raise ValueError("training.weight_decay 不能为负数")
    warmup = _require(raw, "training.warmup_steps", int)
    if warmup < 0 or warmup >= raw["training"]["train_steps"]:
        raise ValueError("warmup_steps 必须在 [0, train_steps) 内")
    max_batches = raw["validation"].get("max_batches")
    if max_batches is not None and (not isinstance(max_batches, int) or max_batches < 1):
        raise ValueError("validation.max_batches 必须为 null 或正整数")
    return raw


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    paths = sorted((PROJECT_ROOT / "src").glob("**/*.py"))
    paths += sorted((PROJECT_ROOT / "scripts").glob("*.py"))
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit() -> str | None:
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


def _normalization_dict(stats: NormalizationStats) -> dict[str, Any]:
    return {
        "position_mean": stats.position_mean.tolist(),
        "position_std": stats.position_std.tolist(),
        "action_mean": stats.action_mean.tolist(),
        "action_std": stats.action_std.tolist(),
        "std_floor": stats.std_floor,
    }


class DeterministicBatchSampler(Sampler[list[int]]):
    """Infinite, resumable epoch sampler with deterministic per-epoch shuffling.

    The next batch position is advanced before yielding. With ``num_workers=0``
    this makes a checkpoint taken after a training step resume at the exact next
    batch without replaying or skipping data.
    """

    def __init__(self, dataset_size: int, batch_size: int, seed: int) -> None:
        if dataset_size < 1 or batch_size < 1:
            raise ValueError("dataset_size 和 batch_size 必须为正数")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.batch_in_epoch = 0

    @property
    def batches_per_epoch(self) -> int:
        return math.ceil(self.dataset_size / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        while True:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(self.dataset_size, generator=generator).tolist()
            for batch_index in range(self.batch_in_epoch, self.batches_per_epoch):
                start = batch_index * self.batch_size
                batch = order[start : start + self.batch_size]
                if batch_index + 1 == self.batches_per_epoch:
                    self.epoch += 1
                    self.batch_in_epoch = 0
                else:
                    self.batch_in_epoch = batch_index + 1
                yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def state_dict(self) -> dict[str, int]:
        return {
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"采样器 {key} 不兼容：{state.get(key)} != {value}")
        epoch = int(state["epoch"])
        batch_in_epoch = int(state["batch_in_epoch"])
        if epoch < 0 or not 0 <= batch_in_epoch < self.batches_per_epoch:
            raise ValueError("checkpoint 中的采样器位置非法")
        self.epoch = epoch
        self.batch_in_epoch = batch_in_epoch


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_random_states() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        # Store NumPy's uint32 array as a tensor so torch.load(...,
        # weights_only=True) can load the checkpoint without allowlisting
        # arbitrary Python globals.
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gaussian": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_random_states(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["state"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gaussian"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _make_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, train_steps: int) -> LambdaLR:
    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(train_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=scale)


@torch.inference_mode()
def _evaluate(
    model: ActionOnlyPolicy,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> float:
    model.eval()
    weighted_loss = 0.0
    valid_steps = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        mask = batch["action_valid_mask"].to(device)
        prediction = model(
            batch["observation_history"].to(device),
            batch["agent_position"].to(device),
        )
        loss = masked_smooth_l1_loss(prediction, batch["action_chunk"].to(device), mask)
        count = int(mask.sum().item())
        weighted_loss += float(loss.item()) * count
        valid_steps += count
    if not valid_steps:
        raise RuntimeError("验证集没有有效动作")
    return weighted_loss / valid_steps


def _append_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _environment_summary(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "git_commit": _git_commit(),
        "source_fingerprint": _source_fingerprint(),
    }


def _checkpoint_payload(
    *,
    model: ActionOnlyPolicy,
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
) -> dict[str, Any]:
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
        "random_states": _capture_random_states(),
        "split_hash": split_hash,
        "source_commit": environment["git_commit"],
        "source_fingerprint": environment["source_fingerprint"],
        "metadata": {
            "architecture": "action_only_v1",
            "task": "pusht",
            "history_frames": 2,
            "image_shape": [3, 96, 96],
            "state_dim": 2,
            "action_horizon": 16,
            "action_dim": 2,
            "action_range": [0.0, 512.0],
            "dataset_fingerprint": dataset_fingerprint,
            "normalization": _normalization_dict(stats),
        },
    }


def _default_run_dir(seed: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "runs" / "action_only" / str(seed) / timestamp


def _log(run_dir: Path, message: str) -> None:
    print(message, flush=True)
    with (run_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def train_action_only(
    config_path: str | Path,
    *,
    resume_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    stop_after_step: int | None = None,
    device_name: str = "auto",
) -> Path:
    """Train, validate, checkpoint, and optionally resume ``ActionOnlyPolicy``."""
    config_path = Path(config_path).expanduser().resolve()
    config = load_training_config(config_path)
    training = config["training"]
    seed = int(training["seed"])
    train_steps = int(training["train_steps"])
    if stop_after_step is not None and not 1 <= stop_after_step <= train_steps:
        raise ValueError("stop_after_step 必须在 [1, train_steps] 内")
    target_step = stop_after_step or train_steps

    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    resume = Path(resume_path).expanduser().resolve() if resume_path else None
    if resume and run_dir is None:
        run_path = resume.parent.parent
    else:
        run_path = Path(run_dir).expanduser().resolve() if run_dir else _default_run_dir(seed)
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "checkpoints").mkdir(exist_ok=True)

    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / ".cache" / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "data" / ".cache" / "hf_datasets"))
    _seed_everything(seed)

    data_config = config["data"]
    dataset_root = _resolve_project_path(data_config["dataset_root"])
    split_path = _resolve_project_path(data_config["split_path"])
    normalization_path = _resolve_project_path(data_config["normalization_path"])
    stats = NormalizationStats.from_audit_file(normalization_path)
    train_dataset = MiniWAMDataset(
        dataset_root,
        load_episode_split(split_path, "train"),
        stats,
        action_horizon=int(data_config["action_horizon"]),
        future_horizon=int(data_config["future_horizon"]),
    )
    validation_dataset = MiniWAMDataset(
        dataset_root,
        load_episode_split(split_path, "validation"),
        stats,
        action_horizon=int(data_config["action_horizon"]),
        future_horizon=int(data_config["future_horizon"]),
    )
    sampler = DeterministicBatchSampler(len(train_dataset), int(training["batch_size"]), seed)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    model = ActionOnlyPolicy(pretrained=bool(config["model"]["pretrained"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = _make_scheduler(optimizer, int(training["warmup_steps"]), train_steps)
    environment = _environment_summary(device)
    split_hash = _sha256_file(split_path)
    dataset_fingerprint = validate_dataset(dataset_root, "lerobot/pusht_image").fingerprint
    best_validation_loss = math.inf
    last_validation_loss: float | None = None
    step = 0

    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("checkpoint 版本不兼容")
        if checkpoint.get("config") != config:
            raise ValueError("恢复失败：当前配置与 checkpoint 配置不同")
        if checkpoint.get("split_hash") != split_hash:
            raise ValueError("恢复失败：数据划分已经变化")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        sampler.load_state_dict(checkpoint["sampler"])
        step = int(checkpoint["step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        last_validation_loss = checkpoint.get("validation_loss")
        _restore_random_states(checkpoint["random_states"])
        if step >= target_step:
            raise ValueError(f"checkpoint 已在第 {step} 步，不早于目标第 {target_step} 步")

    shutil.copy2(config_path, run_path / "config.yaml") if not (run_path / "config.yaml").exists() else None
    (run_path / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_path / "normalization.json").write_text(
        json.dumps(_normalization_dict(stats), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log(run_path, f"device={device} step={step} target_step={target_step} run_dir={run_path}")

    checkpoint_frequency = int(config["checkpoint"]["frequency"])
    validation_frequency = int(config["validation"]["frequency"])
    max_validation_batches = config["validation"].get("max_batches")
    iterator = iter(train_loader)
    while step < target_step:
        batch = next(iterator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        mask = batch["action_valid_mask"].to(device)
        prediction = model(
            batch["observation_history"].to(device),
            batch["agent_position"].to(device),
        )
        loss = masked_smooth_l1_loss(prediction, batch["action_chunk"].to(device), mask)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"第 {step + 1} 步出现非有限 loss")
        loss.backward()
        gradient_norm = float(clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"])))
        optimizer.step()
        scheduler.step()
        step += 1

        _append_csv(
            run_path / "train_metrics.csv",
            ["step", "loss", "learning_rate", "gradient_norm"],
            {
                "step": step,
                "loss": float(loss.item()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": gradient_norm,
            },
        )

        validation_loss: float | None = None
        if step % validation_frequency == 0 or step == target_step:
            validation_loss = _evaluate(model, validation_loader, device, max_validation_batches)
            last_validation_loss = validation_loss
            _append_csv(
                run_path / "val_metrics.csv",
                ["step", "validation_loss", "batches"],
                {
                    "step": step,
                    "validation_loss": validation_loss,
                    "batches": max_validation_batches if max_validation_batches is not None else "all",
                },
            )
            _log(run_path, f"step={step} train_loss={loss.item():.6f} val_loss={validation_loss:.6f}")
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                payload = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    step=step,
                    config=config,
                    best_validation_loss=best_validation_loss,
                    validation_loss=last_validation_loss,
                    stats=stats,
                    split_hash=split_hash,
                    dataset_fingerprint=dataset_fingerprint,
                    environment=environment,
                )
                _atomic_torch_save(payload, run_path / "checkpoints" / "best.pt")

        if step % checkpoint_frequency == 0 or step == target_step:
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                step=step,
                config=config,
                best_validation_loss=best_validation_loss,
                validation_loss=last_validation_loss,
                stats=stats,
                split_hash=split_hash,
                dataset_fingerprint=dataset_fingerprint,
                environment=environment,
            )
            _atomic_torch_save(payload, run_path / "checkpoints" / f"step_{step:07d}.pt")
            _atomic_torch_save(payload, run_path / "checkpoints" / "last.pt")

    _log(run_path, f"completed target step {target_step}")
    return run_path
