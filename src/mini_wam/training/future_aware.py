"""FutureHeadPolicy 的可恢复训练总流程。"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from mini_wam.models.future_head import FutureHeadPolicy
from mini_wam.studio.datasets import validate_dataset

from .artifacts import (
    CHECKPOINT_VERSION,
    append_csv,
    atomic_torch_save,
    checkpoint_payload,
    environment_summary,
    log,
    normalization_dict,
    prepare_run_directory,
    sha256_file,
    sync_run_directory,
)
from .config import PROJECT_ROOT, load_training_config
from .data import build_training_data
from .future_head import evaluate, train_step
from .reproducibility import restore_random_states, seed_everything
from .runtime import select_device
from .steps import make_scheduler


def train_future_aware(
    config_path: str | Path,
    *,
    resume_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    mirror_dir: str | Path | None = None,
    stop_after_step: int | None = None,
    device_name: str = "auto",
    num_workers: int = 0,
) -> Path:
    """训练、验证、保存并可选恢复 FutureHeadPolicy。"""
    config_path = Path(config_path).expanduser().resolve()
    config = load_training_config(
        config_path,
        expected_model_name="future_aware",
    )
    training = config["training"]
    seed = int(training["seed"])
    train_steps = int(training["train_steps"])
    lambda_future = float(training["lambda_future"])
    if stop_after_step is not None and not 1 <= stop_after_step <= train_steps:
        raise ValueError("stop_after_step 必须在 [1, train_steps] 内")
    if num_workers < 0:
        raise ValueError("num_workers 不能为负数")
    target_step = stop_after_step or train_steps
    device = select_device(device_name)

    resume, run_path, mirror_path = prepare_run_directory(
        seed=seed,
        model_name="future_aware",
        resume_path=resume_path,
        run_dir=run_dir,
        mirror_dir=mirror_dir,
    )
    os.environ.setdefault(
        "HF_HOME", str(PROJECT_ROOT / "data" / ".cache" / "huggingface")
    )
    os.environ.setdefault(
        "HF_DATASETS_CACHE", str(PROJECT_ROOT / "data" / ".cache" / "hf_datasets")
    )
    seed_everything(seed)

    data = build_training_data(
        config=config,
        seed=seed,
        num_workers=num_workers,
        include_future_observations=True,
    )
    model = FutureHeadPolicy(
        pretrained=bool(config["model"]["pretrained"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = make_scheduler(
        optimizer,
        int(training["warmup_steps"]),
        train_steps,
    )
    environment = environment_summary(device)
    environment["data_loader"] = {
        "train_num_workers": num_workers,
        "validation_num_workers": 0,
        "prefetch_factor": 2 if num_workers > 0 else None,
        "persistent_workers": num_workers > 0,
        "normalized_image_cache": (
            {
                "path": str(data.train_loader.dataset._image_cache.path),
                "manifest_sha256": sha256_file(
                    data.train_loader.dataset._image_cache.path / "manifest.json"
                ),
            }
            if data.train_loader.dataset._image_cache is not None else None
        ),
    }
    split_hash = sha256_file(data.split_path)
    dataset_fingerprint = validate_dataset(
        data.dataset_root, "lerobot/pusht_image"
    ).fingerprint
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
        if checkpoint.get("metadata", {}).get("architecture") != "future_aware_v1":
            raise ValueError("恢复失败：checkpoint 不是 future_aware_v1")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        data.sampler.load_state_dict(checkpoint["sampler"])
        step = int(checkpoint["step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        last_validation_loss = checkpoint.get("validation_loss")
        restore_random_states(checkpoint["random_states"])
        if step >= target_step:
            raise ValueError(
                f"checkpoint 已在第 {step} 步，不早于目标第 {target_step} 步"
            )

    if not (run_path / "config.yaml").exists():
        shutil.copy2(config_path, run_path / "config.yaml")
    (run_path / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_path / "normalization.json").write_text(
        json.dumps(normalization_dict(data.stats), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(
        run_path,
        f"device={device} num_workers={num_workers} step={step} "
        f"target_step={target_step} lambda_future={lambda_future} run_dir={run_path}",
    )
    if mirror_path:
        sync_run_directory(run_path, mirror_path)

    checkpoint_frequency = int(config["checkpoint"]["frequency"])
    archive_frequency = int(
        config["checkpoint"].get("archive_frequency", checkpoint_frequency)
    )
    validation_frequency = int(config["validation"]["frequency"])
    max_validation_batches = config["validation"].get("max_batches")
    iterator = iter(data.train_loader)

    def current_checkpoint() -> dict[str, Any]:
        return checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=data.sampler,
            step=step,
            config=config,
            best_validation_loss=best_validation_loss,
            validation_loss=last_validation_loss,
            stats=data.stats,
            split_hash=split_hash,
            dataset_fingerprint=dataset_fingerprint,
            environment=environment,
            architecture="future_aware_v1",
            metadata_extra={
                "future_horizon": 4,
                "future_feature_dim": 512,
                "lambda_future": lambda_future,
                "future_supervision": "training_only",
            },
        )

    while step < target_step:
        metrics = train_step(
            model=model,
            batch=next(iterator),
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=data.sampler,
            device=device,
            lambda_future=lambda_future,
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            next_step=step + 1,
        )
        step += 1
        append_csv(
            run_path / "train_metrics.csv",
            [
                "step",
                "total_loss",
                "action_loss",
                "future_loss",
                "learning_rate",
                "gradient_norm",
            ],
            {
                "step": step,
                **metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
            },
        )

        if step % validation_frequency == 0 or step == target_step:
            validation = evaluate(
                model,
                data.validation_loader,
                device,
                max_validation_batches,
                lambda_future,
            )
            last_validation_loss = validation["total_loss"]
            append_csv(
                run_path / "val_metrics.csv",
                [
                    "step",
                    "total_loss",
                    "action_loss",
                    "future_loss",
                    "batches",
                ],
                {
                    "step": step,
                    **validation,
                    "batches": (
                        max_validation_batches
                        if max_validation_batches is not None
                        else "all"
                    ),
                },
            )
            log(
                run_path,
                f"step={step} train_total={metrics['total_loss']:.6f} "
                f"val_total={validation['total_loss']:.6f} "
                f"val_action={validation['action_loss']:.6f} "
                f"val_future={validation['future_loss']:.6f}",
            )
            if last_validation_loss < best_validation_loss:
                best_validation_loss = last_validation_loss
                atomic_torch_save(
                    current_checkpoint(), run_path / "checkpoints" / "best.pt"
                )
            if mirror_path:
                sync_run_directory(run_path, mirror_path)

        if step % checkpoint_frequency == 0 or step == target_step:
            payload = current_checkpoint()
            atomic_torch_save(payload, run_path / "checkpoints" / "last.pt")
            if step % archive_frequency == 0 or step == train_steps:
                atomic_torch_save(
                    payload,
                    run_path / "checkpoints" / f"step_{step:07d}.pt",
                )
            if mirror_path:
                sync_run_directory(run_path, mirror_path)

    log(run_path, f"completed target step {target_step}")
    if mirror_path:
        sync_run_directory(run_path, mirror_path)
    return run_path
