"""Measure action-only data, transfer, compute, and end-to-end training time."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / ".cache" / "huggingface"))
os.environ.setdefault(
    "HF_DATASETS_CACHE", str(PROJECT_ROOT / "data" / ".cache" / "hf_datasets")
)

from mini_wam.data import MiniWAMDataset, NormalizationStats, load_episode_split  # noqa: E402
from mini_wam.models.action_only import ActionOnlyPolicy, masked_smooth_l1_loss  # noqa: E402
from mini_wam.training.config import (  # noqa: E402
    load_training_config,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="定位 action_only 每个训练 step 的耗时"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/action_only_seed0.yaml"
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="可选：加载现有 checkpoint 的模型权重"
    )
    parser.add_argument("--device", default="auto", help="auto、cpu、mps 或 cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--measure-batches", type=int, default=20)
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="只比较数据路径；用于快速筛选 num_workers=0/2/4",
    )
    parser.add_argument("--output", type=Path, help="可选 JSON 结果路径")
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "mean_ms": statistics.mean(values) * 1000.0,
        "median_ms": statistics.median(values) * 1000.0,
        "p95_ms": ordered[p95_index] * 1000.0,
        "min_ms": ordered[0] * 1000.0,
        "max_ms": ordered[-1] * 1000.0,
    }


def _fixed_batches(
    dataset_size: int, batch_size: int, count: int, seed: int
) -> list[list[int]]:
    generator = torch.Generator().manual_seed(seed)
    required = batch_size * count
    batches: list[list[int]] = []
    while required > 0:
        order = torch.randperm(dataset_size, generator=generator).tolist()
        for start in range(0, dataset_size, batch_size):
            batch = order[start : start + batch_size]
            if len(batch) != batch_size:
                continue
            batches.append(batch)
            required -= len(batch)
            if required <= 0:
                return batches
    return batches


def _measure_loader(
    loader: DataLoader, warmup: int, measured: int
) -> tuple[dict[str, float], dict[str, Any]]:
    iterator = iter(loader)
    for _ in range(warmup):
        next(iterator)
    times: list[float] = []
    batch: dict[str, Any] | None = None
    for _ in range(measured):
        start = time.perf_counter()
        batch = next(iterator)
        times.append(time.perf_counter() - start)
    assert batch is not None
    return _summary(times), batch


def _loader(
    dataset: MiniWAMDataset,
    batches: list[list[int]],
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=batches,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(303),
    )


def _move_used_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "observation_history": batch["observation_history"].to(device),
        "agent_position": batch["agent_position"].to(device),
        "action_chunk": batch["action_chunk"].to(device),
        "action_valid_mask": batch["action_valid_mask"].to(device),
    }


def _train_step(
    model: ActionOnlyPolicy,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    gradient_clip_norm: float,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(batch["observation_history"], batch["agent_position"])
    loss = masked_smooth_l1_loss(
        prediction, batch["action_chunk"], batch["action_valid_mask"]
    )
    loss.backward()
    clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()


def _measure_end_to_end(
    loader: DataLoader,
    model: ActionOnlyPolicy,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float,
    warmup: int,
    measured: int,
) -> dict[str, float]:
    iterator = iter(loader)
    for _ in range(warmup):
        batch = _move_used_batch(next(iterator), device)
        _train_step(model, optimizer, batch, gradient_clip_norm)
    _synchronize(device)
    times: list[float] = []
    for _ in range(measured):
        start = time.perf_counter()
        batch = _move_used_batch(next(iterator), device)
        _train_step(model, optimizer, batch, gradient_clip_norm)
        _synchronize(device)
        times.append(time.perf_counter() - start)
    return _summary(times)


def main() -> None:
    args = parse_args()
    if args.warmup_batches < 1 or args.measure_batches < 1:
        raise ValueError("warmup-batches 和 measure-batches 必须为正数")
    if args.num_workers < 0:
        raise ValueError("num-workers 不能为负数")

    config = load_training_config(args.config)
    data_config = config["data"]
    training = config["training"]
    device = _device(args.device)
    dataset_root = resolve_project_path(data_config["dataset_root"])
    split_path = resolve_project_path(data_config["split_path"])
    stats = NormalizationStats.from_audit_file(
        resolve_project_path(data_config["normalization_path"])
    )
    common = {
        "dataset_root": dataset_root,
        "episode_ids": load_episode_split(split_path, "train"),
        "normalization": stats,
        "action_horizon": int(data_config["action_horizon"]),
        "future_horizon": int(data_config["future_horizon"]),
    }
    full_dataset = MiniWAMDataset(**common, include_future_observations=True)
    action_only_dataset = MiniWAMDataset(**common, include_future_observations=False)
    total_batches = args.warmup_batches + args.measure_batches
    batches = _fixed_batches(
        len(full_dataset),
        int(training["batch_size"]),
        total_batches,
        int(training["seed"]),
    )
    full_loader = _loader(full_dataset, batches, args.num_workers)
    action_only_loader = _loader(action_only_dataset, batches, args.num_workers)

    full_data, full_batch = _measure_loader(
        full_loader, args.warmup_batches, args.measure_batches
    )
    lean_data, lean_batch = _measure_loader(
        action_only_loader, args.warmup_batches, args.measure_batches
    )
    used_keys = (
        "observation_history",
        "agent_position",
        "action_chunk",
        "action_valid_mask",
        "episode_id",
        "start_step",
    )
    for key in used_keys:
        if not torch.equal(full_batch[key], lean_batch[key]):
            raise RuntimeError(f"完整模式和 action_only 模式的 {key} 不一致")

    data_result = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "batch_size": int(training["batch_size"]),
        "num_workers": args.num_workers,
        "warmup_batches": args.warmup_batches,
        "measure_batches": args.measure_batches,
        "data_full_with_future": full_data,
        "data_action_only": lean_data,
        "data_speedup": full_data["mean_ms"] / lean_data["mean_ms"],
        "used_fields_exactly_equal": True,
    }
    if args.data_only:
        output = json.dumps(data_result, ensure_ascii=False, indent=2)
        print(output)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n", encoding="utf-8")
        return

    model = ActionOnlyPolicy(pretrained=False).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    transfer_times: list[float] = []
    for _ in range(args.measure_batches):
        _synchronize(device)
        start = time.perf_counter()
        gpu_batch = _move_used_batch(lean_batch, device)
        _synchronize(device)
        transfer_times.append(time.perf_counter() - start)

    gpu_batch = _move_used_batch(lean_batch, device)
    for _ in range(args.warmup_batches):
        _train_step(model, optimizer, gpu_batch, float(training["gradient_clip_norm"]))
    _synchronize(device)
    compute_times: list[float] = []
    for _ in range(args.measure_batches):
        start = time.perf_counter()
        _train_step(model, optimizer, gpu_batch, float(training["gradient_clip_norm"]))
        _synchronize(device)
        compute_times.append(time.perf_counter() - start)

    gradient_clip_norm = float(training["gradient_clip_norm"])
    end_to_end_full = _measure_end_to_end(
        _loader(full_dataset, batches, args.num_workers),
        model,
        optimizer,
        device,
        gradient_clip_norm,
        args.warmup_batches,
        args.measure_batches,
    )
    end_to_end_lean = _measure_end_to_end(
        _loader(action_only_dataset, batches, args.num_workers),
        model,
        optimizer,
        device,
        gradient_clip_norm,
        args.warmup_batches,
        args.measure_batches,
    )

    result = {
        **data_result,
        "host_to_device_used_tensors": _summary(transfer_times),
        "compute_only": _summary(compute_times),
        "end_to_end_full_current": end_to_end_full,
        "end_to_end_action_only": end_to_end_lean,
        "end_to_end_speedup": end_to_end_full["mean_ms"] / end_to_end_lean["mean_ms"],
    }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
