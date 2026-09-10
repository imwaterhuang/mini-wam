"""Verify cached data/resume and measure future-aware throughput on the target GPU.

Run only when the production trainer is paused. All updates are discarded.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mini_wam.data.image_cache import NormalizedImageCache
from mini_wam.models.future_head import FutureHeadPolicy
from mini_wam.training.data import build_training_data
from mini_wam.training.future_head import FUTURE_BATCH_KEYS, train_step
from mini_wam.training.reproducibility import (
    DeterministicBatchSampler, capture_random_states, restore_random_states, seed_everything,
)
from mini_wam.training.runtime import move_tensors_to_device
from mini_wam.training.steps import make_scheduler


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def equal(left, right, path="state"):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            equal(a, b, f"{path}[{i}]")
    else:
        assert left == right, (path, left, right)


def state_objects(checkpoint, size):
    config = checkpoint["config"]
    training = config["training"]
    seed_everything(int(training["seed"]))
    model = FutureHeadPolicy(pretrained=False, target_pretrained=False).cuda()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=training["learning_rate"], weight_decay=training["weight_decay"],
    )
    scheduler = make_scheduler(optimizer, training["warmup_steps"], training["train_steps"])
    sampler = DeterministicBatchSampler(size, training["batch_size"], training["seed"])
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    scheduler.load_state_dict(checkpoint["scheduler"])
    sampler.load_state_dict(checkpoint["sampler"])
    restore_random_states(checkpoint["random_states"])
    return model, optimizer, scheduler, sampler


def update(checkpoint, objects, batch, step):
    model, optimizer, scheduler, sampler = objects
    training = checkpoint["config"]["training"]
    return train_step(model=model, batch=batch, optimizer=optimizer,
                      scheduler=scheduler, sampler=sampler, device=torch.device("cuda"),
                      lambda_future=training["lambda_future"],
                      gradient_clip_norm=training["gradient_clip_norm"], next_step=step)


def continuation(checkpoint, dataset, workers, steps):
    objects = state_objects(checkpoint, len(dataset))
    model, optimizer, scheduler, sampler = objects
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                        generator=torch.Generator().manual_seed(100_000))
    iterator = iter(loader)
    metrics, identities = [], []
    for i in range(steps):
        batch = next(iterator)
        identities.append(torch.stack((batch["episode_id"], batch["start_step"]), -1))
        metrics.append(update(checkpoint, objects, batch, checkpoint["step"] + i + 1))
    result = {**checkpoint, "step": checkpoint["step"] + steps,
              "model": cpu_copy(model.state_dict()), "optimizer": cpu_copy(optimizer.state_dict()),
              "scheduler": cpu_copy(scheduler.state_dict()), "sampler": sampler.state_dict(),
              "random_states": capture_random_states()}
    del iterator, loader, objects, model, optimizer, scheduler, sampler
    gc.collect()
    return result, metrics, identities


def summary(times):
    return {"mean_ms": statistics.mean(times) * 1000,
            "median_ms": statistics.median(times) * 1000,
            "max_ms": max(times) * 1000}


def benchmark(checkpoint, dataset, workers, warmup, measured):
    objects = state_objects(checkpoint, len(dataset))
    sampler = objects[-1]
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                        generator=torch.Generator().manual_seed(100_000))
    iterator = iter(loader)
    e2e, waits = [], []
    for i in range(warmup + measured):
        torch.cuda.synchronize()
        start = time.perf_counter()
        batch = next(iterator)
        received = time.perf_counter()
        update(checkpoint, objects, batch, checkpoint["step"] + i + 1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if i >= warmup:
            e2e.append(elapsed)
            waits.append(received - start)
    del iterator, loader
    gc.collect()

    transfers, compute = [], []
    # Isolated transfer and compute use the same real batch; pre-mark sampler
    # requests because train_step advances its consumed cursor on every update.
    for i in range(warmup + measured):
        torch.cuda.synchronize()
        start = time.perf_counter()
        device_batch = move_tensors_to_device(batch, FUTURE_BATCH_KEYS, torch.device("cuda"))
        torch.cuda.synchronize()
        transfer_elapsed = time.perf_counter() - start
        next(iter(sampler))
        start = time.perf_counter()
        update(checkpoint, objects, device_batch, checkpoint["step"] + i + 1)
        torch.cuda.synchronize()
        compute_elapsed = time.perf_counter() - start
        if i >= warmup:
            transfers.append(transfer_elapsed)
            compute.append(compute_elapsed)
    del objects, device_batch, batch
    gc.collect()

    fixed_sampler = DeterministicBatchSampler(len(dataset), checkpoint["config"]["training"]["batch_size"], 0)
    fixed_sampler.load_state_dict(checkpoint["sampler"])
    data_loader = DataLoader(dataset, batch_sampler=fixed_sampler, num_workers=workers,
                             generator=torch.Generator().manual_seed(100_000))
    data_iterator = iter(data_loader)
    preparation = []
    for i in range(warmup + measured):
        start = time.perf_counter()
        next(data_iterator)
        if i >= warmup:
            preparation.append(time.perf_counter() - start)
    del data_iterator, data_loader
    gc.collect()
    return {"workers": workers, "end_to_end": summary(e2e), "data_wait": summary(waits),
            "data_only": summary(preparation), "transfer": summary(transfers), "compute": summary(compute)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=20)
    args = parser.parse_args()
    assert torch.cuda.is_available()
    # Isolate data/resume equivalence from nondeterministic CUDA kernels.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.pop("MINI_WAM_IMAGE_CACHE", None)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    assert config["model"]["name"] == "future_aware"
    data = build_training_data(config=config, seed=config["training"]["seed"], num_workers=0,
                               include_future_observations=True)
    original = data.train_loader.dataset
    cached = copy.copy(original)
    cached._image_cache = NormalizedImageCache(args.cache, original.dataset_root, len(original._states))
    # Cover random interior windows and both boundaries of every train episode.
    indices = set(torch.randperm(len(original), generator=torch.Generator().manual_seed(91))[:1024].tolist())
    for i in range(len(original)):
        if i == 0 or original.source_indices(i)[0] != original.source_indices(i - 1)[0]:
            indices.add(i)
            if i:
                indices.add(i - 1)
    indices.add(len(original) - 1)
    for i in sorted(indices):
        equal(original[i], cached[i], f"sample[{i}]")
    print(f"TENSORS_EXACT samples={len(indices)} all_fields=True", flush=True)
    val_original = data.validation_loader.dataset
    val_cached = copy.copy(val_original)
    val_cached._image_cache = cached._image_cache
    val_indices = set([0, len(val_original) - 1])
    for i in range(1, len(val_original)):
        if val_original.source_indices(i)[0] != val_original.source_indices(i - 1)[0]:
            val_indices.update((i - 1, i))
    for i in sorted(val_indices):
        equal(val_original[i], val_cached[i], f"validation[{i}]")
    expected, expected_metrics, expected_ids = continuation(checkpoint, original, 4, 3)
    prefix, prefix_metrics, prefix_ids = continuation(checkpoint, original, 4, 1)
    verified = []
    for workers in (0, 2, 4):
        resumed, metrics, ids = continuation(prefix, cached, workers, 2)
        equal(prefix_metrics + metrics, expected_metrics, "metrics")
        equal(prefix_ids + ids, expected_ids, "sample_order")
        for field in ("model", "optimizer", "scheduler", "sampler", "random_states"):
            equal(resumed[field], expected[field], field)
        verified.append(workers)
        print(f"RESUME_EXACT workers={workers} model optimizer scheduler sampler random_states losses samples", flush=True)
        del resumed
    del prefix, expected
    gc.collect()
    results = {"checkpoint_step": checkpoint["step"], "gpu": torch.cuda.get_device_name(),
               "cpu_count": os.cpu_count(), "torch_threads": torch.get_num_threads(),
               "train_samples_exact": len(indices), "validation_samples_exact": len(val_indices),
               "resume_exact_workers": verified,
               "cudnn_deterministic": torch.backends.cudnn.deterministic,
               "cudnn_benchmark": torch.backends.cudnn.benchmark, "benchmarks": []}
    for name, dataset, workers in [("original", original, 4)] + [("cache", cached, w) for w in (0, 2, 4)]:
        result = {"path": name, **benchmark(checkpoint, dataset, workers, args.warmup, args.measure)}
        results["benchmarks"].append(result)
        print("BENCHMARK", json.dumps(result), flush=True)
    fastest = min((r for r in results["benchmarks"] if r["path"] == "cache"),
                  key=lambda r: r["end_to_end"]["mean_ms"])
    results["selected_workers"] = fastest["workers"]
    results["end_to_end_speedup"] = results["benchmarks"][0]["end_to_end"]["mean_ms"] / fastest["end_to_end"]["mean_ms"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print("PROFILE_VERIFIED", json.dumps({k: results[k] for k in ("selected_workers", "end_to_end_speedup")}), flush=True)


if __name__ == "__main__":
    main()
