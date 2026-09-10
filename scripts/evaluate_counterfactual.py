"""Reproduce the original 64-window-batch offline action counterfactual check."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/lerobot/pusht_image")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from torch.utils.data import DataLoader
    from mini_wam.data import MiniWAMDataset, load_episode_split
    from mini_wam.evaluation.closure import EVIDENCE_ROOT, write_json, sha256_file
    from mini_wam.evaluation.closure_runner import counterfactual_batch, load_frozen_policy, runtime_metadata
    from mini_wam.training.runtime import select_device
    from mini_wam.studio.datasets import validate_dataset

    if args.output_dir.resolve().is_relative_to(EVIDENCE_ROOT.resolve()) or args.output_dir.exists():
        parser.error("Use a new output directory outside the immutable archive")
    device = select_device(args.device)
    model, norm, identity = load_frozen_policy(args.checkpoint, "future_aware", device)
    if validate_dataset(args.dataset_root, "lerobot/pusht_image").fingerprint != identity["dataset_fingerprint"]:
        raise ValueError("Dataset fingerprint differs from the checkpoint")
    split = ROOT / "splits/episodes_seed42.json"
    dataset = MiniWAMDataset(args.dataset_root, load_episode_split(split, "validation"), norm,
                             action_horizon=16, future_horizon=4, include_future_observations=True)
    # Historical protocol: preserve order, roll prefixes within each complete
    # batch, and exclude the incomplete final batch. Do not silently change it.
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=True)
    if len(loader) * 64 != 2368:
        raise ValueError("Validation window count differs from the archived protocol")
    args.output_dir.mkdir(parents=True)
    print(f"SPLIT {split}\nOUTPUT {args.output_dir.resolve()}", flush=True)
    write_json(args.output_dir / "runtime.json", runtime_metadata(device))
    totals = dict.fromkeys(("correct", "shuffled", "reversed"), 0.0)
    valid_steps = 0
    started = time.perf_counter()
    try:
        for i, batch in enumerate(loader):
            losses, count = counterfactual_batch(model, batch, norm, device)
            for name in totals:
                totals[name] += losses[name] * count
            valid_steps += count
            if i % 10 == 0:
                print(f"BATCHES {i + 1}/{len(loader)}", flush=True)
        if valid_steps != 9352:
            raise ValueError("Valid future-step count differs from the archived protocol")
        losses = {k: v / valid_steps for k, v in totals.items()}
        write_json(args.output_dir / "results.json", {
            "status": "complete", "checkpoint": identity, "split": "offline_validation",
            "split_sha256": sha256_file(split), "windows_available": len(dataset),
            "windows_evaluated": len(loader) * 64, "windows_dropped": len(dataset) - len(loader) * 64,
            "batch_size": 64, "drop_last": True, "shuffle": False,
            "valid_future_steps": valid_steps, "mean_masked_cosine_loss": losses,
            "deltas_vs_correct": {k: losses[k] - losses["correct"] for k in ("shuffled", "reversed")},
            "elapsed_seconds": time.perf_counter() - started, "sealed_final_test_used": False})
    except Exception as exc:
        write_json(args.output_dir / "failure.json", {"error": repr(exc), "valid_future_steps": valid_steps})
        raise
    print("COUNTERFACTUAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
