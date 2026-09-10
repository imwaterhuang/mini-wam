"""Replay both frozen checkpoints on the 20 archived extra development scenes."""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-checkpoint", type=Path, required=True)
    parser.add_argument("--future-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="cpu reproduces the original evaluation device class")
    args = parser.parse_args()
    from mini_wam.evaluation.closure import EVIDENCE_ROOT, MODELS, load_closure_scenes, paired_statistics, sha256_file, write_json
    from mini_wam.evaluation.closure_runner import load_frozen_policy, runtime_metadata
    from mini_wam.studio.pusht import SceneState, run_rollout
    from mini_wam.training.runtime import select_device

    scenes = load_closure_scenes()
    if args.output_dir.resolve().is_relative_to(EVIDENCE_ROOT.resolve()):
        parser.error("Use a new output directory outside the immutable archive")
    if args.output_dir.exists():
        parser.error("Output directory already exists; preserve earlier runs")
    device = select_device(args.device)
    print(f"SCENES {EVIDENCE_ROOT / 'scenes.json'}\nOUTPUT {args.output_dir.resolve()}", flush=True)
    models, norms, checkpoints = {}, {}, {}
    for name, path in zip(MODELS, (args.action_checkpoint, args.future_checkpoint)):
        models[name], norms[name], checkpoints[name] = load_frozen_policy(path, name, device)
    for key in ("position_mean", "position_std", "action_mean", "action_std"):
        if not (getattr(norms[MODELS[0]], key) == getattr(norms[MODELS[1]], key)).all():
            raise ValueError("The two policies have different normalization")
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "runtime.json", runtime_metadata(device))
    write_json(args.output_dir / "scenes.json", scenes)
    results = {name: [] for name in MODELS}
    start = time.perf_counter()
    try:
        for index, record in enumerate(scenes["scenes"]):
            for name in (MODELS if index % 2 == 0 else MODELS[::-1]):
                video = args.output_dir / name / "videos" / (record["scene_id"] + ".mp4")
                for update in run_rollout(models[name], norms[name], SceneState(**record["state"]), video, max_steps=300, execute_steps=4):
                    final = update
                results[name].append({"scene_id": record["scene_id"], "seed": record["seed"],
                                      **final.metrics.to_dict(), "video": str(video.relative_to(args.output_dir))})
                gc.collect()
            write_json(args.output_dir / "partial_results.json", results)
            print(f"PAIRS {index + 1}/20", flush=True)
        stats = paired_statistics(results, scenes)
        write_json(args.output_dir / "results.json", {
            "status": "complete", "protocol": {"split": "additional_development", "episode_count": 20,
            "max_steps": 300, "execute_steps": 4, "action_horizon": 16, "sealed_final_test_used": False},
            "scene_file_sha256": sha256_file(EVIDENCE_ROOT / "scenes.json"),
            "checkpoints": checkpoints, "episodes": results, **stats,
            "elapsed_seconds": time.perf_counter() - start})
    except Exception as exc:
        write_json(args.output_dir / "failure.json", {"error": repr(exc), "episodes": results})
        raise
    print("PAIRED_COMPARISON_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
