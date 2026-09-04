"""Evaluate a trained action-only checkpoint on frozen development scenes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from mini_wam.evaluation import evaluate_action_only_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto、cpu、mps 或 cuda")
    parser.add_argument("--max-steps", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_action_only_policy(
        args.checkpoint,
        PROJECT_ROOT / "splits" / "dev_scenes.json",
        PROJECT_ROOT / "data" / "lerobot" / "pusht_image",
        PROJECT_ROOT / "artifacts" / "data_audit.json",
        args.output_dir,
        device_name=args.device,
        max_steps=args.max_steps,
    )
    summary = result["summary"]
    print(f"回合：{result['protocol']['episode_count']}")
    print(f"成功率：{summary['success_rate']:.1%}")
    print(f"平均最终覆盖率：{summary['mean_final_coverage']:.6f}")
    print(f"平均最大覆盖率：{summary['mean_max_coverage']:.6f}")
    print(f"平均回报：{summary['mean_reward']:.6f}")
    print(f"平均推理耗时：{summary['mean_inference_ms']:.3f} ms")


if __name__ == "__main__":
    main()
