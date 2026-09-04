"""Command-line entry point for reproducible ActionOnlyPolicy training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "data" / ".cache" / "hf_datasets"))

from mini_wam.training import train_action_only


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练或恢复 Mini-WAM action_only 基线")
    parser.add_argument("--config", required=True, type=Path, help="YAML 训练配置")
    parser.add_argument("--resume", type=Path, help="从 last.pt 或其他正式 checkpoint 恢复")
    parser.add_argument("--run-dir", type=Path, help="新训练的输出目录；恢复时通常无需指定")
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="仅用于可恢复性检查：提前停在指定全局训练步",
    )
    parser.add_argument("--device", default="auto", help="auto、cpu、mps 或 cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = train_action_only(
        args.config,
        resume_path=args.resume,
        run_dir=args.run_dir,
        stop_after_step=args.stop_after_step,
        device_name=args.device,
    )
    print(f"训练产物：{run_dir}")


if __name__ == "__main__":
    main()

