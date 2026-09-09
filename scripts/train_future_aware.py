"""Command-line entry point for reproducible FutureHeadPolicy training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / ".cache" / "huggingface"))
os.environ.setdefault(
    "HF_DATASETS_CACHE", str(PROJECT_ROOT / "data" / ".cache" / "hf_datasets")
)

from mini_wam.training import train_future_aware  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练或恢复 Mini-WAM future_aware 模型")
    parser.add_argument("--config", required=True, type=Path, help="YAML 训练配置")
    parser.add_argument("--resume", type=Path, help="从 last.pt 或其他 checkpoint 恢复")
    parser.add_argument("--run-dir", type=Path, help="新训练的输出目录")
    parser.add_argument("--mirror-dir", type=Path, help="可选持久化镜像目录")
    parser.add_argument("--stop-after-step", type=int, help="提前停在指定全局训练步")
    parser.add_argument("--device", default="auto", help="auto、cpu、mps 或 cuda")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载工作进程数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = train_future_aware(
        args.config,
        resume_path=args.resume,
        run_dir=args.run_dir,
        mirror_dir=args.mirror_dir,
        stop_after_step=args.stop_after_step,
        device_name=args.device,
        num_workers=args.num_workers,
    )
    print(f"训练产物：{run_dir}")


if __name__ == "__main__":
    main()
