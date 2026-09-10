"""Build an optional normalized-frame cache on the runtime's local disk."""
from pathlib import Path
import argparse
import os
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mini_wam.training.config import load_training_config
from mini_wam.training.data import build_training_data
from mini_wam.data.image_cache import build_image_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.pop("MINI_WAM_IMAGE_CACHE", None)
    torch.set_num_threads(1)
    config = load_training_config(args.config, expected_model_name="future_aware")
    data = build_training_data(config=config, seed=0, num_workers=0,
                               include_future_observations=True)
    path = build_image_cache(data.train_loader.dataset, args.output)
    print(f"IMAGE_CACHE_READY {path}", flush=True)


if __name__ == "__main__":
    main()
