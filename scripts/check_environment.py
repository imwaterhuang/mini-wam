"""Verify the local Mini-WAM runtime and record reproducibility metadata."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import gym_pusht  # noqa: F401 - importing registers the Push-T environment
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "environment.json"
LEROBOT_COMMIT = "30da8e687a6dfc617fcd94afc367ac7071c376ce"


def describe_value(value: Any) -> dict[str, Any]:
    """Return a small JSON-safe description of an observation or frame."""
    if isinstance(value, dict):
        return {key: describe_value(item) for key, item in value.items()}

    array = np.asarray(value)
    return {
        "type": type(value).__name__,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def main() -> None:
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )
    try:
        observation, _ = env.reset(seed=42)
        frame = env.render()
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, _ = env.step(action)
    finally:
        env.close()

    metadata = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "torchvision",
                "lerobot",
                "gymnasium",
                "gym-pusht",
                "numpy",
            )
        },
        "lerobot_upstream_commit": LEROBOT_COMMIT,
        "torch_device": {
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "selected": "mps" if torch.backends.mps.is_available() else "cpu",
        },
        "pusht_smoke_test": {
            "observation": describe_value(observation),
            "next_observation": describe_value(next_observation),
            "rendered_frame": describe_value(frame),
            "action": describe_value(action),
            "action_low": np.asarray(env.action_space.low).tolist(),
            "action_high": np.asarray(env.action_space.high).tolist(),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        },
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"\nEnvironment check passed: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
