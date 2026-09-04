"""Run a deterministic random Push-T episode and save a short animation."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import gym_pusht  # noqa: F401 - importing registers the environment
import imageio.v2 as imageio
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "random_rollout.gif"


def main() -> None:
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )
    env.action_space.seed(42)
    observation, _ = env.reset(seed=42)

    frames: list[np.ndarray] = [env.render()]
    total_reward = 0.0
    action = env.action_space.sample()

    try:
        for step in range(120):
            # Hold each target for 15 simulator steps so its effect is visible.
            if step % 15 == 0:
                action = env.action_space.sample()

            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            frames.append(env.render())

            if terminated or truncated:
                break
    finally:
        env.close()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(OUTPUT_PATH, frames, duration=50, loop=0)

    print(f"pixels shape: {observation['pixels'].shape}")
    print(f"agent_pos: {observation['agent_pos']}")
    print(f"last action: {action}")
    print(f"episode steps: {len(frames) - 1}")
    print(f"sum of per-step rewards: {total_reward:.4f}")
    print(f"saved animation: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

