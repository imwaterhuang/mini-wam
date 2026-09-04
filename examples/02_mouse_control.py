"""Control the Push-T agent with the mouse."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import gym_pusht  # noqa: F401 - importing registers the environment
import numpy as np
import pygame


WINDOW_SIZE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Exit after this many steps; 0 keeps running until you quit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        visualization_width=WINDOW_SIZE,
        visualization_height=WINDOW_SIZE,
    )
    observation, info = env.reset(seed=42)

    pygame.init()
    window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Mini-WAM Push-T — move mouse, R reset, Esc quit")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    running = True
    step = 0
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    observation, info = env.reset()
                    step = 0

            mouse_x, mouse_y = pygame.mouse.get_pos()
            action = np.array([mouse_x, mouse_y], dtype=np.float32)
            observation, reward, terminated, truncated, info = env.step(action)

            frame = env.render()
            surface = pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))
            window.blit(surface, (0, 0))

            status = f"reward={reward:.3f}  coverage={info['coverage']:.3f}"
            window.blit(font.render(status, True, (20, 20, 20)), (10, 10))
            pygame.display.flip()
            clock.tick(10)

            step += 1
            if terminated:
                print("Success! Press R to try another starting state.")
            if truncated or (args.max_steps and step >= args.max_steps):
                running = False
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()

