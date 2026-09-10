"""Push-T scene handling and closed-loop ActionOnlyPolicy rollout."""

from __future__ import annotations

import os
import importlib.util
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event
from typing import Generator

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import gym_pusht  # noqa: F401 - importing registers gym_pusht/PushT-v0
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn

from mini_wam.data.dataset import IMAGENET_MEAN, IMAGENET_STD, NormalizationStats


@dataclass(frozen=True)
class SceneState:
    agent_x: float
    agent_y: float
    block_x: float
    block_y: float
    block_angle: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.agent_x, self.agent_y, self.block_x, self.block_y, self.block_angle],
            dtype=np.float32,
        )

    @classmethod
    def from_array(cls, value: np.ndarray) -> "SceneState":
        return cls(*(float(item) for item in value))


@dataclass(frozen=True)
class RolloutMetrics:
    is_success: bool
    final_coverage: float
    max_coverage: float
    reward: float
    steps: int
    mean_inference_ms: float
    stopped: bool
    model_calls: int

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutUpdate:
    frame: np.ndarray
    metrics: RolloutMetrics
    video_path: Path | None = None


def make_pusht_env(max_steps: int = 300):
    """Create the canonical Push-T environment used by every evaluator."""
    return gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=96,
        observation_height=96,
        visualization_width=384,
        visualization_height=384,
        max_episode_steps=max_steps,
    )


# Backward-compatible private alias for the existing workbench code.
_make_env = make_pusht_env


def validate_scene(scene: SceneState) -> None:
    values = scene.as_array()
    if not np.isfinite(values).all():
        raise ValueError("场景坐标与角度必须是有限数值")
    if not (50 <= scene.agent_x <= 450 and 50 <= scene.agent_y <= 450):
        raise ValueError("智能体中心必须位于环境 [50, 450] 范围内")
    if not (100 <= scene.block_x <= 400 and 100 <= scene.block_y <= 400):
        raise ValueError("T 块中心必须位于环境 [100, 400] 范围内")
    if not (-np.pi <= scene.block_angle <= np.pi):
        raise ValueError("T 块角度必须位于 [-π, π]")
    if np.hypot(scene.agent_x - scene.block_x, scene.agent_y - scene.block_y) < 45:
        raise ValueError("智能体与 T 块初始位置重叠")
    # The T is not circular, so center distance alone cannot catch its long bar
    # or a rotated block touching a wall. Let the real physics engine perform
    # one reset step and reject any initial contact.
    env = _make_env(max_steps=1)
    try:
        _, info = env.reset(options={"reset_to_state": scene.as_array()})
        if int(info.get("n_contacts", 0)) > 0:
            raise ValueError("智能体、T 块或边界存在非法初始接触")
    finally:
        env.close()


def canvas_to_environment(x: float, y: float, canvas_size: int = 384) -> tuple[float, float]:
    if not (0 <= x <= canvas_size and 0 <= y <= canvas_size):
        raise ValueError("点击位置超出画布")
    return x * 512.0 / canvas_size, y * 512.0 / canvas_size


def random_scene(seed: int) -> SceneState:
    # Match PushTEnv.reset's public sampling procedure. We keep the requested
    # reset state rather than reading the block's center-of-mass-shifted pose
    # back from info, so replaying the scene is exact.
    rng = np.random.default_rng(int(seed))
    return SceneState(
        agent_x=float(rng.integers(50, 450)),
        agent_y=float(rng.integers(50, 450)),
        block_x=float(rng.integers(100, 400)),
        block_y=float(rng.integers(100, 400)),
        block_angle=float(rng.uniform(-np.pi, np.pi)),
    )


def render_scene(scene: SceneState) -> np.ndarray:
    validate_scene(scene)
    env = _make_env()
    try:
        env.reset(options={"reset_to_state": scene.as_array()})
        return np.asarray(env.render(), dtype=np.uint8)
    finally:
        env.close()


def _image_tensor(image: np.ndarray) -> Tensor:
    resized = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((96, 96), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def prepare_model_input(
    observations: list[dict[str, np.ndarray]],
    normalization: NormalizationStats,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if len(observations) != 2:
        raise ValueError("ActionOnlyPolicy 必须接收恰好两帧历史 observation（观测）")
    images = torch.stack([_image_tensor(item["pixels"]) for item in observations]).unsqueeze(0)
    positions = torch.from_numpy(
        np.stack([np.asarray(item["agent_pos"], dtype=np.float32) for item in observations])
    ).unsqueeze(0)
    positions = normalization.normalize_position(positions)
    return images.to(device), positions.to(device)


def denormalize_actions(prediction: Tensor, normalization: NormalizationStats) -> np.ndarray:
    if tuple(prediction.shape) != (1, 16, 2):
        raise ValueError(f"模型输出必须为 [1,16,2]，实际为 {tuple(prediction.shape)}")
    actions = normalization.denormalize_action(prediction.detach().cpu()).squeeze(0).numpy()
    return np.clip(actions, 0.0, 512.0).astype(np.float32)


def _device_of(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _annotate_frame(frame: np.ndarray, label: str | None) -> np.ndarray:
    if not label:
        return frame
    image = Image.fromarray(frame)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 6, 238, 31), fill=(255, 255, 255))
    draw.text((12, 11), label, fill=(180, 35, 35))
    return np.asarray(image)


def run_rollout(
    model: nn.Module,
    normalization: NormalizationStats,
    scene: SceneState,
    output_path: str | Path,
    stop_event: Event | None = None,
    max_steps: int = 300,
    execute_steps: int = 4,
    initial_history: list[dict[str, np.ndarray]] | None = None,
    scene_is_body_pose: bool = False,
    frame_label: str | None = None,
) -> Generator[RolloutUpdate, None, RolloutMetrics]:
    """Stream a real closed-loop rollout and write its final MP4 video."""
    if scene_is_body_pose:
        if not np.isfinite(scene.as_array()).all():
            raise ValueError("恢复场景含无效数值")
    else:
        validate_scene(scene)
    if initial_history is not None and len(initial_history) != 2:
        raise ValueError("恢复训练窗口必须提供恰好两帧历史")
    if execute_steps != 4:
        raise ValueError("当前执行器固定每次执行动作块前 4 步")
    stop_event = stop_event or Event()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    device = _device_of(model)
    env = _make_env(max_steps=max_steps)
    frames: list[np.ndarray] = []
    inference_ms: list[float] = []
    reward_sum = 0.0
    max_coverage = 0.0
    final_coverage = 0.0
    steps = 0
    model_calls = 0
    stopped = False
    success = False
    try:
        if scene_is_body_pose:
            observation, info = env.reset(seed=0)
            unwrapped = env.unwrapped
            unwrapped.agent.position = [scene.agent_x, scene.agent_y]
            unwrapped.agent.velocity = (0, 0)
            # The recovered pose is the rendered Pymunk body pose. Set angle
            # first, then position, to avoid center-of-gravity compensation.
            unwrapped.block.angle = scene.block_angle
            unwrapped.block.position = [scene.block_x, scene.block_y]
            unwrapped.block.velocity = (0, 0)
            unwrapped.block.angular_velocity = 0
            unwrapped.space.step(unwrapped.dt)
            observation = unwrapped.get_obs()
            info = unwrapped._get_info()
            info["is_success"] = False
        else:
            observation, info = env.reset(options={"reset_to_state": scene.as_array()})
        history: deque[dict[str, np.ndarray]] = deque(
            initial_history if initial_history is not None else [observation, observation],
            maxlen=2,
        )
        frame = _annotate_frame(np.asarray(env.render(), dtype=np.uint8), frame_label)
        frames.append(frame)
        initial = RolloutMetrics(False, 0.0, 0.0, 0.0, 0, 0.0, False, 0)
        yield RolloutUpdate(frame=frame, metrics=initial)

        terminated = truncated = False
        while steps < max_steps and not terminated and not truncated:
            if stop_event.is_set():
                stopped = True
                break
            images, positions = prepare_model_input(list(history), normalization, device)
            start = time.perf_counter()
            with torch.inference_mode():
                prediction = model(images, positions)
            inference_ms.append((time.perf_counter() - start) * 1000.0)
            model_calls += 1
            actions = denormalize_actions(prediction, normalization)
            for action in actions[:execute_steps]:
                if stop_event.is_set():
                    stopped = True
                    break
                observation, reward, terminated, truncated, info = env.step(action)
                history.append(observation)
                steps += 1
                reward_sum += float(reward)
                final_coverage = float(info.get("coverage", 0.0))
                max_coverage = max(max_coverage, final_coverage)
                success = bool(info.get("is_success", False))
                frame = _annotate_frame(np.asarray(env.render(), dtype=np.uint8), frame_label)
                frames.append(frame)
                current = RolloutMetrics(
                    success,
                    final_coverage,
                    max_coverage,
                    reward_sum,
                    steps,
                    float(np.mean(inference_ms)),
                    stopped,
                    model_calls,
                )
                yield RolloutUpdate(frame=frame, metrics=current)
                if terminated or truncated or steps >= max_steps:
                    break
            if stopped:
                break

        # PyAV is present in the local lock; Colab may only have imageio-ffmpeg.
        # Their pixel-format keyword names differ. Choose the backend explicitly.
        if importlib.util.find_spec("av") is not None:
            iio.imwrite(output, np.stack(frames), plugin="pyav", fps=10,
                        codec="libx264", out_pixel_format="yuv420p")
        else:
            iio.imwrite(output, np.stack(frames), plugin="FFMPEG", fps=10,
                        codec="libx264", pixelformat="yuv420p")
        final = RolloutMetrics(
            success,
            final_coverage,
            max_coverage,
            reward_sum,
            steps,
            float(np.mean(inference_ms)) if inference_ms else 0.0,
            stopped,
            model_calls,
        )
        yield RolloutUpdate(frame=frames[-1], metrics=final, video_path=output)
        return final
    finally:
        env.close()
