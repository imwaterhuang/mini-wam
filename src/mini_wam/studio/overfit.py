"""Diagnostics for the 64 windows saved in action_only_overfit.pt."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageDraw

from mini_wam.data import MiniWAMDataset, NormalizationStats, load_episode_split
from mini_wam.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from mini_wam.models.action_only import ActionOnlyPolicy
from mini_wam.studio.pusht import SceneState


@dataclass(frozen=True)
class WindowDiagnostic:
    slot: int
    dataset_index: int
    episode_id: int
    episode_step: int
    global_index: int
    valid_steps: int
    normalized_loss: float
    action_error_mean: float
    first_four_error_mean: float
    history_image: np.ndarray
    action_overlay: np.ndarray


@dataclass(frozen=True)
class AggregateDiagnostic:
    sample_count: int
    valid_steps: int
    stored_initial_loss: float
    stored_final_loss: float
    recomputed_loss: float
    action_error_mean: float
    action_error_median: float
    action_error_p90: float
    first_four_error_mean: float


@dataclass(frozen=True)
class RecoveredWindow:
    scene: SceneState
    initial_history: list[dict[str, np.ndarray]]
    block_mask_iou: float


class OverfitEvaluator:
    """Recreate the exact dataset windows named by the legacy checkpoint."""

    def __init__(self, project_root: str | Path) -> None:
        self.root = Path(project_root).resolve()
        self.checkpoint_path = self.root / "artifacts" / "action_only_overfit.pt"
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        indices = payload.get("indices")
        if not isinstance(indices, list) or len(indices) != 64:
            raise ValueError("过拟合 checkpoint 没有保存完整的 64 个训练索引")
        self.indices = [int(value) for value in indices]
        self.stored_initial_loss = float(payload["initial_loss"])
        self.stored_final_loss = float(payload["final_loss"])
        self.stats = NormalizationStats.from_audit_file(self.root / "artifacts" / "data_audit.json")
        self.dataset = MiniWAMDataset(
            self.root / "data" / "lerobot" / "pusht_image",
            load_episode_split(self.root / "splits" / "episodes_seed42.json", "train"),
            self.stats,
        )
        self.model = ActionOnlyPolicy(pretrained=False)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval()

    @staticmethod
    def _to_uint8(image: torch.Tensor) -> np.ndarray:
        value = image * IMAGENET_STD + IMAGENET_MEAN
        return (
            value.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        )

    @staticmethod
    def _history_montage(images: torch.Tensor) -> np.ndarray:
        frames = [
            np.asarray(
                Image.fromarray(OverfitEvaluator._to_uint8(image)).resize(
                    (384, 384), Image.Resampling.NEAREST
                )
            )
            for image in images
        ]
        return np.concatenate(frames, axis=1)

    def _overlay(
        self,
        current_image: torch.Tensor,
        current_position: torch.Tensor,
        expert: torch.Tensor,
        prediction: torch.Tensor,
    ) -> np.ndarray:
        base = Image.fromarray(self._to_uint8(current_image)).resize(
            (384, 384), Image.Resampling.NEAREST
        )
        draw = ImageDraw.Draw(base)

        def point(value: torch.Tensor) -> tuple[float, float]:
            return float(value[0]) * 0.75, float(value[1]) * 0.75

        origin = point(current_position)
        expert_points = [origin, *(point(value) for value in expert)]
        predicted_points = [origin, *(point(value) for value in prediction)]
        draw.line(expert_points, fill=(0, 190, 80), width=4)
        draw.line(predicted_points, fill=(235, 55, 55), width=4)
        for value in expert_points[1:]:
            draw.ellipse((value[0] - 3, value[1] - 3, value[0] + 3, value[1] + 3), fill=(0, 190, 80))
        for value in predicted_points[1:]:
            draw.ellipse((value[0] - 3, value[1] - 3, value[0] + 3, value[1] + 3), fill=(235, 55, 55))
        draw.rectangle((8, 8, 210, 54), fill=(255, 255, 255))
        draw.text((16, 14), "Expert: green", fill=(0, 150, 60))
        draw.text((16, 32), "Model: red", fill=(210, 30, 30))
        return np.asarray(base)

    @staticmethod
    def _block_mask(image: np.ndarray) -> np.ndarray:
        red = image[:, :, 0].astype(np.int16)
        green = image[:, :, 1].astype(np.int16)
        blue = image[:, :, 2].astype(np.int16)
        return (
            (blue - red >= 10)
            & (blue - red <= 80)
            & (green - red >= 5)
            & (green - red <= 50)
            & (blue < 220)
        )

    @staticmethod
    def _render_block_mask(position: np.ndarray, angle: float) -> np.ndarray:
        polygons = (
            np.asarray([[-60, 0], [60, 0], [60, 30], [-60, 30]], dtype=np.float64),
            np.asarray([[-15, 30], [15, 30], [15, 120], [-15, 120]], dtype=np.float64),
        )
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        mask = np.zeros((96, 96), dtype=np.uint8)
        for polygon in polygons:
            pixels = ((polygon @ rotation.T + position) * 96 / 512).round().astype(np.int32)
            cv2.fillPoly(mask, [pixels], 1)
        return mask.astype(bool)

    @classmethod
    def _recover_block_pose(cls, image: np.ndarray) -> tuple[np.ndarray, float, float]:
        observed = cls._block_mask(image)
        y_pixels, x_pixels = np.where(observed)
        if len(x_pixels) < 100:
            raise ValueError("训练图像中无法可靠识别 T 块")
        observed_centroid = np.asarray([x_pixels.mean(), y_pixels.mean()]) * 512 / 96
        local_centroid = np.asarray([0.0, 40.714285714285715])
        best_score = -1.0
        best_angle = 0.0
        best_position = np.zeros(2)

        def consider(angle: float, position: np.ndarray) -> None:
            nonlocal best_score, best_angle, best_position
            predicted = cls._render_block_mask(position, angle)
            union = np.logical_or(predicted, observed).sum()
            score = float(np.logical_and(predicted, observed).sum() / union)
            if score > best_score:
                best_score, best_angle, best_position = score, angle, position.copy()

        for angle in np.linspace(-np.pi, np.pi, 360, endpoint=False):
            rotation = np.asarray(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            consider(angle, observed_centroid - rotation @ local_centroid)

        coarse_angle = best_angle
        for angle in np.linspace(coarse_angle - np.radians(2), coarse_angle + np.radians(2), 81):
            rotation = np.asarray(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            center_position = observed_centroid - rotation @ local_centroid
            for dx in np.linspace(-4, 4, 9):
                for dy in np.linspace(-4, 4, 9):
                    consider(angle, center_position + [dx, dy])
        return best_position, best_angle, best_score

    def recover_rollout_start(self, slot: int) -> RecoveredWindow:
        if not 1 <= int(slot) <= len(self.indices):
            raise ValueError("训练窗口编号必须在 1 到 64 之间")
        sample = self.dataset[self.indices[int(slot) - 1]]
        raw_images = [self._to_uint8(image) for image in sample["observation_history"]]
        raw_positions = self.stats.denormalize_position(sample["agent_position"]).numpy()
        block_position, block_angle, mask_iou = self._recover_block_pose(raw_images[-1])
        history = [
            {"pixels": image, "agent_pos": position.astype(np.float64)}
            for image, position in zip(raw_images, raw_positions, strict=True)
        ]
        return RecoveredWindow(
            scene=SceneState(
                agent_x=float(raw_positions[-1, 0]),
                agent_y=float(raw_positions[-1, 1]),
                block_x=float(block_position[0]),
                block_y=float(block_position[1]),
                block_angle=float(block_angle),
            ),
            initial_history=history,
            block_mask_iou=mask_iou,
        )

    @torch.inference_mode()
    def inspect(self, slot: int) -> WindowDiagnostic:
        if not 1 <= int(slot) <= len(self.indices):
            raise ValueError("训练窗口编号必须在 1 到 64 之间")
        dataset_index = self.indices[int(slot) - 1]
        sample = self.dataset[dataset_index]
        prediction = self.model(
            sample["observation_history"].unsqueeze(0),
            sample["agent_position"].unsqueeze(0),
        )[0]
        mask = sample["action_valid_mask"]
        normalized_loss = F.smooth_l1_loss(
            prediction[mask], sample["action_chunk"][mask], reduction="mean"
        ).item()
        predicted_actions = self.stats.denormalize_action(prediction[mask])
        expert_actions = self.stats.denormalize_action(sample["action_chunk"][mask])
        errors = torch.linalg.vector_norm(predicted_actions - expert_actions, dim=-1)
        first_four = errors[:4]
        current_position = self.stats.denormalize_position(sample["agent_position"][-1])
        episode_id, episode_step, global_index, _ = self.dataset.source_indices(dataset_index)
        return WindowDiagnostic(
            slot=int(slot),
            dataset_index=dataset_index,
            episode_id=episode_id,
            episode_step=episode_step,
            global_index=global_index,
            valid_steps=int(mask.sum()),
            normalized_loss=float(normalized_loss),
            action_error_mean=float(errors.mean()),
            first_four_error_mean=float(first_four.mean()),
            history_image=self._history_montage(sample["observation_history"]),
            action_overlay=self._overlay(
                sample["observation_history"][-1],
                current_position,
                expert_actions,
                predicted_actions,
            ),
        )

    @torch.inference_mode()
    def evaluate_all(self) -> AggregateDiagnostic:
        errors: list[float] = []
        first_four_errors: list[float] = []
        loss_sum = 0.0
        valid_steps = 0
        for dataset_index in self.indices:
            sample = self.dataset[dataset_index]
            prediction = self.model(
                sample["observation_history"].unsqueeze(0),
                sample["agent_position"].unsqueeze(0),
            )[0]
            mask = sample["action_valid_mask"]
            count = int(mask.sum())
            # Match masked_smooth_l1_loss: average the two coordinates per step.
            loss_sum += float(
                F.smooth_l1_loss(
                    prediction[mask], sample["action_chunk"][mask], reduction="sum"
                )
                / 2
            )
            valid_steps += count
            predicted_actions = self.stats.denormalize_action(prediction[mask])
            expert_actions = self.stats.denormalize_action(sample["action_chunk"][mask])
            sample_errors = torch.linalg.vector_norm(predicted_actions - expert_actions, dim=-1)
            errors.extend(float(value) for value in sample_errors)
            first_four_errors.extend(float(value) for value in sample_errors[:4])
        values = np.asarray(errors)
        return AggregateDiagnostic(
            sample_count=len(self.indices),
            valid_steps=valid_steps,
            stored_initial_loss=self.stored_initial_loss,
            stored_final_loss=self.stored_final_loss,
            recomputed_loss=loss_sum / valid_steps,
            action_error_mean=float(values.mean()),
            action_error_median=float(np.median(values)),
            action_error_p90=float(np.percentile(values, 90)),
            first_four_error_mean=float(np.mean(first_four_errors)),
        )
