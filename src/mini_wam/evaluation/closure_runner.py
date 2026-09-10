"""Model-backed reproduction of the frozen closure protocol (no training)."""
from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path

import torch

from mini_wam.data import NormalizationStats
from mini_wam.models.action_only import ActionOnlyPolicy
from mini_wam.models.future_head import FutureHeadPolicy, masked_cosine_loss
from .closure import EVIDENCE_ROOT, PROJECT_ROOT, read_json, sha256_file


def load_frozen_policy(path, name, device):
    expected = read_json(EVIDENCE_ROOT / "FINAL_REPORT.json")["checkpoints"][name]
    if sha256_file(path) != expected["sha256"]:
        raise ValueError(f"{name}: checkpoint hash differs from the frozen selection")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["step"] != expected["step"]:
        raise ValueError("Checkpoint step mismatch")
    if payload["split_hash"] != sha256_file(PROJECT_ROOT / "splits/episodes_seed42.json"):
        raise ValueError("Checkpoint training/validation split mismatch")
    values = read_json(EVIDENCE_ROOT / "training" / name / "normalization.json")
    if values != payload["metadata"]["normalization"]:
        raise ValueError("Checkpoint normalization differs from archived training statistics")
    norm = NormalizationStats(**{k: torch.tensor(v, dtype=torch.float32) if isinstance(v, list) else v for k, v in values.items()})
    model = (ActionOnlyPolicy(pretrained=False) if name == "action_only" else
             FutureHeadPolicy(pretrained=False, target_pretrained=False))
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, norm, {**expected, "dataset_fingerprint": payload["metadata"]["dataset_fingerprint"]}


def runtime_metadata(device):
    packages = {}
    for name in ("torch", "torchvision", "gym-pusht", "gymnasium", "pymunk", "imageio", "imageio-ffmpeg", "lerobot"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "dtype": "float32", "source_sha256": {
                str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in
                sorted((PROJECT_ROOT / "src/mini_wam").rglob("*.py"))},
            "timing_note": "Archived rollout timing measures forward only; it is not the prescribed end-to-end latency benchmark."}


def reflected_action_prefix(prefix, normalized_positions, normalization):
    """Reflect absolute targets about the current agent position, then clip."""
    raw = normalization.denormalize_action(prefix.cpu())
    position = normalization.denormalize_position(normalized_positions[:, -1].cpu()).unsqueeze(1)
    return normalization.normalize_action((2 * position - raw).clamp(0, 512)).to(prefix.device)


@torch.inference_mode()
def counterfactual_batch(model, batch, norm, device):
    model.eval()
    obs, pos, prefix, future, mask = (
        batch["observation_history"].to(device), batch["agent_position"].to(device),
        batch["action_chunk"][:, :4].to(device), batch["future_observations"].to(device),
        batch["future_valid_mask"].to(device))
    if prefix.shape[0] < 2:
        raise ValueError("Shuffling requires at least two windows")
    fused = model.history_fusion(model.visual_encoder(obs), model.state_encoder(pos))
    target = model.frozen_target_encoder(future)
    variants = {"correct": prefix, "shuffled": prefix.roll(1, 0),
                "reversed": reflected_action_prefix(prefix, pos, norm)}
    losses = {}
    for name, actions in variants.items():
        pred = model.future_head(torch.cat([fused, model.action_prefix_encoder(actions)], dim=-1))
        losses[name] = float(masked_cosine_loss(pred, target, mask))
    return losses, int(mask.sum())
