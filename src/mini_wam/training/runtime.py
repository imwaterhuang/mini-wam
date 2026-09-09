"""训练设备选择与 batch 张量搬运。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch


def select_device(device_name: str = "auto") -> torch.device:
    """选择可用设备，并对显式但不可用的设备立即报错。"""
    normalized = device_name.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 cuda，但当前运行环境没有可用的 CUDA 设备")
    if normalized == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("请求了 mps，但当前运行环境没有可用的 MPS 设备")
    if normalized not in {"cpu", "cuda", "mps"}:
        raise ValueError("device 必须是 auto、cpu、cuda 或 mps")
    return torch.device(normalized)


def move_tensors_to_device(
    batch: Mapping[str, torch.Tensor],
    keys: Iterable[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """只搬运一次训练步骤真正使用的张量。"""
    missing = [key for key in keys if key not in batch]
    if missing:
        raise KeyError(f"batch 缺少字段：{', '.join(missing)}")
    return {key: batch[key].to(device) for key in keys}
