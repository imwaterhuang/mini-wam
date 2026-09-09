"""确定性采样与随机状态保存/恢复。"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch.utils.data import Sampler


SAMPLER_STATE_VERSION = 2


class DeterministicBatchSampler(Sampler[list[int]]):
    """无限、可恢复且不受 DataLoader 预取位置影响的确定性 batch 采样器。"""

    def __init__(self, dataset_size: int, batch_size: int, seed: int) -> None:
        if dataset_size < 1 or batch_size < 1:
            raise ValueError("dataset_size 和 batch_size 必须为正数")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.batch_in_epoch = 0
        self._outstanding_batches = 0

    @property
    def batches_per_epoch(self) -> int:
        return math.ceil(self.dataset_size / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        epoch = self.epoch
        batch_in_epoch = self.batch_in_epoch
        while True:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(self.dataset_size, generator=generator).tolist()
            for batch_index in range(batch_in_epoch, self.batches_per_epoch):
                start = batch_index * self.batch_size
                self._outstanding_batches += 1
                yield order[start : start + self.batch_size]
            epoch += 1
            batch_in_epoch = 0

    def __len__(self) -> int:
        return self.batches_per_epoch

    def state_dict(self) -> dict[str, int]:
        return {
            "sampler_state_version": SAMPLER_STATE_VERSION,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        state_version = state.get("sampler_state_version")
        if state_version not in (None, SAMPLER_STATE_VERSION):
            raise ValueError(f"不支持的 sampler_state_version={state_version}")
        expected = {
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"采样器 {key} 不兼容：{state.get(key)} != {value}")
        epoch = int(state["epoch"])
        batch_in_epoch = int(state["batch_in_epoch"])
        if epoch < 0 or not 0 <= batch_in_epoch < self.batches_per_epoch:
            raise ValueError("checkpoint 中的采样器位置非法")
        self.epoch = epoch
        self.batch_in_epoch = batch_in_epoch
        self._outstanding_batches = 0

    def mark_consumed(self) -> None:
        """仅在参数更新完成后推进持久化采样位置。"""
        if self._outstanding_batches < 1:
            raise RuntimeError("没有已生成但尚未标记完成的 batch")
        self._outstanding_batches -= 1
        if self.batch_in_epoch + 1 == self.batches_per_epoch:
            self.epoch += 1
            self.batch_in_epoch = 0
        else:
            self.batch_in_epoch += 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_random_states() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gaussian": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_random_states(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["state"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gaussian"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda"]])
