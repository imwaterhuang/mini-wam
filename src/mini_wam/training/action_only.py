"""``ActionOnlyPolicy`` 的可复现、可恢复训练器。

算法核心其实很短：读取 batch（批次）→ policy（策略）预测 action chunk
（动作块）→ 计算 masked loss（带掩码损失）→ backward（反向传播）→ 更新参数。

本文件其余代码负责把这个最小循环变成可信的工程实验：

1. 严格校验配置，避免静默使用错误的训练参数；
2. 固定数据顺序与所有 RNG（Random Number Generator，随机数生成器）状态；
3. 记录代码、数据划分和数据集指纹；
4. 执行验证、日志记录和最优模型选择；
5. 原子保存完整 checkpoint（检查点），并支持精确恢复；
6. 将本地高速运行目录增量镜像到持久化目录。

各功能区块从上到下排列，主入口 ``train_action_only`` 位于文件末尾。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler

from mini_wam.data import MiniWAMDataset, NormalizationStats, load_episode_split
from mini_wam.models.action_only import ActionOnlyPolicy, masked_smooth_l1_loss
from mini_wam.studio.datasets import validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_VERSION = 1
SAMPLER_STATE_VERSION = 2


# =============================================================================
# 1. 训练配置：读取 YAML（YAML Ain't Markup Language，配置文件格式）并严格校验
# =============================================================================


def _require(mapping: dict[str, Any], path: str, expected_type: type) -> Any:
    """读取 ``a.b.c`` 形式的嵌套配置项，并检查其 Python 类型。"""
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"配置缺少 {path}")
        value = value[part]
    if expected_type is float and isinstance(value, (int, float)):
        return float(value)
    if expected_type is int and isinstance(value, int) and not isinstance(value, bool):
        return value
    if expected_type is bool and isinstance(value, bool):
        return value
    if expected_type is str and isinstance(value, str):
        return value
    raise TypeError(f"配置 {path} 必须是 {expected_type.__name__}，实际为 {type(value).__name__}")


def load_training_config(path: str | Path) -> dict[str, Any]:
    """读取并严格校验 action-only 训练配置。

    这里主动拒绝与当前模型契约不一致的配置，例如不是两帧历史或不是
    16 步 action chunk，避免错误配置进入长时间正式训练后才暴露。
    """
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("训练配置顶层必须是字典")

    if _require(raw, "model.name", str) != "action_only":
        raise ValueError("当前训练器只支持 model.name=action_only")
    _require(raw, "model.pretrained", bool)
    _require(raw, "data.dataset_root", str)
    _require(raw, "data.split_path", str)
    _require(raw, "data.normalization_path", str)
    if _require(raw, "data.history_length", int) != 2:
        raise ValueError("ActionOnlyPolicy 固定使用两帧历史")
    if _require(raw, "data.action_horizon", int) != 16:
        raise ValueError("ActionOnlyPolicy 固定输出 16 步动作")
    if _require(raw, "data.future_horizon", int) < 1:
        raise ValueError("future_horizon 必须为正数")

    positive_ints = (
        "training.batch_size",
        "training.train_steps",
        "validation.frequency",
        "checkpoint.frequency",
    )
    for key in positive_ints:
        if _require(raw, key, int) < 1:
            raise ValueError(f"配置 {key} 必须为正数")
    archive_frequency = raw["checkpoint"].get("archive_frequency")
    if archive_frequency is not None and (
        not isinstance(archive_frequency, int)
        or isinstance(archive_frequency, bool)
        or archive_frequency < raw["checkpoint"]["frequency"]
        or archive_frequency % raw["checkpoint"]["frequency"] != 0
    ):
        raise ValueError("checkpoint.archive_frequency 必须是 frequency 的正整数倍")
    _require(raw, "training.seed", int)
    for key in ("training.learning_rate", "training.gradient_clip_norm"):
        if _require(raw, key, float) <= 0:
            raise ValueError(f"配置 {key} 必须大于 0")
    if _require(raw, "training.weight_decay", float) < 0:
        raise ValueError("training.weight_decay 不能为负数")
    warmup = _require(raw, "training.warmup_steps", int)
    if warmup < 0 or warmup >= raw["training"]["train_steps"]:
        raise ValueError("warmup_steps 必须在 [0, train_steps) 内")
    max_batches = raw["validation"].get("max_batches")
    if max_batches is not None and (not isinstance(max_batches, int) or max_batches < 1):
        raise ValueError("validation.max_batches 必须为 null 或正整数")
    return raw


# =============================================================================
# 2. 路径、哈希与实验来源：保证训练产物能够追溯到代码和数据
# =============================================================================


def _resolve_project_path(value: str) -> Path:
    """把相对路径解释为相对于项目根目录的绝对路径。"""
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _sha256_file(path: Path) -> str:
    """分块计算文件的 SHA-256（Secure Hash Algorithm 256-bit，安全哈希算法）指纹。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint() -> str:
    """对训练相关 Python 源码计算整体指纹，记录实际运行的代码内容。"""
    digest = hashlib.sha256()
    paths = sorted((PROJECT_ROOT / "src").glob("**/*.py"))
    paths += sorted((PROJECT_ROOT / "scripts").glob("*.py"))
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit() -> str | None:
    """返回当前 Git 提交；目录不是仓库或 Git 不可用时返回 ``None``。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _normalization_dict(stats: NormalizationStats) -> dict[str, Any]:
    """把训练集归一化统计量转换为可序列化字典。"""
    return {
        "position_mean": stats.position_mean.tolist(),
        "position_std": stats.position_std.tolist(),
        "action_mean": stats.action_mean.tolist(),
        "action_std": stats.action_std.tolist(),
        "std_floor": stats.std_floor,
    }


# =============================================================================
# 3. 确定性与精确恢复：数据顺序和随机状态必须接在中断位置之后
# =============================================================================


class DeterministicBatchSampler(Sampler[list[int]]):
    """无限、可恢复且不受 DataLoader 预取位置影响的确定性 batch 采样器。

    ``DataLoader`` 可以提前向 sampler 请求多个 batch，但可恢复位置只在训练循环调用
    :meth:`mark_consumed` 后推进。因此 checkpoint 记录的是“下一个尚未完成训练的 batch”，
    而不是“下一个尚未被 worker 预取的 batch”。
    """

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
        # 迭代器使用局部游标向前生成预取 batch；self 中的持久游标只由
        # mark_consumed() 推进，二者必须刻意分离。
        epoch = self.epoch
        batch_in_epoch = self.batch_in_epoch
        while True:
            # 每个 epoch 使用 seed + epoch，既能改变顺序，又能从相同状态重建顺序。
            generator = torch.Generator().manual_seed(self.seed + epoch)
            order = torch.randperm(self.dataset_size, generator=generator).tolist()
            for batch_index in range(batch_in_epoch, self.batches_per_epoch):
                start = batch_index * self.batch_size
                batch = order[start : start + self.batch_size]
                self._outstanding_batches += 1
                yield batch
            epoch += 1
            batch_in_epoch = 0

    def __len__(self) -> int:
        return self.batches_per_epoch

    def state_dict(self) -> dict[str, int]:
        """导出足以重建下一批数据位置的采样器状态。"""
        return {
            "sampler_state_version": SAMPLER_STATE_VERSION,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """校验并恢复采样器状态，拒绝不同数据规模、batch 或 seed。"""
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
        """在一次 optimizer 更新完成后推进 checkpoint 应保存的 batch 位置。"""
        if self._outstanding_batches < 1:
            raise RuntimeError("没有已生成但尚未标记完成的 batch")
        self._outstanding_batches -= 1
        if self.batch_in_epoch + 1 == self.batches_per_epoch:
            self.epoch += 1
            self.batch_in_epoch = 0
        else:
            self.batch_in_epoch += 1


def _seed_everything(seed: int) -> None:
    """统一初始化 Python、NumPy、PyTorch 和 CUDA 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_random_states() -> dict[str, Any]:
    """捕获所有随机数生成器的当前位置，供 checkpoint 精确恢复。"""
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        # Store NumPy's uint32 array as a tensor so torch.load(...,
        # weights_only=True) can load the checkpoint without allowlisting
        # arbitrary Python globals.
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


def _restore_random_states(state: dict[str, Any]) -> None:
    """恢复 checkpoint 保存时的全部随机数生成器状态。"""
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
    # ``map_location=device`` also moves RNG ByteTensors to CUDA, while the
    # RNG restore APIs require their state tensors to live on the CPU.
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda"]])


# =============================================================================
# 4. 优化与离线验证：学习率曲线和验证口径在训练前固定
# =============================================================================


def _make_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, train_steps: int) -> LambdaLR:
    """创建线性 warmup（预热）后接余弦衰减的逐步学习率调度器。"""

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            # 预热阶段从较小学习率线性升到配置中的基础学习率。
            return max(step, 1) / warmup_steps
        # 预热结束后，学习率按半个余弦周期平滑衰减到 0。
        progress = (step - warmup_steps) / max(train_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=scale)


@torch.inference_mode()
def _evaluate(
    model: ActionOnlyPolicy,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> float:
    """在验证集上计算按有效 action step 加权的平均 masked loss。"""
    model.eval()
    weighted_loss = 0.0
    valid_steps = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        mask = batch["action_valid_mask"].to(device)
        prediction = model(
            batch["observation_history"].to(device),
            batch["agent_position"].to(device),
        )
        loss = masked_smooth_l1_loss(prediction, batch["action_chunk"].to(device), mask)
        # 尾部 padding（填充）长度不同，因此按有效 action 数而不是 batch 数加权。
        count = int(mask.sum().item())
        weighted_loss += float(loss.item()) * count
        valid_steps += count
    if not valid_steps:
        raise RuntimeError("验证集没有有效动作")
    return weighted_loss / valid_steps


# =============================================================================
# 5. 日志与 checkpoint：持续记录指标，并避免写出半个损坏文件
# =============================================================================


def _append_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    """向 CSV（Comma-Separated Values，逗号分隔值）指标文件追加一行。"""
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """先写临时文件再原子替换，避免中断时留下不完整 checkpoint。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _environment_summary(device: torch.device) -> dict[str, Any]:
    """记录复现实验所需的运行环境、Git 提交和源码指纹。"""
    return {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "git_commit": _git_commit(),
        "source_fingerprint": _source_fingerprint(),
    }


def _checkpoint_payload(
    *,
    model: ActionOnlyPolicy,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    sampler: DeterministicBatchSampler,
    step: int,
    config: dict[str, Any],
    best_validation_loss: float,
    validation_loss: float | None,
    stats: NormalizationStats,
    split_hash: str,
    dataset_fingerprint: str,
    environment: dict[str, Any],
) -> dict[str, Any]:
    """组装完整 checkpoint，而不只保存模型权重。

    optimizer、scheduler（学习率调度器）、sampler（采样器）和随机状态共同保证
    中断恢复后的下一步与不中断训练一致。metadata 记录模型的数据契约，便于评估器
    在加载模型前做兼容性检查。
    """
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "sampler": sampler.state_dict(),
        "step": step,
        "config": config,
        "best_validation_loss": best_validation_loss,
        "validation_loss": validation_loss,
        "random_states": _capture_random_states(),
        "split_hash": split_hash,
        "source_commit": environment["git_commit"],
        "source_fingerprint": environment["source_fingerprint"],
        "metadata": {
            "architecture": "action_only_v1",
            "task": "pusht",
            "history_frames": 2,
            "image_shape": [3, 96, 96],
            "state_dim": 2,
            "action_horizon": 16,
            "action_dim": 2,
            "action_range": [0.0, 512.0],
            "dataset_fingerprint": dataset_fingerprint,
            "normalization": _normalization_dict(stats),
        },
    }


def _default_run_dir(seed: int) -> Path:
    """按随机种子和当前时间生成默认运行目录。"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return PROJECT_ROOT / "runs" / "action_only" / str(seed) / timestamp


def _log(run_dir: Path, message: str) -> None:
    """同时输出到终端并追加到运行目录的 ``run.log``。"""
    print(message, flush=True)
    with (run_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


# =============================================================================
# 6. 运行目录镜像：本地高速训练，持久化目录负责抵抗 runtime 丢失
# =============================================================================


def _sync_run_directory(source: Path, destination: Path) -> None:
    """把变化过的运行产物增量镜像到持久化目录。

    Colab 中 ``source`` 通常位于高速但临时的 ``/content``，``destination`` 通常
    位于 Google Drive。复制同样使用临时文件加原子替换，避免远程中断损坏旧文件。
    """
    if source.resolve() == destination.resolve():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for source_path in source.rglob("*"):
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
            continue
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        should_copy = not destination_path.exists()
        if not should_copy:
            source_stat = source_path.stat()
            destination_stat = destination_path.stat()
            should_copy = (
                source_stat.st_size != destination_stat.st_size
                or source_stat.st_mtime_ns != destination_stat.st_mtime_ns
            )
        if should_copy:
            temporary = destination_path.with_suffix(destination_path.suffix + ".sync-tmp")
            shutil.copy2(source_path, temporary)
            temporary.replace(destination_path)


# =============================================================================
# 7. 主训练流程：初始化 → 数据 → 模型 → 恢复 → 训练 → 验证 → 保存
# =============================================================================


def train_action_only(
    config_path: str | Path,
    *,
    resume_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    mirror_dir: str | Path | None = None,
    stop_after_step: int | None = None,
    device_name: str = "auto",
    num_workers: int = 0,
) -> Path:
    """训练、验证、保存并可选恢复 ``ActionOnlyPolicy``。

    参数：
        config_path: 训练配置文件。
        resume_path: 可选的 ``last.pt`` 或其他正式 checkpoint。
        run_dir: 当前机器上的高速运行目录。
        mirror_dir: 可选的持久化镜像目录，Colab 中通常指向 Google Drive。
        stop_after_step: 仅用于恢复性测试的提前停止位置。
        device_name: ``auto``、``cpu``、``mps`` 或 ``cuda``。
        num_workers: DataLoader 的并行数据工作进程数；0 表示主进程串行取数。

    返回：
        实际写入训练产物的本地运行目录。
    """

    # -------------------------------------------------------------------------
    # 7.1 读取配置、确定目标 step 和计算设备
    # -------------------------------------------------------------------------
    config_path = Path(config_path).expanduser().resolve()
    config = load_training_config(config_path)
    training = config["training"]
    seed = int(training["seed"])
    train_steps = int(training["train_steps"])
    if stop_after_step is not None and not 1 <= stop_after_step <= train_steps:
        raise ValueError("stop_after_step 必须在 [1, train_steps] 内")
    target_step = stop_after_step or train_steps
    if num_workers < 0:
        raise ValueError("num_workers 不能为负数")

    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    # -------------------------------------------------------------------------
    # 7.2 确定本地/镜像目录；恢复时先把持久化产物复制回高速本地磁盘
    # -------------------------------------------------------------------------
    resume = Path(resume_path).expanduser().resolve() if resume_path else None
    mirror_path = Path(mirror_dir).expanduser().resolve() if mirror_dir else None
    if resume and run_dir is None:
        run_path = resume.parent.parent
    else:
        run_path = Path(run_dir).expanduser().resolve() if run_dir else _default_run_dir(seed)
    if resume and mirror_path and run_path != mirror_path and mirror_path.exists() and not run_path.exists():
        shutil.copytree(mirror_path, run_path)
        if resume.is_relative_to(mirror_path):
            mirrored_resume = run_path / resume.relative_to(mirror_path)
            if mirrored_resume.exists():
                resume = mirrored_resume
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "checkpoints").mkdir(exist_ok=True)

    # Hugging Face 缓存放进项目数据目录，避免默认缓存位置难以追踪。
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "data" / ".cache" / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "data" / ".cache" / "hf_datasets"))
    _seed_everything(seed)

    # -------------------------------------------------------------------------
    # 7.3 构建训练/验证数据：固定 episode 划分，共享训练集归一化统计量
    # -------------------------------------------------------------------------
    data_config = config["data"]
    dataset_root = _resolve_project_path(data_config["dataset_root"])
    split_path = _resolve_project_path(data_config["split_path"])
    normalization_path = _resolve_project_path(data_config["normalization_path"])
    stats = NormalizationStats.from_audit_file(normalization_path)
    train_dataset = MiniWAMDataset(
        dataset_root,
        load_episode_split(split_path, "train"),
        stats,
        action_horizon=int(data_config["action_horizon"]),
        future_horizon=int(data_config["future_horizon"]),
        include_future_observations=False,
    )
    validation_dataset = MiniWAMDataset(
        dataset_root,
        load_episode_split(split_path, "validation"),
        stats,
        action_horizon=int(data_config["action_horizon"]),
        future_horizon=int(data_config["future_horizon"]),
        include_future_observations=False,
    )
    sampler = DeterministicBatchSampler(len(train_dataset), int(training["batch_size"]), seed)
    # DataLoader 使用独立 generator 生成 worker seed，避免创建/重建 worker 时推进
    # 模型训练所使用的全局 PyTorch RNG 状态。
    train_worker_generator = torch.Generator().manual_seed(seed + 100_000)
    validation_worker_generator = torch.Generator().manual_seed(seed + 200_000)
    train_loader_options = {
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        generator=train_worker_generator,
        **train_loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        generator=validation_worker_generator,
        num_workers=0,
    )

    # -------------------------------------------------------------------------
    # 7.4 构建 model、optimizer 和 scheduler，并记录实验身份
    # -------------------------------------------------------------------------
    model = ActionOnlyPolicy(pretrained=bool(config["model"]["pretrained"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = _make_scheduler(optimizer, int(training["warmup_steps"]), train_steps)
    environment = _environment_summary(device)
    environment["data_loader"] = {
        "train_num_workers": num_workers,
        "validation_num_workers": 0,
        "prefetch_factor": 2 if num_workers > 0 else None,
        "persistent_workers": num_workers > 0,
    }
    split_hash = _sha256_file(split_path)
    dataset_fingerprint = validate_dataset(dataset_root, "lerobot/pusht_image").fingerprint
    best_validation_loss = math.inf
    last_validation_loss: float | None = None
    step = 0

    # -------------------------------------------------------------------------
    # 7.5 可选恢复：拒绝不兼容 checkpoint，再恢复所有会影响下一步的状态
    # -------------------------------------------------------------------------
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=True)
        if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
            raise ValueError("checkpoint 版本不兼容")
        if checkpoint.get("config") != config:
            raise ValueError("恢复失败：当前配置与 checkpoint 配置不同")
        if checkpoint.get("split_hash") != split_hash:
            raise ValueError("恢复失败：数据划分已经变化")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        sampler.load_state_dict(checkpoint["sampler"])
        step = int(checkpoint["step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        last_validation_loss = checkpoint.get("validation_loss")
        _restore_random_states(checkpoint["random_states"])
        if step >= target_step:
            raise ValueError(f"checkpoint 已在第 {step} 步，不早于目标第 {target_step} 步")

    # -------------------------------------------------------------------------
    # 7.6 写出不可缺少的复现元数据，并在训练前做第一次镜像
    # -------------------------------------------------------------------------
    shutil.copy2(config_path, run_path / "config.yaml") if not (run_path / "config.yaml").exists() else None
    (run_path / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_path / "normalization.json").write_text(
        json.dumps(_normalization_dict(stats), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _log(
        run_path,
        f"device={device} num_workers={num_workers} step={step} "
        f"target_step={target_step} run_dir={run_path}",
    )
    if mirror_path:
        _sync_run_directory(run_path, mirror_path)

    checkpoint_frequency = int(config["checkpoint"]["frequency"])
    archive_frequency = int(config["checkpoint"].get("archive_frequency", checkpoint_frequency))
    validation_frequency = int(config["validation"]["frequency"])
    max_validation_batches = config["validation"].get("max_batches")
    iterator = iter(train_loader)

    # -------------------------------------------------------------------------
    # 7.7 算法核心：forward → masked loss → backward → 参数更新
    # -------------------------------------------------------------------------
    while step < target_step:
        batch = next(iterator)
        model.train()
        # set_to_none=True 减少不必要的梯度清零写操作；下一次 backward 会重新创建梯度。
        optimizer.zero_grad(set_to_none=True)

        # Tensor 数据流：
        # observation_history [B, 2, 3, 96, 96]
        # agent_position      [B, 2, 2]
        # prediction/target   [B, 16, 2]
        # action_valid_mask   [B, 16]，False 对应 episode 尾部的 padding
        mask = batch["action_valid_mask"].to(device)
        prediction = model(
            batch["observation_history"].to(device),
            batch["agent_position"].to(device),
        )
        loss = masked_smooth_l1_loss(prediction, batch["action_chunk"].to(device), mask)
        # NaN（Not a Number，非数值）或无穷大会让后续训练失去意义，立即失败更安全。
        if not torch.isfinite(loss):
            raise FloatingPointError(f"第 {step + 1} 步出现非有限 loss")
        loss.backward()
        # 在 optimizer 更新前裁剪整体梯度范数，降低异常 batch 引起梯度爆炸的风险。
        gradient_norm = float(clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"])))
        optimizer.step()
        # 本调度器按训练 step 更新，因此每次参数更新后推进一次。
        scheduler.step()
        # 只有完成参数更新的 batch 才能进入 checkpoint 的“已消费”位置；worker
        # 提前请求但尚未训练的 batch 不得影响恢复点。
        sampler.mark_consumed()
        step += 1

        # 每一步记录 loss、实际学习率和裁剪前梯度范数，供训练曲线与故障分析使用。
        _append_csv(
            run_path / "train_metrics.csv",
            ["step", "loss", "learning_rate", "gradient_norm"],
            {
                "step": step,
                "loss": float(loss.item()),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": gradient_norm,
            },
        )

        # ---------------------------------------------------------------------
        # 7.8 定期验证，并只在验证损失刷新时覆盖 best.pt
        # ---------------------------------------------------------------------
        validation_loss: float | None = None
        if step % validation_frequency == 0 or step == target_step:
            validation_loss = _evaluate(model, validation_loader, device, max_validation_batches)
            last_validation_loss = validation_loss
            _append_csv(
                run_path / "val_metrics.csv",
                ["step", "validation_loss", "batches"],
                {
                    "step": step,
                    "validation_loss": validation_loss,
                    "batches": max_validation_batches if max_validation_batches is not None else "all",
                },
            )
            _log(run_path, f"step={step} train_loss={loss.item():.6f} val_loss={validation_loss:.6f}")
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                payload = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sampler=sampler,
                    step=step,
                    config=config,
                    best_validation_loss=best_validation_loss,
                    validation_loss=last_validation_loss,
                    stats=stats,
                    split_hash=split_hash,
                    dataset_fingerprint=dataset_fingerprint,
                    environment=environment,
                )
                _atomic_torch_save(payload, run_path / "checkpoints" / "best.pt")
            if mirror_path:
                _sync_run_directory(run_path, mirror_path)

        # ---------------------------------------------------------------------
        # 7.9 定期保存可恢复的 last.pt；更稀疏地保留编号归档
        # ---------------------------------------------------------------------
        if step % checkpoint_frequency == 0 or step == target_step:
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                step=step,
                config=config,
                best_validation_loss=best_validation_loss,
                validation_loss=last_validation_loss,
                stats=stats,
                split_hash=split_hash,
                dataset_fingerprint=dataset_fingerprint,
                environment=environment,
            )
            _atomic_torch_save(payload, run_path / "checkpoints" / "last.pt")
            if step % archive_frequency == 0 or step == train_steps:
                _atomic_torch_save(payload, run_path / "checkpoints" / f"step_{step:07d}.pt")
            if mirror_path:
                _sync_run_directory(run_path, mirror_path)

    # 训练正常到达目标 step 后再写完成日志，并执行最后一次镜像。
    _log(run_path, f"completed target step {target_step}")
    if mirror_path:
        _sync_run_directory(run_path, mirror_path)
    return run_path
