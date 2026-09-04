"""Safe checkpoint discovery and loading for ActionOnlyPolicy."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from mini_wam.data.dataset import NormalizationStats
from mini_wam.models.action_only import ActionOnlyPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_IMPORT_ROOT = PROJECT_ROOT / "artifacts" / "studio" / "models"
ARCHITECTURE_ID = "action_only_v1"


class CheckpointValidationError(ValueError):
    """Raised for unsafe or incompatible checkpoint files."""


@dataclass(frozen=True)
class CheckpointInfo:
    name: str
    path: Path
    status: str
    architecture: str
    step: int | None
    validation_loss: float | None
    dataset_fingerprint: str | None
    legacy: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path)
        return result


def _safe_payload(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".pt":
        raise CheckpointValidationError("只接受 .pt checkpoint（检查点）")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CheckpointValidationError(f"无法安全读取 checkpoint：{exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointValidationError("checkpoint 顶层必须是字典，拒绝序列化 Python 模型")
    return payload


def _state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    candidate = payload.get("state_dict", payload.get("model"))
    if not isinstance(candidate, dict) or not candidate:
        raise CheckpointValidationError("checkpoint 缺少 state_dict 或 model 参数")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in candidate.items()):
        raise CheckpointValidationError("模型参数必须是 tensor 字典")
    return candidate


def _validate_metadata(payload: dict[str, Any]) -> tuple[str, str | None, bool]:
    metadata = payload.get("metadata")
    if metadata is None:
        return ARCHITECTURE_ID, None, True
    if not isinstance(metadata, dict):
        raise CheckpointValidationError("metadata 必须是字典")
    expected = {
        "architecture": ARCHITECTURE_ID,
        "task": "pusht",
        "history_frames": 2,
        "image_shape": [3, 96, 96],
        "state_dim": 2,
        "action_horizon": 16,
        "action_dim": 2,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise CheckpointValidationError(
                f"checkpoint 元数据 {key!r} 必须为 {value!r}，实际为 {metadata.get(key)!r}"
            )
    action_range = metadata.get("action_range", [0.0, 512.0])
    if action_range != [0.0, 512.0]:
        raise CheckpointValidationError("动作范围必须为 [0, 512]")
    normalization = metadata.get("normalization")
    if not isinstance(normalization, dict):
        raise CheckpointValidationError("新格式 checkpoint 缺少 normalization")
    for key in ("position_mean", "position_std", "action_mean", "action_std"):
        value = normalization.get(key)
        if not isinstance(value, list) or len(value) != 2:
            raise CheckpointValidationError(f"normalization.{key} 必须含两个数值")
        try:
            numbers = [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise CheckpointValidationError(f"normalization.{key} 含非数值") from exc
        if key.endswith("_std") and any(item <= 0 for item in numbers):
            raise CheckpointValidationError(f"normalization.{key} 必须大于 0")
    return ARCHITECTURE_ID, metadata.get("dataset_fingerprint"), False


def inspect_checkpoint(path: str | Path) -> CheckpointInfo:
    path = Path(path).expanduser().resolve()
    payload = _safe_payload(path)
    state = _state_dict(payload)
    expected = ActionOnlyPolicy(pretrained=False).state_dict()
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    wrong_shapes = [
        f"{key}: {tuple(state[key].shape)} != {tuple(expected[key].shape)}"
        for key in expected.keys() & state.keys()
        if tuple(state[key].shape) != tuple(expected[key].shape)
    ]
    if missing or unexpected or wrong_shapes:
        details = "; ".join(
            part for part in [f"缺少 {missing}" if missing else "", f"多出 {unexpected}" if unexpected else "", ", ".join(wrong_shapes)] if part
        )
        raise CheckpointValidationError(f"参数名或形状不兼容：{details}")
    architecture, fingerprint, legacy = _validate_metadata(payload)
    status = "可执行"
    if path.name == "action_only_full.pt":
        status = "闭环未验证"
    elif "overfit" in path.stem.lower():
        status = "过拟合诊断模型"
    validation_loss = payload.get("validation_loss")
    return CheckpointInfo(
        name=path.name,
        path=path,
        status=status,
        architecture=architecture,
        step=int(payload["step"]) if payload.get("step") is not None else None,
        validation_loss=float(validation_loss) if validation_loss is not None else None,
        dataset_fingerprint=fingerprint,
        legacy=legacy,
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[ActionOnlyPolicy, CheckpointInfo]:
    info = inspect_checkpoint(path)
    payload = _safe_payload(info.path)
    model = ActionOnlyPolicy(pretrained=False)
    model.load_state_dict(_state_dict(payload), strict=True)
    model.to(device).eval()
    return model, info


def checkpoint_normalization(
    path: str | Path,
    fallback: NormalizationStats,
    active_dataset_fingerprint: str,
    legacy_dataset_fingerprint: str | None = None,
) -> NormalizationStats:
    """Use bundle statistics, or the active dataset for current legacy files."""
    payload = _safe_payload(Path(path).expanduser().resolve())
    _, fingerprint, legacy = _validate_metadata(payload)
    if legacy:
        if legacy_dataset_fingerprint and active_dataset_fingerprint != legacy_dataset_fingerprint:
            raise CheckpointValidationError(
                "旧格式 checkpoint 只能搭配项目内置训练数据；外部数据需要含数据集指纹与归一化统计的新格式 bundle"
            )
        return fallback
    if fingerprint and fingerprint != active_dataset_fingerprint:
        raise CheckpointValidationError("checkpoint 与当前数据集指纹不一致")
    values = payload["metadata"]["normalization"]
    return NormalizationStats(
        position_mean=torch.tensor(values["position_mean"], dtype=torch.float32),
        position_std=torch.tensor(values["position_std"], dtype=torch.float32),
        action_mean=torch.tensor(values["action_mean"], dtype=torch.float32),
        action_std=torch.tensor(values["action_std"], dtype=torch.float32),
        std_floor=float(values.get("std_floor", 1e-6)),
    )


def scan_checkpoints() -> tuple[list[CheckpointInfo], list[str]]:
    paths = sorted((PROJECT_ROOT / "artifacts").glob("*.pt"))
    paths += sorted(MODEL_IMPORT_ROOT.glob("*.pt")) if MODEL_IMPORT_ROOT.exists() else []
    valid: list[CheckpointInfo] = []
    errors: list[str] = []
    for path in paths:
        try:
            valid.append(inspect_checkpoint(path))
        except CheckpointValidationError as exc:
            errors.append(f"{path.name}: {exc}")
    return valid, errors


def import_checkpoint(source: str | Path) -> CheckpointInfo:
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise CheckpointValidationError("上传文件不存在")
    if source.stat().st_size > 1_000_000_000:
        raise CheckpointValidationError("checkpoint 超过 1 GB，已拒绝")
    inspect_checkpoint(source)
    MODEL_IMPORT_ROOT.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:10]
    destination = MODEL_IMPORT_ROOT / f"{source.stem}-{digest}.pt"
    if source != destination:
        shutil.copy2(source, destination)
    return inspect_checkpoint(destination)
