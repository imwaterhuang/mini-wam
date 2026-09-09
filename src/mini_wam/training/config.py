"""Action-only 训练配置与项目路径解析。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


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


def load_training_config(
    path: str | Path,
    *,
    expected_model_name: str = "action_only",
) -> dict[str, Any]:
    """读取并严格校验 Mini-WAM 训练配置。"""
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("训练配置顶层必须是字典")

    model_name = _require(raw, "model.name", str)
    if model_name != expected_model_name:
        raise ValueError(
            f"当前训练器要求 model.name={expected_model_name}，实际为 {model_name}"
        )
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
    if expected_model_name == "future_aware":
        if _require(raw, "training.lambda_future", float) <= 0:
            raise ValueError("training.lambda_future 必须大于 0")
    warmup = _require(raw, "training.warmup_steps", int)
    if warmup < 0 or warmup >= raw["training"]["train_steps"]:
        raise ValueError("warmup_steps 必须在 [0, train_steps) 内")
    max_batches = raw["validation"].get("max_batches")
    if max_batches is not None and (not isinstance(max_batches, int) or max_batches < 1):
        raise ValueError("validation.max_batches 必须为 null 或正整数")
    return raw


def resolve_project_path(value: str) -> Path:
    """把相对路径解释为相对于项目根目录的绝对路径。"""
    path = Path(value).expanduser()
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
