from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch import nn

from mini_wam.data.dataset import IMAGENET_MEAN, IMAGENET_STD, NormalizationStats
from mini_wam.studio.checkpoints import (
    CheckpointValidationError,
    checkpoint_normalization,
    inspect_checkpoint,
    scan_checkpoints,
)
from mini_wam.studio.datasets import (
    BUILTIN_DATASET_ROOT,
    DatasetValidationError,
    validate_dataset,
)
from mini_wam.studio.pusht import (
    SceneState,
    canvas_to_environment,
    denormalize_actions,
    prepare_model_input,
    random_scene,
    run_rollout,
    validate_scene,
)
from mini_wam.studio.overfit import OverfitEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_dataset(
    root: Path,
    image_shape: list[int] | None = None,
    state_shape: list[int] | None = None,
    action_shape: list[int] | None = None,
    task: str = "Push the T-shaped block onto the target.",
) -> Path:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "fps": 10,
        "features": {
            "observation.image": {"shape": image_shape or [96, 96, 3]},
            "observation.state": {"shape": state_shape or [2]},
            "action": {"shape": action_shape or [2]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    pq.write_table(
        pa.table({"task_index": [0], "task": [task]}),
        root / "meta" / "tasks.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1],
                "dataset_from_index": [0, 2],
                "dataset_to_index": [2, 4],
            }
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "observation.state": [[10.0, 20.0], [11.0, 21.0], [30.0, 40.0], [31.0, 41.0]],
                "action": [[12.0, 22.0], [13.0, 23.0], [32.0, 42.0], [33.0, 43.0]],
                "episode_index": [0, 0, 1, 1],
                "frame_index": [0, 1, 0, 1],
                "index": [0, 1, 2, 3],
            }
        ),
        root / "data" / "chunk-000" / "file-000.parquet",
    )
    return root


def _stats() -> NormalizationStats:
    return NormalizationStats(
        position_mean=torch.tensor([10.0, 20.0]),
        position_std=torch.tensor([2.0, 4.0]),
        action_mean=torch.tensor([100.0, 200.0]),
        action_std=torch.tensor([10.0, 20.0]),
    )


def test_builtin_dataset_contract() -> None:
    if not BUILTIN_DATASET_ROOT.exists():
        pytest.skip("Local Push-T dataset is not downloaded")
    summary = validate_dataset(BUILTIN_DATASET_ROOT, "lerobot/pusht_image")
    assert summary.episode_count == 206
    assert summary.frame_count == 25_650
    assert summary.image_shape == (96, 96, 3)
    assert summary.state_dim == summary.action_dim == 2
    assert summary.position_mean == pytest.approx((228.9086, 292.79214), abs=1e-4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"image_shape": [64, 64, 3]}, "96×96×3"),
        ({"state_shape": [3]}, "state 必须为 2 维"),
        ({"action_shape": [3]}, "action 必须为 2 维"),
        ({"task": "Pick up the cube."}, "只支持 Push-T"),
    ],
)
def test_dataset_rejects_incompatible_contract(tmp_path: Path, kwargs: dict, message: str) -> None:
    root = _make_dataset(tmp_path / "data", **kwargs)
    with pytest.raises(DatasetValidationError, match=message):
        validate_dataset(root)


def test_dataset_rejects_non_contiguous_episode_frames(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path / "data")
    path = root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path).set_column(3, "frame_index", pa.array([0, 2, 0, 1]))
    pq.write_table(table, path)
    with pytest.raises(DatasetValidationError, match="frame_index 不连续"):
        validate_dataset(root)


def test_existing_checkpoints_are_compatible() -> None:
    expected = {"action_only_overfit.pt", "action_only_full.pt"}
    if not all((PROJECT_ROOT / "artifacts" / name).exists() for name in expected):
        pytest.skip("Current checkpoints are not available")
    models, errors = scan_checkpoints()
    assert not errors
    assert expected <= {item.name for item in models}
    assert next(item for item in models if item.name == "action_only_full.pt").status == "闭环未验证"


def test_legacy_checkpoint_rejects_another_dataset_fingerprint() -> None:
    path = PROJECT_ROOT / "artifacts" / "action_only_overfit.pt"
    if not path.exists():
        pytest.skip("Legacy checkpoint is not available")
    with pytest.raises(CheckpointValidationError, match="只能搭配项目内置训练数据"):
        checkpoint_normalization(path, _stats(), "external", "builtin")


def test_checkpoint_rejects_missing_and_wrong_parameters(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    torch.save({"model": {"wrong.weight": torch.zeros(3, 4)}}, bad)
    with pytest.raises(CheckpointValidationError, match="参数名或形状不兼容"):
        inspect_checkpoint(bad)


def test_checkpoint_rejects_pickled_python_model(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.pt"
    torch.save(nn.Linear(2, 2), unsafe)
    with pytest.raises(CheckpointValidationError, match="安全读取"):
        inspect_checkpoint(unsafe)


def test_seed_and_canvas_mapping_are_deterministic() -> None:
    assert random_scene(42) == random_scene(42)
    assert canvas_to_environment(0, 0) == (0.0, 0.0)
    assert canvas_to_environment(384, 384) == (512.0, 512.0)
    assert canvas_to_environment(192, 96) == (256.0, 128.0)


def test_invalid_scene_is_rejected() -> None:
    with pytest.raises(ValueError, match="重叠"):
        validate_scene(SceneState(200, 200, 200, 200, 0))
    with pytest.raises(ValueError, match="智能体中心"):
        validate_scene(SceneState(0, 200, 250, 250, 0))
    with pytest.raises(ValueError, match="非法初始接触"):
        validate_scene(SceneState(250, 250, 320, 250, 0))


def test_model_input_and_action_transform_contract() -> None:
    observations = [
        {"pixels": np.zeros((96, 96, 3), dtype=np.uint8), "agent_pos": np.array([10, 20])},
        {"pixels": np.full((96, 96, 3), 255, dtype=np.uint8), "agent_pos": np.array([12, 24])},
    ]
    images, positions = prepare_model_input(observations, _stats(), torch.device("cpu"))
    assert images.shape == (1, 2, 3, 96, 96)
    assert positions.shape == (1, 2, 2)
    assert torch.allclose(images[0, 0, :, 0, 0], (torch.zeros(3) - IMAGENET_MEAN[:, 0, 0]) / IMAGENET_STD[:, 0, 0])
    assert torch.allclose(positions[0], torch.tensor([[0.0, 0.0], [1.0, 1.0]]))

    prediction = torch.zeros((1, 16, 2))
    prediction[0, 0] = torch.tensor([-100.0, 100.0])
    actions = denormalize_actions(prediction, _stats())
    assert actions.shape == (16, 2)
    assert actions[0].tolist() == [0.0, 512.0]


class _ConstantPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, images: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert images.shape == (1, 2, 3, 96, 96)
        assert positions.shape == (1, 2, 2)
        return torch.zeros((1, 16, 2), device=images.device) + self.anchor


def test_rollout_executes_four_steps_per_model_call(tmp_path: Path) -> None:
    updates = list(
        run_rollout(
            _ConstantPolicy(),
            _stats(),
            random_scene(7),
            tmp_path / "rollout.mp4",
            max_steps=8,
        )
    )
    final = updates[-1]
    assert final.metrics.steps == 8
    assert final.metrics.model_calls == 2
    assert final.video_path is not None and final.video_path.exists()


def test_overfit_checkpoint_recreates_its_exact_training_windows() -> None:
    if not (PROJECT_ROOT / "artifacts" / "action_only_overfit.pt").exists():
        pytest.skip("Overfit checkpoint is not available")
    evaluator = OverfitEvaluator(PROJECT_ROOT)
    sample = evaluator.inspect(1)
    assert sample.dataset_index == 12_604
    assert sample.episode_id == 117
    assert sample.episode_step == 51
    assert sample.history_image.shape == (384, 768, 3)
    assert sample.action_overlay.shape == (384, 384, 3)

    recovered = evaluator.recover_rollout_start(1)
    assert len(recovered.initial_history) == 2
    assert recovered.block_mask_iou > 0.85
    assert recovered.initial_history[0]["pixels"].shape == (96, 96, 3)

    aggregate = evaluator.evaluate_all()
    assert aggregate.sample_count == 64
    assert aggregate.recomputed_loss == pytest.approx(aggregate.stored_final_loss, abs=1e-7)
    assert aggregate.action_error_mean == pytest.approx(13.61, abs=0.02)
