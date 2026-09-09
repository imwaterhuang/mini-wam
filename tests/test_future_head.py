from __future__ import annotations

import pytest
import torch

from mini_wam.models.future_head import FutureHeadPolicy, masked_cosine_loss


def test_masked_cosine_loss_ignores_invalid_future_steps() -> None:
    target = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]
    )
    prediction = torch.tensor(
        [[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]]
    )
    valid_mask = torch.tensor([[True, False, True, False]])

    loss = masked_cosine_loss(prediction, target, valid_mask)

    assert torch.allclose(loss, torch.tensor(0.0))


def test_masked_cosine_loss_rejects_mismatched_prediction_and_target() -> None:
    prediction = torch.randn(2, 4, 512)
    target = torch.randn(2, 3, 512)
    valid_mask = torch.ones(2, 4, dtype=torch.bool)

    with pytest.raises(ValueError, match="same shape"):
        masked_cosine_loss(prediction, target, valid_mask)


def test_masked_cosine_loss_requires_three_dimensions() -> None:
    prediction = torch.randn(4, 512)
    target = torch.randn(4, 512)
    valid_mask = torch.ones(4, dtype=torch.bool)

    with pytest.raises(ValueError, match=r"\[B, T, feature_dim\]"):
        masked_cosine_loss(prediction, target, valid_mask)


def test_masked_cosine_loss_rejects_wrong_mask_shape() -> None:
    prediction = torch.randn(2, 4, 512)
    target = torch.randn_like(prediction)
    valid_mask = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="valid_mask must have shape"):
        masked_cosine_loss(prediction, target, valid_mask)


def test_masked_cosine_loss_requires_boolean_mask() -> None:
    prediction = torch.randn(2, 4, 512)
    target = torch.randn_like(prediction)
    valid_mask = torch.ones(2, 4)

    with pytest.raises(TypeError, match="dtype bool"):
        masked_cosine_loss(prediction, target, valid_mask)


def test_masked_cosine_loss_rejects_empty_mask() -> None:
    prediction = torch.randn(2, 4, 512)
    target = torch.randn_like(prediction)
    valid_mask = torch.zeros(2, 4, dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one valid future step"):
        masked_cosine_loss(prediction, target, valid_mask)


def test_policy_training_and_inference_contracts() -> None:
    model = FutureHeadPolicy(pretrained=False, target_pretrained=False)
    observation_history = torch.randn(1, 2, 3, 96, 96)
    position_history = torch.randn(1, 2, 2)

    model.eval()
    with torch.inference_mode():
        action_prediction = model(observation_history, position_history)
    assert action_prediction.shape == (1, 16, 2)

    model.train()
    outputs = model(
        observation_history,
        position_history,
        torch.randn(1, 4, 2),
        torch.randn(1, 4, 3, 96, 96),
        compute_future=True,
    )
    assert [output.shape for output in outputs] == [
        (1, 16, 2),
        (1, 4, 512),
        (1, 4, 512),
    ]
    assert not model.frozen_target_encoder.training
    assert not model.frozen_target_encoder.feature_extractor.training


def test_policy_requires_future_inputs_when_requested() -> None:
    model = FutureHeadPolicy(pretrained=False, target_pretrained=False)
    with pytest.raises(ValueError, match="必须提供"):
        model(
            torch.randn(1, 2, 3, 96, 96),
            torch.randn(1, 2, 2),
            compute_future=True,
        )
