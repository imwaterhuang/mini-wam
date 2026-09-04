from __future__ import annotations

import pytest
import torch

from mini_wam.models.action_only import ActionOnlyPolicy, masked_smooth_l1_loss


@pytest.mark.parametrize("batch_size", [1, 3])
def test_action_only_policy_output_shape(batch_size: int) -> None:
    policy = ActionOnlyPolicy(pretrained=False)
    observation_history = torch.randn(batch_size, 2, 3, 96, 96)
    agent_position = torch.randn(batch_size, 2, 2)

    predicted_actions = policy(observation_history, agent_position)

    assert predicted_actions.shape == (batch_size, 16, 2)
    assert predicted_actions.dtype == torch.float32


def test_masked_smooth_l1_ignores_padding_and_padding_gradient() -> None:
    prediction = torch.tensor(
        [[[1.0, -1.0], [100.0, -100.0]]],
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)
    valid_mask = torch.tensor([[True, False]])

    loss = masked_smooth_l1_loss(prediction, target, valid_mask)
    loss.backward()

    assert loss.item() == pytest.approx(0.5)
    assert torch.count_nonzero(prediction.grad[0, 0]).item() == 2
    assert torch.count_nonzero(prediction.grad[0, 1]).item() == 0


def test_masked_smooth_l1_rejects_an_empty_mask() -> None:
    prediction = torch.zeros((1, 2, 2))
    target = torch.zeros_like(prediction)
    valid_mask = torch.zeros((1, 2), dtype=torch.bool)

    with pytest.raises(ValueError, match="at least one valid action step"):
        masked_smooth_l1_loss(prediction, target, valid_mask)
