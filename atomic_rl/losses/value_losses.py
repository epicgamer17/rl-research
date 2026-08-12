"""
Value Function Loss Primitives.

Pure mathematical loss functions for value estimation networks (e.g. Clipped MSE for PPO Value Updates).
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def compute_clipped_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    old_predictions: torch.Tensor,
    clip_coef: float,
) -> Tuple[torch.Tensor, dict]:
    """
    Computes a Clipped MSE loss, commonly used in PPO for value function updates.

    This loss clips the predicted value to be within a certain range of the old value,
    and then takes the maximum of the unclipped and clipped MSE losses. This provides
    a pessimistic estimate of the value, which can help stabilize training.

    Args:
        predictions: The current predictions (e.g. new state values).
        targets: The target values (e.g. returns).
        old_predictions: The predictions from the previous iteration.
        clip_coef: The clipping coefficient.

    Returns:
        Tuple[torch.Tensor, dict]: The clipped MSE loss (raw) and logging info.
    """
    assert (
        predictions.shape == old_predictions.shape == targets.shape
    ), f"Shape mismatch: pred {predictions.shape}, old {old_predictions.shape}, targets {targets.shape}"

    # Explicitly detach old_predictions to prevent gradient leak
    old_predictions = old_predictions.detach()

    # Unclipped MSE
    v_loss_unclipped = F.mse_loss(predictions, targets, reduction="none")

    # Clipped MSE: clip the predicted value to be within [old - clip, old + clip]
    v_clipped = old_predictions + torch.clamp(
        predictions - old_predictions, -clip_coef, clip_coef
    )
    v_loss_clipped = F.mse_loss(v_clipped, targets, reduction="none")

    # The value loss is the max of clipped and unclipped losses (pessimistic estimate)
    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
    loss = 0.5 * v_loss_max

    info = {
        "loss/value": loss.mean().detach(),
        "value/unclipped_loss": v_loss_unclipped.mean().detach(),
        "value/clipped_loss": v_loss_clipped.mean().detach(),
    }

    return loss, info
