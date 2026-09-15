"""
Temporal Difference (TD) Error Primitives.

Computes the 1-step Temporal Difference (TD) error between predictions and targets.
"""

import torch


def compute_td_error(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates the 1-step Temporal Difference error: delta = targets - predictions.

    Args:
        predictions: Value or Q-value predictions [*Batch, *Channels] or [*Batch].
        targets: Target values [*Batch, *Channels] or [*Batch].

    Returns:
        torch.Tensor: The TD error tensor delta matching the shape of predictions and targets.
    """
    assert (
        predictions.shape == targets.shape
    ), f"Exact shape mismatch in compute_td_error! predictions: {predictions.shape}, targets: {targets.shape}"

    return targets - predictions
