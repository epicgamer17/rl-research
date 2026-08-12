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

    Args:
        predictions: The current predictions (e.g. new state values).
        targets: The target values (e.g. returns).
        old_predictions: The predictions from the previous iteration.
        clip_coef: The clipping coefficient.

    Returns:
        Tuple[torch.Tensor, dict]: The clipped MSE loss (raw) and logging info.
    """
    pass
