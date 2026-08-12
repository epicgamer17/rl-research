"""
Loss Wrappers and Probability Ratio Helpers.

Higher-order functions that modify or augment base loss functions (PER importance sampling, sequence masking).
"""

import torch
from typing import Callable, Tuple


def with_per_weights(base_loss_fn: Callable, is_weights: torch.Tensor) -> Callable:
    """
    Higher-order function that wraps a standard loss function to apply
    PER Importance Sampling weights and extract TD errors.

    Args:
        base_loss_fn (Callable): Function to calculate loss.
        is_weights (torch.Tensor): Importance sampling weights.

    Returns:
        Callable: The loss function with PER weights applied.
    """
    pass


def with_sequence_mask(base_loss_fn: Callable, mask: torch.Tensor) -> Callable:
    """
    Higher-order function that masks out invalid sequence transitions before calculating the mean loss.

    Args:
        base_loss_fn (Callable): Function to calculate loss.
        mask (torch.Tensor): Mask indicating valid transitions.

    Returns:
        Callable: The loss function with sequence mask applied.
    """
    pass


# TODO: does this belong here?
def probability_ratio(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the probability ratio between a new and an old log probability.

    Args:
        old_log_probs (torch.Tensor): Tensor of old log probabilities.
        new_log_probs (torch.Tensor): Tensor of new log probabilities.

    Returns:
        torch.Tensor: The probability ratio.
    """
    pass
