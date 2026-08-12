"""
Policy Loss Primitives.

Pure mathematical loss functions for policy optimization (Policy Gradient, PPO Clipped Surrogate, Entropy).
"""

import torch
import torch.distributions as D
from typing import Tuple


def compute_policy_gradient_loss(
    advantages: torch.Tensor,
    log_probs: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """
    Calculate the policy gradient loss for a batch of transitions.

    Args:
        advantages (torch.Tensor): Tensor of advantages.
        log_probs (torch.Tensor): Tensor of log probabilities of actions.

    Returns:
        Tuple[torch.Tensor, dict]: A tuple containing the raw losses and the info dictionary.

    Expects flat tensors [B * T] or [B].
    """
    pass


def compute_entropy_loss(
    dist: D.Distribution,
) -> Tuple[torch.Tensor, dict]:
    """
    Calculate the entropy loss for an action distribution.
    Returns the NEGATIVE mean entropy, so that minimizing this "loss" maximizes entropy.

    Args:
        dist: The PyTorch distribution object.

    Returns:
        Tuple[torch.Tensor, dict]: The negative mean entropy and logging info.
    """
    pass


def compute_clipped_surrogate_loss(
    ratio: torch.Tensor, advantages: torch.Tensor, clip_coef: float
) -> Tuple[torch.Tensor, dict]:
    """
    Computes the PPO clipped surrogate loss.

    Args:
        ratio: The probability ratio r_t(θ) [B] or [B * T]
        advantages: The estimated advantages [B] or [B * T]
        clip_coef: The clipping coefficient (epsilon)

    Returns:
        Tuple containing the loss tensor and an info dictionary for logging.
    """
    pass
