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
    assert (
        advantages.shape == log_probs.shape
    ), f"Shape mismatch: {advantages.shape} vs {log_probs.shape}"
    # PG Loss: -log_prob * advantage (Advantage is treated as constant)
    loss = -log_probs * advantages.detach()
    # NOTE: no priorities, PG is on-policy and so PER doesn't really apply.abs
    # TODO: what about A-PPO or A3C?
    info = {"loss/policy_gradient": loss.mean().detach()}

    return loss, info


def compute_entropy_loss(
    dist: D.Distribution,
) -> Tuple[torch.Tensor, dict]:
    """
    Calculate the entropy loss for an action distribution.
    Returns the NEGATIVE mean entropy, so that minimizing this "loss" maximizes entropy.

    This function expects the distribution's `event_shape` to be configured correctly.
    If you have a multi-dimensional action space and want joint entropy, you MUST wrap your base distribution in `torch.distributions.Independent` before passing it here.
    We strictly avoid guessing whether to sum over extra dimensions.

    Args:
        dist: The PyTorch distribution object.

    Returns:
        Tuple[torch.Tensor, dict]: The negative mean entropy and logging info.
    """
    # 1. dist.entropy() automatically handles the underlying math based on event_shape
    entropy = dist.entropy()

    # Fail Fast: If the user didn't use `Independent`, entropy will incorrectly retain
    # the action dimension as a batch dimension. We check against the expected batch dim.
    # Note: We assume scalar entropy per transition here.
    assert entropy.ndim == 1, (
        f"Expected 1D entropy [B], got {entropy.shape}. If your action space is "
        "multi-dimensional, ensure you wrap your distribution in `Independent`."
    )

    loss = -entropy.mean()
    return loss, {"loss/entropy": loss.detach()}


def compute_clipped_surrogate_loss(
    ratio: torch.Tensor, advantages: torch.Tensor, clip_coef: float
) -> Tuple[torch.Tensor, dict]:
    """
    Computes the PPO clipped surrogate loss.

    Mathematically, PPO defines L^CLIP as an objective to be maximized.
    Because PyTorch optimizers minimize by default, this function returns
    the NEGATED objective so it can be directly minimized. L^CLIP is based on L^CLI which is the same loss without the penalty for moving the ratio away from 1 (the min statement with the clip coefficient)

    Args:
        ratio: The probability ratio r_t(θ) [B] or [B * T]
        advantages: The estimated advantages [B] or [B * T]
        clip_coef: The clipping coefficient (epsilon)

    Returns:
        Tuple containing the loss tensor and an info dictionary for logging.
    """
    # FAIL FAST: Prevent catastrophic broadcasting
    assert (
        ratio.shape == advantages.shape
    ), f"Shape mismatch: ratio {ratio.shape} vs adv {advantages.shape}"

    # Calculate the objective
    # TODO: should this be its own function?
    unclipped_objective = ratio * advantages.detach()
    clipped_objective = (
        torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages.detach()
    )

    # The PPO Objective to maximize
    objective = torch.min(unclipped_objective, clipped_objective)

    # Negate to turn the objective into a loss for gradient descent
    loss = -objective

    # Metrics for debugging and evaluating the clip_coef
    with torch.no_grad():
        log_ratio = torch.log(ratio)
        # Standard PPO approximate KL divergence: ((ratio - 1.0) - log_ratio).mean()
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        # Fraction of the batch that was clipped
        clip_fraction = (torch.abs(ratio - 1.0) > clip_coef).float().mean()

    info = {
        "loss/policy": loss.mean().detach(),
        "policy/approx_kl": approx_kl.detach(),
        "policy/clip_fraction": clip_fraction.detach(),
        "objective/unclipped": unclipped_objective.mean().detach(),
        "objective/clipped": clipped_objective.mean().detach(),
    }

    return loss, info


def probability_ratio(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
) -> torch.Tensor:
    r"""
    Calculate the probability ratio between a new and an old log probability.
    Used in algorithms like PPO, TRPO, and V-trace.
    NOTE: The probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is mathematically an importance sampling weight.

    Args:
        old_log_probs (torch.Tensor): Tensor of old log probabilities.
        new_log_probs (torch.Tensor): Tensor of new log probabilities.

    Returns:
        torch.Tensor: The probability ratio.

    Expects flat tensors [B * T] or [B].
    """
    assert (
        old_log_probs.shape == new_log_probs.shape
    ), f"Shape mismatch: {old_log_probs.shape} vs {new_log_probs.shape}"
    return torch.exp(new_log_probs - old_log_probs.detach())

