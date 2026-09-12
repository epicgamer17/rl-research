"""
Loss Weighting, Importance Sampling, and Policy Likelihood Ratios.

This module consolidates all sample-wise, step-wise, and trajectory-level loss weighting
and importance sampling mechanisms across on-policy, off-policy, and offline RL.

Functions to be moved here include:
1. compute_is_weights (Prioritized Experience Replay)
2. compute_vtrace_weights (V-trace clipped importance weights)
3. compute_ope_weights (Off-Policy Evaluation cumulative trajectory weights)
4. probability_ratio (PPO policy likelihood ratio)
"""

from typing import Tuple, Optional, Union
import torch

# ==============================================================================
# SAMPLE-WISE IMPORTANCE SAMPLING (To be moved from buffers/replay/write.py)
# ==============================================================================


# TODO: rename to compute_per_weights?
def compute_is_weights(
    leaf_priorities: torch.Tensor,
    min_prob: Union[float, torch.Tensor],
    total_priority: Union[float, torch.Tensor],
    beta: torch.Tensor,
) -> torch.Tensor:
    r"""
    Computes Importance Sampling (IS) weights for Prioritized Experience Replay (PER).

    IS weights correct for the bias introduced by non-uniform sampling in PER.
    Mathematically: $w_i = ( (1/N) * (1/P_i) )^\beta / \max_j w_j$, where $P_i$ is the
    sampling probability of transition $i$. This simplifies to:
    $w_i = ( P_i / \min_j P_j )^{-\beta}$.

    Args:
        leaf_priorities (torch.Tensor): The priority values of the sampled transitions.
        min_prob (Union[float, torch.Tensor]): The minimum sampling probability to normalize by.
            Passing the batch minimum probability (`leaf_priorities.min() / total_priority`)
            computes batch normalization (scaling batch max weight to 1.0).
            Passing the global buffer minimum probability (`min_tree[0] / total_priority`)
            computes global buffer normalization.
        total_priority (Union[float, torch.Tensor]): The sum of all priorities in the tree.
        beta (torch.Tensor): The correction coefficient (usually scheduled from 0.4 to 1.0).

    Returns:
        torch.Tensor: The normalized IS weights for the batch.
    """
    probs = leaf_priorities / total_priority
    # IS weight: (P_i / min_P)^-beta
    # This simplifies the (1/N * 1/P_i)^beta / max_w normalization
    is_weights = torch.pow(probs / min_prob, -beta)
    return is_weights


# ==============================================================================
# STEP-WISE IMPORTANCE WEIGHTS & RATIOS
# ==============================================================================


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


def compute_vtrace_weights(
    target_log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    rho_clip: Optional[float] = 1.0,
    c_clip: Optional[float] = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes clipped V-trace importance sampling weights (rho_bar, c_bar).

    Reference:
        Espeholt et al., "IMPALA: Scalable Distributed Deep-RL with
        Importance Weighted Actor-Learner Architectures" (2018), Section 4.1.

    Formula:
        rho_t = exp(target_log_probs - behavior_log_probs)
        rho_bar_t = min(rho_clip, rho_t)   # Truncated IS weight for TD error / PG advantage
        c_bar_t   = min(c_clip, rho_t)     # Truncated IS weight for trace continuation product

    Args:
        target_log_probs: Log probabilities under current target policy [*Batch, Time, *Channels].
        behavior_log_probs: Log probabilities under rollout behavior policy [*Batch, Time, *Channels].
        rho_clip: Upper clip bound for rho_bar. Defaults to 1.0 (pass None for unclipped).
        c_clip: Upper clip bound for c_bar. Defaults to 1.0 (pass None for unclipped).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (rho_bar, c_bar) matching input shape.
    """
    assert (
        target_log_probs.shape == behavior_log_probs.shape
    ), f"Exact shape mismatch in compute_vtrace_weights! target: {target_log_probs.shape} vs behavior: {behavior_log_probs.shape}"

    # NOTE: Skeleton stub pending full IMPALA integration.
    # Potential internal reuse: atomic_rl.losses.probability_ratio(behavior_log_probs, target_log_probs)
    # can be reused internally to compute the importance sampling ratio rho_t.
    raise NotImplementedError(
        "compute_vtrace_weights kernel implementation is pending."
    )


# ==============================================================================
# TRAJECTORY-LEVEL IMPORTANCE SAMPLING (To be moved from evaluation/ope.py)
# ==============================================================================


def compute_ope_weights(
    target_log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    time_dim: int = 1,
    max_weight: Optional[float] = None,
) -> torch.Tensor:
    """
    Computes cumulative trajectory importance sampling weights for Off-Policy Evaluation (OPE).

    Reference:
        Precup et al., "Eligibility Traces for Off-Policy Policy Evaluation" (2000).

    Formula:
        rho_t = exp(target_log_probs - behavior_log_probs)
        w_t = prod_{k=0}^t rho_k = exp(cumsum(target_log_probs - behavior_log_probs))

    Args:
        target_log_probs: Log probabilities under the evaluation target policy [*Batch, Time, *Channels].
        behavior_log_probs: Log probabilities under the behavior rollout policy [*Batch, Time, *Channels].
        time_dim: Time step dimension along which to accumulate product (default: 1).
        max_weight: Optional maximum weight threshold for truncated importance sampling.

    Returns:
        torch.Tensor: Cumulative importance weights w_t matching input tensor shape.
    """
    assert (
        target_log_probs.shape == behavior_log_probs.shape
    ), f"Exact shape mismatch in compute_ope_weights! target: {target_log_probs.shape} vs behavior: {behavior_log_probs.shape}"

    # NOTE: Skeleton stub pending implementation.
    # Potential internal reuse: atomic_rl.losses.probability_ratio(behavior_log_probs, target_log_probs)
    # can be reused internally to compute the per-step importance ratio rho_t before accumulating products.
    raise NotImplementedError("compute_ope_weights kernel implementation is pending.")
