"""
Off-Policy Evaluation (OPE) Primitives.

Pure mathematical kernels for importance sampling and off-policy evaluation across trajectories.
"""

import torch
from typing import Optional


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
