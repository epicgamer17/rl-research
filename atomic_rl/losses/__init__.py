"""
Loss Functions for Reinforcement Learning.

Pure functional implementations of common RL loss functions (Q-learning, Policy Gradient,
Entropy, PPO Clipped Surrogate, Clipped MSE).
"""

from .q_losses import (
    compute_mse_loss,
    compute_cross_entropy_loss,
    compute_huber_loss,
)
from .policy_losses import (
    compute_policy_gradient_loss,
    compute_entropy_loss,
    compute_clipped_surrogate_loss,
)
from .value_losses import (
    compute_clipped_mse_loss,
)
from .vq_losses import (
    compute_commitment_loss,
)
from .weighting import (
    compute_is_weights,
    compute_vtrace_weights,
    compute_ope_weights,
    probability_ratio,
)

__all__ = [
    "compute_mse_loss",
    "compute_cross_entropy_loss",
    "compute_huber_loss",
    "compute_policy_gradient_loss",
    "compute_entropy_loss",
    "compute_clipped_surrogate_loss",
    "compute_clipped_mse_loss",
    "compute_commitment_loss",
    "compute_is_weights",
    "compute_vtrace_weights",
    "compute_ope_weights",
    "probability_ratio",
]

"""
NOTE: Higher-order closure wrappers (with_per_weights, with_sequence_mask) were removed
in favor of direct, idiomatic tensor operations in the orchestration loop:
    - Sample weighting: loss = (raw_losses * is_weights).mean()
    - Sequence masking: loss = (raw_losses * mask).sum() / mask.sum().clamp(min=1.0)

It may be worth adding them back for semantic understanding, though that is unlikely.
"""

# NOTE: although a lot of the functions simply wrap existing torch functions, they are there for a consistent API to prevent the user from using a more complex loss function, and then trying to switch it to a simpler one and having to switch to torch api or conventions which would be confusing and annoying. This is a core principle, for ease of testing.

# TODO: add a weighting file or something, for vtrace weights or other elementwise loss weights (importance sampling weights). i think it would be nice to have those all in one file.
