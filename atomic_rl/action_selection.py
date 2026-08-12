import math
from typing import Tuple, Callable, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


# Reference: https://github.com/mohmdelsayed/streaming-drl/blob/main/src/action_selection.py
#   The authors' action selection (argmax / epsilon-greedy / distribution sampling) is
#   implemented there — consult it when aligning behavior with the released code.


def compute_expected_value(
    predictions: torch.Tensor, support: torch.Tensor
) -> torch.Tensor:
    """
    Calculate expected values from a categorical distribution over a support.

    Works for both Q-value prediction [*Batch, A, N] and state-value prediction [*Batch, N].

    Args:
        predictions: Logits output from the model [*Batch, A, N] or [*Batch, N].
        support: The support for the distribution [N], [*Batch, N], or [*Batch, A, N].

    Returns:
        The expected values of shape [*Batch, A] or [*Batch].
    """
    assert (
        predictions.shape[-1] == support.shape[-1]
    ), f"Atoms dimension mismatch in compute_expected_value: predictions {predictions.shape} vs support {support.shape}"

    probs = F.softmax(predictions, dim=-1)

    # Broadcast support safely to match probs
    if support.ndim == 1:
        support = support.expand_as(probs)
    else:
        assert (
            support.shape == probs.shape
        ), f"If support is not 1D, it must match predictions shape exactly. Got {support.shape} vs {probs.shape}"

    # Dot product over the atoms dimension
    return (probs * support).sum(dim=-1)


def argmax_selector(
    predictions: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """
    Selects the action with the maximum value along dim=-1 (with random tie-breaking).
    Supports any [*Batch, Num_Actions] prediction shape.

    Args:
        predictions: The model predictions (e.g. Q-values or logits) [*Batch, Num_Actions].

    Returns:
        A tuple containing:
            - The selected actions of shape [*Batch, 1].
            - An empty info dictionary.
    """
    assert (
        predictions.ndim >= 1
    ), f"Expected at least 1D predictions, got {predictions.shape}"

    # Add microscopic noise to break ties randomly
    noise = torch.rand_like(predictions) * 1e-8
    vals_with_noise = predictions + noise

    # Force [*Batch, 1] output for consistency
    action = torch.argmax(vals_with_noise, dim=-1, keepdim=True)
    return action, {}


# TODO: is multidiscrete sampling handled and is it handled well? should we make a helper function for making the multi discrete distribution?
def sample_distribution(
    dist: torch.distributions.Distribution, explore: bool = True
) -> Tuple[torch.Tensor, dict]:
    """
    Samples an action from a generic PyTorch distribution.

    Rule Enforcement (Explicit over Implicit):
    The caller is responsible for constructing the appropriate distribution
    (e.g., Categorical, Normal) and wrapping it in `Independent` if joint
    probabilities are required.

    Args:
        dist: The PyTorch distribution object.
        explore: Whether to sample from the distribution or take the mode/mean.

    Returns:
        A tuple containing:
            - The selected actions.
            - An info dictionary containing "log_prob".
    """
    if not explore:
        # Note: This is one of the rare cases where isinstance/hasattr is unavoidable.
        # PyTorch's torch.distributions does not enforce a uniform interface for the 'mode'
        # or greedy action across continuous/discrete distributions.
        if isinstance(dist, torch.distributions.Categorical):
            action = torch.argmax(dist.probs, dim=-1)
        elif hasattr(dist, "mean"):
            action = dist.mean
        else:
            raise NotImplementedError(
                f"Deterministic selection for {type(dist)} is not implemented."
            )
    else:
        action = dist.sample()

    log_prob = dist.log_prob(action)

    if action.ndim == 1:
        action = action.unsqueeze(-1)
    if log_prob.ndim == 1:
        log_prob = log_prob.unsqueeze(-1)

    return action, {"log_prob": log_prob}


def with_epsilon_greedy(selector_fn: Callable) -> Callable:
    """
    Higher-order function that augments a selector with epsilon-greedy logic.
    Supports optional action masking during the random exploration phase.

    Args:
        selector_fn (Callable): The function to use for action selection.
            Must return (actions [B, 1], info_dict).
    """

    def epsilon_greedy_selector(
        predictions: torch.Tensor,
        epsilon: float,
        num_actions: int,
        generator: torch.Generator = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        # Expected [B, 1], dict. The selector_fn handles masked predictions naturally
        greedy_actions, info = selector_fn(predictions)

        if epsilon <= 0.0:
            return greedy_actions, {**info, "generator": generator}

        batch_size = predictions.shape[0]

        if mask is not None:
            # Fail Fast: Ensure mask matches [Batch, Actions]
            assert mask.shape == (
                batch_size,
                num_actions,
            ), f"Mask shape {mask.shape} does not match expected ({batch_size}, {num_actions})"

            # Fail Fast: Ensure no environment has zero valid actions
            assert (
                mask.sum(dim=-1) > 0
            ).all(), "Encountered a mask where an environment has 0 valid actions."

            # Convert bool mask to probability weights (1.0 for valid, 0.0 for invalid)
            valid_weights = mask.to(predictions.dtype)

            # Uniformly sample 1 valid action per batch element
            random_actions = torch.multinomial(valid_weights, 1, generator=generator)
        else:
            # Standard uniform sampling over all actions
            random_actions = torch.randint(
                0,
                num_actions,
                (batch_size, 1),
                generator=generator,
                device=predictions.device,
            )

        # Decide which actions are random vs greedy
        random_mask = (
            torch.rand((batch_size, 1), generator=generator, device=predictions.device)
            < epsilon
        )

        final_actions = torch.where(random_mask, random_actions, greedy_actions)
        return final_actions, {**info, "generator": generator}

    return epsilon_greedy_selector


def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Explicitly replaces logits of invalid actions with a large negative number (-1e8).
    This forces the probability of these actions to approach 0 after softmax.

    Supports single discrete, MultiDiscrete, sequence [B, T, A], and multi-agent logits [B, N, A].
    """
    assert (
        logits.shape == mask.shape
    ), f"Exact shape mismatch in apply_action_mask! logits: {logits.shape}, mask: {mask.shape}"
    assert (
        mask.sum(dim=-1).gt(0).all()
    ), "Encountered a mask where an environment/state has 0 valid actions."

    mask_bool = mask.to(torch.bool)
    return torch.where(
        mask_bool,
        logits,
        logits.new_tensor(-1e8),
    )


def compute_masked_entropy(
    logits: torch.Tensor, probs: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Computes categorical entropy manually to prevent NaNs.
    PyTorch's Categorical.entropy() can yield NaNs if logits are extremely negative.

    Math: entropy = -sum(p * log(p))
    """
    mask_bool = mask.to(torch.bool)
    p_log_p = logits * probs
    # Force masked out p_log_p elements to 0.0 before summing
    p_log_p = torch.where(
        mask_bool,
        p_log_p,
        logits.new_zeros(()),
    )
    return -p_log_p.sum(dim=-1)


def gather_q_values(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    """
    Gathers Q-values for selected action indices along the specified action dimension (`dim`).

    Rule Enforcement (Explicit over Implicit):
    Expects Q-values to have an action dimension at index `dim`.
    Supports single-agent [B, A], distributional [B, A, Atoms], multi-player [B, A, N_players],
    and sequence Q-values [B, T, A].

    Args:
        q_values: Q-values tensor [*Batch, Num_Actions, *Channels].
        actions: Action indices tensor [*Batch] or [*Batch, 1].
        dim: Dimension index of the action space in `q_values` (default: 1).

    Returns:
        The Q-values for the selected actions [*Batch, *Channels] with the action dimension removed.
    """
    assert q_values.ndim >= 2, f"Expected at least 2D q_values, got {q_values.shape}"

    # Normalize negative dim to positive rank index
    if dim < 0:
        dim = q_values.ndim + dim

    # Standardize actions to have a 1-sized dimension at `dim` if passed as [*Batch]
    # TODO: instead of this check, should we just enforce this behaviour with an assert?
    if actions.ndim == q_values.ndim - 1:
        actions = actions.unsqueeze(dim)

    # Expand actions across any trailing channel dimensions (e.g. Atoms, N_players)
    actions_expanded = actions
    while actions_expanded.ndim < q_values.ndim:
        actions_expanded = actions_expanded.unsqueeze(-1)

    target_shape = list(q_values.shape)
    target_shape[dim] = 1
    actions_expanded = actions_expanded.expand(target_shape).long()

    return q_values.gather(dim, actions_expanded).squeeze(dim)


def add_dirichlet_noise(
    logits: torch.Tensor,
    epsilon: float,
    alpha: float,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Adds Dirichlet noise to the given logits for exploration.
    Formula: P(s, a) = (1 - epsilon) * P(s, a) + epsilon * noise
    where noise ~ Dirichlet(alpha).

    Args:
        logits: The logits to add noise to [..., Num_Actions].
        epsilon: The weight of the noise.
        alpha: The concentration parameter of the Dirichlet distribution.
        mask: Optional boolean mask [..., Num_Actions] of valid actions.

    Returns:
        The logits with added noise.
    """

    # 0. Convert logits to probs
    masked_logits = logits.clone()
    if mask is not None:
        masked_logits = masked_logits.masked_fill(mask == 0, -float("inf"))
    probs = torch.softmax(masked_logits, dim=-1)

    # 1. Sample noise from Dirichlet distribution
    num_actions = probs.shape[-1]
    alphas = torch.full((num_actions,), alpha, device=probs.device, dtype=probs.dtype)

    # torch.distributions.Dirichlet handles batch shapes correctly via .sample()
    dist = torch.distributions.Dirichlet(alphas)
    noise = dist.sample(probs.shape[:-1])  # [..., Num_Actions]

    if mask is not None:
        # Mask out noise for illegal actions
        noise = noise * mask.to(probs.dtype)

        # Re-normalize the noise so it still sums to 1.0 across valid actions
        # Add 1e-8 to prevent division by zero in case of severe precision issues
        noise = noise / torch.clamp(noise.sum(dim=-1, keepdim=True), min=1e-8)

    noisy_probs = (1.0 - epsilon) * probs + epsilon * noise
    noisy_logits = torch.log(torch.clamp(noisy_probs, min=1e-8))

    # Preserve the mask in log-space so masked/illegal actions get exactly 0.0
    # probability after softmax (log(1e-8) would leak a tiny nonzero prior).
    if mask is not None:
        noisy_logits = noisy_logits.masked_fill(mask == 0, -float("inf"))

    return noisy_logits


def reduce_ensemble_q_values(
    q_ensemble: torch.Tensor,
    reduction: str = "min",
    ensemble_dim: int = 0,
) -> torch.Tensor:
    """
    Reduces an ensemble of Q-function predictions (e.g. Double DQN, SAC, REDQ, TD3).

    Args:
        q_ensemble: Tensor of Q-values containing an ensemble dimension (e.g. [N_ensemble, B, A]).
        reduction: Reduction method: "min" (pessimistic Q-learning), "mean", or "std".
        ensemble_dim: Index of the ensemble dimension (default: 0).

    Returns:
        torch.Tensor: Reduced Q-values tensor with the ensemble dimension removed.
    """
    assert q_ensemble.ndim >= 2, f"Expected at least 2D q_ensemble, got {q_ensemble.shape}"
    if reduction == "min":
        return torch.min(q_ensemble, dim=ensemble_dim).values
    elif reduction == "mean":
        return torch.mean(q_ensemble, dim=ensemble_dim)
    elif reduction == "std":
        return torch.std(q_ensemble, dim=ensemble_dim)
    else:
        raise ValueError(f"Unknown reduction '{reduction}'. Expected 'min', 'mean', or 'std'.")
