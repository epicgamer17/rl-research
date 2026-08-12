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
        base_loss_fn (Callable): Function to calculate loss. Must return a tuple of
            (raw_losses, info_dict).
        is_weights (torch.Tensor): Importance sampling weights.

    Returns:
        Callable: The loss function with PER weights applied.
    """

    def per_loss_fn(
        predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Calculate the weighted loss for a batch of transitions.

        Args:
            predictions (torch.Tensor): Predicted Q-values.
            targets (torch.Tensor): Target Q-values.

        Returns:
            torch.Tensor: The weighted loss for the batch.
            info_dict (dict): A dictionary containing the weighted loss and priorities.
        """
        # 1. Compute raw loss and priorities
        raw_losses, info_dict = base_loss_fn(predictions, targets)

        # 2. Compute weighted loss for the optimizer
        weighted_loss = (raw_losses * is_weights).mean()

        # Update info with weighted loss and priorities
        info_dict["loss/weighted"] = weighted_loss.detach()

        return weighted_loss, info_dict

    return per_loss_fn


def with_sequence_mask(base_loss_fn: Callable, mask: torch.Tensor) -> Callable:
    """
    Higher-order function that masks out invalid sequence transitions before calculating the mean loss.

    Args:
        base_loss_fn (Callable): Function to calculate loss. Must return a tuple of (raw_losses, info_dict).
        mask (torch.Tensor): Mask indicating valid transitions. In other words masking out the loss for observations or predictions past terminal or truncated states or that are for another episode (an artifact of our how our buffer handles sequences)

    Returns:
        Callable: The loss function with sequence mask applied.
    """

    def masked_loss_fn(
        predictions: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        # 1. Fail Fast: Check input volume matches mask volume
        assert predictions.numel() == mask.numel(), (
            f"Shape mismatch: predictions total elements ({predictions.numel()}) "
            f"must match mask total elements ({mask.numel()})."
        )
        assert targets.numel() == mask.numel(), (
            f"Shape mismatch: targets total elements ({targets.numel()}) "
            f"must match mask total elements ({mask.numel()})."
        )
        # predictions and targets are [B * T]
        raw_losses, info_dict = base_loss_fn(predictions, targets)

        flat_mask = mask.view(-1).float()

        # 3. Fail Fast: Ensure base_loss_fn output matches the flattened mask shape
        assert raw_losses.shape == flat_mask.shape, (
            f"Loss geometry mismatch: base_loss_fn output shape {raw_losses.shape} "
            f"does not match flattened mask shape {flat_mask.shape}. Ensure inputs are flattened"
        )
        masked_losses = raw_losses * flat_mask

        # Mean over only valid transitions to avoid diluting the gradient
        valid_count = flat_mask.sum().clamp(min=1.0)
        mean_loss = masked_losses.sum() / valid_count

        info_dict["loss/masked_mean"] = mean_loss.detach()
        return masked_losses, info_dict

    return masked_loss_fn


# TODO: does this belong here?
# TODO: is this good? should this take in torch.Distribution objects instead?
def probability_ratio(
    old_log_probs: torch.Tensor,
    new_log_probs: torch.Tensor,
):
    """
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
