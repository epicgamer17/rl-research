"""
Normalization Utilities for Reinforcement Learning.

Pure functional routines for running statistics (Welford algorithm) and tensor standardization.
"""

import torch
from typing import Tuple


# TODO: should this be inplace?
# TODO: there is a tension between Gym which uses numpy and training loops which use torch. If this is to be used in a wrapper for the environment, then it should probably be numpy, but if we want to use it in simple training loops, torch is nicer.
# TODO: Rename to sample mean var?
# Reference: https://github.com/mohmdelsayed/streaming-drl/blob/main/src/normalization_wrappers.py
#   The authors' `SampleMeanVar` / `SampleMeanStd` implement the same running statistics
#   used by the paper (Algorithm 5); consult them to verify this Welford update.
def compute_welford_stats(
    mean: torch.Tensor, sq_diff: torch.Tensor, count: torch.Tensor, batch: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pure mathematical update for running mean and variance using Welford's online algorithm.

    Args:
        mean: current running mean of the features, initially 0.0. [*F]
        sq_diff: current running sum of squared differences from the mean, initially 1.0. [*F]
        count: current number of samples seen. (1)
        batch: the new batch of features to update with. (B, *F)

    Returns:
        new_mean, new_sq_diff, new_var, new_count
    """

    # Fail Fast: Strict layout validation to catch dimension errors immediately
    assert batch.ndim > mean.ndim, (
        f"Shape Mismatch: Incoming batch must contain an explicit leading batch dimension. "
        f"Batch shape: {batch.shape}, Mean shape: {mean.shape}"
    )
    assert batch.shape[1:] == mean.shape == sq_diff.shape, (
        f"Geometry mismatch. Batch features: {batch.shape[1:]}, "
        f"Mean features: {mean.shape}, Sq_diff features: {sq_diff.shape}"
    )

    batch_size = batch.size(0)

    new_count = count + batch_size
    delta = batch - mean.unsqueeze(0)
    new_mean = mean + delta.sum(dim=0) / new_count

    delta2 = batch - new_mean.unsqueeze(0)
    new_sq_diff = sq_diff + (delta * delta2).sum(dim=0)
    new_var = new_sq_diff / (new_count - 1)
    if new_count < 2:
        new_var = torch.ones_like(new_var)

    return new_mean, new_sq_diff, new_var, new_count


def standardize_tensor(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Mean-centers and scales a tensor by its standard deviation.
    Used when the baseline does not perfectly center the current batch (e.g., PPO Critic).
    """
    if tensor.numel() <= 1:
        return torch.zeros_like(tensor)

    return (tensor - tensor.mean()) / (tensor.std() + eps)


def scale_tensor_by_std(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scales a tensor by its standard deviation WITHOUT mean-centering.
    Used when the data is already centered (e.g., EMA Advantages).
    """
    if tensor.numel() <= 1:
        return torch.zeros_like(tensor)

    return tensor / (tensor.std() + eps)
