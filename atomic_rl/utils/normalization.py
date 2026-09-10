"""
Normalization Utilities for Reinforcement Learning.

Pure functional routines for running statistics (Welford algorithm) and tensor standardization.
"""

import torch
from typing import Optional, Tuple, Union


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


def min_max_normalize(
    tensor: torch.Tensor,
    dim: Union[int, Tuple[int, ...]] = -1,
    min_val: float = 0.0,
    max_val: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scales tensor values along specified dimension(s) `dim` to [min_val, max_val].

    Used in MuZero (Appendix G) to bound latent activations to [0, 1] across recurrent steps.

    Args:
        tensor: Input tensor of shape [*Batch, *Features].
        dim: Dimension or tuple of dimensions along which to compute min/max (default: -1).
        min_val: Target minimum value (default 0.0).
        max_val: Target maximum value (default 1.0).
        eps: Epsilon to prevent division by zero when min == max.

    Returns:
        Tensor of identical shape [*Batch, *Features] scaled to [min_val, max_val].
    """
    assert max_val > min_val, f"max_val ({max_val}) must be strictly greater than min_val ({min_val})"
    assert eps > 0, f"eps must be strictly positive, got {eps}"

    t_min = tensor.amin(dim=dim, keepdim=True)
    t_max = tensor.amax(dim=dim, keepdim=True)

    norm = (tensor - t_min) / torch.clamp(t_max - t_min, min=eps)
    if min_val == 0.0 and max_val == 1.0:
        return norm
    return min_val + norm * (max_val - min_val)
