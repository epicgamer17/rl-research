"""Value and Target Transformation Primitives.

This module provides pure mathematical transformations between continuous scalars,
contractive-scaled representations, and discrete two-hot categorical supports (MuZero, R2D2, DreamerV3).
"""

import torch


# TODO: is there a better name for this?
def compute_contractive_transform(
    x: torch.Tensor,
    epsilon: float = 0.001,
) -> torch.Tensor:
    """Computes the contractive invertible scaling transformation h(x) (R2D2 / Pohlen et al., 2018 / MuZero Appendix F).

    Formula (R2D2 Section 2 & Table 2):
        h(x) = sign(x) * (sqrt(|x| + 1) - 1) + eps * x

    Note:
        In MuZero Appendix F, the formula has a typesetting error where the parenthesis
        mistakenly wraps the linear term (+ eps * x). R2D2 correctly places the linear
        term outside the square root parenthesis, preserving strict odd-symmetry h(-x) == -h(x)
        and ensuring exact analytical invertibility for all negative values.

    This non-linear scaling compresses large dynamic range reward and value targets,
    stabilizing neural network training in unbounded reward environments (like Atari).

    Args:
        x: Continuous scalar tensor of arbitrary shape [*Batch, *Channels]. Must be a floating-point tensor.
        epsilon: Small linear regularization term (default: 0.001).

    Returns:
        Contractive-scaled tensor of identical shape [*Batch, *Channels] and dtype.
    """
    assert isinstance(
        x, torch.Tensor
    ), f"Expected x to be a torch.Tensor, got {type(x)}"
    assert torch.is_floating_point(
        x
    ), f"x must be a floating-point tensor, got dtype {x.dtype}"
    assert (
        isinstance(epsilon, (float, int)) and epsilon > 0.0
    ), f"epsilon must be positive number, got {epsilon}"

    abs_x = torch.abs(x)
    eps = float(epsilon)
    return torch.sign(x) * (torch.sqrt(abs_x + 1.0) - 1.0) + eps * x


# TODO: is there a better name for this?
def compute_inverse_contractive_transform(
    y: torch.Tensor,
    epsilon: float = 0.001,
) -> torch.Tensor:
    """Computes the exact analytical inverse of the contractive scaling transformation h^{-1}(y) (R2D2 / MuZero Appendix F).

    Formula (solving eps*t^2 + t - (|y| + 1 + eps) = 0 for t = sqrt(|x| + 1)):
        t = (sqrt(1 + 4*eps*(|y| + 1 + eps)) - 1) / (2*eps)
        h^{-1}(y) = sign(y) * (t^2 - 1)

    Recovers unscaled continuous values from scaled network predictions.

    Args:
        y: Scaled value tensor of arbitrary shape [*Batch, *Channels]. Must be a floating-point tensor.
        epsilon: Small linear regularization term matching forward transform (default: 0.001).

    Returns:
        Unscaled continuous value tensor of identical shape [*Batch, *Channels] and dtype.
    """
    assert isinstance(
        y, torch.Tensor
    ), f"Expected y to be a torch.Tensor, got {type(y)}"
    assert torch.is_floating_point(
        y
    ), f"y must be a floating-point tensor, got dtype {y.dtype}"
    assert (
        isinstance(epsilon, (float, int)) and epsilon > 0.0
    ), f"epsilon must be positive number, got {epsilon}"

    abs_y = torch.abs(y)
    eps = float(epsilon)
    discriminant = 1.0 + 4.0 * eps * (abs_y + 1.0 + eps)
    t = (torch.sqrt(discriminant) - 1.0) / (2.0 * eps)
    return torch.sign(y) * (torch.square(t) - 1.0)


def compute_two_hot_distribution(
    scalar: torch.Tensor,
    support_min: int = -300,
    support_max: int = 300,
) -> torch.Tensor:
    """Projects continuous scalar targets onto discrete adjacent supports (two-hot representation).

    Under this transformation, each scalar is represented as a linear combination of its
    two adjacent integer supports such that:
        scalar = x_low * p_low + x_high * p_high
        where p_low + p_high = 1.0

    For example, a scalar target of 3.7 with integer support yields:
        weight 0.3 on support 3, weight 0.7 on support 4.

    Args:
        scalar: Continuous scalar tensor of shape [*Batch, *Channels] (or [*Batch]).
        support_min: Minimum integer support bound (default: -300).
        support_max: Maximum integer support bound (default: 300).

    Returns:
        Categorical probability distribution of shape [*Batch, *Channels, num_atoms]
        where num_atoms = support_max - support_min + 1.
    """
    assert isinstance(
        scalar, torch.Tensor
    ), f"Expected scalar to be a torch.Tensor, got {type(scalar)}"
    assert torch.is_floating_point(
        scalar
    ), f"scalar must be a floating-point tensor, got dtype {scalar.dtype}"
    assert (
        isinstance(support_min, int)
        and isinstance(support_max, int)
        and support_max > support_min
    ), f"support_max ({support_max}) must be an int strictly greater than support_min ({support_min})"

    num_atoms = support_max - support_min + 1

    # Clamp continuous values to the valid support bounds
    clamped = torch.clamp(scalar, min=float(support_min), max=float(support_max))

    # Decompose into lower integer atom and linear interpolation weights
    floor = torch.floor(clamped)
    p_high = clamped - floor
    p_low = 1.0 - p_high

    # Map atom values to categorical indices [0, num_atoms - 1]
    floor_idx = (floor - support_min).long().unsqueeze(-1)
    ceil_idx = torch.clamp(floor_idx + 1, max=num_atoms - 1)

    # Accumulate two-hot probabilities
    distribution = scalar.new_zeros(*scalar.shape, num_atoms)
    distribution.scatter_add_(dim=-1, index=floor_idx, src=p_low.unsqueeze(-1))
    distribution.scatter_add_(dim=-1, index=ceil_idx, src=p_high.unsqueeze(-1))

    return distribution
