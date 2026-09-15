"""
Quantization and Discretization Transformations.

Pure functional primitives for discrete latent variables and straight-through gradient estimation
(e.g., Stochastic MuZero chance outcome discretization, VQ codebook lookups).
"""

import torch
import torch.nn.functional as F
from typing import Tuple


# TODO: Implement compute_straight_through_one_hot
# Reference:
# - Stochastic MuZero (Antonoglou et al., ICLR 2022, Section 3.2: Discrete Latent State)
# - Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation (Bengio et al., 2013)
#
# Mathematical definition:
#   Given continuous logits/features z in R^[..., M]:
#   1. Discrete index: k = argmax_i(z_i)
#   2. Hard one-hot: c = one_hot(k, num_classes=M)
#   3. Straight-Through Estimator (STE):
#      c_ST = z + stop_gradient(c - z)
#   In the forward pass, c_ST equals the hard discrete one-hot vector c.
#   In the backward pass, gradients flow directly to z: dL/dz = dL/dc_ST.
def compute_straight_through_one_hot(
    logits: torch.Tensor,
    dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Discretizes continuous representations into one-hot codes with straight-through gradient estimation.

    Args:
        logits: Continuous unquantized logits or features [*Batch, NumClasses].
        dim: Dimension along which to perform argmax and one-hot discretization (default: -1).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - one_hot_st: Discretized one-hot tensor [*Batch, NumClasses] with gradient attached.
            - indices: Integer class indices [*Batch] resulting from the argmax.
    """
    assert (
        logits.ndim >= 1
    ), f"logits must have at least 1 dimension, got shape {tuple(logits.shape)}"
    num_classes = logits.shape[dim]
    assert num_classes >= 1, f"num_classes must be >= 1, got {num_classes}"

    # 1. Compute discrete argmax indices:
    indices = logits.argmax(dim=dim)
    # 2. Form hard one-hot:
    hard_one_hot = F.one_hot(indices, num_classes=num_classes).to(dtype=logits.dtype)
    if dim != -1 and dim != (logits.ndim - 1):
        hard_one_hot = hard_one_hot.movedim(-1, dim)
    # 3. Apply straight-through trick:
    one_hot_st = logits + (hard_one_hot - logits).detach()
    # 4. Return one_hot_st, indices

    return one_hot_st, indices
