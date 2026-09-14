"""
Vector Quantization (VQ) and Codebook Loss Primitives.

Pure mathematical loss functions for quantized latent representations and codebook commitment
(e.g., Stochastic MuZero chance encoder commitment, VQ-VAE codebook learning).
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any


# TODO: Implement compute_commitment_loss
# Reference:
# - Stochastic MuZero (Antonoglou et al., ICLR 2022, Eq. 2 & Section 3.2)
# - VQ-VAE (van den Oord et al., NeurIPS 2017)
#
# Contract per CONTRIBUTING.MD:
# 1. Pure function named `compute_commitment_loss`.
# 2. Strict shape assertion: continuous_embeddings and discrete_codes must match in shape [*Batch, D].
# 3. Prevent gradient leakage: detach discrete_codes (and stop gradients where appropriate).
# 4. Return tuple: (raw_unreduced_losses [*Batch], info_dict).
#
# Mathematical definition:
#   loss = beta * || continuous_embeddings - stop_gradient(discrete_codes) ||_2^2
#   If discrete_codes are one-hot, continuous_embeddings can either be the raw encoder logits
#   or continuous latent features.
def compute_commitment_loss(
    continuous_embeddings: torch.Tensor,
    discrete_codes: torch.Tensor,
    beta: float = 0.25,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Computes Vector Quantization commitment loss for representation learning.

    Encourages the continuous encoder outputs to commit to the discrete representation,
    preventing representation collapse or arbitrary drift.

    Args:
        continuous_embeddings: Continuous tensor from encoder [*Batch, FeatureDim].
        discrete_codes: Quantized one-hot or discrete code representation [*Batch, FeatureDim].
        beta: Scaling coefficient for the commitment loss (default: 0.25).

    Returns:
        Tuple[torch.Tensor, dict]:
            - raw_loss: Unreduced element-wise or feature-averaged loss [*Batch].
            - info: Dictionary containing logging metrics (e.g. 'loss/commitment').
    """
    assert (
        continuous_embeddings.shape == discrete_codes.shape
    ), f"Shape mismatch: continuous {continuous_embeddings.shape} vs discrete {discrete_codes.shape}"
    assert beta >= 0.0, f"beta must be non-negative, got {beta}"

    # TODO: Calculate squared L2 norm between continuous_embeddings and detached discrete_codes:
    # detached_codes = discrete_codes.detach()
    # raw_loss = beta * F.mse_loss(continuous_embeddings, detached_codes, reduction="none").sum(dim=-1)
    #
    # info = {
    #     "loss/commitment": raw_loss.mean().detach(),
    # }
    # return raw_loss, info

    raise NotImplementedError("TODO: Implement compute_commitment_loss")
