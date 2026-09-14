"""
Vector Quantization & Codebook Diagnostic Metrics.

Diagnostic tools for detecting codebook collapse, measuring discrete representation capacity,
and monitoring latent chance usage in stochastic model-based reinforcement learning.
"""

import torch
import torch.nn.functional as F


# TODO: Implement compute_codebook_entropy
# Reference:
# - Stochastic MuZero (Antonoglou et al., ICLR 2022, Appendix D: Codebook Dynamics)
# - Jukebox / VQ-VAE: Monitoring Codebook Utilization & Active Code Count
#
# Mathematical definition:
#   Given discrete codes c in {0, ..., M-1} across a batch of N samples:
#   1. Compute empirical frequencies: p_k = count(c == k) / N  for k in 0..M-1
#   2. Compute categorical Shannon entropy:
#      H(p) = - \sum_{k=0}^{M-1} p_k * log(p_k + eps)
#   3. Normalized perplexity / active codes:
#      Perplexity = exp(H(p))
#   If H(p) -> 0, the codebook has collapsed to a single code.
#   If H(p) -> log(M), all codes are used uniformly.
# TODO: semantically it may be nice, but is this the same as our entropy_loss?
def compute_codebook_entropy(
    chance_codes: torch.Tensor,
    codebook_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes empirical Shannon entropy of discrete codebook usage across a batch.

    Used to detect and monitor codebook collapse in Stochastic MuZero and VQ-VAEs.

    Args:
        chance_codes: Tensor of integer code indices [*Batch] or one-hot codes [*Batch, CodebookSize].
        codebook_size: Total number of discrete codes in the codebook (M).
        eps: Small epsilon for numerical stability in the logarithm.

    Returns:
        torch.Tensor: Scalar 0D tensor containing the Shannon entropy H in nats.
    """
    assert (
        isinstance(codebook_size, int) and codebook_size >= 1
    ), f"codebook_size must be >= 1, got {codebook_size}"
    assert chance_codes.numel() > 0, "chance_codes cannot be empty"

    # TODO:
    # 1. If chance_codes is integer indices [*Batch], convert to flat one-hot or use bincount:
    #    if chance_codes.shape[-1] == codebook_size and chance_codes.is_floating_point():
    #        probs = chance_codes.view(-1, codebook_size).mean(dim=0)
    #    else:
    #        flat_codes = chance_codes.reshape(-1)
    #        one_hot = F.one_hot(flat_codes, num_classes=codebook_size).float()
    #        probs = one_hot.mean(dim=0)
    # 2. Calculate Shannon entropy:
    #    entropy = -torch.sum(probs * torch.log(probs + eps))
    # 3. Return scalar 0D tensor entropy

    raise NotImplementedError("TODO: Implement compute_codebook_entropy")
