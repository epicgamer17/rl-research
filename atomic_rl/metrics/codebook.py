"""
Vector Quantization & Codebook Diagnostic Metrics.

Diagnostic tools for detecting codebook collapse, measuring discrete representation capacity,
and monitoring latent chance usage in stochastic model-based reinforcement learning.
"""

import os
from typing import Optional, Dict, Any, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb


# ==============================================================================
# COMPUTATION PRIMITIVES
# ==============================================================================


def compute_codebook_usage(
    chance_codes: torch.Tensor,
    codebook_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes empirical probability distribution and active code count across a batch.

    Args:
        chance_codes: Tensor of integer code indices [*Batch] or one-hot codes [*Batch, CodebookSize].
        codebook_size: Total number of discrete codes in the codebook (M).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - probs: 1D FloatTensor of empirical probabilities of shape [codebook_size].
            - active_codes: Scalar LongTensor representing the number of codes used at least once.
    """
    assert (
        isinstance(codebook_size, int) and codebook_size >= 1
    ), f"codebook_size must be >= 1, got {codebook_size}"
    assert chance_codes.numel() > 0, "chance_codes cannot be empty"

    if chance_codes.shape[-1] == codebook_size and chance_codes.is_floating_point():
        probs = chance_codes.reshape(-1, codebook_size).mean(dim=0)
    else:
        flat_codes = chance_codes.reshape(-1).long()
        assert (flat_codes >= 0).all() and (flat_codes < codebook_size).all(), (
            f"chance_codes indices must be in range [0, {codebook_size - 1}]"
        )
        one_hot = F.one_hot(flat_codes, num_classes=codebook_size).to(dtype=torch.float32)
        probs = one_hot.mean(dim=0)

    active_codes = (probs > 0.0).sum()
    return probs, active_codes


def compute_codebook_entropy(
    chance_codes: torch.Tensor,
    codebook_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes empirical Shannon entropy of discrete codebook usage across a batch.

    Used to detect and monitor codebook collapse in Stochastic MuZero and VQ-VAEs.
    Formula:
        H(p) = - sum_{k=0}^{M-1} p_k * log(p_k + eps)

    Args:
        chance_codes: Tensor of integer code indices [*Batch] or one-hot codes [*Batch, CodebookSize].
        codebook_size: Total number of discrete codes in the codebook (M).
        eps: Small epsilon for numerical stability in the logarithm.

    Returns:
        torch.Tensor: Scalar 0D tensor containing the Shannon entropy H in nats.
    """
    assert eps > 0.0, f"eps must be positive, got {eps}"
    probs, _ = compute_codebook_usage(chance_codes, codebook_size)
    entropy = -torch.sum(probs * torch.log(probs + eps))
    return entropy


def compute_codebook_perplexity(
    chance_codes: torch.Tensor,
    codebook_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes empirical codebook perplexity (effective number of active codes).

    Formula:
        Perplexity = exp(H(p))
    Range:
        [1.0, M], where 1.0 indicates complete codebook collapse (single active code)
        and M indicates perfectly uniform discrete code utilization.

    Args:
        chance_codes: Tensor of integer code indices [*Batch] or one-hot codes [*Batch, CodebookSize].
        codebook_size: Total number of discrete codes in the codebook (M).
        eps: Small epsilon for numerical stability in the logarithm.

    Returns:
        torch.Tensor: Scalar 0D tensor containing the codebook perplexity.
    """
    entropy = compute_codebook_entropy(chance_codes, codebook_size, eps=eps)
    return torch.exp(entropy)


# ==============================================================================
# PAPER-QUALITY VISUALIZATION (Matplotlib)
# ==============================================================================


def plot_codebook_distribution(
    chance_codes_or_probs: Union[torch.Tensor, np.ndarray],
    codebook_size: int,
    title: str = "Discrete Codebook Usage Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots a publication-quality bar chart of discrete codebook utilization.

    Includes a horizontal dashed reference line for uniform distribution (1 / M),
    per-bar percentage annotations, and entropy / perplexity statistics.

    Args:
        chance_codes_or_probs: Either discrete codes [*Batch] or 1D probability mass [M].
        codebook_size: Total number of discrete codes (M).
        title: Chart title.
        save_path: Optional file path to save the generated figure.

    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    if isinstance(chance_codes_or_probs, np.ndarray):
        tensor_input = torch.from_numpy(chance_codes_or_probs)
    else:
        tensor_input = chance_codes_or_probs.detach().cpu()

    # Determine if input is already a 1D probability distribution of length M
    if tensor_input.ndim == 1 and tensor_input.shape[0] == codebook_size and torch.is_floating_point(tensor_input) and torch.isclose(tensor_input.sum(), torch.tensor(1.0), atol=1e-2):
        probs = tensor_input
        active_codes = (probs > 0.0).sum()
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
    else:
        probs, active_codes = compute_codebook_usage(tensor_input, codebook_size)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))

    perplexity = torch.exp(entropy).item()
    entropy_val = entropy.item()
    max_entropy = np.log(codebook_size)
    uniformity = (entropy_val / max_entropy) if max_entropy > 0 else 1.0

    probs_np = probs.cpu().numpy()
    code_indices = np.arange(codebook_size)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    bars = ax.bar(
        code_indices,
        probs_np,
        color="royalblue",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
        label="Observed Usage",
    )

    # Reference uniform line
    uniform_prob = 1.0 / codebook_size
    ax.axhline(
        uniform_prob,
        color="crimson",
        linestyle="--",
        linewidth=1.8,
        label=f"Uniform Ref ({uniform_prob:.3f})",
    )

    # Bar percentage labels
    for bar in bars:
        height = bar.get_height()
        if height > 0.005:
            ax.annotate(
                f"{height * 100:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="medium",
            )

    subtitle = (
        f"Active: {active_codes.item()}/{codebook_size} | "
        f"Perplexity: {perplexity:.2f}/{codebook_size} | "
        f"Entropy: {entropy_val:.3f} nats ({uniformity * 100:.1f}% max)"
    )

    ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Discrete Chance Code Index ($c$)", fontsize=10)
    ax.set_ylabel("Empirical Probability Mass", fontsize=10)
    ax.set_xticks(code_indices)
    ax.set_xticklabels([f"c_{i}" for i in range(codebook_size)])
    ax.set_ylim(0, max(probs_np.max() * 1.15, uniform_prob * 1.25, 0.1))
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig


# ==============================================================================
# EXPERIMENT TRACKING (Weights & Biases)
# ==============================================================================


@torch.no_grad()
def create_wandb_codebook_metrics(
    chance_codes: torch.Tensor,
    codebook_size: int,
    step: Optional[int] = None,
    log_histogram: bool = True,
    log_bar_chart: bool = True,
) -> Dict[str, Any]:
    """
    Generates structured Weights & Biases codebook diagnostics and visualizations.

    Metrics generated:
        - codebook/entropy: Shannon entropy in nats
        - codebook/perplexity: Effective active code count in [1.0, M]
        - codebook/active_codes: Integer count of distinct codes observed
        - codebook/uniformity: Normalized entropy H(p) / log(M) in [0.0, 1.0]
        - codebook/code_{k}_prob: Probability of code k for k in 0..M-1
        - codebook/histogram: wandb.Histogram of discrete code indices (optional)
        - codebook/distribution_bar: wandb.plot.bar chart of code usage (optional)

    Args:
        chance_codes: Tensor of integer code indices [*Batch] or one-hot codes [*Batch, CodebookSize].
        codebook_size: Number of discrete codes (M).
        step: Optional training step for chart titles.
        log_histogram: Whether to include a wandb.Histogram artifact.
        log_bar_chart: Whether to include a wandb.plot.bar chart artifact.

    Returns:
        Dict[str, Any]: Dictionary ready to be passed to wandb.log().
    """
    probs, active_codes = compute_codebook_usage(chance_codes, codebook_size)
    entropy = compute_codebook_entropy(chance_codes, codebook_size)
    perplexity = torch.exp(entropy)
    max_entropy = np.log(codebook_size)
    uniformity = (entropy.item() / max_entropy) if max_entropy > 0 else 1.0

    metrics: Dict[str, Any] = {
        "codebook/entropy": entropy.detach().cpu().item(),
        "codebook/perplexity": perplexity.detach().cpu().item(),
        "codebook/active_codes": int(active_codes.item()),
        "codebook/uniformity": float(uniformity),
    }

    # Per-code probability tracking
    probs_cpu = probs.detach().cpu().numpy()
    for k in range(codebook_size):
        metrics[f"codebook/code_{k}_prob"] = float(probs_cpu[k])

    # W&B Histogram of sampled discrete codes
    if log_histogram:
        if chance_codes.shape[-1] == codebook_size and chance_codes.is_floating_point():
            indices_np = chance_codes.reshape(-1, codebook_size).argmax(dim=-1).detach().cpu().numpy()
        else:
            indices_np = chance_codes.reshape(-1).long().detach().cpu().numpy()
        metrics["codebook/histogram"] = wandb.Histogram(indices_np)

    # W&B Bar Chart
    if log_bar_chart:
        data = [[f"c_{k}", float(probs_cpu[k])] for k in range(codebook_size)]
        table = wandb.Table(data=data, columns=["Chance Code", "Probability"])
        step_suffix = f" (Step {step})" if step is not None else ""
        metrics["codebook/distribution_bar"] = wandb.plot.bar(
            table,
            "Chance Code",
            "Probability",
            title=f"Codebook Distribution{step_suffix}",
        )

    return metrics
