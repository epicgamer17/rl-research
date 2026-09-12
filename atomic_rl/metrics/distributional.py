"""
Metrics and Visualizations for Distributional Reinforcement Learning.

Provides evaluation metrics and distribution plotting for categorical (C51) and
quantile (QR-DQN, IQN) value representations.
"""

import os
from typing import Optional, Dict, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb


# ==============================================================================
# COMPUTATION PRIMITIVES
# ==============================================================================


from ..transforms.value_transform import compute_expected_value


# ==============================================================================
# PAPER-QUALITY VISUALIZATION (Matplotlib)
# ==============================================================================


def plot_categorical_distribution(
    support: np.ndarray,
    probs: np.ndarray,
    expected_q: Optional[float] = None,
    title: str = "Value Distribution",
    xlabel: str = "Return Support (z)",
    ylabel: str = "Probability Mass",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots a publication-quality categorical probability mass distribution over support atoms.

    Args:
        support: 1D array of atom support values [Atoms].
        probs: 1D array of probabilities matching support [Atoms].
        expected_q: Optional scalar expected Q-value to display as a vertical dashed line.
        title: Chart title.
        xlabel: Label for horizontal axis.
        ylabel: Label for vertical axis.
        save_path: Optional file path to save the generated plot.

    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    bar_width = (support[-1] - support[0]) / (len(support) * 1.2)
    ax.bar(
        support,
        probs,
        width=bar_width,
        color="dodgerblue",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
        label="Probability Mass",
    )

    if expected_q is not None:
        ax.axvline(
            expected_q,
            color="crimson",
            linestyle="--",
            linewidth=2.0,
            label=rf"$\mathbb{{E}}[Q] = {expected_q:.2f}$",
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(frameon=True, facecolor="white", edgecolor="none")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig


# ==============================================================================
# EXPERIMENT TRACKING (Weights & Biases)
# ==============================================================================


# TODO: should this be create_wandb_distributional_metrics?
@torch.no_grad()
def log_distributional_metrics(
    info_dict: dict, support: torch.Tensor, step: int, log_chart: bool = False
) -> dict:
    """
    Generate readable line charts and log expected Q-values for distributional RL.

    Args:
        info_dict (dict): The info dictionary from the loss function.
        support (torch.Tensor): The atom support tensor.
        step (int): The current training step.
        log_chart (bool): Whether to generate the heavy W&B line chart.

    Returns:
        dict: A dictionary of W&B plots and metrics.
    """
    metrics = {}
    if "predictions" in info_dict:
        # 1. Calculate probabilities [Batch, Atoms]
        probs = torch.softmax(info_dict["predictions"], dim=-1)

        # 2. Calculate Expected Q-value via reusable computation primitive
        expected_q_batch = compute_expected_value(probs, support, probs=True)
        metrics["metrics/expected_q_value"] = expected_q_batch.mean().detach()

        if log_chart:
            # 3. Average probabilities over the batch for the distribution curve
            mean_probs = probs.mean(dim=0).detach().cpu().numpy()
            support_np = support.cpu().numpy()

            # 4. Create a Line Plot for readability
            data = [[s, p] for s, p in zip(support_np, mean_probs)]
            table = wandb.Table(data=data, columns=["Support", "Probability"])
            metrics["charts/distribution_curve"] = wandb.plot.line(
                table,
                "Support",
                "Probability",
                title=f"Atom Distribution (Step {step})",
            )

    return metrics
