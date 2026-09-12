"""
Metrics and Visualizations for Value Function Explained Variance.

Measures how well the critic / value network predicts the empirical returns:
    Explained Variance = 1 - Var(y_true - y_pred) / Var(y_true)
"""

import os
from typing import Optional
import numpy as np
import torch
import matplotlib.pyplot as plt


# ==============================================================================
# COMPUTATION PRIMITIVES
# ==============================================================================


def compute_explained_variance(
    y_true: torch.Tensor, y_pred: torch.Tensor
) -> torch.Tensor:
    """
    Computes the explained variance of a value prediction / regression problem.
    Formula: 1 - Var(y_true - y_pred) / Var(y_true)

    Single Contract (Torch-Native First):
    Expects y_true and y_pred to be torch.Tensors of identical shape [*Batch, *Channels].

    Args:
        y_true: Ground truth target returns [*Batch, *Channels].
        y_pred: Predicted state values V(s) [*Batch, *Channels].

    Returns:
        torch.Tensor: Scalar 0D tensor containing the explained variance.
                      1.0 represents perfect prediction, 0.0 or negative means poor prediction, NaN for zero variance.
    """
    assert isinstance(y_true, torch.Tensor) and isinstance(
        y_pred, torch.Tensor
    ), f"Expected PyTorch Tensors for y_true and y_pred, got {type(y_true)} and {type(y_pred)}"
    assert (
        y_true.shape == y_pred.shape
    ), f"Exact shape mismatch in compute_explained_variance! y_true: {y_true.shape}, y_pred: {y_pred.shape}"

    var_y = torch.var(y_true)
    if var_y == 0:
        return y_true.new_tensor(float("nan"))

    var_diff = torch.var(y_true - y_pred)
    return 1.0 - (var_diff / var_y)


# ==============================================================================
# PAPER-QUALITY VISUALIZATION (Matplotlib)
# ==============================================================================


def plot_value_prediction_scatter(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    max_samples: int = 1000,
    title: str = "Value Function: Predictions vs. Empirical Returns",
    xlabel: str = "Empirical Return Targets ($G_t$)",
    ylabel: str = "Predicted State Values ($V(s)$)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generates a publication-quality scatter plot comparing predicted state values V(s)
    against empirical return targets G_t with a 45-degree reference line (y = x).

    Args:
        y_true: Ground truth return targets [*Batch].
        y_pred: Predicted state values [*Batch].
        max_samples: Subsampling cap to avoid plotting millions of points.
        title: Chart title.
        xlabel: Horizontal axis label.
        ylabel: Vertical axis label.
        save_path: Optional path to save the figure.

    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    with torch.no_grad():
        ev = compute_explained_variance(y_true, y_pred).item()
        y_t = y_true.flatten().cpu().numpy()
        y_p = y_pred.flatten().cpu().numpy()

    if len(y_t) > max_samples:
        indices = np.random.choice(len(y_t), max_samples, replace=False)
        y_t, y_p = y_t[indices], y_p[indices]

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(y_t, y_p, alpha=0.35, edgecolors="none", color="black", s=18, label="Transitions")

    # Reference diagonal y = x
    min_val = min(y_t.min(), y_p.min())
    max_val = max(y_t.max(), y_p.max())
    padding = (max_val - min_val) * 0.05
    line_min, line_max = min_val - padding, max_val + padding

    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        color="crimson",
        linestyle="--",
        linewidth=1.8,
        label="Ideal ($y = x$)",
    )

    ax.text(
        0.05,
        0.92,
        f"Explained Variance: {ev:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlim(line_min, line_max)
    ax.set_ylim(line_min, line_max)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", frameon=True, facecolor="white")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig


# ==============================================================================
# EXPERIMENT TRACKING (Weights & Biases)
# ==============================================================================


def create_wandb_explained_variance(step: int, explained_variance: float) -> dict:
    """
    Creates a W&B metric dictionary for tracking value function explained variance.

    Args:
        step: Current training step or optimization epoch.
        explained_variance: Computed explained variance scalar.

    Returns:
        dict: Payload suitable for wandb.log(...).
    """
    return {
        "step": step,
        "metrics/explained_variance": float(explained_variance),
    }
