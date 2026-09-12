"""
Training Curves, Statistical Aggregations, and Benchmark Performance Visualizations.

Provides paper-quality publication plots (with multi-seed confidence intervals) and
lightweight W&B logging payloads for training curves, loss curves, and evaluation checkpoints.
"""

import os
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import matplotlib.pyplot as plt


# ==============================================================================
# STATISTICAL BENCHMARK METRICS
# ==============================================================================


def compute_iqm(scores: Union[torch.Tensor, np.ndarray]) -> float:
    r"""
    Computes the Interquartile Mean (IQM) across benchmark seeds or tasks.

    The IQM is the 25% trimmed mean, discarding the bottom 25% and top 25% of runs,
    making it robust to outliers and heavy-tailed performance distributions.

    Reference:
        Agarwal et al., "Deep Reinforcement Learning at the Edge of the
        Statistical Precipice" (NeurIPS 2021).

    Args:
        scores: 1D array or tensor of final performance scores across runs/seeds.

    Returns:
        float: The scalar Interquartile Mean.
    """
    if isinstance(scores, torch.Tensor):
        arr = scores.detach().cpu().numpy().flatten()
    else:
        arr = np.asarray(scores).flatten()

    assert len(arr) > 0, "Cannot compute IQM on an empty score array."
    q25 = np.percentile(arr, 25)
    q75 = np.percentile(arr, 75)
    trimmed = arr[(arr >= q25) & (arr <= q75)]
    return float(np.mean(trimmed))


def compute_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes empirical mean and standard error confidence bounds across runs.

    Args:
        data: Array of shape [Seeds, Steps].
        confidence: Confidence level (default: 0.95, ~1.96 standard errors).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (mean, lower_bound, upper_bound) of shape [Steps].
    """
    mean = np.mean(data, axis=0)
    sem = np.std(data, axis=0) / np.sqrt(data.shape[0])
    z = 1.96 if confidence == 0.95 else 1.0
    lower = mean - z * sem
    upper = mean + z * sem
    return mean, lower, upper


# ==============================================================================
# PAPER-QUALITY VISUALIZATIONS (Matplotlib)
# ==============================================================================


def plot_episode_returns(
    returns_by_system: Dict[str, np.ndarray],
    steps: Optional[np.ndarray] = None,
    xlabel: str = "Environment Steps",
    ylabel: str = "Episode Return",
    title: str = "Performance Across Seeds",
    smooth_window: int = 1,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generates a publication-quality multi-algorithm learning curve with shaded confidence bands.

    Args:
        returns_by_system: Dict mapping system/algorithm names to 2D numpy arrays of shape [Seeds, Steps].
        steps: Optional 1D array of horizontal axis step values matching Steps.
        xlabel: Label for horizontal axis.
        ylabel: Label for vertical axis.
        title: Chart title.
        smooth_window: Moving average smoothing window size (1 = no smoothing).
        save_path: Optional file path to save the generated figure.

    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["black", "dodgerblue", "crimson", "forestgreen", "darkorange", "purple"]

    for idx, (name, runs) in enumerate(returns_by_system.items()):
        runs = np.asarray(runs)
        if runs.ndim == 1:
            runs = runs[np.newaxis, :]  # Treat single seed as [1, Steps]

        x_vals = steps if steps is not None else np.arange(runs.shape[1])
        mean, lower, upper = compute_confidence_interval(runs)

        if smooth_window > 1 and len(mean) >= smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            mean = np.convolve(mean, kernel, mode="valid")
            lower = np.convolve(lower, kernel, mode="valid")
            upper = np.convolve(upper, kernel, mode="valid")
            x_vals = x_vals[: len(mean)]

        color = colors[idx % len(colors)]
        ax.plot(x_vals, mean, label=name, color=color, linewidth=2.0)
        if runs.shape[0] > 1:
            ax.fill_between(x_vals, lower, upper, color=color, alpha=0.18)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(frameon=True, facecolor="white", loc="best")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_loss_curves(
    loss_history: Dict[str, np.ndarray],
    title: str = "Training Losses",
    xlabel: str = "Optimization Step",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generates multi-panel side-by-side subplots tracking individual loss components
    (e.g., Policy Gradient Loss, Value Function MSE, Entropy Bonus).

    Args:
        loss_history: Dict mapping loss names to 1D arrays of scalar loss values.
        title: Super-title for the entire figure.
        xlabel: Label for horizontal axis.
        save_path: Optional file path to save the generated figure.

    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    num_losses = len(loss_history)
    fig, axes = plt.subplots(1, num_losses, figsize=(5 * num_losses, 4), squeeze=False)
    axes = axes.flatten()

    for idx, (loss_name, values) in enumerate(loss_history.items()):
        ax = axes[idx]
        vals = np.asarray(values)
        ax.plot(vals, color="black", linewidth=1.5)
        ax.set_title(loss_name, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Loss Magnitude", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=":")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_eval_vs_train(
    train_steps: np.ndarray,
    train_returns: np.ndarray,
    eval_steps: np.ndarray,
    eval_returns: np.ndarray,
    title: str = "Training Exploration vs. Evaluation Performance",
    xlabel: str = "Environment Steps",
    ylabel: str = "Return",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Overlays exploratory training rollout returns with offline greedy evaluation returns.

    Args:
        train_steps: 1D array of environment steps during training.
        train_returns: 1D array of episode returns collected during exploration.
        eval_steps: 1D array of steps when evaluation checkpoints occurred.
        eval_returns: 1D array of deterministic evaluation checkpoint returns.
        title: Chart title.
        xlabel: Horizontal axis label.
        ylabel: Vertical axis label.
        save_path: Optional path to save figure.

    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        train_steps,
        train_returns,
        color="gray",
        alpha=0.35,
        linewidth=1.0,
        label="Training Rollout (Exploratory)",
    )
    ax.plot(
        eval_steps,
        eval_returns,
        color="dodgerblue",
        marker="o",
        markersize=5,
        linewidth=2.0,
        label="Evaluation Checkpoint (Greedy)",
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(frameon=True, facecolor="white", loc="best")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig


# ==============================================================================
# EXPERIMENT TRACKING (Weights & Biases)
# ==============================================================================


def create_wandb_episode_metrics(
    step: int,
    episode_return: float,
    episode_length: int,
    success_rate: Optional[float] = None,
) -> dict:
    """
    Standardized payload for logging episodic performance metrics to W&B.

    Args:
        step: Total environment steps or training iterations.
        episode_return: Total episodic undiscounted return.
        episode_length: Number of steps the episode lasted.
        success_rate: Optional task completion binary indicator or fraction (0.0 to 1.0).

    Returns:
        dict: Payload suitable for wandb.log(...).
    """
    payload = {
        "step": step,
        "charts/episodic_return": float(episode_return),
        "charts/episodic_length": int(episode_length),
    }
    if success_rate is not None:
        payload["charts/success_rate"] = float(success_rate)
    return payload
