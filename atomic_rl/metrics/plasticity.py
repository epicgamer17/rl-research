"""
Metrics and Visualizations for Continual Learning and Neural Network Plasticity.

Tracks phenomena such as loss of plasticity, dormant/dead units, feature rank collapse,
and weight/gradient explosion/decay during non-stationary RL training.
"""

import os
from typing import Iterable, Optional
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ==============================================================================
# COMPUTATION PRIMITIVES (To be moved from atomic_rl/metrics/metrics.py)
# ==============================================================================


# TODO [MOVE HERE]: compute_dead_units_proportion
@torch.no_grad()
def compute_dead_units_proportion(activations: torch.Tensor) -> float:
    """
    Computes the proportion of dead units in a layer.
    A unit is considered 'dead' if it outputs exactly 0 for an entire batch of data.

    Args:
        activations (torch.Tensor): The output tensor of a ReLU layer of shape [Batch, Features].

    Returns:
        float: The percentage of dead units (0.0 to 1.0).
    """
    # Check if a unit is 0 across the entire batch dimension (dim=0)
    dead_mask = (activations <= 0).all(dim=0)
    return dead_mask.sum().item() / dead_mask.numel()


@torch.no_grad()
def compute_stable_rank(activations: torch.Tensor) -> float:
    """
    Computes the stable rank of the representation matrix.
    Stable Rank = (Frobenius Norm)^2 / (Spectral Norm)^2
    It measures the effective dimensionality/expressivity of the features.

    Args:
        activations (torch.Tensor): Tensor of shape [Batch, Features].

    Returns:
        float: The stable rank.
    """
    # Center the activations
    centered = activations - activations.mean(dim=0, keepdim=True)

    # Compute singular values
    # We use svdvals for numerical stability instead of full SVD
    singular_values = torch.linalg.svdvals(centered)

    squared_sv = singular_values**2

    # Frobenius norm squared is the sum of squared singular values
    # Spectral norm squared is the max squared singular value
    max_sq_sv = squared_sv.max().item()
    if max_sq_sv < 1e-8:
        return 0.0

    return squared_sv.sum().item() / max_sq_sv


@torch.no_grad()
def compute_average_weight_magnitude(parameters: Iterable[nn.Parameter]) -> float:
    """Computes the mean absolute value of all weights in the given parameters."""
    total_mag = 0.0
    total_elements = 0
    with torch.no_grad():
        for param in parameters:
            total_mag += torch.abs(param).sum().item()
            total_elements += param.numel()
    return total_mag / total_elements if total_elements > 0 else 0.0


@torch.no_grad()
def compute_average_gradient_magnitude(parameters: Iterable[nn.Parameter]) -> float:
    """Computes the mean absolute value of all gradients in the given parameters."""
    total_mag = 0.0
    total_elements = 0
    with torch.no_grad():
        for param in parameters:
            if param.grad is not None:
                total_mag += torch.abs(param.grad).sum().item()
                total_elements += param.grad.numel()
    return total_mag / total_elements if total_elements > 0 else 0.0


# ==============================================================================
# VISUALIZATION PRIMITIVES (To be moved from atomic_rl/metrics/visualization.py)
# ==============================================================================


def plot_continual_learning_performance(
    results_dict: dict,
    title: str = "Performance Across Continual Tasks",
    xlabel: str = "Task / Permutation Number",
    ylabel: str = "Average Online Accuracy",
    save_path: str = None,
) -> plt.Figure:
    """
    Plots performance over time for multiple algorithms in a continual learning setting.
    Ideal for visualizing loss of plasticity (performance degrading over tasks).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, metrics in results_dict.items():
        steps = np.arange(len(metrics))
        ax.plot(steps, metrics, label=label, linewidth=1.5, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return fig


def plot_plasticity_correlates(
    metrics_dict: dict, save_path: str = "figures/plasticity_correlates.png"
):
    """
    Dynamically plots up to 4 metrics from the provided dictionary in a 2x2 grid.
    Expects metrics_dict in the format:
    {
        "System A": {"Metric 1": [...], "Metric 2": [...], ...},
        "System B": {"Metric 1": [...], "Metric 2": [...], ...}
    }
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Dynamically grab the keys from the first system in the dictionary
    first_system = list(metrics_dict.keys())[0]
    plot_keys = list(metrics_dict[first_system].keys())

    # Ensure we only plot up to 4 keys to fit the 2x2 grid safely
    plot_keys = plot_keys[:4]

    # Predefined colors for standard comparisons, fallback to gray
    colors = {"Base System": "black", "SWR": "dodgerblue"}

    for ax_idx, key in enumerate(plot_keys):
        ax = axes[ax_idx]

        for system_name, system_metrics in metrics_dict.items():
            if key not in system_metrics:
                continue  # Skip if a system is missing this metric

            data = system_metrics[key]
            steps = np.arange(len(data))

            # Match coloring logic
            color = colors.get(system_name)
            if color is None:
                # Assign a generic color if not Base or SWR
                color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                color = color_cycle[len(ax.lines) % len(color_cycle)]

            ax.plot(
                steps, data, label=system_name, color=color, linewidth=2.0, alpha=0.8
            )

        ax.set_title(f"{key}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Task Number")
        ax.grid(True, alpha=0.3, linestyle="--")

        if ax_idx == 0:
            ax.legend(loc="best")

        # Format Y-axis as percentage for Dead Units specifically
        if "Dead Units" in key:
            ticks = ax.get_yticks()
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{y * 100:.0f}%" for y in ticks])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.close()  # Close figure to prevent memory leaks in loops


# ==============================================================================
# EXPERIMENT TRACKING (Weights & Biases)
# ==============================================================================


def create_wandb_plasticity_metrics(
    step: int,
    dead_units_proportion: Optional[float] = None,
    stable_rank: Optional[float] = None,
    weight_magnitude: Optional[float] = None,
    gradient_magnitude: Optional[float] = None,
) -> dict:
    """
    Creates a standardized dictionary payload for logging plasticity correlates to W&B.

    Args:
        step: Current training step or task index.
        dead_units_proportion: Fraction of dead units in ReLU layers (0.0 to 1.0).
        stable_rank: Continuous feature matrix expressivity metric.
        weight_magnitude: Mean absolute weight value.
        gradient_magnitude: Mean absolute gradient value.

    Returns:
        dict: Payload suitable for wandb.log(...).
    """
    payload = {"step": step}
    if dead_units_proportion is not None:
        payload["plasticity/dead_units_proportion"] = float(dead_units_proportion)
    if stable_rank is not None:
        payload["plasticity/stable_rank"] = float(stable_rank)
    if weight_magnitude is not None:
        payload["plasticity/weight_magnitude"] = float(weight_magnitude)
    if gradient_magnitude is not None:
        payload["plasticity/gradient_magnitude"] = float(gradient_magnitude)
    return payload

