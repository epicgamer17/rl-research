"""
Metrics and Visualizations for Parameter and Optimization Traces.

Visualizes dynamic meta-parameters over training steps, such as step-size / learning-rate
adaptation in IDBD and AutoStep algorithms (comparing relevant vs. irrelevant features).
"""

import os
from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# OPTIMIZATION & STEP-SIZE TRACES (To be moved from visualization.py)
# ==============================================================================


def plot_learning_rate_traces(
    alphas_history: np.ndarray,
    num_relevant: int,
    title: str = "Learning Rates Over Time",
    xlabel: str = "Time Steps",
    ylabel: str = "Learning Rate (alpha)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots the average learning rates for relevant vs. irrelevant features over time.
    Excellent for visualizing IDBD's automatic feature selection.

    Args:
        alphas_history (np.ndarray): Array of shape [Steps, Features] containing learning rates.
        num_relevant (int): The first N features are considered relevant.
        title (str): Chart title.
        save_path (str, optional): If provided, saves the figure to this path.

    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = np.arange(alphas_history.shape[0])

    # Calculate means
    relevant_alphas = alphas_history[:, :num_relevant].mean(axis=1)
    irrelevant_alphas = alphas_history[:, num_relevant:].mean(axis=1)

    ax.plot(
        steps, relevant_alphas, label="Relevant Features", color="black", linewidth=1.5
    )
    ax.plot(
        steps,
        irrelevant_alphas,
        label="Irrelevant Features",
        color="gray",
        linewidth=1.5,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return fig


def create_wandb_lr_plot(step: int, relevant_lr: float, irrelevant_lr: float) -> dict:
    """
    Creates a simple payload for wandb logging of learning rates.
    """
    return {
        "step": step,
        "metrics/relevant_lr_mean": relevant_lr,
        "metrics/irrelevant_lr_mean": irrelevant_lr,
    }
