import numpy as np
import torch
import torch.nn as nn
from typing import Iterable

# TODO: should we merge this with visualzations.py? should stuff from visualizations.py be moved here?


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
    assert isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor), (
        f"Expected PyTorch Tensors for y_true and y_pred, got {type(y_true)} and {type(y_pred)}"
    )
    assert y_true.shape == y_pred.shape, (
        f"Exact shape mismatch in compute_explained_variance! y_true: {y_true.shape}, y_pred: {y_pred.shape}"
    )

    var_y = torch.var(y_true)
    if var_y == 0:
        return y_true.new_tensor(float("nan"))

    var_diff = torch.var(y_true - y_pred)
    return 1.0 - (var_diff / var_y)



