"""
Environment Feature Encoders.

Functions for tile coding and continuous-to-discrete state representation mappings.
"""

import numpy as np
import torch
from typing import Optional


def compute_tile_coding_features(
    state: np.ndarray,
    action: int,
    num_actions: int,
    num_tilings: int = 10,
    tiles_per_tiling: int = 10,
    state_low: Optional[np.ndarray] = None,
    state_high: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Computes a sparse binary feature vector phi(s, a) using tile coding.

    Args:
        state: The continuous state vector.
        action: Action index.
        num_actions: Total number of discrete actions.
        num_tilings: Number of overlapping tilings.
        tiles_per_tiling: Resolution per tiling dimension.
        state_low: Lower bounds of state space. Defaults to [-1.2, -0.07] if None.
        state_high: Upper bounds of state space. Defaults to [0.6, 0.07] if None.

    Returns:
        torch.Tensor: Sparse binary feature vector.
    """
    if state_low is None:
        state_low = np.array([-1.2, -0.07])
    if state_high is None:
        state_high = np.array([0.6, 0.07])
    pass
