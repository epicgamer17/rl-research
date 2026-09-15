"""
Environment Feature Encoders.

Functions for tile coding and continuous-to-discrete state representation mappings.
"""

import numpy as np
import torch
from typing import Optional


# TODO: where to put this?
# TODO: kind of messy, untested, and very mountaincar specific (in respect to defaults and dimensionality)
# TODO: does this belong as an env util function or something?
def compute_tile_coding_features(
    state: np.ndarray,
    action: int,
    num_actions: int,
    num_tilings: int = 10,
    tiles_per_tiling: int = 10,
    state_low: np.ndarray = np.array([-1.2, -0.07]),
    state_high: np.ndarray = np.array([0.6, 0.07]),
) -> torch.Tensor:
    """
    Computes a sparse binary feature vector phi(s, a) using tile coding.

    According to the Stream RL paper, tile coding has been shown to reduce forgetting, as a form of "sparse initialization" but really sparse representation.
    """
    # 1. Clip the state to the bounds strictly
    clipped_state = np.clip(state, state_low, state_high)

    # 2. Normalize state to [0, tiles_per_tiling]
    state_normalized = (
        (clipped_state - state_low) / (state_high - state_low) * tiles_per_tiling
    )

    # 2. Compute asymmetrical offsets for each tiling to prevent diagonal artifacts
    # Using 1 and 3 as displacement multipliers (Sutton's recommendation for 2D)
    offsets = np.zeros((num_tilings, len(state)))
    for i in range(num_tilings):
        offsets[i, 0] = ((i * 1) % num_tilings) / num_tilings
        if len(state) > 1:
            offsets[i, 1] = ((i * 3) % num_tilings) / num_tilings

    # 3. Find the active tile in each tiling
    active_tiles = []
    for i in range(num_tilings):
        tile_coords = np.floor(state_normalized + offsets[i]).astype(int)
        # Map 2D coordinates to a 1D index for this specific tiling
        # Handling up to 2D for Mountain Car.
        if len(state) == 2:
            tile_idx = (
                i * ((tiles_per_tiling + 1) ** 2)
                + tile_coords[0] * (tiles_per_tiling + 1)
                + tile_coords[1]
            )
        else:
            tile_idx = i * (tiles_per_tiling + 1) + tile_coords[0]

        active_tiles.append(tile_idx)

    # 4. Create the state-action flat vector
    features_per_action = num_tilings * ((tiles_per_tiling + 1) ** 2)
    total_features = features_per_action * num_actions

    phi = torch.zeros(total_features, dtype=torch.float64)

    # Offset by the action to make it phi(s, a)
    action_offset = action * features_per_action
    for tile_idx in active_tiles:
        phi[action_offset + tile_idx] = 1.0

    return phi
