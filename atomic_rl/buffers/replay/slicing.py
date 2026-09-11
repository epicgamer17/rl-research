"""Functional dynamic unroll slicing for sequence replay buffers.

Provides vectorized, pure functional utilities to extract K-step unroll trajectories
from stored sequences with boundary checking and absorbing state padding.
"""

from typing import Optional, Tuple, Union
import torch
from tensordict import TensorDict


def slice_unrolls(
    data: TensorDict,
    coords: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    unroll_steps: int,
    valid_lengths: torch.Tensor,
) -> Tuple[TensorDict, torch.Tensor]:
    """Slices dynamic K-step unrolls from stored sequences using sampled coordinates.

    Args:
        data (TensorDict): Stored sequences with batch_size (e.g. [capacity, sequence_length]
            or N-D grids [runs, agents, steps]).
        coords (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): Sample coordinates.
            For 2D buffers: (seq_indices, step_indices).
            For N-D buffers: (coord_0, ..., step_indices).
            For 1D buffers: step_indices.
        unroll_steps (int): Number of unroll steps K (yielding K+1 total steps).
        valid_lengths (torch.Tensor): Tensor of actual sequence lengths.
            Steps at or beyond valid_lengths are marked invalid in valid_mask.

    Returns:
        Tuple[TensorDict, torch.Tensor]:
            - unrolls: TensorDict with batch_size [B, unroll_steps + 1] containing gathered data.
            - valid_mask: Boolean tensor of shape [B, unroll_steps + 1] indicating which steps
              fall within the valid sequence boundary.
    """
    leading_coords = coords[:-1] if isinstance(coords, tuple) else ()
    step_indices = coords[-1] if isinstance(coords, tuple) else coords

    batch_size = step_indices.shape[0]
    device = step_indices.device

    # Vectorized offset grid: [B, K + 1]
    offsets = torch.arange(unroll_steps + 1, device=device).unsqueeze(0)
    target_steps = step_indices.unsqueeze(1) + offsets  # [B, K + 1]

    seq_len = data.batch_size[-1]
    actual_lengths = valid_lengths[leading_coords].unsqueeze(1)
    valid_mask = target_steps < actual_lengths

    # Clamp target steps to valid range for gathering without indexing crashes
    clamped_steps = torch.clamp(target_steps, 0, seq_len - 1)
    seq_data = data[leading_coords] if len(leading_coords) > 0 else data

    # Vectorized advanced indexing across [B, K + 1]
    batch_idx = (
        torch.arange(batch_size, device=device)
        .unsqueeze(1)
        .expand(-1, unroll_steps + 1)
    )
    unrolls = seq_data[batch_idx, clamped_steps]

    # Zero out invalid/absorbing steps
    for key, tensor in unrolls.items():
        mask = valid_mask[(...,) + (None,) * (tensor.ndim - valid_mask.ndim)]
        unrolls[key] = tensor * mask

    return unrolls, valid_mask
