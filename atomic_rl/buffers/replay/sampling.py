from ...utils import _allocate_tensordict
from .state import BufferState, PERBufferState
from .write import compute_is_weights
import torch
from tensordict import TensorDict
import math
from typing import Tuple, Callable, List, Optional, Union
import random
from collections import deque
from dataclasses import dataclass


# Option Comparison (Option A vs Option B):
# - Option B: Maintain a full 2D boolean mask valid_mask: Tensor[num_seqs, max_steps] and call torch.nonzero(valid_mask) to gather coordinates. While conceptually simple, Option B must scan and allocate large 2D coordinate matrices on every sample call, adding memory traffic and CPU scanning overhead.
# - Option A (Chosen): Maintain 1D sequence lengths valid_lengths: Tensor[num_seqs]. Using torch.cumsum + torch.searchsorted maps a 1D uniform integer u in [0, total_steps) directly into (seq_idx, step_idx) in O(B log N) time (< 10 microseconds).
#   Every valid transition has exact 1/total_steps probability, zero padded dead space is sampled,
#   and no large 2D index tables are allocated.
def uniform_sample(
    buffer_state: BufferState,
    rng_key: torch.Generator,
    batch_size: int,
) -> Tuple[TensorDict, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
    """Uniform sampling supporting 1D items (transitions) or multi-dimensional items (sequences/observations).

    Args:
        buffer_state (BufferState): The buffer state.
        rng_key (torch.Generator): RNG generator.
        batch_size (int): Number of items to sample.

    Returns:
        Tuple[TensorDict, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
            (batch, coords) where coords is a 1D tensor for 1D sampling, or a tuple of index tensors
            (coord_dim0, ..., step_indices) for multi-dimensional sampling.
    """
    if buffer_state.size < 1:
        raise ValueError("Cannot sample from an empty buffer.")

    if buffer_state.size < batch_size:
        raise ValueError(
            f"Total valid buffer items ({buffer_state.size}) is smaller than batch size ({batch_size})."
        )

    # 1. 1D Buffer: valid_lengths is a 0D scalar tracking total transitions
    if buffer_state.valid_lengths.ndim == 0:
        coords = torch.randint(0, buffer_state.size, (batch_size,), generator=rng_key)
        return buffer_state.data[coords], coords

    # 2. Multi-dimensional Buffer via Prefix Sum over valid transitions

    flat_lengths = buffer_state.valid_lengths.view(-1)
    cum_lengths = torch.cumsum(flat_lengths, dim=0)

    u = torch.randint(
        0,
        buffer_state.size,
        (batch_size,),
        generator=rng_key,
        device=flat_lengths.device,
    )
    flat_prefix_idx = torch.searchsorted(cum_lengths, u, right=True)
    prev_cum = torch.where(
        flat_prefix_idx > 0, cum_lengths[flat_prefix_idx - 1], torch.zeros_like(u)
    )
    step_idx = u - prev_cum

    leading_coords = torch.unravel_index(
        flat_prefix_idx, buffer_state.storage_shape[:-1]
    )
    coords = (*leading_coords, step_idx)

    return buffer_state.data[coords], coords


# TODO: unify sample API and make sample_per use an RNG key
def sample_per(
    buffer_state: PERBufferState,
    batch_size: int,
    beta: torch.Tensor,
) -> Tuple[
    TensorDict,
    Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    torch.Tensor,
]:
    """Prioritized Experience Replay sampling supporting 1D and multi-dimensional sample_dims.

    Args:
        buffer_state (PERBufferState): The PER buffer state.
        batch_size (int): The size of the batch to sample.
        beta (torch.Tensor): The beta parameter for PER.
            Using a tensor for beta avoids torch.compile recompilation.

    Returns:
        Tuple[TensorDict, Union[torch.Tensor, Tuple[torch.Tensor, ...]], torch.Tensor]:
            (batch, coords, is_weights).
    """
    total_priority = buffer_state.sum_tree[0]

    # Generate batch_size random values between 0 and total_priority
    segment_length = total_priority / batch_size
    targets = (
        torch.rand(batch_size, device=buffer_state.sum_tree.device)
        + torch.arange(batch_size, device=buffer_state.sum_tree.device)
    ) * segment_length

    # Vectorized Tree Traversal
    indices = buffer_state.sum_tree.new_zeros(batch_size, dtype=torch.long)
    depth = int(math.log2(buffer_state.tree_capacity))

    for _ in range(depth):
        left_children = indices * 2 + 1
        right_children = indices * 2 + 2

        left_priorities = buffer_state.sum_tree[left_children]

        # If target > left_priority, go right and subtract left_priority
        go_right = targets > left_priorities

        targets = torch.where(go_right, targets - left_priorities, targets)
        indices = torch.where(go_right, right_children, left_children)

    # 'indices' are now the tree indices (leaves). We need the data indices.
    data_indices = indices - (buffer_state.tree_capacity - 1)

    # Importance Sampling (IS) Weights
    leaf_priorities = buffer_state.sum_tree[indices]
    min_prob = buffer_state.min_tree[0] / total_priority

    is_weights = compute_is_weights(leaf_priorities, min_prob, total_priority, beta)

    # Generic coordinate decomposition over storage dimensions
    storage_shape = buffer_state.storage_shape
    max_valid_flat = math.prod(storage_shape) - 1
    data_indices = torch.clamp(data_indices, 0, max_valid_flat)

    if len(storage_shape) == 1:
        coords: Union[torch.Tensor, Tuple[torch.Tensor, ...]] = data_indices
    else:
        coords = torch.unravel_index(data_indices, storage_shape)

    batch = buffer_state.data[coords]

    return batch, coords, is_weights
