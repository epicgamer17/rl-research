from ...utils import _allocate_tensordict
from .state import BufferState, PERBufferState
import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional, Union
import math
import random
from collections import deque
from dataclasses import dataclass

# TODO: Should valid_mask be in the exlicit signature? or should we add some kind of higher level wrapper for it? I know its an optional arg, but it kind of assumes we are storing padded data (more implicit less explicit). We could instead for example have "make_padded_write" or something like that (maybe thats actually cleaner??) or is it cleaner to say all write should allow for padded data? really not sure. how "native" do we want it to be? can you make padded writes generic? i'm actually not sure if you can (for example with PER it has to be handled specially), so likely current approach is fine, but should be documented with why.


def circular_write_strategy_(
    buffer_state: BufferState,
    batch: TensorDict,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[BufferState, torch.Tensor]:
    """Writes a batch of transitions or sequences to the buffer using circular (FIFO) overwriting.

    Validity & Sequence Length Handling:
        - Dense / Unpadded Data: If valid_mask is None, every step in the batch is assumed
          to be real data. The valid length is inferred dynamically from the batch shape
          (batch.batch_size[-1] for multi-dimensional sequences).
        - Ragged / Padded Sequences: When storing variable-length episodes padded into rectangular
          tensors [batch_size, max_seq_len, ...], provide `valid_mask` (shape [batch_size, max_seq_len],
          boolean). The buffer uses `valid_mask.sum(dim=-1)` to record the true valid length of each
          sequence, ensuring padding steps are never sampled.

    Args:
        buffer_state (BufferState): The buffer state to write into.
        batch (TensorDict): A TensorDict of batched transitions or sequences to write.
        valid_mask (Optional[torch.Tensor]): Optional boolean mask of shape matching batch leading
            dimensions where True = real step, False = padding.

    Returns:
        Tuple[BufferState, torch.Tensor]: The updated buffer state and written indices.
    """
    batch_size = batch.batch_size[0]
    start_idx = buffer_state.pointer
    end_idx = start_idx + batch_size

    if end_idx <= buffer_state.capacity:
        indices = torch.arange(start_idx, end_idx, dtype=torch.long)
        buffer_state.data[start_idx:end_idx] = batch
    else:
        overflow = end_idx - buffer_state.capacity
        first_chunk_size = buffer_state.capacity - start_idx
        indices = torch.cat(
            [
                torch.arange(start_idx, buffer_state.capacity, dtype=torch.long),
                torch.arange(0, overflow, dtype=torch.long),
            ]
        )
        buffer_state.data[start_idx : buffer_state.capacity] = batch[:first_chunk_size]
        buffer_state.data[0:overflow] = batch[first_chunk_size:]

    buffer_state.pointer = end_idx % buffer_state.capacity
    buffer_state.steps_seen += batch_size

    if buffer_state.valid_lengths.ndim == 0:
        new_size = min(buffer_state.size + batch_size, buffer_state.capacity)
        buffer_state.valid_lengths.fill_(new_size)
    else:
        if valid_mask is not None:
            buffer_state.valid_lengths[indices] = valid_mask.sum(dim=-1).long()
        else:
            buffer_state.valid_lengths[indices] = batch.batch_size[-1]

    return buffer_state, indices


def reservoir_write_strategy_(
    buffer_state: BufferState,
    batch: TensorDict,
    valid_mask: Optional[torch.Tensor] = None,
) -> Tuple[BufferState, torch.Tensor]:
    """Writes data using Reservoir Sampling (uniform probability over an unbounded stream).

    Validity & Sequence Length Handling:
        - Dense / Unpadded Data: If valid_mask is None, every step in the batch is assumed
          to be real data. The valid length is inferred dynamically from the batch shape.
        - Ragged / Padded Sequences: When storing variable-length episodes padded into rectangular
          tensors, provide `valid_mask` (shape [batch_size, max_seq_len], boolean). The buffer
          uses `valid_mask[i].sum(dim=-1)` to record the true valid length of each written sequence.

    Args:
        buffer_state (BufferState): The buffer state to write into.
        batch (TensorDict): A TensorDict of batched transitions or sequences to write.
        valid_mask (Optional[torch.Tensor]): Optional boolean mask where True = real step,
            False = padding.

    Returns:
        Tuple[BufferState, torch.Tensor]: The updated buffer state and written indices.
    """
    batch_size = batch.batch_size[0]

    written_indices = []
    for i in range(batch_size):
        # 1. Decide if and where to write
        if buffer_state.steps_seen < buffer_state.capacity:
            idx = buffer_state.steps_seen
        else:
            # Standard reservoir math: keep item with probability (capacity / steps_seen)
            j = random.randint(0, buffer_state.steps_seen)
            if j < buffer_state.capacity:
                idx = j
            else:
                idx = None

        if idx is not None:
            # 2. Write to TensorDict
            buffer_state.data[idx] = batch[i]
            written_indices.append(idx)
            if buffer_state.valid_lengths.ndim > 0:
                if valid_mask is not None:
                    buffer_state.valid_lengths[idx] = valid_mask[i].sum(dim=-1).long()
                else:
                    buffer_state.valid_lengths[idx] = batch.batch_size[-1]

        buffer_state.steps_seen += 1

    if buffer_state.valid_lengths.ndim == 0:
        buffer_state.valid_lengths.fill_(
            min(buffer_state.steps_seen, buffer_state.capacity)
        )

    indices_tensor = (
        torch.tensor(written_indices, dtype=torch.long)
        if written_indices
        else torch.empty(0, dtype=torch.long)
    )
    return buffer_state, indices_tensor


def with_per_tracking(
    write_strategy_fn: Callable,
    epsilon: float,
) -> Callable:
    """Higher-order function composing a base writing strategy with PER logic.

    Args:
        write_strategy_fn (Callable): The base write strategy (e.g. circular_write_strategy_).
        epsilon (float): Minimum priority floor for valid transitions.
    """

    def per_add(
        buffer_state: PERBufferState,
        batch: TensorDict,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> PERBufferState:
        # 1. Execute the base writing strategy
        new_state, written_indices = write_strategy_fn(
            buffer_state, batch, valid_mask=valid_mask
        )

        # 2. If data was written, update the PER sum/min trees
        if written_indices.numel() > 0:
            block_size = (
                math.prod(new_state.storage_shape[1:])
                if len(new_state.storage_shape) > 1
                else 1
            )
            if block_size == 1:
                coords = written_indices
            else:
                r = written_indices.repeat_interleave(block_size)
                s = torch.arange(block_size, device=written_indices.device).repeat(
                    len(written_indices)
                )
                coords = (r, s)

            tree_indices = _coords_to_tree_indices(new_state, coords)

            # TODO: we accept the valid mask as an explicit arg instead of a batch key/item, should we do the same with priorities? Reason for: its more explicit, reason against, it might make it harder to use for example stacks of wrappers (i dont think thats actually true), but at least making OOP orchestrators for the functions will be harder, as signatures will differ so composition might be harder, but pure functional orchestration just as easy (easier as signatures are more obvious and clear).
            # Check for explicit priorities in the batch
            if "priority" in batch.keys():
                priorities = batch["priority"].reshape(-1)
                priorities = torch.clamp(priorities, min=epsilon)
            else:
                # Use max priority for all new additions, bounded by epsilon
                init_prio = torch.clamp(new_state.max_priority, min=epsilon)
                priorities = torch.full(
                    (tree_indices.shape[0],),
                    init_prio,
                    dtype=torch.float32,
                    device=written_indices.device,
                )

            if valid_mask is not None:
                mask_flat = valid_mask.reshape(-1).to(device=priorities.device).bool()
                priorities = torch.where(
                    mask_flat,
                    priorities,
                    torch.zeros_like(priorities),
                )

            new_sum_tree, new_min_tree = _update_tree(
                new_state.sum_tree, new_state.min_tree, tree_indices, priorities
            )

            new_state.sum_tree = new_sum_tree
            new_state.min_tree = new_min_tree
            positive_prios = priorities[priorities > 0]
            if positive_prios.numel() > 0:
                new_state.max_priority = torch.maximum(
                    new_state.max_priority, torch.max(positive_prios)
                )

        return new_state

    return per_add


@torch.inference_mode()
def _update_tree(
    sum_tree: torch.Tensor,
    min_tree: torch.Tensor,
    tree_indices: torch.Tensor,
    priorities: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the tree nodes and propagates up to the root.
    """
    # This modifies the tensors in-place for speed, but maintains a functional signature
    sum_tree[tree_indices] = priorities
    min_tree[tree_indices] = torch.where(
        priorities > 0,
        priorities,
        torch.tensor(float("inf"), device=priorities.device),
    )

    # Propagate up
    parent_indices = (tree_indices - 1) // 2

    while parent_indices.numel() > 0 and (parent_indices >= 0).all():
        left_children = parent_indices * 2 + 1
        right_children = parent_indices * 2 + 2

        sum_tree[parent_indices] = sum_tree[left_children] + sum_tree[right_children]
        min_tree[parent_indices] = torch.minimum(
            min_tree[left_children], min_tree[right_children]
        )

        parent_indices = (parent_indices - 1) // 2

    return sum_tree, min_tree


def _coords_to_tree_indices(
    buffer_state: PERBufferState,
    coords: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
) -> torch.Tensor:
    """Converts multi-dimensional sample coordinates to binary sum/min tree leaf indices.

    Args:
        buffer_state (PERBufferState): The PER buffer state.
        coords (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): Coordinates in buffer storage.
            For 1D buffers, this is a 1D tensor of data indices.
            For multi-dimensional buffers, this is a tuple of index tensors (e.g. (seq_idx, step_idx)).

    Returns:
        torch.Tensor: Absolute leaf indices in the binary tree.
    """
    if isinstance(coords, tuple):
        flat_data_indices = coords[0]
        for coord, dim_size in zip(coords[1:], buffer_state.storage_shape[1:]):
            flat_data_indices = flat_data_indices * dim_size + coord
    else:
        flat_data_indices = coords

    return flat_data_indices + buffer_state.tree_capacity - 1


def update_priorities(
    buffer_state: PERBufferState,
    coords: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    td_errors: torch.Tensor,
    alpha: float,
    epsilon: float,
) -> PERBufferState:
    """Update the PER sum/min trees with TD errors using sample coordinates.

    Args:
        buffer_state (PERBufferState): The PER buffer state.
        coords (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): Coordinates of sampled transitions.
        td_errors (torch.Tensor): TD errors for priority calculation.
        alpha (float): PER exponent alpha.
        epsilon (float): Minimum priority floor for valid transitions.
    """
    tree_indices = _coords_to_tree_indices(buffer_state, coords)
    priorities = torch.pow(torch.clamp(torch.abs(td_errors), min=epsilon), alpha)

    new_sum_tree, new_min_tree = _update_tree(
        buffer_state.sum_tree, buffer_state.min_tree, tree_indices, priorities
    )

    new_max_priority = torch.maximum(buffer_state.max_priority, torch.max(priorities))

    return PERBufferState(
        data=buffer_state.data,
        sum_tree=new_sum_tree,
        min_tree=new_min_tree,
        max_priority=new_max_priority,
        pointer=buffer_state.pointer,
        tree_capacity=buffer_state.tree_capacity,
        steps_seen=buffer_state.steps_seen,
        valid_lengths=buffer_state.valid_lengths,
    )
