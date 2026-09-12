import math
from ...utils import _allocate_tensordict
import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional, Union, Sequence
from dataclasses import dataclass


@dataclass(kw_only=True)
class BufferState:
    """
    State of a standard replay buffer.

    Attributes:
        data (TensorDict): The stored transitions or sequences.
        pointer (int): Current write position along dimension 0.
        valid_lengths (torch.Tensor): Tensor tracking active length along the trailing dimension.
            Shape is storage_shape[:-1]: () for 1D, (capacity,) for 2D, (capacity, agents) for 3D.
        steps_seen (int): Total number of transitions/sequences processed by this buffer.
            Useful for Reservoir Sampling logic.
    """

    data: TensorDict
    pointer: int
    valid_lengths: torch.Tensor
    steps_seen: int = 0

    @property
    def capacity(self) -> int:
        """Maximum number of slots along dimension 0."""
        return self.data.shape[0]

    @property
    def storage_shape(self) -> Tuple[int, ...]:
        """Full shape of the stored buffer [capacity, ...]."""
        return tuple(self.data.batch_size)

    @property
    def size(self) -> int:
        """Total number of valid samplable elements/transitions currently in the buffer."""
        return int(self.valid_lengths.sum().item())


# TODO: CONSIDER Option 3 API: decouple write slots from slot shape (e.g. num_slots: int, slot_shape: Tuple[int, ...])
# so that capacity = (num_slots, *slot_shape), making the distinction between stream capacity and sequence dimensions even more explicit.
def init_buffer(
    capacity: Union[int, Sequence[int]],
    shapes: dict,
    device: Optional[Union[str, torch.device]] = None,
) -> BufferState:
    """Initializes a standard buffer with 1D or multi-dimensional batch capacity."""
    if isinstance(capacity, int):
        batch_size = [capacity]
    else:
        batch_size = list(capacity)

    data = _allocate_tensordict(shapes, batch_size, device)
    valid_shape = tuple(batch_size[:-1])
    valid_lengths = torch.zeros(valid_shape, dtype=torch.long, device=device)
    return BufferState(
        data=data,
        pointer=0,
        valid_lengths=valid_lengths,
        steps_seen=0,
    )


@dataclass(kw_only=True)
class PERBufferState(BufferState):
    """
    State of a Prioritized Experience Replay (PER) buffer.
    """

    sum_tree: torch.Tensor
    min_tree: torch.Tensor
    max_priority: torch.Tensor
    tree_capacity: int


def init_per_buffer(
    capacity: Union[int, Sequence[int]],
    shapes: dict,
    device: Optional[Union[str, torch.device]] = None,
) -> PERBufferState:
    """Initializes a PER buffer where priorities are tracked for every element in storage_shape."""
    if isinstance(capacity, int):
        batch_size = [capacity]
    else:
        batch_size = list(capacity)

    num_elements = math.prod(batch_size)
    tree_capacity = 1 << (num_elements - 1).bit_length() if num_elements > 0 else 1
    empty_data = _allocate_tensordict(shapes, batch_size, device)
    valid_shape = tuple(batch_size[:-1])
    valid_lengths = torch.zeros(valid_shape, dtype=torch.long, device=device)

    return PERBufferState(
        data=empty_data,
        sum_tree=torch.zeros(2 * tree_capacity - 1, device=device),
        min_tree=torch.full((2 * tree_capacity - 1,), float("inf"), device=device),
        max_priority=torch.tensor(1.0, device=device),
        pointer=0,
        tree_capacity=tree_capacity,
        steps_seen=0,
        valid_lengths=valid_lengths,
    )


