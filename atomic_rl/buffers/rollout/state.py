import math
from ...utils import _allocate_tensordict
import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional, Union, Sequence
from dataclasses import dataclass, field


# TODO: maybe remove/make this like PERBufferState instead of its own base class, it extends a standard buffer state (or make a new base state).
@dataclass
class RolloutBufferState:
    """
    State of a rollout buffer used for on-policy algorithms (PPO, A2C).

    Attributes:
        data (TensorDict): The stored transitions with geometry [Batch, Time].
        truncation_records (List[Tuple[int, int, torch.Tensor]]): Records of
            truncations and their final observations for patching next_values.
    """

    data: TensorDict
    truncation_records: List[Tuple[int, int, torch.Tensor]] = field(
        default_factory=list
    )

    @property
    def capacity(self) -> int:
        """Total capacity (number of transitions) across all environments and steps."""
        return math.prod(self.data.batch_size)

    @property
    def storage_shape(self) -> Tuple[int, ...]:
        """Explicit batch geometry of the buffer [num_envs, steps_per_env]."""
        return tuple(self.data.batch_size)

    @property
    def size(self) -> int:
        """Total number of transitions in the rollout buffer."""
        return math.prod(self.data.batch_size)

    @property
    def valid_lengths(self) -> torch.Tensor:
        """Active lengths along the trailing dimension for slice_unrolls compatibility."""
        return torch.full(
            self.data.batch_size[:-1],
            self.data.batch_size[-1],
            dtype=torch.long,
            device=self.data.device,
        )


def init_rollout_buffer(
    capacity: Sequence[int],
    shapes: dict,
    device: Optional[Union[str, torch.device]] = None,
) -> RolloutBufferState:
    """Initializes a rollout buffer with batch geometry `capacity` [num_envs, steps_per_env]."""
    data = _allocate_tensordict(shapes, list(capacity), device)
    return RolloutBufferState(
        data=data,
        truncation_records=[],
    )
