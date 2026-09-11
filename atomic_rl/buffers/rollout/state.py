from ...utils import _allocate_tensordict
import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional, Union, Sequence
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RolloutBufferState:
    """
    State of a rollout buffer used for on-policy algorithms (PPO, A2C).

    Attributes:
        data (TensorDict): The stored transitions with geometry [Batch, Time].
        truncation_records (List[Tuple[int, int, torch.Tensor]]): Records of
            truncations and their final observations for patching next_values.
        storage_shape (Tuple[int, ...]): Explicit batch geometry of the buffer.
    """

    data: TensorDict
    truncation_records: List[Tuple[int, int, torch.Tensor]] = field(
        default_factory=list
    )
    storage_shape: Tuple[int, ...] = ()


def init_rollout_buffer(
    steps_per_env: Union[int, Sequence[int]],
    num_envs: Optional[Union[int, dict]] = None,
    shapes: Optional[dict] = None,
    device: str = "cpu",
) -> RolloutBufferState:
    """Initializes a rollout buffer [Batch, Time].

    Accepts either (steps_per_env, num_envs, shapes) or capacity tuple (capacity, shapes)
    where capacity=(num_envs, steps_per_env).
    """
    if isinstance(steps_per_env, (tuple, list)):
        num_e, steps_e = steps_per_env[0], steps_per_env[1]
        actual_shapes = num_envs if isinstance(num_envs, dict) else shapes
    else:
        steps_e = steps_per_env
        num_e = num_envs
        actual_shapes = shapes

    batch_size = [num_e, steps_e]
    data = _allocate_tensordict(actual_shapes, batch_size, device)
    return RolloutBufferState(
        data=data,
        truncation_records=[],
        storage_shape=tuple(batch_size),
    )
