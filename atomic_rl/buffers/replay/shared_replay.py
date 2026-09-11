"""Process-safe Shared Replay Buffers for Multiprocessing RL algorithms."""

from typing import Any, Dict, Optional, Tuple, Union, Sequence
from tensordict import TensorDict
import torch
import torch.multiprocessing as mp

from .state import init_buffer, init_per_buffer
from .write import circular_write_strategy_, update_priorities, with_per_tracking
from .sampling import sample_per, uniform_sample


# TODO: Clean this up. Should we add other Buffer classes? Can we make these more configurable. ie allowing another sampling function instead of inlining them and hardcoding them, wherever possible. Its an orchestration wrapper.
class SharedReplayBuffer:
    """
    Thread- and process-safe uniform replay buffer using torch.multiprocessing.

    Shares underlying TensorDict buffer memory across worker processes and
    synchronizes pointers/size via an mp.Lock. Supports 1D and multidimensional
    capacities (e.g. [capacity, sequence_length]).
    """

    # TODO: CONSIDER Option 3 API: decouple write slots from slot shape (e.g. num_slots: int, slot_shape: Tuple[int, ...])
    # so that capacity = (num_slots, *slot_shape), making the distinction between stream capacity and sequence dimensions even more explicit.
    def __init__(
        self,
        capacity: Union[int, Sequence[int]],
        shapes: Dict[str, Any],
        device: torch.device = torch.device("cpu"),
    ):
        self.buffer_state = init_buffer(capacity, shapes, device=device)
        self.buffer_state.data.share_memory_()
        self.buffer_state.valid_lengths.share_memory_()

        self.lock = mp.Lock()
        self._pointer = mp.Value("i", 0)
        self._size = mp.Value("i", 0)

    def add_transitions(
        self,
        samples_td: TensorDict,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        """Adds transitions or sequences to the shared buffer.

        Args:
            samples_td (TensorDict): Transitions or sequences to write.
            valid_mask (Optional[torch.Tensor]): Optional boolean mask where True = real step,
                False = padding. If None, full length is assumed.
        """
        with self.lock:
            self.buffer_state.pointer = self._pointer.value
            self.buffer_state, _ = circular_write_strategy_(
                self.buffer_state, samples_td, valid_mask=valid_mask
            )
            self._pointer.value = self.buffer_state.pointer
            self._size.value = self.buffer_state.size

    def sample_batch(
        self,
        rng_key: torch.Generator,
        batch_size: int,
        min_size: int = 0,
    ) -> Optional[Tuple[TensorDict, Union[torch.Tensor, Tuple[torch.Tensor, ...]]]]:
        with self.lock:
            if self._size.value < min_size:
                return None
            return uniform_sample(
                self.buffer_state,
                rng_key,
                batch_size,
            )

    @property
    def size(self) -> int:
        return self._size.value


class SharedPERBuffer:
    """
    Thread- and process-safe Prioritized Experience Replay (PER) buffer using torch.multiprocessing.
    """

    def __init__(
        self,
        capacity: Union[int, Sequence[int]],
        shapes: dict,
        device: torch.device = torch.device("cpu"),
    ):
        self.buffer_state = init_per_buffer(capacity, shapes, device=device)
        self.buffer_state.data.share_memory_()
        self.buffer_state.sum_tree.share_memory_()
        self.buffer_state.min_tree.share_memory_()
        self.buffer_state.valid_lengths.share_memory_()

        self.lock = mp.Lock()
        self._pointer = mp.Value("i", 0)
        self._size = mp.Value("i", 0)
        self._max_priority = mp.Value("d", 1.0)
        self.per_add = with_per_tracking(circular_write_strategy_)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "per_add" in state:
            del state["per_add"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.per_add = with_per_tracking(circular_write_strategy_)

    def add_transitions(
        self,
        transitions: TensorDict,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        """Adds transitions or sequences to the shared PER buffer.

        Args:
            transitions (TensorDict): Transitions or sequences to write.
            valid_mask (Optional[torch.Tensor]): Optional boolean mask where True = real step,
                False = padding. If None and 'valid' not in transitions, full length is assumed.
        """
        with self.lock:
            self.buffer_state.pointer = self._pointer.value
            self.buffer_state.max_priority = torch.tensor(
                self._max_priority.value, dtype=torch.float32
            )

            self.buffer_state = self.per_add(
                self.buffer_state, transitions, valid_mask=valid_mask
            )

            self._pointer.value = self.buffer_state.pointer
            self._size.value = self.buffer_state.size
            max_p = self.buffer_state.max_priority
            self._max_priority.value = float(
                max_p.item() if isinstance(max_p, torch.Tensor) else max_p
            )


    def sample_batch(
        self, batch_size: int, beta: float, min_size: int = 0
    ) -> Tuple[
        Optional[TensorDict],
        Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]],
        Optional[torch.Tensor],
    ]:
        with self.lock:
            if self._size.value < min_size:
                return None, None, None
            beta_tensor = torch.tensor(beta, dtype=torch.float32)
            return sample_per(self.buffer_state, batch_size, beta_tensor)

    @property
    def size(self) -> int:
        return self._size.value

    def update_priorities(
        self,
        coords: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        td_errors: torch.Tensor,
        alpha: float,
    ):
        with self.lock:
            self.buffer_state = update_priorities(
                self.buffer_state, coords, td_errors, alpha
            )
