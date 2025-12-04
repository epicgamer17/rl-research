# modules/world_model_interface.py (Revised)
from abc import ABC, abstractmethod
from torch import Tensor
from typing import Tuple, Dict, Any


class WorldModelInterface(ABC):
    """
    Abstract Interface for any model/simulator used within the MuZero training loop.
    All implementations (MuZero, Dreamer, PerfectSim) must adhere to these methods.
    """

    @abstractmethod
    def initial_inference(self, observation: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculates the initial hidden state from an observation.

        Returns: (hidden_state)
        """
        pass

    @abstractmethod
    def recurrent_inference(
        self, hidden_state: Tensor, action: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculates the next hidden state, immediate reward
        from a hidden state and an action.

        Returns: (next_hidden_state, reward)
        """
        pass

    @abstractmethod
    def get_networks(self) -> Dict[str, nn.Module]:
        """
        Returns a dictionary of all trainable PyTorch networks within this model.
        Used by the main training loop for optimization and checkpointing.
        """
        pass
