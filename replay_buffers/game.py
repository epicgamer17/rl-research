import numpy as np
from typing import NamedTuple, Optional, Any


# 1. THE INTERFACE
# A lightweight, immutable container.
# Great for passing data between functions: process_step(step)
class TimeStep(NamedTuple):
    observation: Any
    info: dict
    action: Optional[Any] = None
    reward: Optional[float] = 0.0
    value: Optional[float] = 0.0
    policy: Optional[Any] = None


class Game:
    def __init__(self, num_players: int):
        self.num_players = num_players
        self.length = 0

        # 2. THE STORAGE (Columnar)
        # We store data in separate lists internally.
        # This is CRITICAL for speed when converting to Tensors later.
        self._obs = []
        self._info = []
        self._rewards = []
        self._actions = []
        self._values = []
        self._policies = []

    def add(self, step: TimeStep):
        """
        Takes a TimeStep object, breaks it down, and stores it efficiently.
        """
        self._obs.append(step.observation)
        self._info.append(step.info)
        self._rewards.append(step.reward if step.reward is not None else 0.0)
        self._actions.append(step.action)
        self._values.append(step.value)
        self._policies.append(step.policy)
        self.length += 1

    def __getitem__(self, index) -> TimeStep:
        """
        Reconstructs the TimeStep object when you ask for it.
        This allows you to treat the Game like a list of objects.
        """
        return TimeStep(
            observation=self._obs[index],
            info=self._info[index],
            reward=self._rewards[index],
            action=self._actions[index],
            value=self._values[index],
            policy=self._policies[index],
        )

    def set_rewards(self):
        """
        Optimized reward setting. Because self._rewards is a simple list,
        we can iterate over it extremely fast without object overhead.
        """
        if self.length == 0:
            return

        final_reward_vector = self._rewards[-1]

        # Vectorized-style logic is easier here
        for i in reversed(range(self.length)):
            if self._rewards[i] is not None:
                # Direct list access is faster than object attribute access
                self._rewards[i] = final_reward_vector[i % self.num_players]

    def get_batch(self):
        """
        The Scaling Win: Convert columns to Numpy/Tensors instantly.
        This is O(1) complexity overhead vs O(N) for list-of-objects.
        """
        return {
            "obs": np.array(self._obs),
            "actions": np.array(self._actions),
            "rewards": np.array(self._rewards, dtype=np.float32),
            "values": np.array(self._values, dtype=np.float32),
        }

    def __len__(self):
        return self.length
