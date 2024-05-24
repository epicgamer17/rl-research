import numpy as np
from collections import deque

from replay_buffers.base_replay_buffer import BaseDQNReplayBuffer, BaseReplayBuffer


class NStepReplayBuffer(BaseDQNReplayBuffer):
    def __init__(
        self,
        observation_dimensions: tuple,
        max_size: int,
        batch_size: int = 32,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        print("NStep Replay Buffer Init")
        self.n_step = n_step
        self.gamma = gamma
        super().__init__(
            observation_dimensions,
            max_size,
            batch_size,
        )

    def store(
        self,
        observation,
        action,
        reward: float,
        next_observation,
        done: bool,
        id=None,
        legal_moves=None,
    ):
        transition = (observation, action, reward, next_observation, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return ()

        # compute n-step return and store
        reward, next_observation, done = self._get_n_step_info()
        observation, action = self.n_step_buffer[0][:2]
        super().store(
            observation,
            action,
            reward,
            next_observation,
            done,
            id=id,
            legal_moves=legal_moves,
        )
        return self.n_step_buffer[0]

    def clear(self):
        super().clear()
        self.n_step_buffer = deque(maxlen=self.n_step)

    def _get_n_step_info(self):
        reward, next_observation, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]
            reward = r + self.gamma * reward * (1 - d)
            next_observation, done = (n_o, d) if d else (next_observation, done)

        return reward, next_observation, done
