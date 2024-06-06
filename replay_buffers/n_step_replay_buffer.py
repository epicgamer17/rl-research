import numpy as np
from collections import deque

from replay_buffers.base_replay_buffer import BaseDQNReplayBuffer, BaseReplayBuffer


class NStepReplayBuffer(BaseDQNReplayBuffer):
    def __init__(
        self,
        observation_dimensions: tuple,
        observation_dtype: np.dtype,
        max_size: int,
        batch_size: int = 32,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        self.n_step = n_step
        self.gamma = gamma
        super().__init__(
            observation_dimensions,
            observation_dtype,
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
        """Store a (s_t, a, r, s_t+1) transtion to the replay buffer.
           Returns a valid generated n-step transition (s_t-n, a, r, s_t) with the
           inputted observation as the next_observation (s_t)

        Returns:
            (s_t-n, a, r, s_t): where r is the n-step return calculated with the replay buffer's gamma
        """
        transition = (observation, action, reward, next_observation, done)
        # print("store t:", transition)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return None

        # compute n-step return and store
        reward, next_observation, done = self._get_n_step_info()
        observation, action = self.n_step_buffer[0][:2]
        n_step_transition = (observation, action, reward, next_observation, done)
        super().store(*n_step_transition, id=id, legal_moves=legal_moves)
        return n_step_transition

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
