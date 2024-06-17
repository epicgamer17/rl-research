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
        compressed_observations: bool = False,
    ):
        self.n_step = n_step
        self.gamma = gamma
        super().__init__(
            observation_dimensions=observation_dimensions,
            observation_dtype=observation_dtype,
            max_size=max_size,
            batch_size=batch_size,
            compressed_observations=compressed_observations,
        )

    def store(
        self,
        observation,
        info: dict,
        action,
        reward: float,
        next_observation,
        next_info: dict,
        done: bool,
        id=None,
    ):
        """Store a (s_t, a, r, s_t+1) transtion to the replay buffer.
           Returns a valid generated n-step transition (s_t-n, a, r, s_t) with the
           inputted observation as the next_observation (s_t)

        Returns:
            (s_t-n, a, r, s_t): where r is the n-step return calculated with the replay buffer's gamma
        """
        transition = (
            observation,
            info,
            action,
            reward,
            next_observation,
            next_info,
            done,
        )
        # print("store t:", transition)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return None

        # compute n-step return and store
        reward, next_observation, next_info, done = self._get_n_step_info()
        observation, info, action = self.n_step_buffer[0][:3]
        n_step_transition = (
            observation,
            info,
            action,
            reward,
            next_observation,
            next_info,
            done,
        )
        super().store(*n_step_transition, id=id)
        return n_step_transition

    def clear(self):
        super().clear()
        self.n_step_buffer = deque(maxlen=self.n_step)

    def _get_n_step_info(self):
        reward, next_observation, next_info, done = self.n_step_buffer[-1][-4:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, n_i, d = transition[-4:]
            reward = r + self.gamma * reward * (1 - d)
            next_observation, next_info, done = (
                (n_o, n_i, d) if d else (next_observation, next_info, done)
            )

        return reward, next_observation, next_info, done
