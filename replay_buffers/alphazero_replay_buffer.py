import numpy as np
import copy

from replay_buffers.base_replay_buffer import BaseReplayBuffer, Game


class AlphaZeroReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        max_size: int,
        batch_size: int,
    ):
        super().__init__(max_size=max_size, batch_size=batch_size)

    def store(self, game: Game):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(game)
        self.size += 1

    def sample(self):
        move_sum = float(sum([len(game) for game in self.buffer]))
        games: list[Game] = np.random.choice(
            self.buffer,
            self.batch_size,
            p=[len(game) / move_sum for game in self.buffer],
        )

        game_indices: list[int] = [
            (game, np.random.randint(len(game))) for game in games
        ]

        return dict(
            observations=[game.observation_history[i] for game, i in game_indices],
            rewards=[[game.rewards[i]] for game, i in game_indices],
            policy=[game.policy_history[i] for game, i in game_indices],
        )

    def clear(self):
        self.buffer: list[Game] = []
        self.size = 0
