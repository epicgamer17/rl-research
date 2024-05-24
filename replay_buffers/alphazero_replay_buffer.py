import numpy as np
from replay_buffers.base_replay_buffer import Game, BaseGameReplayBuffer


class AlphaZeroReplayBuffer(BaseGameReplayBuffer):
    def sample(self):
        game_indices = super().sample()

        return dict(
            observations=[game.observation_history[i] for game, i in game_indices],
            rewards=[[game.rewards[i]] for game, i in game_indices],
            policy=[game.policy_history[i] for game, i in game_indices],
        )
