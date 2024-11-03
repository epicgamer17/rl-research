import numpy as np
from replay_buffers.base_replay_buffer import Game, BaseGameReplayBuffer


class AlphaZeroReplayBuffer(BaseGameReplayBuffer):
    def sample(self):
        game_indices = super().sample()
        print("Game Indices", game_indices)
        return dict(
            observations=np.array(
                [game.observation_history[i] for game, i in game_indices]
            ),
            policies=np.array([game.policy_history[i] for game, i in game_indices]),
            rewards=np.array([[game.rewards[i]] for game, i in game_indices]),
            infos=[game.info_history[i] for game, i in game_indices],
        )
