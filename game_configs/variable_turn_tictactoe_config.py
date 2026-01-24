from .game_config import GameConfig
import sys

sys.path.append("../../")
from wrappers import (
    ActionMaskInInfoWrapper,
    ChannelLastToFirstWrapper,
    FrameStackWrapper,
    TwoPlayerPlayerPlaneWrapper,
)
from custom_gym_envs.envs import VariableTurnTicTacToeEnv


def make_env(render_mode="rgb_array", min_moves=2, max_moves=2):
    env = VariableTurnTicTacToeEnv(render_mode=render_mode, min_moves=min_moves, max_moves=max_moves)
    env = ActionMaskInInfoWrapper(env)
    env = FrameStackWrapper(env, 4, channel_first=False)
    env = TwoPlayerPlayerPlaneWrapper(env, channel_first=False)
    env = ChannelLastToFirstWrapper(env)
    return env


class VariableTurnTicTacToeConfig(GameConfig):
    def __init__(self, make_env=make_env):
        super(VariableTurnTicTacToeConfig, self).__init__(
            max_score=1,
            min_score=-1,
            is_discrete=True,
            is_image=True,
            is_deterministic=False,
            has_legal_moves=True,
            perfect_information=False,
            multi_agent=True,
            num_players=2,
            make_env=make_env,
        )
