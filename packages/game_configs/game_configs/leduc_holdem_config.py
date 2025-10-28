from .game_config import GameConfig


class LeducHoldemConfig(GameConfig):
    def __init__(self, make_env=None):
        super(LeducHoldemConfig, self).__init__(
            max_score=10,
            min_score=-10,
            is_discrete=True,
            is_image=False,
            is_deterministic=False,
            has_legal_moves=False,
            perfect_information=False,
            multi_agent=True,
            num_players=2,
            # has_intermediate_rewards=False,
            make_env=make_env,
        )
