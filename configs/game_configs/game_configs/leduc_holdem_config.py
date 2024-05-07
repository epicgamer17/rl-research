from .game_config import GameConfig


class LeducHoldemConfig(GameConfig):
    def __init__(self):
        super(LeducHoldemConfig, self).__init__(
            max_score=10,
            min_score=-10,
            is_discrete=True,
            is_image=False,
            is_deterministic=False,
            has_legal_moves=False,
            perfect_information=False,
            multi_agent=True,
        )
