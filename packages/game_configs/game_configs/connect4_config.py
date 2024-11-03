from .game_config import GameConfig


class Connect4Config(GameConfig):
    def __init__(self):
        super(Connect4Config, self).__init__(
            max_score=1,
            min_score=-1,
            is_discrete=True,
            is_image=True,
            is_deterministic=True,
            has_legal_moves=True,
            perfect_information=True,
            multi_agent=True,
            num_players=2,
        )
