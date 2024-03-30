from game_config import GameConfig


class TicTacToeConfig(GameConfig):
    def __init__(self):
        super(TicTacToeConfig, self).__init__(
            max_score=1,
            min_score=-1,
            is_discrete=True,
            is_image=True,
            is_deterministic=True,
            has_legal_moves=True,
        )
