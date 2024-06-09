from .game_config import GameConfig


class BlackjackConfig(GameConfig):
    def __init__(self):
        super(BlackjackConfig, self).__init__(
            max_score=1,
            min_score=-1,
            is_discrete=True,
            is_image=False,
            is_deterministic=False,
            has_legal_moves=False,
            perfect_information=False,
            multi_agent=False,
        )
