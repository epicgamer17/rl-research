from .game_config import GameConfig


class MississippiMarblesConfig(GameConfig):
    def __init__(self):
        super(MississippiMarblesConfig, self).__init__(
            max_score=20000,  # technically infinite, but this is a good enough approximation
            min_score=0,
            is_discrete=True,
            is_image=False,
            is_deterministic=False,
            has_legal_moves=True,
            perfect_information=False,
            multi_agent=True,
            num_players=2,  # could be more
            has_intermediate_rewards=False,
        )
