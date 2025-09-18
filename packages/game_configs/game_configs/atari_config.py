from .game_config import GameConfig


class AtariConfig(GameConfig):
    def __init__(self):
        super(AtariConfig, self).__init__(
            max_score=10,  # FROM CATEGORICAL DQN PAPER
            min_score=-10,
            is_discrete=True,
            is_image=True,
            is_deterministic=False,  # if no frameskip, then deterministic
            has_legal_moves=False,
            perfect_information=True,  # although it is not deterministic, it is so close to it that it is considered perfect information
            multi_agent=False,
            num_players=1,
            has_intermediate_rewards=True,
        )
