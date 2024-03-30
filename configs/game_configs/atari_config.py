from configs.game_configs.game_config import GameConfig


class AtariConfig(GameConfig):
    def __init__(self):
        super(AtariConfig, self).__init__(
            max_score=None,
            min_score=0,
            is_discrete=True,
            is_image=True,
            is_deterministic=False,  # if no frameskip, then deterministic
            has_legal_moves=False,
        )
