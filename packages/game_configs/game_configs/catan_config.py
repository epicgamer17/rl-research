from .game_config import GameConfig


class CatanConfig(GameConfig):
    def __init__(self, make_env=None):
        super(CatanConfig, self).__init__(
            max_score=1,
            min_score=-1,
            is_discrete=True,
            is_image=False,
            is_deterministic=False,
            has_legal_moves=True,
            perfect_information=False,
            multi_agent=True,
            num_players=2,
            make_env=make_env,
        )


class SinglePlayerCatanConfig(GameConfig):
    def __init__(self, make_env=None):
        super(CatanConfig, self).__init__(
            max_score=1,
            min_score=-1,
            is_discrete=True,
            is_image=False,
            is_deterministic=False,
            has_legal_moves=True,
            perfect_information=False,
            multi_agent=False,
            num_players=1,
            make_env=make_env,
        )
