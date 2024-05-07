from .game_config import GameConfig


class ClassicControlConfig(GameConfig):
    def __init__(self):
        super(ClassicControlConfig, self).__init__(
            max_score=500,
            min_score=0,
            is_discrete=True,
            is_image=False,
            is_deterministic=True,  # i think it is deterministic (pretty sure if you input the same actions the same thing will happen, it just has a random start state)
            has_legal_moves=False,
            perfect_information=True,
            multi_agent=False,
        )
