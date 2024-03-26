from game_config import GameConfig


class CartPoleConfig(GameConfig):
    def __init__(self):
        self.max_score = 500
        self.min_score = 0
        self.discrete = True
