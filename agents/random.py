import numpy as np


class RandomAgent:
    def __init__(self, model_name="random", action_space=None):
        self.model_name = model_name
        self.action_space = action_space

    def predict(self, observation, info, env=None):
        return observation, info

    def select_actions(self, prediction, info):
        if self.action_space is not None:
            return self.action_space.sample()
        return np.random.choice(info["legal_moves"])
