from copy import deepcopy
from math import log, sqrt, inf
import copy
import numpy as np


class MCTS:
    def __init__(
        self, env, observation, done, parent, parent_action, legal_moves, reward
    ):
        self.env = copy.deepcopy(env)
        self.env.window = None  # to stop rendering when render mode is human
        self.env.close()
        self.observation = observation
        self.done = done
        self.reward = reward
        self.parent = parent
        self.parent_action = parent_action
        self.children = np.array([None] * len(legal_moves))
        self.visits = 0
        self.legal_moves = legal_moves
        self.score = 0

    def return_score(self):
        return self.score

    def set_score(self, score):
        self.score = score

    def create_children(self):
        for i in range(len(self.legal_moves)):
            child_env = copy.deepcopy(self.env)
            observation, reward, terminated, truncated, info = child_env.step(
                self.legal_moves[i]
            )
            done = terminated or truncated
            # print(info["legal_moves"])
            child_legal_moves = (
                info["legal_moves"] if "legal_moves" in info else self.legal_moves
            )
            self.children[i] = MCTS(
                child_env,
                observation,
                done,
                self,
                self.legal_moves[i],
                child_legal_moves,
                reward,
            )
            # print(self.children[i])
        # print(self.children)
