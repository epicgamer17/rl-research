from copy import deepcopy
from math import log, sqrt, inf
import copy


class Node:
    def __init__(self, env, observation, done, parent, parent_action, possible_actions):
        self.env = copy.deepcopy(env)
        self.env.close()
        self.observation = observation
        self.done = done
        self.parent = parent
        self.parent_action = parent_action
        self.children = {}
        self.visits = 0
        self.possible_actions = possible_actions
        self.score = 0

    def return_score(self):
        return self.score

    def set_score(self, score):
        self.score = score

    def create_children(self):
        if self.done:
            return None
        for action in range(self.possible_actions):
            observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            child_possible_actions = (
                info["possible_actions"]
                if "possible_actions" in info
                else self.possible_actions
            )
            self.children[action] = Node(
                self.env, observation, done, self, action, child_possible_actions
            )


