from copy import deepcopy
from math import log, sqrt, inf
import copy
import numpy as np


class Node:
    def __init__(self, prior_policy, env, state, legal_moves):
        self.visits = 0
        self.to_play = -1
        self.prior_policy = prior_policy
        self.value_sum = 0
        self.children = {}
        self.env = env
        self.state = state
        self.legal_moves = legal_moves

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits
