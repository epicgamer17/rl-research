from copy import deepcopy
from math import log, sqrt, inf
import copy
import numpy as np


class Node:
    def __init__(self, prior_policy, state, legal_moves):
        self.visits = 0
        self.to_play = -1
        self.prior_policy = prior_policy
        self.value_sum = 0
        self.children = {}
        self.state = copy.deepcopy(state)
        self.legal_moves = legal_moves

    def expand(self, policy, env):
        print("Expanding")
        base_env = copy.deepcopy(env)
        env = copy.deepcopy(base_env)
        policy = {a: policy[a] for a in self.legal_moves}
        policy_sum = sum(policy.values())

        for action, p in policy.items():
            child_state, reward, terminated, truncated, info = env.step(action)
            child_legal_moves = (
                info["legal_moves"]
                if "legal_moves" in info
                else range(self.num_actions)
            )
            # Create Children Nodes (New Leaf Nodes)
            self.children[action] = Node(p / policy_sum, child_state, child_legal_moves)
            env = copy.deepcopy(base_env)

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits

    def add_noise(self, dirichlet_alpha, exploration_fraction):
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior_policy = (1 - frac) * self.children[
                a
            ].prior_policy + frac * n
