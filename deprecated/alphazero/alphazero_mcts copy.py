from copy import deepcopy
from math import log, sqrt, inf
import copy
import numpy as np


class Node:
    def __init__(self, prior_policy, state, info):
        self.visits = 0
        self.to_play = -1
        # print("Prior Policy", prior_policy)
        self.prior_policy = prior_policy
        self.value_sum = 0
        self.children = {}
        self.state = copy.deepcopy(state)
        self.info = info

    def expand(self, policy, env):
        # print("Expanding")
        base_env = copy.deepcopy(env)
        # print("MCTS Policy", policy)
        # print("MCTS legal moves", self.info["legal_moves"])
        policy = {a: policy[a] for a in self.info["legal_moves"]}
        policy_sum = sum(policy.values())
        # print("Legal Policy", policy)
        # print("Legal Moves", self.info["legal_moves"])
        # print("Policy Sum", policy_sum)
        for action, p in policy.items():
            # print("Action", action)
            env = copy.deepcopy(base_env)
            child_state, reward, terminated, truncated, child_info = env.step(action)
            # Create Children Nodes (New Leaf Nodes)
            self.children[action] = Node(p / policy_sum, child_state, child_info)

        env.close()
        base_env.close()
        del env
        del base_env

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits

    def add_noise(self, dirichlet_alpha, exploration_fraction):
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        for a, n in zip(actions, noise):
            self.children[a].prior_policy = (1 - exploration_fraction) * self.children[
                a
            ].prior_policy + exploration_fraction * n

    def select_child(self, pb_c_base, pb_c_init):
        # Select the child with the highest UCB
        _, action, child = max(
            [
                (self.child_ucb_score(child, pb_c_base, pb_c_init), action, child)
                for action, child in self.children.items()
            ]
        )
        # print("Selected Action", action)
        # print("Selected Child Visits", child.visits)
        # print("Selected Child Value", child.value())
        # print("Selected Child Prior Policy", child.prior_policy)
        # print("Selected Child UCB", self.child_ucb_score(child, pb_c_base, pb_c_init))

        # for a, c in self.children.items():
        #     print("Child Action", a)
        #     print("Child Visits", c.visits)
        #     print("Child Value", c.value())
        #     print("Child Prior Policy", c.prior_policy)
        #     print("Child UCB", self.child_ucb_score(c, pb_c_base, pb_c_init))

        return action, child

    def child_ucb_score(self, child, pb_c_base, pb_c_init):
        pb_c = log((self.visits + pb_c_base + 1) / pb_c_base) + pb_c_init
        pb_c *= sqrt(self.visits) / (child.visits + 1)

        prior_score = pb_c * child.prior_policy * sqrt(self.visits) / (child.visits + 1)
        value_score = child.value()
        return prior_score + value_score
