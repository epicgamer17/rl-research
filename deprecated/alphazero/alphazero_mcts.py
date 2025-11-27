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
        print("created node with: ")
        print("node state", state)
        print("node info", info)
        self.state = copy.deepcopy(state)
        self.info = copy.deepcopy(info)

    def expand(self, policy, base_env):
        print(self.info["legal_moves"])
        policy = {a: policy[a] for a in self.info["legal_moves"]}
        policy_sum = sum(policy.values())
        print("expanding")
        print(base_env.last())
        for action, p in policy.items():
            print("Action", action)
            env = copy.deepcopy(base_env)
            print("env last", env.last())
            # child_state, reward, terminated, truncated, child_info = env.step(action)
            env.step(action)
            child_state, reward, terminated, truncated, child_info = env.last()
            agent_id = env.agent_selection
            current_player = env.agents.index(agent_id)
            child_state, child_info = process_petting_zoo_obs(
                child_state, child_info, current_player
            )
            # Create Children Nodes (New Leaf Nodes)
            self.children[action] = Node(p / policy_sum, child_state, child_info)

        # env.close()
        del env

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
