from copy import deepcopy
from math import log, sqrt, inf
import copy
import math
import numpy as np
import torch


class Node:
    def __init__(self, prior_policy, parent=None):
        self.visits = 0
        self.to_play = -1
        self.prior_policy = prior_policy
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
        self.parent = parent

        self.root_score = None
        self.network_policy = None  # dense policy vector (numpy or torch)
        self.network_value = None  # network scalar value estimate (float)

    def expand(self, legal_moves, to_play, policy, hidden_state, reward, value=None):
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        self.network_policy = policy.detach().cpu()

        self.network_value = value
        legal_policy = {a: policy[a] for a in legal_moves}
        legal_policy_sum = sum(legal_policy.values())

        for action, p in legal_policy.items():
            self.children[action] = Node((p / (legal_policy_sum + 1e-10)).item(), self)

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

    def select_child(
        self,
        min_max_stats,
        pb_c_base,
        pb_c_init,
        discount,
        num_players,
        cvisit,
        cscale,
        allowed_actions=None,
        gumbel=False,
    ):
        actions = list(self.children.keys())
        if allowed_actions is not None and self.parent is None:
            actions = [a for a in allowed_actions]

        if gumbel and self.parent != None:
            # grab probabilities for candidate actions
            visits = torch.tensor([float(self.children[a].visits) for a in actions])
            sum_N = float(visits.sum())

            # visited action list indices
            visited_mask = visits > 0
            visited_idxs = np.where(visited_mask)[0]

            # q(a) for visited actions (use child.value() empirical)
            q_vals = torch.zeros(len(actions))
            for i, a in enumerate(actions):
                if self.children[a].visits > 0:
                    q_vals[i] = float(self.children[a].value())
                else:
                    q_vals[i] = 0.0

            p_vis_sum = float(
                self.network_policy[visited_mask].sum()
            )  # pi mass on visited actions
            expected_q_vis = float(
                (self.network_policy * q_vals).sum()
            )  # sum_pi(a) * q(a) but q(a)=0 for unvisited
            term = sum_N * (p_vis_sum * expected_q_vis)
            v_mix = (self.network_value + term) / (1.0 + sum_N)
            # completedQ: visited keep q(a), unvisited get v_mix
            completedQ = torch.full((len(actions),), v_mix)
            for i in visited_idxs:
                completedQ[i] = q_vals[i]

            parent_perspective = torch.zeros(len(actions))
            for i, a in enumerate(actions):
                # child.reward (if child exists) else 0
                child = self.children.get(a, None)
                r = float(child.reward) if child is not None else 0.0
                # child.value() if visited else v_mix
                q_est = completedQ[i]
                # sign = +1 if child.to_play == self.to_play else -1 (multi-agent). For single-player it's +1.
                if child is not None:
                    child_to_play = child.to_play
                else:
                    # if child not created, assume alternating or same player? fallback to assuming same player
                    child_to_play = self.to_play
                if num_players == 1:
                    sign = 1.0
                else:
                    sign = 1.0 if child_to_play == self.to_play else -1.0
                parent_perspective[i] = r + discount * (sign * q_est)

            # normalize each parent_perspective via min_max_stats
            normalized_parent = torch.tensor(
                [min_max_stats.normalize(float(x)) for x in parent_perspective]
            )

            # compute sigma: (cvisit + max_visits) * cscale * normalized_parent
            max_visits = (
                max([ch.visits for ch in self.children.values()])
                if len(self.children) > 0
                else 0
            )
            sigma = (cvisit + max_visits) * cscale * normalized_parent

            # combine logits (network) and sigma to get pi0 (only inside candidate_actions)
            eps = 1e-12
            logits = torch.log(self.network_policy + eps)
            pi0_logits = logits + sigma
            # softmax
            # exp_logits = np.exp(pi0_logits - np.max(pi0_logits))
            # pi0 = exp_logits / (exp_logits.sum() + eps)
            pi0 = torch.softmax(pi0_logits, dim=0)

            # compute selection metric: pi0(a) - N(a) / (1 + sum_N)
            denom = 1.0 + sum_N
            selection_scores = pi0 - (visits / denom)

            # pick argmax (random tie-break among maxima)
            max_score = float(selection_scores.max())
            candidate_indices = np.where(np.isclose(selection_scores, max_score))[0]
            chosen_idx = np.random.choice(candidate_indices)
            action = actions[int(chosen_idx)]

        else:
            child_ucbs = [
                self.child_score(
                    self.children[action],
                    min_max_stats,
                    pb_c_base,
                    pb_c_init,
                    discount,
                    num_players,
                    cvisit,
                    cscale,
                    gumbel=gumbel,
                )
                for action in actions
            ]
            action_index = np.random.choice(
                np.where(np.isclose(child_ucbs, max(child_ucbs)))[0]
            )
            action = list(actions)[action_index]
        return action, self.children[action]

    def child_score(
        self,
        child,
        min_max_stats,
        pb_c_base,
        pb_c_init,
        discount,
        num_players,
        cvisit,
        cscale,
        gumbel,
    ):
        if gumbel:
            if self.parent is None:
                # compute normalized q across current root children
                # q_vals = [ch.value() for ch in self.children.values()]
                # max_q = max(q_vals) if len(q_vals) > 0 else 0.0
                # min_q = min(q_vals) if len(q_vals) > 0 else 0.0
                # range_q = max(1e-6, max_q - min_q)
                # normalized_q = (child.value() - min_q) / range_q
                if num_players == 1:
                    sign = 1.0
                else:
                    sign = 1.0 if child.to_play == self.to_play else -1.0

                # parent_value_contrib = child.reward + discount * (sign * child.value())
                parent_value_contrib = child.reward + discount * (sign * child.value())

                # normalize using MinMaxStats (preferred over local min/max)
                normalized_q = min_max_stats.normalize(parent_value_contrib)

                max_visits = (
                    max([ch.visits for ch in self.children.values()])
                    if len(self.children) > 0
                    else 0
                )
                sigma = (cvisit + max_visits) * cscale * normalized_q
                return float(child.root_score + sigma)
            else:
                # existing PUCT-like scoring (unchanged)
                pb_c = log((self.visits + pb_c_base + 1) / pb_c_base) + pb_c_init
                pb_c *= sqrt(self.visits) / (child.visits + 1)

                prior_score = pb_c * child.prior_policy
                if child.visits > 0:
                    if num_players == 1:
                        sign = 1.0
                    else:
                        sign = 1.0 if child.to_play == self.to_play else -1.0

                    value_score = min_max_stats.normalize(
                        child.reward + discount * (sign * child.value())
                    )
                else:
                    value_score = 0.0
        else:

            pb_c = log((self.visits + pb_c_base + 1) / pb_c_base) + pb_c_init
            pb_c *= sqrt(self.visits) / (child.visits + 1)

            prior_score = pb_c * child.prior_policy
            if child.visits > 0:
                if num_players == 1:
                    sign = 1.0
                else:
                    sign = 1.0 if child.to_play == self.to_play else -1.0

                value_score = min_max_stats.normalize(
                    child.reward + discount * (sign * child.value())
                )
            else:
                value_score = 0.0

            # check if value_score is nan
        assert (
            value_score == value_score
        ), "value_score is nan, child value is {}, and reward is {},".format(
            child.value(),
            child.reward,
        )
        assert prior_score == prior_score, "prior_score is nan"
        return prior_score + value_score
