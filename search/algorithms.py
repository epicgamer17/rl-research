from abc import ABC, abstractmethod
import torch
import math

from modules.utils import support_to_scalar
from search.muzero_mcts import ChanceNode, DecisionNode
from search.muzero_minmax_stats import MinMaxStats
from utils.utils import get_legal_moves


class SearchAlgorithm(ABC):
    def __init__(self, config, device, num_actions):
        self.config = config
        self.device = device
        self.num_actions = num_actions

    @abstractmethod
    def run(
        self,
        state,
        info,
        to_play,
        inference_fns,
        trajectory_action=None,
        inference_model=None,
    ):
        pass

    @abstractmethod
    def _expand_root(self):
        pass

    @abstractmethod
    def _inject_trajectory_action(self):
        pass

    @abstractmethod
    def _select_initial_action_set(self):
        pass

    def _set_node_configs(self):
        ChanceNode.estimation_method = self.config.q_estimation_method
        ChanceNode.discount = self.config.discount_factor
        ChanceNode.value_prefix = self.config.value_prefix
        DecisionNode.estimation_method = self.config.q_estimation_method
        DecisionNode.discount = self.config.discount_factor
        DecisionNode.value_prefix = self.config.value_prefix
        DecisionNode.pb_c_init = self.config.pb_c_init
        DecisionNode.pb_c_base = self.config.pb_c_base
        DecisionNode.gumbel = self.config.gumbel
        DecisionNode.cvisit = self.config.gumbel_cvisit
        DecisionNode.cscale = self.config.gumbel_cscale
        DecisionNode.stochastic = self.config.stochastic


class UCTSearch(SearchAlgorithm):
    def run(
        self,
        state,
        info,
        to_play,
        inference_fns,
        trajectory_action=None,
        inference_model=None,
    ):
        self._set_node_configs()
        root = DecisionNode(0.0)

        # TODO: add sample muzero for complex action spaces (right now just use gumbel)
        self._expand_root(
            root, state, info, to_play, inference_fns, inference_model=inference_model
        )

        if trajectory_action is None or self.config.reanalyze_noise:
            root.add_noise(
                self.config.root_dirichlet_alpha,
                self.config.root_exploration_fraction,
            )

        if trajectory_action is not None:
            self._inject_trajectory_action(root, trajectory_action)

        min_max_stats = MinMaxStats(self.config.known_bounds)

        for _ in range(self.config.num_simulations):
            self._run_single_simulation(
                root,
                min_max_stats,
                inference_fns,
                inference_model=inference_model,
                allowed_actions=None,
            )

        visit_counts = torch.tensor(
            [child.visits for action, child in root.children.items()]
        )
        actions = [action for action, child in root.children.items()]

        policy = torch.zeros(self.num_actions)
        policy[actions] = visit_counts / torch.sum(visit_counts)

        return (
            root.value(),
            policy,
            policy,
            torch.argmax(policy),
        )

    def _expand_root(self, root, state, info, to_play, inference_fns, inference_model):
        _, policy, hidden_state = inference_fns["initial"](
            state,
            model=inference_model,
        )
        policy = policy[0]
        reward_h_state = torch.zeros(1, 1, self.config.lstm_hidden_size).to(self.device)
        reward_c_state = torch.zeros(1, 1, self.config.lstm_hidden_size).to(self.device)

        legal_moves = get_legal_moves(info)[0]  # [0]
        if legal_moves is None:
            legal_moves = list(range(self.num_actions))
            to_play = 0
        root.visits += 1

        root.expand(
            legal_moves,
            to_play,
            policy,
            hidden_state,
            0.0,
            reward_h_state=reward_h_state,
            reward_c_state=reward_c_state,
            is_reset=True,
        )

    def _inject_trajectory_action(self, root, trajectory_action):
        # ensure action exists as child
        assert (
            trajectory_action in root.children
        ), f"trajectory_action not in root.children, make sure if there is one it is garaunteed to be in the sampled actions, trajectory action: {trajectory_action}, root.children: {root.children}"
        inject_frac = self.config.injection_frac  # 0.25 as paper used
        # renormalize priors: put (1-inject_frac) of current mass on existing priors, add inject_frac on the trajectory action
        # compute sum of current priors
        total_prior = sum(child.prior_policy for child in root.children.values())
        for a, child in root.children.items():
            child.prior_policy = (1.0 - inject_frac) * (
                child.prior_policy / total_prior
            )
        # boost injected action
        root.children[trajectory_action].prior_policy += inject_frac

    def _run_single_simulation(
        self,
        root: DecisionNode,
        min_max_stats: MinMaxStats,
        inference_fns,
        inference_model=None,
        allowed_actions=None,
    ):
        node = root
        search_path = [node]
        to_play = root.to_play
        horizon_index = 0
        # old_to_play = to_play
        # GO UNTIL A LEAF NODE IS REACHED
        # while node.expanded():
        #     action, node = node.select_child(
        #         min_max_stats=min_max_stats,
        #         allowed_actions=allowed_actions,
        #     )
        #     # old_to_play = (old_to_play + 1) % self.config.game.num_players
        #     search_path.append(node)
        #     horizon_index = (horizon_index + 1) % self.config.lstm_horizon_len
        # ---------------------------------------------------------------------
        # 1. SELECTION PHASE
        # ---------------------------------------------------------------------
        # We descend until we hit a leaf DecisionNode OR a ChanceNode that needs a new code.
        while True:
            if not node.expanded():
                break  # Reached a leaf state (DecisionNode)
            if isinstance(node, DecisionNode):
                # Decision -> Select Action -> ChanceNode
                action, node = node.select_child(
                    min_max_stats=min_max_stats,
                    allowed_actions=allowed_actions,
                )
                horizon_index = (horizon_index + 1) % self.config.lstm_horizon_len

            elif isinstance(node, ChanceNode):
                # Chance -> Select Code -> DecisionNode

                code, node = node.select_child(
                    # TODO: Gumbel Top-K on chance nodes?
                )
            search_path.append(node)

        parent = search_path[-2]
        # if to_play != old_to_play and self.training_step > 1000:
        #     print("WRONG TO PLAY", onehot_to_play)
        if isinstance(node, DecisionNode):
            if isinstance(parent, DecisionNode):
                (
                    reward,
                    hidden_state,
                    value,
                    policy,
                    to_play,
                    reward_h_state,
                    reward_c_state,
                ) = inference_fns["recurrent"](
                    parent.hidden_state,
                    torch.tensor(action).to(parent.hidden_state.device).unsqueeze(0),
                    parent.reward_h_state,
                    parent.reward_c_state,
                    model=inference_model,
                )
                if self.config.support_range is not None:
                    reward = support_to_scalar(reward, self.config.support_range).item()
                    value = support_to_scalar(value, self.config.support_range).item()
                else:
                    reward = reward.item()
                    value = value.item()

                # onehot_to_play = to_play
                to_play = int(to_play.argmax().item())
                is_reset = horizon_index == 0
                if self.config.value_prefix and is_reset:
                    reward_h_state = torch.zeros_like(reward_h_state).to(self.device)
                    reward_c_state = torch.zeros_like(reward_c_state).to(self.device)

                node.expand(
                    list(range(self.num_actions)),
                    to_play,
                    policy[0],
                    hidden_state,
                    reward,
                    value=value,
                    reward_h_state=reward_h_state,
                    reward_c_state=reward_c_state,
                    is_reset=is_reset,
                )
            elif isinstance(parent, ChanceNode):
                # assert (
                #     node.value_prefix == False
                # ), "value prefix not implemented with chance nodes"
                # TODO: make value prefix work with chance nodes
                # print("code before recurrent inference", code.shape)
                (
                    reward,
                    hidden_state,
                    value,
                    policy,
                    to_play,
                    reward_h_state,
                    reward_c_state,
                ) = inference_fns["recurrent"](
                    parent.afterstate,
                    code.to(parent.afterstate.device)
                    .unsqueeze(0)
                    .float(),  # a sampled code instead of an action
                    parent.reward_h_state,
                    parent.reward_c_state,
                    model=inference_model,
                )
                if self.config.support_range is not None:
                    reward = support_to_scalar(reward, self.config.support_range).item()
                    value = support_to_scalar(value, self.config.support_range).item()
                else:
                    reward = reward.item()
                    value = value.item()

                # onehot_to_play = to_play
                to_play = int(to_play.argmax().item())
                is_reset = horizon_index == 0
                if self.config.value_prefix and is_reset:
                    reward_h_state = torch.zeros_like(reward_h_state).to(self.device)
                    reward_c_state = torch.zeros_like(reward_c_state).to(self.device)

                node.expand(
                    list(range(self.num_actions)),
                    to_play,
                    policy[0],
                    hidden_state,
                    reward,
                    value=value,
                    reward_h_state=reward_h_state,
                    reward_c_state=reward_c_state,
                    is_reset=is_reset,
                )
        elif isinstance(node, ChanceNode):
            # CASE B: Stochastic Expansion (The Core Change)
            # We are at (State, Action). We need to:
            # 1. Get Afterstate Value & Code Priors (Expand ChanceNode)
            # 2. Sample a Code
            # 3. Get Next State & Reward (Create DecisionNode)
            afterstate, value, code_priors = inference_fns[
                "afterstate"
            ](  # <--- YOU NEED THIS METHOD
                parent.hidden_state,
                torch.tensor(action)
                .to(parent.hidden_state.device)
                .unsqueeze(0)
                .float(),
                model=inference_model,
            )

            if self.config.support_range:
                value = support_to_scalar(value, self.config.support_range).item()
            else:
                value = value.item()

            # Expand the Chance Node with these priors
            node.expand(
                parent.to_play,
                afterstate,
                value,
                code_priors[0],
                reward_h_state=parent.reward_h_state,
                reward_c_state=parent.reward_c_state,
            )

        n = len(search_path)
        if n == 0:
            return []

        # --- 1) Build per-player accumulator array acc[p] = Acc_p(i) for current i (starting from i = n-1) ---
        # Acc_p(i) definition: discounted return from node i for a node whose player is p:
        # Acc_p(i) = sum_{j=i+1..n-1} discount^{j-i-1} * sign(p, j) * reward_j
        #            + discount^{n-1-i} * sign(p, leaf) * leaf_value
        # Where sign(p, j) = +1 if acting_player_at_j (which is search_path[j-1].to_play) == p else -1.
        #
        # We compute Acc_p(n-1) = sign(p, leaf) * leaf_value as base, then iterate backward:
        # Acc_p(i-1) = s(p, i) * reward_i + discount * Acc_p(i)

        # Initialize acc for i = n-1 (base: discounted exponent 0 for leaf value)
        # acc is a Python list of floats length num_players
        acc = [0.0] * self.config.game.num_players
        for p in range(self.config.game.num_players):
            acc[p] = value if to_play == p else -value

        # totals[i] will hold Acc_{node_player}(i)
        totals = [0.0] * n
        # Iterate from i = n-1 down to 0
        for i in range(n - 1, -1, -1):
            node = search_path[i]
            node_player = node.to_play
            # totals for this node = acc[node_player] (current Acc_p(i))
            # print(totals[i])
            # print(acc[node_player])
            totals[i] = acc[node_player]

            node.value_sum += totals[i]
            node.visits += 1

            # Prepare acc for i-1 (if any)
            if i > 0:
                # reward at index i belongs to acting_player = search_path[i-1].to_play
                acting_player = search_path[i - 1].to_play
                if isinstance(search_path[i], DecisionNode):
                    r_i = search_path[i - 1].child_reward(search_path[i])

                    # # Update per-player accumulators in O(num_players)
                    # # Acc_p(i-1) = sign(p, i) * r_i + discount * Acc_p(i)
                    # # sign(p, i) = +1 if acting_player == p else -1
                    # # We overwrite acc[p] in-place to be Acc_p(i-1)
                    for p in range(self.config.game.num_players):
                        sign = 1.0 if acting_player == p else -1.0
                        acc[p] = sign * r_i + self.config.discount_factor * acc[p]
                elif isinstance(search_path[i], ChanceNode):
                    for p in range(self.config.game.num_players):
                        # sign = 1.0 if acting_player == p else -1.0
                        # acc[p] = sign * r_i + self.config.discount_factor * acc[p]
                        # chance nodes can be thought to have 0 reward, and no discounting (as its like the roll after the action, or another way of thinking of it is that only on decision nodes do we discount expected reward, a chance node is not a decision point)
                        acc[p] = acc[p]
                child_q = search_path[i - 1].get_child_q_from_parent(search_path[i])
                min_max_stats.update(child_q)
            else:
                min_max_stats.update(search_path[i].value())

    def _select_initial_action_set(self):
        # TODO: FOR SAMPLE MUZERO
        pass


class GumbelSequentialHalving(SearchAlgorithm):
    def run(
        self,
        state,
        info,
        to_play,
        inference_fns,
        trajectory_action=None,
        inference_model=None,
    ):
        self._set_node_configs()
        root = DecisionNode(0.0)

        v_pi_raw, policy, hidden_state = inference_fns["initial"](
            state,
            model=inference_model,
        )
        policy = policy[0]
        reward_h_state = torch.zeros(1, 1, self.config.lstm_hidden_size).to(self.device)
        reward_c_state = torch.zeros(1, 1, self.config.lstm_hidden_size).to(self.device)

        if self.config.support_range is not None:
            # support_to_scalar expects a support vector and returns a scalar tensor
            v_pi = support_to_scalar(v_pi_raw, self.config.support_range)
            # ensure it's a Python float where you use .item() later
            v_pi_scalar = float(v_pi.item())
        else:
            v_pi_scalar = float(v_pi_raw.item())

        # if self.config.game.num_players != 1:
        #     legal_moves = get_legal_moves(info)[0]  # [0]
        #     # print(legal_moves)
        # else:
        #     legal_moves = list(range(self.num_actions))
        #     to_play = 0
        legal_moves = get_legal_moves(info)[0]  # [0]
        if legal_moves is None:
            legal_moves = list(range(self.num_actions))
            to_play = 0

        # print("traj_action", trajectory_action)
        # print("legal moves", legal_moves)
        root.visits += 1

        if self.config.gumbel:
            actions = legal_moves  # list of ints
            # policy is a tensor/prob vector for all actions (shape num_actions)
            if trajectory_action is not None:
                inject_frac = self.config.injection_frac  # 0.25 as paper used
                # renormalize priors: put (1-inject_frac) of current mass on existing priors, add inject_frac on the trajectory action
                # compute sum of current priors
                for i, prior in enumerate(policy):
                    if actions[i] == trajectory_action:
                        policy[i] += 0.25
                    else:
                        policy[i] *= 1 - inject_frac
                assert policy.sum() == torch.ones_like(policy.sum())
            logits = torch.log(policy + 1e-12).cpu()  # use numpy for gumbel math

            # legal_moves is the list of available actions

            # --- Gumbel sampling ---
            k = len(actions)
            m = min(
                self.config.gumbel_m, k
            )  # add config param gumbel_m, e.g., min(n,16)
            # sample Gumbel noise
            g = -torch.log(-torch.log(torch.rand(k)))  # shape (k,)
            # compute g + logits (only for legal actions)
            scores = g + logits[actions]
            # find top-m indices (indices into 'actions')
            top_idx = torch.argsort(scores, descending=True)[:m]
            sampled_actions = [actions[i] for i in top_idx]
            if trajectory_action is None:
                sampled_g_values = {
                    actions[i]: float(g[i] + logits[actions[i]]) for i in top_idx
                }
            else:
                assert (
                    trajectory_action in actions
                ), f"trajectory_action {trajectory_action} not in legal actions {actions}"
                if trajectory_action not in sampled_actions:
                    sampled_actions += [trajectory_action]

                    top_idx = torch.concat(
                        (
                            top_idx,
                            torch.tensor([actions.index(trajectory_action)]),
                        )
                    )

                if self.config.reanalyze_noise:
                    sampled_g_values = {
                        actions[i]: float(g[i] + logits[actions[i]]) for i in top_idx
                    }
                else:
                    sampled_g_values = {
                        actions[i]: float(logits[actions[i]]) for i in top_idx
                    }
                assert trajectory_action in actions

            # expand root with only sampled_actions; pass in the per-child root_score
            root.expand(
                sampled_actions,
                to_play,
                policy,
                hidden_state,
                0.0,
                value=v_pi_scalar,
                reward_h_state=reward_h_state,
                reward_c_state=reward_c_state,
                is_reset=True,
            )

            # attach the root_score to the created children
            for a in sampled_actions:
                root.children[a].root_score = sampled_g_values[a]  # store Gumbel+logit

        min_max_stats = MinMaxStats(self.config.known_bounds)

        best_action = self.sequential_halving(
            root,
            min_max_stats,
            list(root.children.keys()),
            inference_fns,
            inference_model=inference_model,
        )

        visit_counts = torch.tensor(
            [child.visits for action, child in root.children.items()]
        )
        actions = [action for action, child in root.children.items()]
        policy = torch.zeros(self.num_actions)
        policy[actions] = visit_counts / torch.sum(visit_counts)
        target_policy = root.get_gumbel_improved_policy(min_max_stats).to(self.device)

        return (
            root.value(),
            policy,
            target_policy,
            torch.tensor(best_action),
        )

    def sequential_halving(
        self,
        root: DecisionNode,
        min_max_stats: MinMaxStats,
        candidates: list,
        inference_fns,
        inference_model=None,
    ):
        """
        Perform Sequential Halving among `candidates` (list of action ints).
        It splits self.config.num_simulations across rounds that eliminate ~half of candidates each round.
        Survivors remain in root.children and accumulate visits/values as usual.
        """

        # Helper function for sorting scores by value (used for both priority and elimination)
        def sort_by_score(item):
            # item is a tuple (action, score)
            return item[1]

        m = len(candidates)

        # number of rounds to reduce to 1 (ceil log2)
        survivors = candidates.copy()
        scores = []
        for a in survivors:
            child = root.children[a]
            scores.append((a, root.get_gumbel_root_child_score(child, min_max_stats)))
        scores.sort(key=sort_by_score, reverse=True)
        survivors = [a for a, _ in scores]

        sims_used = 0
        while sims_used < self.config.num_simulations:
            if len(survivors) > 2:
                # TODO: should this be a min of 1 visit per thing per round?
                sims_this_round = max(
                    1,
                    math.floor(
                        self.config.num_simulations / (math.log2(m) * (len(survivors)))
                    ),
                ) * len(survivors)

            else:
                sims_this_round = self.config.num_simulations - sims_used

            if sims_used + sims_this_round > self.config.num_simulations:
                sims_this_round = self.config.num_simulations - sims_used
            # run sims_per_round simulations restricted to current survivors
            sims_used += sims_this_round
            for i in range(sims_this_round):
                # The modulo operation cycles through the sorted survivors list
                action = survivors[i % len(survivors)]

                # Run a single simulation, but ONLY allow the current `action` to be selected at the root
                self._run_single_simulation(
                    root,
                    min_max_stats,
                    inference_fns,
                    inference_model=inference_model,
                    allowed_actions=[action],
                )  # recompute a score per survivor for elimination (use visits primarily)

            scores = []
            for a in survivors:
                child = root.children[a]
                scores.append(
                    (a, root.get_gumbel_root_child_score(child, min_max_stats))
                )
            scores.sort(key=sort_by_score, reverse=True)
            survivors = [a for a, _ in scores]
            # print("scores", scores)
            # print("survivors", survivors)

            num_to_eliminate = math.ceil(len(survivors) / 2.0)
            # leave 2 survivors
            if len(survivors) - num_to_eliminate < 2:
                # num_to_eliminate = len(survivors) - 2
                # print(num_to_eliminate)
                survivors = survivors[:2]
            else:
                survivors = survivors[:-num_to_eliminate]
            # eliminated = [a for a, _ in scores[:num_to_eliminate]]
            # survivors = [a for a in survivors if a not in eliminated]

            # print("survivors after elimination", survivors)
            # print(survivors, scores)

        # return survivors[0]
        final_scores = []
        for a in survivors:
            child = root.children[a]
            final_scores.append(
                (a, root.get_gumbel_root_child_score(child, min_max_stats))
            )

        final_scores.sort(key=sort_by_score, reverse=True)

        # Return the BEST action (Index 0), not the worst!
        return final_scores[0][0]

    def _run_single_simulation(
        self,
        root: DecisionNode,
        min_max_stats: MinMaxStats,
        inference_fns,
        inference_model=None,
        allowed_actions=None,
    ):
        node = root
        search_path = [node]
        to_play = root.to_play
        horizon_index = 0
        # old_to_play = to_play
        # GO UNTIL A LEAF NODE IS REACHED
        # while node.expanded():
        #     action, node = node.select_child(
        #         min_max_stats=min_max_stats,
        #         allowed_actions=allowed_actions,
        #     )
        #     # old_to_play = (old_to_play + 1) % self.config.game.num_players
        #     search_path.append(node)
        #     horizon_index = (horizon_index + 1) % self.config.lstm_horizon_len
        # ---------------------------------------------------------------------
        # 1. SELECTION PHASE
        # ---------------------------------------------------------------------
        # We descend until we hit a leaf DecisionNode OR a ChanceNode that needs a new code.
        while True:
            if not node.expanded():
                break  # Reached a leaf state (DecisionNode)
            if isinstance(node, DecisionNode):
                # Decision -> Select Action -> ChanceNode
                action, node = node.select_child(
                    min_max_stats=min_max_stats,
                    allowed_actions=allowed_actions,
                )
                horizon_index = (horizon_index + 1) % self.config.lstm_horizon_len

            elif isinstance(node, ChanceNode):
                # Chance -> Select Code -> DecisionNode

                code, node = node.select_child(
                    # TODO: Gumbel Top-K on chance nodes?
                )
            search_path.append(node)

        parent = search_path[-2]
        # if to_play != old_to_play and self.training_step > 1000:
        #     print("WRONG TO PLAY", onehot_to_play)
        if isinstance(node, DecisionNode):
            if isinstance(parent, DecisionNode):
                (
                    reward,
                    hidden_state,
                    value,
                    policy,
                    to_play,
                    reward_h_state,
                    reward_c_state,
                ) = inference_fns["recurrent"](
                    parent.hidden_state,
                    torch.tensor(action).to(parent.hidden_state.device).unsqueeze(0),
                    parent.reward_h_state,
                    parent.reward_c_state,
                    model=inference_model,
                )
                if self.config.support_range is not None:
                    reward = support_to_scalar(reward, self.config.support_range).item()
                    value = support_to_scalar(value, self.config.support_range).item()
                else:
                    reward = reward.item()
                    value = value.item()

                # onehot_to_play = to_play
                to_play = int(to_play.argmax().item())
                is_reset = horizon_index == 0
                if self.config.value_prefix and is_reset:
                    reward_h_state = torch.zeros_like(reward_h_state).to(self.device)
                    reward_c_state = torch.zeros_like(reward_c_state).to(self.device)

                node.expand(
                    list(range(self.num_actions)),
                    to_play,
                    policy[0],
                    hidden_state,
                    reward,
                    value=value,
                    reward_h_state=reward_h_state,
                    reward_c_state=reward_c_state,
                    is_reset=is_reset,
                )
            elif isinstance(parent, ChanceNode):
                # assert (
                #     node.value_prefix == False
                # ), "value prefix not implemented with chance nodes"
                # TODO: make value prefix work with chance nodes
                # print("code before recurrent inference", code.shape)
                (
                    reward,
                    hidden_state,
                    value,
                    policy,
                    to_play,
                    reward_h_state,
                    reward_c_state,
                ) = inference_fns["recurrent"](
                    parent.afterstate,
                    code.to(parent.afterstate.device)
                    .unsqueeze(0)
                    .float(),  # a sampled code instead of an action
                    parent.reward_h_state,
                    parent.reward_c_state,
                    model=inference_model,
                )
                if self.config.support_range is not None:
                    reward = support_to_scalar(reward, self.config.support_range).item()
                    value = support_to_scalar(value, self.config.support_range).item()
                else:
                    reward = reward.item()
                    value = value.item()

                # onehot_to_play = to_play
                to_play = int(to_play.argmax().item())
                is_reset = horizon_index == 0
                if self.config.value_prefix and is_reset:
                    reward_h_state = torch.zeros_like(reward_h_state).to(self.device)
                    reward_c_state = torch.zeros_like(reward_c_state).to(self.device)

                node.expand(
                    list(range(self.num_actions)),
                    to_play,
                    policy[0],
                    hidden_state,
                    reward,
                    value=value,
                    reward_h_state=reward_h_state,
                    reward_c_state=reward_c_state,
                    is_reset=is_reset,
                )
        elif isinstance(node, ChanceNode):
            # CASE B: Stochastic Expansion (The Core Change)
            # We are at (State, Action). We need to:
            # 1. Get Afterstate Value & Code Priors (Expand ChanceNode)
            # 2. Sample a Code
            # 3. Get Next State & Reward (Create DecisionNode)
            afterstate, value, code_priors = inference_fns[
                "afterstate"
            ](  # <--- YOU NEED THIS METHOD
                parent.hidden_state,
                torch.tensor(action)
                .to(parent.hidden_state.device)
                .unsqueeze(0)
                .float(),
                model=inference_model,
            )

            if self.config.support_range:
                value = support_to_scalar(value, self.config.support_range).item()
            else:
                value = value.item()

            # Expand the Chance Node with these priors
            node.expand(
                parent.to_play,
                afterstate,
                value,
                code_priors[0],
                reward_h_state=parent.reward_h_state,
                reward_c_state=parent.reward_c_state,
            )

        n = len(search_path)
        if n == 0:
            return []

        # --- 1) Build per-player accumulator array acc[p] = Acc_p(i) for current i (starting from i = n-1) ---
        # Acc_p(i) definition: discounted return from node i for a node whose player is p:
        # Acc_p(i) = sum_{j=i+1..n-1} discount^{j-i-1} * sign(p, j) * reward_j
        #            + discount^{n-1-i} * sign(p, leaf) * leaf_value
        # Where sign(p, j) = +1 if acting_player_at_j (which is search_path[j-1].to_play) == p else -1.
        #
        # We compute Acc_p(n-1) = sign(p, leaf) * leaf_value as base, then iterate backward:
        # Acc_p(i-1) = s(p, i) * reward_i + discount * Acc_p(i)

        # Initialize acc for i = n-1 (base: discounted exponent 0 for leaf value)
        # acc is a Python list of floats length num_players
        acc = [0.0] * self.config.game.num_players
        for p in range(self.config.game.num_players):
            acc[p] = value if to_play == p else -value

        # totals[i] will hold Acc_{node_player}(i)
        totals = [0.0] * n
        # Iterate from i = n-1 down to 0
        for i in range(n - 1, -1, -1):
            node = search_path[i]
            node_player = node.to_play
            # totals for this node = acc[node_player] (current Acc_p(i))
            # print(totals[i])
            # print(acc[node_player])
            totals[i] = acc[node_player]

            node.value_sum += totals[i]
            node.visits += 1

            # Prepare acc for i-1 (if any)
            if i > 0:
                # reward at index i belongs to acting_player = search_path[i-1].to_play
                acting_player = search_path[i - 1].to_play
                if isinstance(search_path[i], DecisionNode):
                    r_i = search_path[i - 1].child_reward(search_path[i])

                    # # Update per-player accumulators in O(num_players)
                    # # Acc_p(i-1) = sign(p, i) * r_i + discount * Acc_p(i)
                    # # sign(p, i) = +1 if acting_player == p else -1
                    # # We overwrite acc[p] in-place to be Acc_p(i-1)
                    for p in range(self.config.game.num_players):
                        sign = 1.0 if acting_player == p else -1.0
                        acc[p] = sign * r_i + self.config.discount_factor * acc[p]
                elif isinstance(search_path[i], ChanceNode):
                    for p in range(self.config.game.num_players):
                        # sign = 1.0 if acting_player == p else -1.0
                        # acc[p] = sign * r_i + self.config.discount_factor * acc[p]
                        # chance nodes can be thought to have 0 reward, and no discounting (as its like the roll after the action, or another way of thinking of it is that only on decision nodes do we discount expected reward, a chance node is not a decision point)
                        acc[p] = acc[p]
                child_q = search_path[i - 1].get_child_q_from_parent(search_path[i])
                min_max_stats.update(child_q)
            else:
                min_max_stats.update(search_path[i].value())

    def _expand_root(self):
        pass

    def _inject_trajectory_action(self):
        pass

    def _select_initial_action_set(self):
        pass
