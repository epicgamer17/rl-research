from abc import ABC, abstractmethod
import torch
import numpy as np
import math
import torch.nn.functional as F
from modules.utils import support_to_scalar
from search.action_selectors import (
    GumbelMuZeroSelectionStrategy,
    SamplingSelectionStrategy,
    UCBSelectionStrategy,
)
from search.initial_action_sets import SelectAll, SelectTopK
from search.prior_injectors import (
    ActionTargetInjector,
    DirichletInjector,
    GumbelInjector,
)
from search.nodes import ChanceNode, DecisionNode
from search.min_max_stats import MinMaxStats
from search.root_policies import GumbelRootPolicy, VisitFrequencyPolicy
from utils.utils import action_mask, get_legal_moves
from math import log, sqrt


class SearchAlgorithm(ABC):
    def __init__(self, config, device, num_actions):
        self.config = config
        self.device = device
        self.num_actions = num_actions

        # Default selection strategies
        self.root_selection_strategy = UCBSelectionStrategy()
        self.chance_selection_strategy = SamplingSelectionStrategy()
        # Set default internal strategy based on Gumbel flag
        if self.config.gumbel:
            self.decision_selection_strategy = GumbelMuZeroSelectionStrategy(config)
            self.root_policy = VisitFrequencyPolicy(
                config, device, num_actions, temperature=1
            )
            self.prior_injectors = [ActionTargetInjector(), DirichletInjector()]
            self.inital_action_set = SelectTopK()
        else:
            self.decision_selection_strategy = UCBSelectionStrategy()
            self.root_policy = GumbelRootPolicy(config, device, num_actions)
            self.prior_injectors = [ActionTargetInjector(), GumbelInjector()]
            self.inital_action_set = SelectAll()

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

    def _backpropogate(self, search_path, leaf_value, leaf_to_play, min_max_stats):
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
            acc[p] = leaf_value if leaf_to_play == p else -leaf_value

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
                # Decision -> Select Action -> ChanceNode

            # Use root strategy if parent is None, otherwise use internal strategy
            if node.parent is None:
                action_or_code, node = self.root_selection_strategy.select_child(
                    node,
                    allowed_actions=allowed_actions,
                    min_max_stats=min_max_stats,
                )
            else:
                if isinstance(node, DecisionNode):
                    action_or_code, node = (
                        self.decision_selection_strategy.select_child(
                            node,
                            allowed_actions=allowed_actions,
                            min_max_stats=min_max_stats,
                        )
                    )
                elif isinstance(node, ChanceNode):
                    action_or_code, node = self.chance_selection_strategy.select_child(
                        node,
                        allowed_actions=allowed_actions,
                        min_max_stats=min_max_stats,
                    )

            horizon_index = (horizon_index + 1) % self.config.lstm_horizon_len
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
                    torch.tensor(action_or_code)
                    .to(parent.hidden_state.device)
                    .unsqueeze(0),
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
                    action_or_code.to(parent.afterstate.device)
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
                torch.tensor(action_or_code)
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

        self._backpropogate(search_path, value, to_play, min_max_stats)


class UCTSearch(SearchAlgorithm):
    """
    Standard UCT Search (Budget = num_simulations, one by one).
    Can be paired with:
    - DirichletStrategy (Standard AlphaZero)
    - GumbelStrategy (Expands top-K at root, then standard UCT on those K).
    """

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
        assert not root.expanded()

        outputs = inference_fns["initial"](state, model=inference_model)
        # Handle unpacking based on return size (Standard vs Gumbel signatures might vary slightly, adapting to provided code)
        if len(outputs) == 3:
            val_raw, policy, hidden_state = outputs
        else:
            raise ValueError("Inference function returned unexpected number of values")

        policy = policy[0]
        policy = policy.cpu()  # ensure CPU for manipulation

        reward_h_state = torch.zeros(1, 1, self.config.lstm_hidden_size).to(self.device)
        reward_c_state = torch.zeros(1, 1, self.config.lstm_hidden_size).to(self.device)

        # 2. Value Processing
        if self.config.support_range is not None:
            v_pi = support_to_scalar(val_raw, self.config.support_range)
            v_pi_scalar = float(v_pi.item())
        else:
            v_pi_scalar = float(val_raw.item())

        # 3. Legal Moves
        legal_moves = get_legal_moves(info)[0]
        if legal_moves is None:
            legal_moves = list(range(self.num_actions))

        # 4. Create Injection Context

        # 5. Apply Prior Injectors (Stackable)
        policy = action_mask(policy.unsqueeze(0), [legal_moves]).squeeze(0)
        for injector in self.prior_injectors:
            policy = injector.inject(policy, self.config, trajectory_action)
            policy = action_mask(policy.unsqueeze(0), [legal_moves]).squeeze(0)

        # 6. Select Actions
        # For Gumbel, 'count' comes from config.gumbel_m. For others, it might be ignored or len(legal_moves).
        selection_count = self.config.gumbel_m
        selected_actions = self.inital_action_set.create_initial_actionset(
            policy, legal_moves, selection_count, trajectory_action
        )

        root.visits += 1

        # 7. Expand Root
        # Note: We use context.policy, which might have been modified by Injectors (e.g. Dirichlet, ActionTarget)
        root.expand(
            selected_actions,
            to_play,
            policy.to(self.device),  # Send back to device if needed by Node
            hidden_state,
            0.0,
            value=v_pi_scalar,
            reward_h_state=reward_h_state,
            reward_c_state=reward_c_state,
            is_reset=True,
        )

        min_max_stats = MinMaxStats(self.config.known_bounds)

        for _ in range(self.config.num_simulations):
            self._run_single_simulation(
                root,
                min_max_stats,
                inference_fns,
                inference_model=inference_model,
                allowed_actions=None,
            )

        # Policy generation is handled by the strategy (e.g. Gumbel needs improvement, Dirichlet uses visits)
        policy = torch.zeros(self.num_actions)

        # NOTE: Standard UCT usually uses visit counts for policy
        # If using Gumbel with UCT, you might still want the Gumbel improved policy?
        # For now, let's just use visit counts for UCT search result, but return target_policy from strategy
        visit_counts = torch.tensor(
            [child.visits for action, child in root.children.items()]
        )
        actions = [action for action, child in root.children.items()]
        policy[actions] = visit_counts / torch.sum(visit_counts)

        # The target policy might differ (e.g. Gumbel improved policy)
        target_policy = self.root_policy.get_policy(root, min_max_stats)
        assert (
            isinstance(target_policy, torch.Tensor)
            and target_policy.shape == policy.shape
        )
        return (
            root.value(),
            policy,
            target_policy,
            torch.argmax(policy),
        )


class SequentialHalvingSearch(SearchAlgorithm):
    """
    Sequential Halving Search (Budget split across rounds, eliminating halves).
    Can be paired with:
    - GumbelStrategy (Standard Gumbel MuZero)
    - DirichletStrategy (Sequential Halving with standard noise/expansion)
    """

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

        # Delegate root expansion to the strategy
        # 1. Inference
        assert not root.expanded()

        outputs = inference_fns["initial"](state, model=inference_model)
        # Handle unpacking based on return size (Standard vs Gumbel signatures might vary slightly, adapting to provided code)
        if len(outputs) == 3:
            val_raw, policy, hidden_state = outputs
        else:
            raise ValueError("Inference function returned unexpected number of values")

        policy = policy[0]
        policy = policy.cpu()  # ensure CPU for manipulation

        reward_h_state = torch.zeros(1, 1, self.config.lstm_hidden_size).to(self.device)
        reward_c_state = torch.zeros(1, 1, self.config.lstm_hidden_size).to(self.device)

        # 2. Value Processing
        if self.config.support_range is not None:
            v_pi = support_to_scalar(val_raw, self.config.support_range)
            v_pi_scalar = float(v_pi.item())
        else:
            v_pi_scalar = float(val_raw.item())

        # 3. Legal Moves
        legal_moves = get_legal_moves(info)[0]
        if legal_moves is None:
            legal_moves = list(range(self.num_actions))

        # 4. Create Injection Context

        # 5. Apply Prior Injectors (Stackable)
        policy = action_mask(policy.unsqueeze(0), [legal_moves]).squeeze(0)
        for injector in self.prior_injectors:
            policy = injector.inject(policy, self.config, trajectory_action)
            policy = action_mask(policy.unsqueeze(0), [legal_moves]).squeeze(0)

        # 6. Select Actions
        # For Gumbel, 'count' comes from config.gumbel_m. For others, it might be ignored or len(legal_moves).
        selection_count = self.config.gumbel_m
        selected_actions = self.action_selector.create_initial_actionset(
            policy, legal_moves, selection_count, trajectory_action
        )

        root.visits += 1

        # 7. Expand Root
        # Note: We use context.policy, which might have been modified by Injectors (e.g. Dirichlet, ActionTarget)
        root.expand(
            selected_actions,
            to_play,
            policy.to(self.device),  # Send back to device if needed by Node
            hidden_state,
            0.0,
            value=v_pi_scalar,
            reward_h_state=reward_h_state,
            reward_c_state=reward_c_state,
            is_reset=True,
        )
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

        target_policy = self.root_policy.get_policy(root, min_max_stats)

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
            # Use strategy to score
            scores.append((a, self.exploration.score_child(root, child, min_max_stats)))
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
                    (a, self.exploration.score_child(root, child, min_max_stats))
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
                (a, self.exploration.score_child(root, child, min_max_stats))
            )

        final_scores.sort(key=sort_by_score, reverse=True)

        # Return the BEST action (Index 0), not the worst!
        return final_scores[0][0]
