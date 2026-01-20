from abc import ABC
import torch
import math
from modules.utils import support_to_scalar
from search.action_selectors import SelectionStrategy
from search.backpropogation import Backpropagator
from search.initial_action_sets import ActionSet
from search.nodes import ChanceNode, DecisionNode
from search.min_max_stats import MinMaxStats
from search.prior_injectors import PriorInjector
from search.root_policies import RootPolicyStrategy
from utils.utils import action_mask, get_legal_moves
from search.pruners import PruningMethod


class SearchAlgorithm:
    def __init__(
        self,
        config,
        device,
        num_actions,
        root_selection_strategy,
        decision_selection_strategy,
        chance_selection_strategy,
        root_target_policy,
        root_exploratory_policy,
        prior_injectors,
        root_actionset,
        internal_actionset,
        pruning_method,
        internal_pruning_method,
        backpropagator,
    ):
        self.config = config
        self.device = device
        self.num_actions = num_actions

        self.root_selection_strategy: SelectionStrategy = root_selection_strategy
        self.decision_selection_strategy: SelectionStrategy = (
            decision_selection_strategy
        )
        self.chance_selection_strategy: SelectionStrategy = chance_selection_strategy
        self.root_target_policy: RootPolicyStrategy = root_target_policy
        self.root_exploratory_policy: RootPolicyStrategy = root_exploratory_policy
        self.prior_injectors: PriorInjector = prior_injectors
        self.root_actionset: ActionSet = root_actionset
        self.internal_actionset: ActionSet = internal_actionset
        self.pruning_method: PruningMethod = pruning_method
        self.internal_pruning_method: PruningMethod = internal_pruning_method
        self.backpropagator: Backpropagator = backpropagator

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
        if len(outputs) == 3:
            val_raw, policy, hidden_state = outputs
        else:
            raise ValueError("Inference function returned unexpected number of values")

        # 3. Legal Moves
        legal_moves = get_legal_moves(info)
        # print("legal smoves", legal_moves)
        if legal_moves is None:
            legal_moves = [list(range(self.num_actions))]

        # TODO: should i action mask?
        policy = action_mask(policy, legal_moves, device=self.device)

        legal_moves = legal_moves[0]
        policy = policy[0]
        policy = policy.cpu()  # ensure CPU for manipulation
        network_policy = policy.clone()

        reward_h_state = torch.zeros(1, 1, self.config.lstm_hidden_size).to(self.device)
        reward_c_state = torch.zeros(1, 1, self.config.lstm_hidden_size).to(self.device)

        # 2. Value Processing
        if self.config.support_range is not None:
            v_pi = support_to_scalar(val_raw, self.config.support_range)
            v_pi_scalar = float(v_pi.item())
        else:
            v_pi_scalar = float(val_raw.item())

        # 5. Apply Prior Injectors (Stackable)
        for injector in self.prior_injectors:
            policy = injector.inject(
                policy, legal_moves, self.config, trajectory_action
            )

        # 6. Select Actions
        selection_count = self.config.gumbel_m
        selected_actions = self.root_actionset.create_initial_actionset(
            policy, legal_moves, selection_count, trajectory_action
        )

        root.visits += 1

        # 7. Expand Root
        root.expand(
            allowed_actions=selected_actions,
            to_play=to_play,
            priors=policy.to(self.device),  # Send back to device if needed by Node
            network_policy=network_policy,
            hidden_state=hidden_state,
            reward=0.0,
            value=v_pi_scalar,
            reward_h_state=reward_h_state,
            reward_c_state=reward_c_state,
            is_reset=True,
        )

        min_max_stats = MinMaxStats(
            self.config.known_bounds,
            soft_update=self.config.soft_update,
            min_max_epsilon=self.config.min_max_epsilon,
        )

        # Initialize pruning state (e.g. Sequential Halving budget)
        # pruning_state = self.pruning_method.initialize(root, self.config)
        pruning_context = {
            "root": self.pruning_method.initialize(root, self.config),
            "internal": {},  # Map node -> state
        }
        # --- Main Simulation Loop ---
        for i in range(self.config.num_simulations):
            # Pruning method determines which actions are allowed for this simulation step
            # allowed_actions, pruning_state = self.pruning_method.step(
            #     root, pruning_state, self.config, min_max_stats, i
            # )

            self._run_single_simulation(
                root,
                min_max_stats,
                inference_fns,
                inference_model=inference_model,
                # pruned_actionset=allowed_actions,
                current_sim_idx=i,
                pruning_context=pruning_context,
            )

        target_policy = self.root_target_policy.get_policy(root, min_max_stats)
        exploratory_policy = self.root_exploratory_policy.get_policy(
            root, min_max_stats
        )

        # Mask target policy if required by pruning method (e.g. for Sequential Halving)
        if self.pruning_method.mask_target_policy:
            target_policy = action_mask(
                target_policy.unsqueeze(0), [legal_moves]
            ).squeeze(0)

        assert (
            isinstance(target_policy, torch.Tensor)
            and target_policy.shape == policy.shape
        )
        return (
            root.value(),
            exploratory_policy,
            target_policy,
            # TODO: BEST ACTION SELECTION, WHERE? WHAT, HOW?
            torch.argmax(target_policy),
        )

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

    def _run_single_simulation(
        self,
        root: DecisionNode,
        min_max_stats: MinMaxStats,
        inference_fns,
        inference_model=None,
        current_sim_idx=0,
        pruning_context=None,
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
        #         pruned_actionset=pruned_actionset,
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
                pruned_actionset, next_state = self.pruning_method.step(
                    node,
                    pruning_context["root"],
                    self.config,
                    min_max_stats,
                    current_sim_idx,
                )
                pruning_context["root"] = next_state
                # TODO: EARLY STOPPING CLASSES
                if pruned_actionset is not None and len(pruned_actionset) == 0:
                    return  # Stop this simulation

                action_or_code, node = self.root_selection_strategy.select_child(
                    node,
                    pruned_actionset=pruned_actionset,
                    min_max_stats=min_max_stats,
                )
            else:
                if isinstance(node, DecisionNode):
                    if node not in pruning_context["internal"]:
                        pruning_context["internal"][node] = (
                            self.internal_pruning_method.initialize(node, self.config)
                        )

                    pruned_actionset, next_state = self.internal_pruning_method.step(
                        node,
                        pruning_context["internal"][node],
                        self.config,
                        min_max_stats,
                        current_sim_idx,
                    )

                    # TODO: EARLY STOPPING CLASSES
                    if pruned_actionset is not None and len(pruned_actionset) == 0:
                        return  # Stop this simulation

                    pruning_context["internal"][node] = next_state
                    action_or_code, node = (
                        self.decision_selection_strategy.select_child(
                            node,
                            pruned_actionset=pruned_actionset,
                            min_max_stats=min_max_stats,
                        )
                    )
                elif isinstance(node, ChanceNode):
                    action_or_code, node = self.chance_selection_strategy.select_child(
                        node,
                        # pruned_actionset=pruned_actionset,
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

                actions_to_expand = self.internal_actionset.create_initial_actionset(
                    policy[0],
                    list(range(self.num_actions)),
                    self.config.gumbel_m,
                    trajectory_action=None,
                )

                node.expand(
                    allowed_actions=actions_to_expand,
                    to_play=to_play,
                    priors=policy[0],
                    network_policy=policy[0],
                    hidden_state=hidden_state,
                    reward=reward,
                    value=value,
                    reward_h_state=reward_h_state,
                    reward_c_state=reward_c_state,
                    is_reset=is_reset,
                )
            elif isinstance(parent, ChanceNode):
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

                actions_to_expand = self.internal_actionset.create_initial_actionset(
                    policy[0],
                    list(range(self.num_actions)),
                    self.config.gumbel_m,
                    trajectory_action=None,
                )

                node.expand(
                    allowed_actions=actions_to_expand,
                    to_play=to_play,
                    priors=policy[0],
                    network_policy=policy[0],
                    hidden_state=hidden_state,
                    reward=reward,
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
            afterstate, value, code_probs = inference_fns[
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
                to_play=parent.to_play,
                afterstate=afterstate,
                network_value=value,
                code_probs=code_probs[0],
                reward_h_state=parent.reward_h_state,
                reward_c_state=parent.reward_c_state,
            )

        self.backpropagator.backpropagate(
            search_path, value, to_play, min_max_stats, self.config
        )
