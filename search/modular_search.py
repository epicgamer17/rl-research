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
        self.device = torch.device(device)
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

    @property
    def use_amp(self):
        return self.config.use_mixed_precision and not getattr(
            self.config, "use_quantization", False
        )

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

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
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
        search_batch_size = self.config.search_batch_size
        if search_batch_size > 0:
            num_batches = math.ceil(self.config.num_simulations / search_batch_size)
            for i in range(num_batches):
                self._run_batched_simulations(
                    root,
                    min_max_stats,
                    inference_fns,
                    batch_size=search_batch_size,
                    inference_model=inference_model,
                    current_sim_idx=i * search_batch_size,
                    pruning_context=pruning_context,
                )
        else:
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
        # Extract root children values for visualization
        root_children_values = torch.zeros(self.num_actions)
        for action, child in root.children.items():
            if isinstance(child, (DecisionNode, ChanceNode)):  # Should be nodes
                root_children_values[action] = child.value()

        return (
            root.value(),
            exploratory_policy,
            target_policy,
            # TODO: BEST ACTION SELECTION, WHERE? WHAT, HOW?
            torch.argmax(target_policy),
            {
                "network_policy": network_policy,
                "network_value": v_pi_scalar,
                "search_policy": target_policy,
                "search_value": root.value(),
                "root_children_values": root_children_values,
            },
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
                with torch.no_grad():
                    with torch.autocast(
                        device_type=self.device.type,
                        enabled=self.use_amp,
                    ):
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
                with torch.no_grad():
                    with torch.autocast(
                        device_type=self.device.type,
                        enabled=self.use_amp,
                    ):
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
            with torch.no_grad():
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                ):
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

    def _run_batched_simulations(
        self,
        root: DecisionNode,
        min_max_stats: MinMaxStats,
        inference_fns,
        batch_size,
        inference_model=None,
        current_sim_idx=0,
        pruning_context=None,
    ):
        use_virtual_mean = self.config.use_virtual_mean
        virtual_loss = self.config.virtual_loss

        sim_data = []

        # Pre-allocate buffers lazily
        rec_states = None
        rec_actions = None
        rec_rhs = None
        rec_rcs = None
        rec_indices = []

        aft_states = None
        aft_actions = None
        aft_indices = []

        # 1. Selection Phase
        for b in range(batch_size):
            node = root
            search_path = [node]
            path_virtual_values = []
            horizon_index = 0

            action_or_code = None

            while True:
                if not node.expanded():
                    break

                parent_node = node

                if node.parent is None:
                    # Root
                    pruned_actionset, next_state = self.pruning_method.step(
                        node,
                        pruning_context["root"],
                        self.config,
                        min_max_stats,
                        current_sim_idx + b,
                    )
                    pruning_context["root"] = next_state
                    if pruned_actionset is not None and len(pruned_actionset) == 0:
                        # Revert virtual update for this failed path
                        if use_virtual_mean:
                            for n, v in zip(search_path, path_virtual_values):
                                n.visits -= 1
                                n.value_sum -= v
                        else:
                            for n in search_path:
                                n.visits -= 1
                                n.value_sum += virtual_loss

                        node = None
                        break

                    action_or_code, node = self.root_selection_strategy.select_child(
                        node,
                        pruned_actionset=pruned_actionset,
                        min_max_stats=min_max_stats,
                    )
                else:
                    if isinstance(node, DecisionNode):
                        if node not in pruning_context["internal"]:
                            pruning_context["internal"][node] = (
                                self.internal_pruning_method.initialize(
                                    node, self.config
                                )
                            )

                        pruned_actionset, next_state = (
                            self.internal_pruning_method.step(
                                node,
                                pruning_context["internal"][node],
                                self.config,
                                min_max_stats,
                                current_sim_idx + b,
                            )
                        )
                        pruning_context["internal"][node] = next_state

                        if pruned_actionset is not None and len(pruned_actionset) == 0:
                            if use_virtual_mean:
                                for n, v in zip(search_path, path_virtual_values):
                                    n.visits -= 1
                                    n.value_sum -= v
                            else:
                                for n in search_path:
                                    n.visits -= 1
                                    n.value_sum += virtual_loss
                            node = None
                            break

                        action_or_code, node = (
                            self.decision_selection_strategy.select_child(
                                node,
                                pruned_actionset=pruned_actionset,
                                min_max_stats=min_max_stats,
                            )
                        )
                    elif isinstance(node, ChanceNode):
                        action_or_code, node = (
                            self.chance_selection_strategy.select_child(
                                node,
                                min_max_stats=min_max_stats,
                            )
                        )

                horizon_index = (horizon_index + 1) % self.config.lstm_horizon_len

                # Apply virtual update to the PARENT (search_path[-1])
                parent_node = search_path[-1]
                if use_virtual_mean:
                    v_val = parent_node.value()
                    parent_node.visits += 1
                    parent_node.value_sum += v_val
                    path_virtual_values.append(v_val)
                else:
                    parent_node.visits += 1
                    parent_node.value_sum -= virtual_loss

                search_path.append(node)

            if node is None:
                sim_data.append(None)
                continue

            # Leaf Node Update
            if use_virtual_mean:
                v_val = node.value()
                node.visits += 1
                node.value_sum += v_val
                path_virtual_values.append(v_val)
            else:
                node.visits += 1
                node.value_sum -= virtual_loss

            sim_entry = {
                "path": search_path,
                "node": node,
                "parent": search_path[-2],
                # "action": action_or_code, # Not stored, we act on it immediately
                "horizon_index": horizon_index,
                "virtual_values": path_virtual_values if use_virtual_mean else None,
                "result": None,
            }
            sim_data.append(sim_entry)

            # --- Prepare Inference Inputs (In-Place Filling) ---
            parent = sim_entry["parent"]
            action = action_or_code

            if isinstance(node, DecisionNode):
                # Recurrent Inference
                if isinstance(parent, DecisionNode):
                    state = parent.hidden_state
                elif isinstance(parent, ChanceNode):
                    state = parent.afterstate

                # Buffers Setup
                if rec_states is None:
                    # Infer shapes from first item
                    rec_states = torch.empty(
                        (batch_size, *state.shape[1:]),
                        device=self.device,
                        dtype=state.dtype,
                    )
                    rec_rhs = torch.empty(
                        (batch_size, *parent.reward_h_state.shape[1:]),
                        device=self.device,
                        dtype=parent.reward_h_state.dtype,
                    )
                    rec_rcs = torch.empty(
                        (batch_size, *parent.reward_c_state.shape[1:]),
                        device=self.device,
                        dtype=parent.reward_c_state.dtype,
                    )
                    # For actions, we use a list to be safe against shape/type variants
                    rec_actions_list = []

                idx = len(rec_indices)
                rec_states[idx] = state.squeeze(0)
                rec_rhs[idx] = parent.reward_h_state.squeeze(0)
                rec_rcs[idx] = parent.reward_c_state.squeeze(0)

                # Handle action safely
                if isinstance(action, torch.Tensor):
                    val = action.clone().detach()  # Detach to be safe
                else:
                    val = torch.tensor(action)
                rec_actions_list.append(val)

                rec_indices.append(b)

            elif isinstance(node, ChanceNode):
                # Afterstate Inference
                state = parent.hidden_state
                if aft_states is None:
                    aft_states = torch.empty(
                        (batch_size, *state.shape[1:]),
                        device=self.device,
                        dtype=state.dtype,
                    )
                    # For actions, we use a list to be safe against shape/type variants
                    aft_actions_list = []

                idx = len(aft_indices)
                aft_states[idx] = state.squeeze(0)

                # Handle action safely
                if isinstance(action, torch.Tensor):
                    val = action.clone().detach()
                else:
                    val = torch.tensor(action)
                aft_actions_list.append(val)

                aft_indices.append(b)

        # 2. Batched Inference
        if rec_indices:
            # Slice to actual size
            count = len(rec_indices)
            states = rec_states[:count]
            rhs = rec_rhs[:count]
            rcs = rec_rcs[:count]

            # Form action tensor from list
            act_list = []
            for val in rec_actions_list:
                # Ensure shape (1, ...) or (1)
                if val.dim() == 0:
                    val = val.unsqueeze(0)
                if val.dim() == 1 and val.shape[0] == 1:
                    val = val  # Already correct
                elif val.dim() == 1:
                    val = val.unsqueeze(0)  # (N) -> (1, N)

                # IMPORTANT: Keep Original Dtype/Value logic if possible,
                # but usually actions are floats for NN inputs
                act_list.append(val.float())

            actions = torch.cat(act_list, dim=0)
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)  # Ensure (B, 1) if simplified

            with torch.no_grad():
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                ):
                    (
                        rewards,
                        hidden_states,
                        values,
                        policies,
                        to_plays,
                        rh_news,
                        rc_news,
                    ) = inference_fns["recurrent"](
                        states, actions, rhs, rcs, model=inference_model
                    )

            # Distribute results
            for i, sim_idx in enumerate(rec_indices):
                sim_data[sim_idx]["result"] = {
                    "reward": rewards[i],
                    "hidden_state": hidden_states[i : i + 1],
                    "value": values[i],
                    "policy": policies[i : i + 1],
                    "to_play": to_plays[i : i + 1],
                    "rh": rh_news[i : i + 1],
                    "rc": rc_news[i : i + 1],
                }

        if aft_indices:
            count = len(aft_indices)
            states = aft_states[:count]

            # Form action tensor from list
            act_list = []
            for val in aft_actions_list:
                if val.dim() == 0:
                    val = val.unsqueeze(0)
                if val.dim() == 1 and val.shape[0] > 1:
                    val = val.unsqueeze(0)

                act_list.append(val.float())

            actions = torch.cat(act_list, dim=0)
            if actions.dim() == 1:
                actions = actions.unsqueeze(1)

            with torch.no_grad():
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.use_amp,
                ):
                    afterstates, values, code_probs_batch = inference_fns["afterstate"](
                        states, actions, model=inference_model
                    )

            for i, sim_idx in enumerate(aft_indices):
                sim_data[sim_idx]["result"] = {
                    "afterstate": afterstates[i : i + 1],
                    "value": values[i],
                    "code_probs": code_probs_batch[i : i + 1],
                }

        # 3. Expansion & Backprop

        # A. Revert Virtual Loss / Virtual Mean (Global Reversion Phase)
        for d in sim_data:
            if d is None:
                continue
            path = d["path"]
            virtual_values = d.get("virtual_values", [])

            # Revert Path VL/VM
            if virtual_values:
                # Virtual Mean Reversion
                for node, v_val in zip(path, virtual_values):
                    node.visits -= 1
                    node.value_sum -= v_val
            else:
                # Virtual Loss Reversion (Constant)
                for node in path:
                    node.visits -= 1
                    node.value_sum += virtual_loss

        # B. Backpropagation Phase
        for d in sim_data:
            if d is None:
                continue
            path = d["path"]
            node = d["node"]

            res = d.get("result")
            if not res:
                continue

            to_play_for_backprop = None

            if isinstance(node, DecisionNode):
                reward = res["reward"]
                value = res["value"]
                if self.config.support_range is not None:
                    reward = support_to_scalar(reward, self.config.support_range).item()
                    value = support_to_scalar(value, self.config.support_range).item()
                else:
                    reward = reward.item()
                    value = value.item()

                to_play = int(res["to_play"].argmax().item())
                to_play_for_backprop = to_play

                is_reset = d["horizon_index"] == 0
                rh = res["rh"]
                rc = res["rc"]

                if self.config.value_prefix and is_reset:
                    rh = torch.zeros_like(rh).to(self.device)
                    rc = torch.zeros_like(rc).to(self.device)

                policy = res["policy"][0]
                actions_to_expand = self.internal_actionset.create_initial_actionset(
                    policy,
                    list(range(self.num_actions)),
                    self.config.gumbel_m,
                    trajectory_action=None,
                )

                node.expand(
                    allowed_actions=actions_to_expand,
                    to_play=to_play,
                    priors=policy,
                    network_policy=policy,
                    hidden_state=res["hidden_state"],
                    reward=reward,
                    value=value,
                    reward_h_state=rh,
                    reward_c_state=rc,
                    is_reset=is_reset,
                )

            elif isinstance(node, ChanceNode):
                value = res["value"]
                if self.config.support_range:
                    value = support_to_scalar(value, self.config.support_range).item()
                else:
                    value = value.item()

                to_play_for_backprop = d["parent"].to_play

                node.expand(
                    to_play=d["parent"].to_play,
                    afterstate=res["afterstate"],
                    network_value=value,
                    code_probs=res["code_probs"][0],
                    reward_h_state=d["parent"].reward_h_state,
                    reward_c_state=d["parent"].reward_c_state,
                )

            self.backpropagator.backpropagate(
                path, value, to_play_for_backprop, min_max_stats, self.config
            )
