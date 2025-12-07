from time import time
import numpy as np
from sympy import N
import torch
from packages.utils.utils.utils import legal_moves_mask, numpy_dtype_to_torch_dtype
from replay_buffers.base_replay_buffer import (
    BaseReplayBuffer,
    Game,
)
from replay_buffers.segment_tree import MinSegmentTree, SumSegmentTree
import torch
import torch.multiprocessing as mp

from time import time
import numpy as np
import torch
import torch.multiprocessing as mp
from packages.utils.utils.utils import legal_moves_mask, numpy_dtype_to_torch_dtype
from replay_buffers.base_replay_buffer import BaseReplayBuffer, Game
from replay_buffers.segment_tree import MinSegmentTree, SumSegmentTree


class MuZeroReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        observation_dimensions,
        observation_dtype: type,
        max_size: int,
        num_actions: int,
        batch_size: int,
        n_step: int,
        num_unroll_steps: int,
        gamma: float,
        num_players: int,
        max_priority: float = 1.0,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 0.0001,
        use_batch_weights: bool = False,
        initial_priority_max: bool = False,
        lstm_horizon_len: int = 5,
        value_prefix: bool = False,
        tau: float = 0.3,
    ):
        assert alpha >= 0 and alpha <= 1
        assert beta >= 0 and beta <= 1
        assert n_step >= 1
        assert gamma >= 0 and gamma <= 1

        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype
        self.write_lock = mp.Lock()
        self.priority_lock = mp.Lock()

        self.num_actions = num_actions
        self.num_players = num_players

        self.n_step = n_step
        self.unroll_steps = num_unroll_steps
        self.gamma = gamma

        self.initial_max_priority = max_priority
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        self.use_batch_weights = use_batch_weights
        self.per_initial_priority_max = initial_priority_max

        self.lstm_horizon_len = lstm_horizon_len
        self.value_prefix = value_prefix
        self.tau = tau

        self.time_to_full = time()

        super().__init__(max_size=max_size, batch_size=batch_size)

    def store_game(self, game: Game, training_step):
        # print(game.info_history)
        # n = number of transitions (actions)
        # obs length is n + 1 (initial -> terminal)
        obs_history = game.observation_history
        n_transitions = len(game.action_history)
        n_states = len(obs_history)  # Should be n_transitions + 1

        # 1. Prepare Observations (Keep ALL of them)
        obs_tensor = torch.from_numpy(np.stack(obs_history)).to(
            self.observation_buffer.device
        )

        # 2. Prepare & Pad Actions, Rewards, Policies, ToPlays
        # We need to pad these to length (n_states) so indices align.
        # The data at the terminal index [n] is "dummy" data.

        # Actions
        acts_raw = torch.tensor(game.action_history, dtype=torch.float16)
        # Pad with 0 (or random, doesn't matter, won't be sampled as root)
        # TODO: PAD WITH RANDOM ACTION
        acts_pad = torch.zeros(1, dtype=torch.float16)
        acts_t = torch.cat([acts_raw, acts_pad])

        # Rewards
        rews_raw = torch.tensor(game.rewards, dtype=torch.float32)
        rews_pad = torch.zeros(1, dtype=torch.float32)
        rews_t = torch.cat([rews_raw, rews_pad])

        # Policies
        pols_raw = torch.stack(game.policy_history).cpu().float()
        # Pad with uniform or zeros
        # TODO: PAD WITH UNIFORM
        pols_pad = torch.zeros((1, self.num_actions), dtype=torch.float32)
        pols_t = torch.cat([pols_raw, pols_pad])

        # Values
        # (Game values usually already have n+1 entries if constructed correctly: v_0...v_terminal)
        vals_t = torch.tensor(game.value_history, dtype=torch.float32)
        if len(vals_t) == n_transitions:
            # If game object didn't store terminal value, append 0
            vals_t = torch.cat([vals_t, torch.zeros(1, dtype=torch.float32)])

        # To Plays
        to_plays = [i.get("player", 0) for i in game.info_history]
        # Ensure length matches obs
        if len(to_plays) < n_states:
            to_plays = to_plays + [0] * (n_states - len(to_plays))
        tps_t = torch.tensor(to_plays[:n_states], dtype=torch.int16)

        # Chances
        chances = [i.get("chance", 0) for i in game.info_history]
        if len(chances) < n_states:
            chances = chances + [0] * (n_states - len(chances))
        chance_t = torch.tensor(chances[:n_states], dtype=torch.int16).unsqueeze(1)
        # Legal Moves Mask
        # We need to construct this.
        # Steps 0 to N-1: Use game info.
        # Step N (Terminal): All False.
        legal_masks = []
        for i in range(n_transitions):
            moves = game.info_history[i].get("legal_moves", [])
            legal_masks.append(legal_moves_mask(self.num_actions, moves))

        # Terminal state mask: All False (No moves allowed)
        legal_masks.append(torch.zeros(self.num_actions, dtype=torch.bool))
        legal_masks_t = torch.stack(legal_masks)

        # 3. Priorities
        if self.per_initial_priority_max:
            priorities = np.full(n_states, self.max_priority)
        else:
            raise NotImplementedError("Non initial max priority not implemented yet")

        # --- WRITE TO BUFFER (Standard logic, just larger N) ---
        with self.write_lock:
            start_idx = int(self.pointer.item())
            end_idx = (start_idx + n_states) % self.max_size

            # Update Pointers
            self.pointer[0] = end_idx
            self.size = min(self.size + n_states, self.max_size)

            start_game_id = int(self._next_game_id.item()) + 1
            self._next_game_id[0] = start_game_id

            start_id = int(self._next_id.item())
            ids = torch.arange(start_id + 1, start_id + n_states + 1, dtype=torch.long)
            self._next_id[0] = start_id + n_states

            # Write Data (Handling Wrap-around)
            if start_idx + n_states <= self.max_size:
                rng = slice(start_idx, start_idx + n_states)
                self.observation_buffer[rng] = obs_tensor
                self.value_buffer[rng] = vals_t
                self.policy_buffer[rng] = pols_t
                self.reward_buffer[rng] = rews_t
                self.action_buffer[rng] = acts_t
                self.to_play_buffer[rng] = tps_t
                self.chance_buffer[rng] = chance_t
                self.legal_moves_mask_buffer[rng] = legal_masks_t

                self.game_id_buffer[rng] = start_game_id
                self.id_buffer[rng] = ids
                self.training_step_buffer[rng] = training_step
            else:
                tail_len = self.max_size - start_idx
                head_len = n_states - tail_len
                tail_rng = slice(start_idx, self.max_size)
                head_rng = slice(0, head_len)

                # Helper lambda for split writing
                def write_split(buf, data):
                    buf[tail_rng] = data[:tail_len]
                    buf[head_rng] = data[tail_len:]

                write_split(self.observation_buffer, obs_tensor)
                write_split(self.value_buffer, vals_t)
                write_split(self.policy_buffer, pols_t)
                write_split(self.reward_buffer, rews_t)
                write_split(self.action_buffer, acts_t)
                write_split(self.to_play_buffer, tps_t)
                write_split(self.chance_buffer, chance_t)
                write_split(self.legal_moves_mask_buffer, legal_masks_t)

                self.game_id_buffer[tail_rng] = start_game_id
                self.game_id_buffer[head_rng] = start_game_id
                self.id_buffer[tail_rng] = ids[:tail_len]
                self.id_buffer[head_rng] = ids[tail_len:]
                self.training_step_buffer[tail_rng] = training_step
                self.training_step_buffer[head_rng] = training_step

        # Update Priorities
        with self.priority_lock:
            indices = [(start_idx + i) % self.max_size for i in range(n_states)]
            for i, (idx, p) in enumerate(zip(indices, priorities)):
                is_terminal = i == n_states - 1
                if is_terminal:
                    # 1. SumTree: 0.0 means probability mass is 0.
                    # The sampler will NEVER pick this index.
                    self.sum_tree[idx] = 0.0

                    # 2. MinTree: Infinity is the neutral element for min().
                    # This ensures min_priority is determined only by valid states.
                    self.min_tree[idx] = float("inf")
                else:
                    # Standard behavior for valid states
                    self.sum_tree[idx] = p**self.alpha
                    self.min_tree[idx] = p**self.alpha

    def store(self, game: Game, training_step):
        self.store_game(game, training_step)
        if self.size < 1000:
            assert self.id_buffer.sum() > 0
            assert self.game_id_buffer.sum() > 0
            print("Buffer size:", self.size)

    def sample(self):
        # Sampling is read-only logic, avoiding locks for throughput
        indices: list[int] = self._sample_proportional()
        weights = torch.tensor(
            [self._calculate_weight(i) for i in indices], dtype=torch.float32
        )
        if self.use_batch_weights:
            weights = weights / weights.max()
        else:
            min_priority = self.min_tree.min() / self.sum_tree.sum()
            max_weight = (min_priority * len(self)) ** (-self.beta)
            weights = weights / max_weight

        assert torch.all(weights > 0), "Non-positive weights found in sampling"

        # Perform the heavy lifting: unpacking sequences and calculating N-step returns on the fly
        samples = self.sample_from_indices(indices)
        samples.update(dict(weights=weights))
        return samples

    def _compute_n_step_targets(
        self, batch_size, raw_rewards, raw_values, raw_to_plays, valid_mask, device
    ):
        """
        Helper: Computes N-step value targets and aligns rewards for the unroll sequence.
        """
        # Prepare output tensors
        target_values = torch.zeros(
            (batch_size, self.unroll_steps + 1), dtype=torch.float32, device=device
        )
        target_rewards = torch.zeros(
            (batch_size, self.unroll_steps + 1), dtype=torch.float32, device=device
        )

        # We need to look ahead up to K + N steps
        lookahead_window = self.unroll_steps + self.n_step

        for u in range(self.unroll_steps + 1):
            is_valid_node = valid_mask[:, u]

            # --- 1. Immediate Reward (Input to Dynamics) ---
            # For Unroll step u, we need the reward received *getting to* u.
            # In standard storage: Reward[i] is reward received after action at i-1.
            if u == 0:
                target_rewards[:, u] = 0.0
            else:
                # If u is valid, we grab the reward stored at u-1 (the transition leading to u)
                target_rewards[is_valid_node, u] = raw_rewards[is_valid_node, u - 1]

            # --- 2. N-Step Value Calculation ---
            # Formula: G_t = Sum(gamma^k * r_{t+k+1}) + gamma^n * v_{t+n}

            computed_value = torch.zeros(batch_size, device=device)
            current_player = raw_to_plays[:, u]

            # Sum Discounted Rewards
            for k in range(self.n_step):
                r_idx = u + k
                if r_idx >= lookahead_window:
                    break

                # Check validity and player perspective
                r_is_valid = valid_mask[:, r_idx]
                step_player = raw_to_plays[:, r_idx]

                # Zero-sum logic: if player changed, flip sign
                sign = torch.where(current_player == step_player, 1.0, -1.0)

                reward_chunk = (self.gamma**k) * raw_rewards[:, r_idx] * sign

                # Accumulate only if the reward index is valid
                computed_value += torch.where(
                    r_is_valid, reward_chunk, torch.tensor(0.0, device=device)
                )

            # Add Bootstrap Value
            boot_idx = u + self.n_step
            if boot_idx < lookahead_window:
                b_is_valid = valid_mask[:, boot_idx]
                boot_player = raw_to_plays[:, boot_idx]
                sign_boot = torch.where(current_player == boot_player, 1.0, -1.0)

                val_chunk = (
                    (self.gamma**self.n_step) * raw_values[:, boot_idx] * sign_boot
                )

                computed_value += torch.where(
                    b_is_valid, val_chunk, torch.tensor(0.0, device=device)
                )

            target_values[is_valid_node, u] = computed_value[is_valid_node]

        return target_values, target_rewards

    def sample_from_indices(self, indices: list[int]):
        try:
            device = self.observation_buffer.device
        except Exception:
            device = torch.device("cpu")

        indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        batch_size = len(indices)

        # 1. Define Window
        lookahead_window = self.unroll_steps + self.n_step
        offsets = torch.arange(
            0, lookahead_window + 1, dtype=torch.long, device=device
        ).unsqueeze(0)
        all_indices = (indices_tensor.unsqueeze(1) + offsets) % self.max_size

        # 2. Fetch Raw Data
        raw_rewards = self.reward_buffer[all_indices]
        raw_values = self.value_buffer[all_indices]
        raw_policies = self.policy_buffer[all_indices]
        raw_actions = self.action_buffer[all_indices]
        raw_to_plays = self.to_play_buffer[all_indices]
        raw_chances = self.chance_buffer[all_indices]
        raw_game_ids = self.game_id_buffer[all_indices]

        # NEW: Fetch Legal Moves for the whole window to determine Policy Validity
        # Shape: (B, Window, Num_Actions)
        raw_legal_masks = self.legal_moves_mask_buffer[all_indices]

        # 3. Consistency Mask (Episodes)
        base_game_ids = raw_game_ids[:, 0].unsqueeze(1)
        # This mask includes the terminal state (True for everything in the episode)
        consistency_mask = raw_game_ids == base_game_ids

        # 4. Policy Mask (Actionable States)
        # A state is valid for policy training if:
        # A) It is part of the episode (Consistency)
        # B) It has at least one legal move (Not Terminal)
        has_legal_moves = raw_legal_masks.sum(dim=-1) > 0  # Shape (B, Window)
        policy_mask = consistency_mask & has_legal_moves

        # 5. Compute N-Step Targets
        # (Pass consistency_mask here, because Value/Reward targets ARE valid at terminal steps)
        target_values, target_rewards = self._compute_n_step_targets(
            batch_size, raw_rewards, raw_values, raw_to_plays, consistency_mask, device
        )

        # 6. Prepare Unroll Targets (0 to K)
        target_policies = torch.zeros(
            (batch_size, self.unroll_steps + 1, self.num_actions),
            dtype=torch.float32,
            device=device,
        )
        target_actions = torch.zeros(
            (batch_size, self.unroll_steps), dtype=torch.int64, device=device
        )
        target_to_plays = torch.zeros(
            (batch_size, self.unroll_steps + 1, self.num_players),
            dtype=torch.float32,
            device=device,
        )
        target_chances = torch.zeros(
            (batch_size, self.unroll_steps + 1, 1), dtype=torch.int64, device=device
        )

        # NEW: Return the specific policy mask for the unroll steps
        target_policy_mask = torch.zeros(
            (batch_size, self.unroll_steps + 1), dtype=torch.bool, device=device
        )

        for u in range(self.unroll_steps + 1):
            is_consistent = consistency_mask[:, u]
            is_policy_valid = policy_mask[:, u]

            # --- Policy ---
            target_policies[is_consistent, u] = raw_policies[is_consistent, u]
            target_policies[~is_consistent, u] = 1.0 / self.num_actions

            # Save the mask for the loss function
            target_policy_mask[:, u] = is_policy_valid

            # --- To Play ---
            tp_indices = torch.clamp(raw_to_plays[:, u].long(), 0, self.num_players - 1)
            target_to_plays[range(batch_size), u, tp_indices] = 1.0
            target_to_plays[~is_consistent, u] = 0

            # --- Chance Codes ---
            target_chances[is_consistent, u, 0] = (
                raw_chances[is_consistent, u].squeeze(-1).long()
            )

            # --- Actions ---
            if u < self.unroll_steps:
                target_actions[:, u] = raw_actions[:, u].long()
                target_actions[~is_consistent, u] = int(
                    np.random.randint(0, self.num_actions)
                )

        # 7. Unroll Observations
        obs_indices = all_indices[:, : self.unroll_steps + 1]
        obs_valid_mask = consistency_mask[:, : self.unroll_steps + 1]

        unroll_observations = self.observation_buffer[obs_indices].clone()

        for step in range(1, self.unroll_steps + 1):
            is_absorbing = ~obs_valid_mask[:, step]
            if is_absorbing.any():
                unroll_observations[is_absorbing, step] = unroll_observations[
                    is_absorbing, step - 1
                ]

        return dict(
            observations=self.observation_buffer[indices_tensor],
            # Mask 1: For Value and Reward Loss (Includes Terminal)
            valid_masks=obs_valid_mask,
            # Mask 2: For Policy Loss (Excludes Terminal)
            policy_masks=target_policy_mask,
            unroll_observations=unroll_observations,
            rewards=target_rewards,
            policy=target_policies,
            values=target_values,
            actions=target_actions,
            to_plays=target_to_plays,
            chance_codes=target_chances,
            ids=self.id_buffer[indices_tensor].clone(),
            legal_moves_masks=self.legal_moves_mask_buffer[indices_tensor],
            indices=indices,
            training_steps=self.training_step_buffer[indices_tensor],
        )

    def sample_game(self):
        """
        Retrieves all stored states for a specific game ID.
        Useful for debugging or visualization, but slow (O(N) scan).
        """
        # Find all indices where game_id_buffer matches game_id
        # Note: This is expensive on CPU, do not use in training loop
        game_ids = list(set(list(self.game_id_buffer[: self.size])))
        # print(self.game_id_buffer[: self.size])
        # print(self.n_step_values_buffer[: self.size])
        # print(game_ids)
        game_id = np.random.choice(game_ids, 1)[0]
        # print("game id", game_id)
        mask = self.game_id_buffer[: self.size] == game_id
        indices = torch.nonzero(mask).view(-1).tolist()
        # print(indices)

        if not indices:
            print("game indices not found")
            return None

        # Sort indices to ensure chronological order (assuming sequential writes)
        indices.sort()
        return self.sample_from_indices(indices)

    def reanalyze_game(
        self,
        indices,
        new_policies,
        new_values,
        traj_rewards=None,  # Not strictly needed if only updating value/policy
        traj_actions=None,
        traj_infos=None,
        ids=None,
        current_training_step=None,
        total_training_steps=None,
    ):
        """
        Updates the raw values and policies in the buffer.
        Much faster now: No N-step recalculation required during write.
        """
        # We can write directly to the raw buffers.
        # indices, new_policies, new_values are lists or tensors

        if len(indices) == 0:
            return

        # Convert to tensor for bulk write if possible, or loop
        # Loop is safer for checking IDs
        for i, idx in enumerate(indices):
            with self.write_lock:
                # Sanity check: ensure the game hasn't been overwritten
                if ids is not None and int(self.id_buffer[idx].item()) != ids[i]:
                    continue

                # Update Raw Values/Policies
                self.value_buffer[idx] = new_values[i]
                self.policy_buffer[idx] = new_policies[i]

                # Note: We do not need to update rewards or actions usually in Reanalyze
                # And we definitely don't need to calc N-step here anymore!

    def clear(self):
        with self.write_lock:
            with self.priority_lock:
                self._size = torch.zeros(1, dtype=torch.int32).share_memory_()
                self.max_priority = self.initial_max_priority
                self.pointer = torch.zeros(1, dtype=torch.int64).share_memory_()
                self.tree_pointer = torch.zeros(1, dtype=torch.int64).share_memory_()
                self._next_id = torch.zeros(1, dtype=torch.int64).share_memory_()
                self._next_game_id = torch.zeros(1, dtype=torch.int64).share_memory_()

                self.observation_buffer = torch.zeros(
                    (self.max_size,) + self.observation_dimensions,
                    dtype=numpy_dtype_to_torch_dtype(self.observation_dtype),
                ).share_memory_()

                self.value_buffer = torch.zeros(
                    (self.max_size,), dtype=torch.float32
                ).share_memory_()
                self.policy_buffer = torch.zeros(
                    (self.max_size, self.num_actions), dtype=torch.float32
                ).share_memory_()
                self.reward_buffer = torch.zeros(
                    (self.max_size,), dtype=torch.float32
                ).share_memory_()
                self.action_buffer = torch.zeros(
                    (self.max_size,), dtype=torch.float16
                ).share_memory_()
                self.to_play_buffer = torch.zeros(
                    (self.max_size,), dtype=torch.int16
                ).share_memory_()

                # NEW: Chance Buffer
                # Storing as int16 assuming codes are small integers
                self.chance_buffer = torch.zeros(
                    (self.max_size, 1), dtype=torch.int16
                ).share_memory_()

                self.id_buffer = torch.zeros(
                    (self.max_size,), dtype=torch.int64
                ).share_memory_()
                self.game_id_buffer = torch.zeros(
                    (self.max_size,), dtype=torch.int64
                ).share_memory_()
                self.training_step_buffer = torch.zeros(
                    (self.max_size,), dtype=torch.int64
                ).share_memory_()
                self.legal_moves_mask_buffer = torch.zeros(
                    (self.max_size, self.num_actions), dtype=torch.bool
                ).share_memory_()

                tree_capacity = 1
                while tree_capacity < self.max_size:
                    tree_capacity *= 2
                self.sum_tree = SumSegmentTree(tree_capacity)
                self.min_tree = MinSegmentTree(tree_capacity)

    def _sample_proportional(self):
        with self.priority_lock:
            indices = []
            total_priority = np.float64(self.sum_tree.sum(0, len(self) - 1))
            priority_segment = total_priority / self.batch_size

            for i in range(self.batch_size):
                a = priority_segment * i
                b = priority_segment * (i + 1)
                upperbound = np.random.uniform(a, b)
                index = self.sum_tree.retrieve(upperbound)
                indices.append(index)
            return indices

    def _calculate_weight(self, index: int):
        priority_sample = self.sum_tree[index] / self.sum_tree.sum()
        weight = (priority_sample * len(self)) ** (-self.beta)
        assert weight > 0, "None positive weight: {}".format(weight)
        return weight

    def update_priorities(self, indices: list[int], priorities: list[float], ids=None):
        with self.priority_lock:
            # necessary for shared replay buffer
            if ids is not None:
                assert (
                    len(priorities) == len(ids) == len(indices)
                    or priorities.shape == ids.shape == indices.shape
                )

                for index, id, priority in zip(indices, ids, priorities):
                    assert (
                        priority > 0
                    ), "Negative priority: {} \n All priorities {}".format(
                        priority, priorities
                    )
                    assert 0 <= index < len(self)

                    if self.id_buffer[index] != id:
                        continue

                    self.sum_tree[index] = priority**self.alpha
                    self.min_tree[index] = priority**self.alpha
                    self.max_priority = max(self.max_priority, priority)
            else:
                assert len(indices) == len(priorities)
                for index, priority in zip(indices, priorities):
                    # print("Updating index", index, "with priority", priority)
                    assert priority > 0, "Negative priority: {}".format(priority)
                    assert 0 <= index < len(self)

                    self.sum_tree[index] = priority**self.alpha
                    self.min_tree[index] = priority**self.alpha
                    self.max_priority = max(
                        self.max_priority, priority
                    )  # could remove and clip priorities in experience replay isntead

            # return priorities**self.alpha

    def set_beta(self, beta: float):
        self.beta = beta

    @property
    def size(self):
        return int(self._size.item())

    @size.setter
    def size(self, val):
        self._size[0] = val

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["write_lock"]
        del state["priority_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.write_lock = mp.Lock()
        self.priority_lock = mp.Lock()


# # TODO: SHOULD BUFFER STORE TERMINAL OBSERVATIONS AND SHOULD MUZERO (ROOT) INITIAL INFERENCE TRAIN ON THEM?
# class MuZeroReplayBuffer(BaseReplayBuffer):
#     def __init__(
#         self,
#         observation_dimensions,
#         observation_dtype: type,
#         max_size: int,
#         num_actions: int,
#         batch_size: int,
#         n_step: int,
#         num_unroll_steps: int,
#         gamma: float,
#         # has_intermediate_rewards: bool,
#         num_players: int,
#         max_priority: float = 1.0,
#         alpha: float = 0.6,
#         beta: float = 0.4,
#         epsilon: float = 0.0001,
#         use_batch_weights: bool = False,
#         initial_priority_max: bool = False,
#         lstm_horizon_len: int = 5,
#         value_prefix: bool = False,
#         tau: float = 0.3,
#         # epsilon=0.01,
#     ):
#         assert alpha >= 0 and alpha <= 1
#         assert beta >= 0 and beta <= 1
#         assert n_step >= 1
#         assert gamma >= 0 and gamma <= 1

#         self.observation_dimensions = observation_dimensions
#         self.observation_dtype = observation_dtype
#         self.write_lock = (
#             mp.Lock()
#         )  # protects pointer, tree_pointer, and size reservation
#         self.priority_lock = (
#             mp.Lock()
#         )  # protects segment trees and max_priority updates

#         self.num_actions = num_actions
#         self.num_players = num_players

#         self.n_step = n_step
#         self.unroll_steps = num_unroll_steps
#         self.gamma = gamma
#         # self.has_intermediate_rewards = has_intermediate_rewards

#         self.initial_max_priority = max_priority
#         self.alpha = alpha
#         self.beta = beta
#         self.epsilon = epsilon

#         self.use_batch_weights = use_batch_weights
#         self.per_initial_priority_max = initial_priority_max

#         self.lstm_horizon_len = lstm_horizon_len
#         self.value_prefix = value_prefix

#         self.tau = tau

#         print("Warning: for board games it is recommnded to have n_step >= game length")
#         self.time_to_full = time()
#         # self.throughput_time = time()
#         # self.prev_buffer_size = 0

#         super().__init__(max_size=max_size, batch_size=batch_size)

#     def store_position(
#         self,
#         game: Game,
#         position: int,
#         training_step,
#         priority: float = None,
#         game_id=None,
#     ):
#         # Reserve an index (atomic with write_lock) and update pointer/tree_pointer/size
#         # with self.write_lock:
#         idx = int(self.pointer.item())
#         tree_idx = int(self.tree_pointer.item())  # advance pointers
#         next_ptr = (idx + 1) % self.max_size
#         next_tree_ptr = (tree_idx + 1) % self.max_size

#         self.pointer[0] = next_ptr
#         self.tree_pointer[0] = next_tree_ptr
#         if self.size < self.max_size:
#             self.size += 1

#         new_id = int(self._next_id.item()) + 1
#         self._next_id[0] = new_id

#         # Write data into buffers at reserved index WITHOUT holding the lock
#         self.observation_buffer[idx] = torch.from_numpy(
#             game.observation_history[position]
#         )
#         # if "legal_moves" not in game.info_history[position]:
#         #     print("warning legal moves not found in info")

#         self.legal_moves_mask_buffer[idx] = legal_moves_mask(
#             self.num_actions, game.info_history[position].get("legal_moves", [])
#         )

#         # _get_n_step_info now returns (values, policies, rewards, actions, to_plays)
#         values, policies, rewards, actions, to_plays = self._get_n_step_info(
#             position,
#             game.value_history,
#             game.policy_history,
#             game.rewards,
#             game.action_history,
#             game.info_history,
#             self.unroll_steps,
#             self.n_step,
#             self.lstm_horizon_len,
#         )
#         self.n_step_values_buffer[idx] = values
#         self.n_step_policies_buffer[idx] = policies
#         self.n_step_rewards_buffer[idx] = rewards
#         self.n_step_actions_buffer[idx] = actions
#         self.n_step_to_plays_buffer[idx] = to_plays  # NEW: store to_play sequence
#         self.id_buffer[idx] = new_id
#         self.game_id_buffer[idx] = game_id

#         self.training_step_buffer[idx] = training_step

#         if priority is None:
#             if self.per_initial_priority_max:
#                 priority = self.max_priority
#             else:
#                 priority = abs(game.value_history[position] - values[0]) + self.epsilon

#         # Update priority trees under priority_lock to avoid races with concurrent tree writes
#         with self.priority_lock:
#             self.sum_tree[tree_idx] = priority**self.alpha
#             self.min_tree[tree_idx] = priority**self.alpha
#             # update shared max_priority safely
#             if priority > self.max_priority:
#                 self.max_priority = priority

#     def store_game(self, game: Game, training_step):
#         n = len(game)

#         # Prepare data arrays locally (CPU) to maximize write speed inside the lock
#         # We assume game.history attributes are already lists or numpy arrays
#         # Stack them into tensors/arrays so we can do bulk slicing
#         obs_tensor = torch.from_numpy(np.stack(game.observation_history)).to(
#             self.observation_buffer.device
#         )[
#             :-1
#         ]  # [:-1] NOT STORING FINAL POSITION #TODO: CHANGE THIS MAYBE?

#         # Pre-calculate n-step returns for the whole game at once to avoid Python loops inside lock
#         # (This uses your existing logic but applied to the whole list)
#         # Note: optimizing _get_n_step_info to be vectorized is a separate task,
#         # but even calling it in a loop BEFORE the lock is better than inside the lock.
#         # For now, let's assume we calculate these lists before acquiring the lock:

#         vals, pols, rews, acts, tps = [], [], [], [], []
#         for i in range(n):
#             v, p, r, a, tp = self._get_n_step_info(
#                 i,
#                 game.value_history,
#                 game.policy_history,
#                 game.rewards,
#                 game.action_history,
#                 game.info_history,
#                 self.unroll_steps,
#                 self.n_step,
#                 self.lstm_horizon_len,
#             )
#             vals.append(v)
#             pols.append(p)
#             rews.append(r)
#             acts.append(a)
#             tps.append(tp)

#         # Stack into tensors for bulk assignment
#         vals_t = torch.stack(vals)
#         pols_t = torch.stack(pols)
#         rews_t = torch.stack(rews)
#         acts_t = torch.stack(acts)
#         tps_t = torch.stack(tps)

#         # Calculate Priorities (PER)
#         # For simplicity using max priority for new data (common optimization)
#         if self.per_initial_priority_max:
#             priorities = np.full(n, self.max_priority)
#         else:
#             # Vectorized priority calc could go here
#             priorities = np.full(n, self.max_priority)

#         # --- CRITICAL SECTION STARTS ---
#         with self.write_lock:
#             start_idx = int(self.pointer.item())
#             end_idx = (start_idx + n) % self.max_size

#             # Update Pointers
#             self.pointer[0] = end_idx
#             self.size += n
#             if self.size > self.max_size:
#                 self.size = self.max_size

#             # Update IDs
#             start_game_id = int(self._next_game_id.item()) + 1
#             self._next_game_id[0] = start_game_id

#             start_id = int(self._next_id.item())
#             ids = torch.arange(start_id + 1, start_id + n + 1, dtype=torch.long)
#             self._next_id[0] = start_id + n

#             # --- WRITE DATA (Handle Wrap-Around) ---

#             # Case 1: No Wrap-around (Contiguous write)
#             if start_idx + n <= self.max_size:
#                 rng = slice(start_idx, start_idx + n)

#                 self.observation_buffer[rng] = obs_tensor
#                 self.n_step_values_buffer[rng] = vals_t
#                 self.n_step_policies_buffer[rng] = pols_t
#                 self.n_step_rewards_buffer[rng] = rews_t
#                 self.n_step_actions_buffer[rng] = acts_t
#                 self.n_step_to_plays_buffer[rng] = tps_t

#                 self.game_id_buffer[rng] = start_game_id
#                 self.id_buffer[rng] = ids
#                 self.training_step_buffer[rng] = training_step

#                 # Create mask for legal moves (assuming you have a vectorized helper or list)
#                 # self.legal_moves_mask_buffer[rng] = ...

#             # Case 2: Wrap-around (Split write)
#             else:
#                 tail_len = self.max_size - start_idx
#                 head_len = n - tail_len

#                 # Tail (end of buffer)
#                 tail_rng = slice(start_idx, self.max_size)
#                 # Head (start of buffer)
#                 head_rng = slice(0, head_len)

#                 # Slice input data
#                 self.observation_buffer[tail_rng] = obs_tensor[:tail_len]
#                 self.observation_buffer[head_rng] = obs_tensor[tail_len:]

#                 self.n_step_values_buffer[tail_rng] = vals_t[:tail_len]
#                 self.n_step_values_buffer[head_rng] = vals_t[tail_len:]

#                 self.n_step_policies_buffer[tail_rng] = pols_t[:tail_len]
#                 self.n_step_policies_buffer[head_rng] = pols_t[tail_len:]

#                 self.n_step_rewards_buffer[tail_rng] = rews_t[:tail_len]
#                 self.n_step_rewards_buffer[head_rng] = rews_t[tail_len:]

#                 self.n_step_actions_buffer[tail_rng] = acts_t[:tail_len]
#                 self.n_step_actions_buffer[head_rng] = acts_t[tail_len:]

#                 self.n_step_to_plays_buffer[tail_rng] = tps_t[:tail_len]
#                 self.n_step_to_plays_buffer[head_rng] = tps_t[tail_len:]

#                 self.game_id_buffer[tail_rng] = start_game_id
#                 self.game_id_buffer[head_rng] = start_game_id

#                 self.id_buffer[tail_rng] = ids[:tail_len]
#                 self.id_buffer[head_rng] = ids[tail_len:]

#                 self.training_step_buffer[tail_rng] = training_step
#                 self.training_step_buffer[head_rng] = training_step

#         # Update Priorities Tree (Needs separate lock usually, or do it inside write_lock if simpler)
#         with self.priority_lock:
#             # You would need to handle wrap-around here for the tree updates too
#             # Simplest is just a loop over the range indices, as tree updates are usually element-wise
#             indices = list(range(start_idx, start_idx + n))
#             indices = [i % self.max_size for i in indices]
#             for idx, p in zip(indices, priorities):
#                 self.sum_tree[idx] = p**self.alpha
#                 self.min_tree[idx] = p**self.alpha

#     def store(self, game: Game, training_step):
#         # store() simply iterates; each store_position reserves its own index so we don't need a global lock here
#         self.store_game(game, training_step)
#         # with self.write_lock:
#         # game_id = int(self._next_game_id.item()) + 1
#         # self._next_game_id[0] = game_id
#         # for i in range(len(game)):
#         # dont store last position
#         # self.store_position(
#         #     game, i, training_step=training_step, game_id=game_id
#         # )
#         # print(self.game_id_buffer)
#         # print(self.id_buffer)
#         if self.size < 1000:
#             assert self.id_buffer.sum() > 0
#             assert self.game_id_buffer.sum() > 0
#             print("Buffer size:", self.size)
#         # print("Added a game to the buffer after {} seconds".format(elapsed_time))

#     def sample(self):
#         # Sampling is read-only. To maximize throughput we intentionally avoid taking locks here.
#         # This can return slightly-stale or concurrently-updated results which is common and acceptable
#         # in prioritized replay.
#         indices: list[int] = self._sample_proportional()
#         weights = torch.tensor(
#             [self._calculate_weight(i) for i in indices], dtype=torch.float32
#         )
#         if self.use_batch_weights:
#             weights = weights / weights.max()
#         else:
#             min_priority = self.min_tree.min() / self.sum_tree.sum()
#             max_weight = (min_priority * len(self)) ** (-self.beta)
#             weights = weights / max_weight

#         samples = self.sample_from_indices(indices)
#         samples.update(dict(weights=weights))
#         return samples

#     def update_priorities(self, indices: list[int], priorities: list[float], ids=None):
#         with self.priority_lock:
#             # necessary for shared replay buffer
#             if ids is not None:
#                 assert (
#                     len(priorities) == len(ids) == len(indices)
#                     or priorities.shape == ids.shape == indices.shape
#                 )

#                 for index, id, priority in zip(indices, ids, priorities):
#                     assert (
#                         priority > 0
#                     ), "Negative priority: {} \n All priorities {}".format(
#                         priority, priorities
#                     )
#                     assert 0 <= index < len(self)

#                     if self.id_buffer[index] != id:
#                         continue

#                     self.sum_tree[index] = priority**self.alpha
#                     self.min_tree[index] = priority**self.alpha
#                     self.max_priority = max(self.max_priority, priority)
#             else:
#                 assert len(indices) == len(priorities)
#                 for index, priority in zip(indices, priorities):
#                     # print("Updating index", index, "with priority", priority)
#                     assert priority > 0, "Negative priority: {}".format(priority)
#                     assert 0 <= index < len(self)

#                     self.sum_tree[index] = priority**self.alpha
#                     self.min_tree[index] = priority**self.alpha
#                     self.max_priority = max(
#                         self.max_priority, priority
#                     )  # could remove and clip priorities in experience replay isntead

#             # return priorities**self.alpha

#     def _sample_proportional(self):
#         with self.priority_lock:
#             indices = []
#             total_priority = np.float64(self.sum_tree.sum(0, len(self) - 1))
#             priority_segment = total_priority / self.batch_size

#             for i in range(self.batch_size):
#                 a = priority_segment * i
#                 b = priority_segment * (i + 1)

#                 if b > total_priority:
#                     print("warning b > total_priority")
#                     b = total_priority
#                 upperbound = np.random.uniform(a, b)
#                 index = self.sum_tree.retrieve(upperbound)
#                 indices.append(index)

#             return indices

#     def _calculate_weight(self, index: int):
#         # print("Sum tree sum:", self.sum_tree.sum())
#         priority_sample = self.sum_tree[index] / self.sum_tree.sum()
#         weight = (priority_sample * len(self)) ** (-self.beta)
#         weight = weight

#         return weight

#     # def _get_n_step_info(
#     #     self,
#     #     index: int,
#     #     values: list,
#     #     policies: list,
#     #     rewards: list,
#     #     actions: list,
#     #     infos: list,
#     #     num_unroll_steps: int,
#     #     n_step: int,
#     # ):
#     #     n_step_values = torch.zeros(num_unroll_steps + 1, dtype=torch.float32)
#     #     n_step_rewards = torch.zeros(num_unroll_steps + 1, dtype=torch.float32)
#     #     n_step_policies = torch.zeros(
#     #         (num_unroll_steps + 1, self.num_actions), dtype=torch.float32
#     #     )
#     #     n_step_actions = torch.zeros(num_unroll_steps, dtype=torch.int16)
#     #     for current_index in range(index, index + num_unroll_steps + 1):
#     #         unroll_step = current_index - index
#     #         bootstrap_index = current_index + n_step
#     #         # print("bootstrapping")
#     #         # value of current position is the value at the position n_steps away + rewards to get to the n_step position
#     #         if bootstrap_index < len(values):
#     #             if (
#     #                 "player" not in infos[current_index]
#     #                 or infos[current_index]["player"]
#     #                 == infos[bootstrap_index]["player"]
#     #             ):
#     #                 value = values[bootstrap_index] * self.gamma**n_step
#     #             else:
#     #                 value = -values[bootstrap_index] * self.gamma**n_step
#     #         else:
#     #             value = 0

#     #         # the rewards at this index to the bootstrap index should be added to the value
#     #         for i, reward in enumerate(rewards[current_index:bootstrap_index]):
#     #             # WHAT IS current_index + i + 1 when current index is the last frame?? IS THIS AN ERROR?
#     #             if (
#     #                 "player" not in infos[current_index]
#     #                 or infos[current_index]["player"]
#     #                 == infos[current_index + i][
#     #                     "player"
#     #                 ]  # + 1 if doing my og thing and i want to go back
#     #             ):
#     #                 value += reward * self.gamma**i
#     #             else:
#     #                 value -= reward * self.gamma**i

#     #         # target reward is the reward before the ones added to the value
#     #         if current_index > 0 and current_index <= len(rewards):
#     #             last_reward = rewards[current_index - 1]
#     #             # if self.has_intermediate_rewards:
#     #             #     last_reward = rewards[current_index - 1]
#     #             # else:
#     #             #     value += (
#     #             #         rewards[current_index - 1]
#     #             #         if infos[current_index]["player"]
#     #             #         == infos[current_index - 1]["player"]
#     #             #         else -rewards[current_index - 1]
#     #             #     )
#     #             #     last_reward = rewards[current_index - 1]  # reward not used
#     #         else:
#     #             last_reward = 0  # self absorbing state 0 reward

#     #         if current_index < len(values):
#     #             n_step_values[unroll_step] = value
#     #             n_step_rewards[unroll_step] = last_reward
#     #             n_step_policies[unroll_step] = policies[current_index]
#     #             if unroll_step < num_unroll_steps:
#     #                 # no action for last unroll step (since you dont act on that state)
#     #                 n_step_actions[unroll_step] = actions[current_index]
#     #         else:
#     #             n_step_values[unroll_step] = (
#     #                 value  # should be value or 0, maybe broken for single player
#     #             )
#     #             n_step_rewards[unroll_step] = last_reward
#     #             n_step_policies[unroll_step] = (
#     #                 torch.ones(self.num_actions) / self.num_actions
#     #             )  # self absorbing state
#     #             if unroll_step < num_unroll_steps:
#     #                 # no action for last unroll step (since you dont act on that state)
#     #                 n_step_actions[unroll_step] = -1  # self absorbing state

#     #     return (
#     #         n_step_values,  # [initial value, recurrent values]
#     #         n_step_policies,  # [initial policy, recurrent policies]
#     #         n_step_rewards,  # [initial reward (0), recurrent rewards] initial reward is useless like the first last action, but we ignore it in the learn function
#     #         n_step_actions,  # [recurrent actions, extra action]
#     #     )  # remove the last actions, as there should be one less action than other stuff

#     def _get_n_step_info(
#         self,
#         index: int,
#         values: list,
#         policies: list,
#         rewards: list,
#         actions: list,
#         infos: list,
#         num_unroll_steps: int,
#         n_step: int,
#         lstm_horizon_len: int = 5,
#     ):
#         """
#         Returns:
#             n_step_values: tensor shape (num_unroll_steps+1,)
#             n_step_policies: tensor shape (num_unroll_steps+1, num_actions)
#             n_step_rewards: tensor shape (num_unroll_steps+1,)  # n_step_rewards[0] == 0
#             n_step_actions: tensor shape (num_unroll_steps,)
#             n_step_to_plays: tensor shape (num_unroll_steps+1,)  # NEW: player id at each unroll step (or -1 if OOB)
#         Conventions:
#             - rewards[t] is the reward from taking action at state t (transition t → t+1)
#             - infos[t]["player"] is the player who acted at state t
#             - n_step_rewards[0] = 0 (no reward leading into root)
#         """
#         n_step_values = torch.zeros(num_unroll_steps + 1, dtype=torch.float32)
#         n_step_rewards = torch.zeros(num_unroll_steps + 1, dtype=torch.float32)
#         n_step_policies = torch.zeros(
#             (num_unroll_steps + 1, self.num_actions), dtype=torch.float32
#         )
#         n_step_actions = torch.zeros(num_unroll_steps, dtype=torch.float16)
#         n_step_to_plays = torch.zeros(
#             (num_unroll_steps + 1, self.num_players), dtype=torch.int16
#         )
#         value_prefix = 0.0
#         horizon_id = 0
#         root_player = (
#             infos[index].get("player", None)
#             if index < len(infos) and "player" in infos[index]
#             else None
#         )

#         for u in range(0, num_unroll_steps + 1):
#             current_index = index + u

#             # 1. discounted n-step value from current_index (same logic as before)
#             value = 0.0
#             for k in range(n_step):
#                 r_idx = current_index + k
#                 if r_idx < len(rewards):
#                     r = rewards[r_idx]
#                     node_player = (
#                         infos[current_index].get("player", None)
#                         if current_index < len(infos)
#                         else None
#                     )
#                     acting_player = (
#                         infos[r_idx].get("player", None) if r_idx < len(infos) else None
#                     )
#                     sign = (
#                         1.0
#                         if (
#                             node_player is None
#                             or acting_player is None
#                             or node_player == acting_player
#                         )
#                         else -1.0
#                     )
#                     value += (self.gamma**k) * (sign * r)
#                 else:
#                     break

#             boot_idx = current_index + n_step
#             if boot_idx < len(values):
#                 v_boot = values[boot_idx]
#                 node_player = (
#                     infos[current_index].get("player", None)
#                     if current_index < len(infos)
#                     else None
#                 )
#                 boot_player = (
#                     infos[boot_idx].get("player", None)
#                     if boot_idx < len(infos)
#                     else None
#                 )
#                 sign_leaf = (
#                     1.0
#                     if (
#                         node_player is None
#                         or boot_player is None
#                         or node_player == boot_player
#                     )
#                     else -1.0
#                 )
#                 value += (self.gamma**n_step) * (sign_leaf * v_boot)

#             n_step_values[u] = value

#             # 2. reward target with first cell zeroed
#             if self.value_prefix:
#                 # 2. Value Prefix (EfficientZero Logic)
#                 # Reset accumulation periodically based on LSTM horizon
#                 if horizon_id % lstm_horizon_len == 0:
#                     value_prefix = 0.0
#                 horizon_id += 1

#                 # Get immediate reward (root u=0 has 0 reward)
#                 current_reward = 0.0
#                 if u > 0:
#                     reward_idx = current_index - 1
#                     if reward_idx < len(rewards):
#                         # <-- **use raw reward directly** (no sign flipping)
#                         current_reward = rewards[reward_idx]

#                 # Accumulate into prefix
#                 value_prefix += current_reward
#                 n_step_rewards[u] = value_prefix
#             else:
#                 if u == 0:
#                     n_step_rewards[u] = 0.0  # root has no preceding reward
#                 else:
#                     reward_idx = current_index - 1
#                     n_step_rewards[u] = (
#                         rewards[reward_idx] if reward_idx < len(rewards) else 0.0
#                     )

#             # 3. policy
#             if current_index < len(policies):
#                 n_step_policies[u] = policies[current_index]
#             else:
#                 n_step_policies[u] = torch.ones(self.num_actions) / self.num_actions

#             # 4. action
#             if u < num_unroll_steps:
#                 # n_step_actions[u] = (
#                 # actions[current_index]
#                 # if current_index < len(actions)
#                 # else torch.nan
#                 # )
#                 if current_index < len(actions):
#                     n_step_actions[u] = actions[current_index]
#                 else:
#                     n_step_actions[u] = int(torch.randint(0, self.num_actions, (1,)))
#             # 5. to_play (NEW): store the player id for this state (or -1 if OOB)
#             if current_index < len(infos):
#                 if "player" in infos[current_index]:
#                     n_step_to_plays[u][infos[current_index]["player"]] = 1

#         return (
#             n_step_values,
#             n_step_policies,
#             n_step_rewards,
#             n_step_actions,
#             n_step_to_plays,
#         )

#     def set_beta(self, beta: float):
#         self.beta = beta

#     @property
#     def size(self):
#         # return self._size.value
#         return int(self._size.item())

#     @size.setter
#     def size(self, val):
#         # self._size.value = val
#         self._size[0] = val

#     def clear(self):
#         with self.write_lock:
#             with self.priority_lock:
#                 self._size = torch.zeros(1, dtype=torch.int32).share_memory_()
#                 self.max_priority = self.initial_max_priority
#                 self.pointer = torch.zeros(1, dtype=torch.int64).share_memory_()
#                 self.tree_pointer = torch.zeros(1, dtype=torch.int64).share_memory_()

#                 self.observation_buffer = torch.zeros(
#                     (self.max_size,) + self.observation_dimensions,
#                     dtype=numpy_dtype_to_torch_dtype(self.observation_dtype),
#                 ).share_memory_()

#                 self.n_step_rewards_buffer = torch.zeros(
#                     (self.max_size, self.unroll_steps + 1),
#                     dtype=torch.float32,
#                 ).share_memory_()
#                 self.n_step_policies_buffer = torch.zeros(
#                     (self.max_size, self.unroll_steps + 1, self.num_actions),
#                     dtype=torch.float32,
#                 ).share_memory_()
#                 self.n_step_values_buffer = torch.zeros(
#                     (self.max_size, self.unroll_steps + 1), dtype=torch.float32
#                 ).share_memory_()
#                 self.n_step_actions_buffer = torch.zeros(
#                     (self.max_size, self.unroll_steps),
#                     dtype=torch.float16,
#                 ).share_memory_()

#                 # NEW: buffer for to_play IDs (one per unroll step)
#                 self.n_step_to_plays_buffer = torch.zeros(
#                     (self.max_size, self.unroll_steps + 1, self.num_players),
#                     dtype=torch.int16,
#                 ).share_memory_()

#                 # add after n_step_to_plays_buffer creation
#                 self.id_buffer = torch.zeros(
#                     (self.max_size,), dtype=torch.int64
#                 ).share_memory_()
#                 self.game_id_buffer = torch.zeros(
#                     (self.max_size,), dtype=torch.int64
#                 ).share_memory_()

#                 self._next_id = torch.zeros(1, dtype=torch.int64).share_memory_()
#                 self._next_game_id = torch.zeros(1, dtype=torch.int64).share_memory_()

#                 self.legal_moves_mask_buffer = torch.zeros(
#                     (self.max_size, self.num_actions), dtype=torch.bool
#                 ).share_memory_()

#                 self.chance_buffer = torch.zeros((self.max_size, 1), dtype=torch.int16)

#                 self.training_step_buffer = torch.zeros(
#                     (self.max_size,), dtype=torch.int64
#                 ).share_memory_()

#                 tree_capacity = 1
#                 while tree_capacity < self.max_size:
#                     tree_capacity *= 2

#                 self.sum_tree = SumSegmentTree(tree_capacity)
#                 self.min_tree = MinSegmentTree(tree_capacity)

#     def sample_from_indices(self, indices: list[int]):
#         # TODO: FIX THIS MASKING STUFF IT SHOULD BE PADDED WITH A TRUE AT THE FRONT, HOWEVER THESE ARE NON TERMINAL -> TERMINAL is valid for everything but consistency loss
#         # --- LOGIC ADDED FOR OBSERVATION ROLLOUTS ---
#         # 1. Convert indices to tensor for vectorized operations (respect device if available)
#         try:
#             device = self.observation_buffer.device
#         except Exception:
#             device = None
#         if device is not None:
#             indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
#         else:
#             indices_tensor = torch.tensor(indices, dtype=torch.long)

#         # 2. Create offsets for unroll steps [1, 2, ..., K]
#         offsets = torch.arange(
#             0, self.unroll_steps + 1, dtype=torch.long, device=indices_tensor.device
#         ).unsqueeze(0)

#         # 3. Compute target indices with circular buffer wrapping
#         # Shape: (Batch, UnrollSteps)
#         target_indices = (indices_tensor.unsqueeze(1) + offsets) % self.max_size
#         # print(target_indices)
#         # 4. Validate Game IDs to prevent crossing episode boundaries or buffer overwrites
#         # Shape: (Batch, 1)
#         current_game_ids = self.game_id_buffer[indices_tensor].unsqueeze(1)
#         # print(current_game_ids)
#         # Shape: (Batch, UnrollSteps)
#         target_game_ids = self.game_id_buffer[target_indices]
#         # print(target_game_ids)
#         # print("current ids = target ids", current_game_ids == target_game_ids)
#         # Create a mask where the game ID matches (Valid transitions)
#         not_self_absorbing_mask = (
#             current_game_ids == target_game_ids
#         )  # dtype: bool, shape (B, U)

#         assert torch.all(
#             not_self_absorbing_mask[:, 0]
#         ), "The first step of the unroll (t -> t+1) must be valid and within the same episode."
#         # 5. Fetch unrolled observations
#         # Shape: (Batch, UnrollSteps, C, H, W)
#         unroll_observations = self.observation_buffer[target_indices].clone()

#         # base (t=0) observation for each sampled index
#         base_observations = self.observation_buffer[
#             indices_tensor
#         ]  # shape: (B, C, H, W)

#         # 6. Replace invalid frames with the last *valid* observation (absorbing-state behavior)
#         # We start from 1 because step 0 (root) is guaranteed valid by the assertion above.
#         # We go up to unroll_steps + 1 to cover the full unroll sequence.
#         output_unroll = unroll_observations  # already cloned above

#         for step in range(1, self.unroll_steps + 1):
#             is_absorbing = ~not_self_absorbing_mask[:, step]  # shape: (B,)

#             if is_absorbing.any():
#                 # Grab the observation from the previous step.
#                 # Because we iterate sequentially, (step - 1) has already been fixed/validated.
#                 prev_obs = output_unroll[:, step - 1]

#                 # FIX 1: Add ", step" to the index so we only write to the current timestep
#                 # FIX 2: prev_obs[is_absorbing] gets the (N, C, H, W) data we need
#                 output_unroll[is_absorbing, step] = prev_obs[is_absorbing]

#         # TODO: REPLACE THIS, TEMPORARY FIX FOR NON TERMINAL TO TERMINAL TRANSITIONS BEING COUNTED AS FALSE SINCE TERMINAL OBSERVATIONS ARE NOT STORED
#         batch_size = not_self_absorbing_mask.shape[0]
#         true_col = torch.ones(
#             (batch_size, 1), dtype=torch.bool, device=not_self_absorbing_mask.device
#         )

#         # Concat True at start, slice off the last column to keep shape (B, U+1)
#         not_self_absorbing_mask = torch.cat(
#             (true_col, not_self_absorbing_mask[:, :-1]), dim=1
#         )
#         # print("not self absorbing mask", not_self_absorbing_mask)

#         # Unroll observations should be (B, unroll_steps + 1, ...)
#         # It must contain the root (0) plus the K future steps
#         assert output_unroll.shape[1] == self.unroll_steps + 1, (
#             f"Unroll obs shape mismatch. Expected dim 1 to be {self.unroll_steps + 1} "
#             f"(Root + {self.unroll_steps} steps), but got {output_unroll.shape[1]}"
#         )

#         # --- Check 2: Verify Mask Topology (True...True -> False...False) ---
#         # We want to ensure we never transition from False (invalid) back to True (valid)
#         # 1. Cast mask to float/int for arithmetic: (B, Unroll+1)
#         mask_numeric = not_self_absorbing_mask.float()

#         # 2. Calculate difference between step t+1 and step t
#         # If True(1) -> False(0), result is -1 (Valid: episode ended)
#         # If False(0) -> False(0), result is 0 (Valid: still ended)
#         # If True(1) -> True(1), result is 0 (Valid: episode ongoing)
#         # If False(0) -> True(1), result is +1 (INVALID: Zombie state)
#         mask_diff = mask_numeric[:, 1:] - mask_numeric[:, :-1]

#         # 3. Assert no values are positive (meaning no 0->1 transitions)
#         if (mask_diff > 0).any():
#             invalid_indices = (mask_diff > 0).nonzero(as_tuple=False)
#             print(
#                 f"CRITICAL ERROR: Found {len(invalid_indices)} Invalid Mask Sequences!"
#             )
#             print(
#                 f"First invalid mask row: {not_self_absorbing_mask[invalid_indices[0][0]]}"
#             )
#             print(
#                 f"First invalid mask row game ids: {target_game_ids[invalid_indices[0][0]]}"
#             )
#             raise ValueError(
#                 "Masks contain 'holes' (False -> True transitions). Indexing logic is broken."
#             )

#         # 4. Check Root Validity
#         # The root (step 0) must always be valid for every sample in the batch
#         if not not_self_absorbing_mask[:, 0].all():
#             invalid_roots = (~not_self_absorbing_mask[:, 0]).sum()
#             raise ValueError(
#                 f"Found {invalid_roots} samples where the Root (t=0) is masked out. "
#                 "You are sampling garbage/padding data from the buffer."
#             )

#         # 7. Return everything (including the explicit unroll_masks)
#         return dict(
#             observations=self.observation_buffer[indices],
#             valid_masks=not_self_absorbing_mask,
#             unroll_observations=output_unroll,  # filled with last valid obs where needed
#             rewards=self.n_step_rewards_buffer[indices],
#             policy=self.n_step_policies_buffer[indices],
#             values=self.n_step_values_buffer[indices],
#             actions=self.n_step_actions_buffer[indices],
#             to_plays=self.n_step_to_plays_buffer[indices],
#             ids=self.id_buffer[indices].clone(),
#             legal_moves_masks=self.legal_moves_mask_buffer[indices],
#             indices=indices,
#             training_steps=self.training_step_buffer[indices],
#         )

#     def sample_game(self):
#         """
#         Retrieves all stored states for a specific game ID.
#         Useful for debugging or visualization, but slow (O(N) scan).
#         """
#         # Find all indices where game_id_buffer matches game_id
#         # Note: This is expensive on CPU, do not use in training loop
#         game_ids = list(set(list(self.game_id_buffer[: self.size])))
#         # print(self.game_id_buffer[: self.size])
#         # print(self.n_step_values_buffer[: self.size])
#         # print(game_ids)
#         game_id = np.random.choice(game_ids, 1)[0]
#         # print("game id", game_id)
#         mask = self.game_id_buffer[: self.size] == game_id
#         indices = torch.nonzero(mask).view(-1).tolist()
#         # print(indices)

#         if not indices:
#             print("game indices not found")
#             return None

#         # Sort indices to ensure chronological order (assuming sequential writes)
#         indices.sort()
#         return self.sample_from_indices(indices)

#     def reanalyze_game(
#         self,
#         indices,
#         new_policies,
#         new_values,
#         traj_rewards,
#         traj_actions,
#         traj_infos,
#         ids,
#         current_training_step,
#         total_training_steps,
#     ):
#         first_idx = indices[0]
#         t_st = int(self.training_step_buffer[first_idx].item())

#         # 2. Calculate Dynamic Horizon l (Eq 12)
#         # l = (k - floor((T_current - T_st) / (tau * T_total))).clip(1, k)
#         k = self.unroll_steps

#         # Avoid div by zero
#         denom = self.tau * total_training_steps

#         age_factor = (current_training_step - t_st) / denom
#         l_calc = k - np.floor(age_factor)

#         # Clip l between [1, k]
#         l = int(np.clip(l_calc, 1, k))

#         # Optional: Print debug info occasionally
#         # if np.random.random() < 0.01:
#         #     print(f"Reanalyze: Age={current_training_step - t_st}, Horizon l={l}")

#         for i, (idx, pid, value) in enumerate(zip(indices, new_policies, new_values)):
#             with self.write_lock:
#                 # check id again
#                 if int(self.id_buffer[idx].item()) != ids[indices.index(idx)]:
#                     continue
#                 # print(i)
#                 values, policies, rewards, actions, to_plays = self._get_n_step_info(
#                     i,
#                     new_values,
#                     new_policies,
#                     traj_rewards,
#                     traj_actions,
#                     traj_infos,
#                     self.unroll_steps,
#                     l,
#                     self.lstm_horizon_len,
#                 )
#                 # print("BEFORE")
#                 # print("value", self.n_step_values_buffer[idx])
#                 # print("policy", self.n_step_policies_buffer[idx])
#                 # print("reward", self.n_step_rewards_buffer[idx])
#                 # print("action", self.n_step_actions_buffer[idx])
#                 # print("to_plays", self.n_step_to_plays_buffer[idx])

#                 self.n_step_values_buffer[idx] = values
#                 self.n_step_policies_buffer[idx] = policies
#                 self.n_step_rewards_buffer[idx] = rewards
#                 self.n_step_actions_buffer[idx] = actions
#                 self.n_step_to_plays_buffer[idx] = (
#                     to_plays  # NEW: store to_play sequence
#                 )

#                 # print("AFTER")
#                 # print("value", self.n_step_values_buffer[idx])
#                 # print("policy", self.n_step_policies_buffer[idx])
#                 # print("reward", self.n_step_rewards_buffer[idx])
#                 # print("action", self.n_step_actions_buffer[idx])
#                 # print("to_plays", self.n_step_to_plays_buffer[idx])

#     def __getstate__(self):
#         state = self.__dict__.copy()

#         del state["write_lock"]
#         del state["priority_lock"]

#         assert "write_lock" not in state
#         assert "priority_lock" not in state
#         return state

#     def __setstate__(self, state):
#         self.__dict__.update(state)
#         self.write_lock = mp.Lock()
#         self.priority_lock = mp.Lock()
