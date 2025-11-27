from ast import Tuple
from time import time
import numpy as np
import torch
from packages.utils.utils.utils import legal_moves_mask, numpy_dtype_to_torch_dtype
from replay_buffers.base_replay_buffer import (
    BaseReplayBuffer,
    Game,
)
from replay_buffers.segment_tree import MinSegmentTree, SumSegmentTree
import torch
import torch.multiprocessing as mp


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
        # has_intermediate_rewards: bool,
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
        # epsilon=0.01,
    ):
        assert alpha >= 0 and alpha <= 1
        assert beta >= 0 and beta <= 1
        assert n_step >= 1
        assert gamma >= 0 and gamma <= 1

        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype
        self.write_lock = (
            mp.Lock()
        )  # protects pointer, tree_pointer, and size reservation
        self.priority_lock = (
            mp.Lock()
        )  # protects segment trees and max_priority updates

        self.num_actions = num_actions
        self.num_players = num_players

        self.n_step = n_step
        self.unroll_steps = num_unroll_steps
        self.gamma = gamma
        # self.has_intermediate_rewards = has_intermediate_rewards

        self.initial_max_priority = max_priority
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

        self.use_batch_weights = use_batch_weights
        self.per_initial_priority_max = initial_priority_max

        self.lstm_horizon_len = lstm_horizon_len
        self.value_prefix = value_prefix

        self.tau = tau

        print("Warning: for board games it is recommnded to have n_step >= game length")
        self.time_to_full = time()
        # self.throughput_time = time()
        # self.prev_buffer_size = 0

        super().__init__(max_size=max_size, batch_size=batch_size)

    def store_position(
        self,
        game: Game,
        position: int,
        training_step,
        priority: float = None,
        game_id=None,
    ):
        # Reserve an index (atomic with write_lock) and update pointer/tree_pointer/size
        with self.write_lock:
            idx = int(self.pointer.item())
            tree_idx = int(self.tree_pointer.item())  # advance pointers
            next_ptr = (idx + 1) % self.max_size
            next_tree_ptr = (tree_idx + 1) % self.max_size

            self.pointer[0] = next_ptr
            self.tree_pointer[0] = next_tree_ptr
            if self.size < self.max_size:
                self.size += 1

            new_id = int(self._next_id.item()) + 1
            self._next_id[0] = new_id

        # Write data into buffers at reserved index WITHOUT holding the lock
        self.observation_buffer[idx] = torch.from_numpy(
            game.observation_history[position]
        )
        # if "legal_moves" not in game.info_history[position]:
        #     print("warning legal moves not found in info")

        self.legal_moves_mask_buffer[idx] = legal_moves_mask(
            self.num_actions, game.info_history[position].get("legal_moves", [])
        )

        # _get_n_step_info now returns (values, policies, rewards, actions, to_plays)
        values, policies, rewards, actions, to_plays = self._get_n_step_info(
            position,
            game.value_history,
            game.policy_history,
            game.rewards,
            game.action_history,
            game.info_history,
            self.unroll_steps,
            self.n_step,
            self.lstm_horizon_len,
        )
        self.n_step_values_buffer[idx] = values
        self.n_step_policies_buffer[idx] = policies
        self.n_step_rewards_buffer[idx] = rewards
        self.n_step_actions_buffer[idx] = actions
        self.n_step_to_plays_buffer[idx] = to_plays  # NEW: store to_play sequence
        self.id_buffer[idx] = new_id
        self.game_id_buffer[idx] = game_id

        self.training_step_buffer[idx] = training_step

        if priority is None:
            if self.per_initial_priority_max:
                priority = self.max_priority
            else:
                priority = abs(game.value_history[position] - values[0]) + self.epsilon

        # Update priority trees under priority_lock to avoid races with concurrent tree writes
        with self.priority_lock:
            self.sum_tree[tree_idx] = priority**self.alpha
            self.min_tree[tree_idx] = priority**self.alpha
            # update shared max_priority safely
            if priority > self.max_priority:
                self.max_priority = priority

    def store(self, game: Game, training_step):
        # store() simply iterates; each store_position reserves its own index so we don't need a global lock here
        with self.write_lock:
            game_id = int(self._next_game_id.item())
            self._next_game_id[0] = game_id
        for i in range(len(game)):
            # dont store last position
            self.store_position(game, i, training_step=training_step, game_id=game_id)
        # print(self.game_id_buffer)
        # print(self.id_buffer)
        if self.size < 1000:
            print("Buffer size:", self.size)
        # print("Added a game to the buffer after {} seconds".format(elapsed_time))

    def sample(self):
        # Sampling is read-only. To maximize throughput we intentionally avoid taking locks here.
        # This can return slightly-stale or concurrently-updated results which is common and acceptable
        # in prioritized replay.
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

        samples = self.sample_from_indices(indices)
        samples.update(dict(weights=weights))
        return samples

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

    def _sample_proportional(self):
        indices = []
        total_priority = self.sum_tree.sum(0, len(self) - 1)
        priority_segment = total_priority / self.batch_size

        for i in range(self.batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            index = self.sum_tree.retrieve(upperbound)
            indices.append(index)

        return indices

    def _calculate_weight(self, index: int):
        # print("Sum tree sum:", self.sum_tree.sum())
        priority_sample = self.sum_tree[index] / self.sum_tree.sum()
        weight = (priority_sample * len(self)) ** (-self.beta)
        weight = weight

        return weight

    # def _get_n_step_info(
    #     self,
    #     index: int,
    #     values: list,
    #     policies: list,
    #     rewards: list,
    #     actions: list,
    #     infos: list,
    #     num_unroll_steps: int,
    #     n_step: int,
    # ):
    #     n_step_values = torch.zeros(num_unroll_steps + 1, dtype=torch.float32)
    #     n_step_rewards = torch.zeros(num_unroll_steps + 1, dtype=torch.float32)
    #     n_step_policies = torch.zeros(
    #         (num_unroll_steps + 1, self.num_actions), dtype=torch.float32
    #     )
    #     n_step_actions = torch.zeros(num_unroll_steps, dtype=torch.int16)
    #     for current_index in range(index, index + num_unroll_steps + 1):
    #         unroll_step = current_index - index
    #         bootstrap_index = current_index + n_step
    #         # print("bootstrapping")
    #         # value of current position is the value at the position n_steps away + rewards to get to the n_step position
    #         if bootstrap_index < len(values):
    #             if (
    #                 "player" not in infos[current_index]
    #                 or infos[current_index]["player"]
    #                 == infos[bootstrap_index]["player"]
    #             ):
    #                 value = values[bootstrap_index] * self.gamma**n_step
    #             else:
    #                 value = -values[bootstrap_index] * self.gamma**n_step
    #         else:
    #             value = 0

    #         # the rewards at this index to the bootstrap index should be added to the value
    #         for i, reward in enumerate(rewards[current_index:bootstrap_index]):
    #             # WHAT IS current_index + i + 1 when current index is the last frame?? IS THIS AN ERROR?
    #             if (
    #                 "player" not in infos[current_index]
    #                 or infos[current_index]["player"]
    #                 == infos[current_index + i][
    #                     "player"
    #                 ]  # + 1 if doing my og thing and i want to go back
    #             ):
    #                 value += reward * self.gamma**i
    #             else:
    #                 value -= reward * self.gamma**i

    #         # target reward is the reward before the ones added to the value
    #         if current_index > 0 and current_index <= len(rewards):
    #             last_reward = rewards[current_index - 1]
    #             # if self.has_intermediate_rewards:
    #             #     last_reward = rewards[current_index - 1]
    #             # else:
    #             #     value += (
    #             #         rewards[current_index - 1]
    #             #         if infos[current_index]["player"]
    #             #         == infos[current_index - 1]["player"]
    #             #         else -rewards[current_index - 1]
    #             #     )
    #             #     last_reward = rewards[current_index - 1]  # reward not used
    #         else:
    #             last_reward = 0  # self absorbing state 0 reward

    #         if current_index < len(values):
    #             n_step_values[unroll_step] = value
    #             n_step_rewards[unroll_step] = last_reward
    #             n_step_policies[unroll_step] = policies[current_index]
    #             if unroll_step < num_unroll_steps:
    #                 # no action for last unroll step (since you dont act on that state)
    #                 n_step_actions[unroll_step] = actions[current_index]
    #         else:
    #             n_step_values[unroll_step] = (
    #                 value  # should be value or 0, maybe broken for single player
    #             )
    #             n_step_rewards[unroll_step] = last_reward
    #             n_step_policies[unroll_step] = (
    #                 torch.ones(self.num_actions) / self.num_actions
    #             )  # self absorbing state
    #             if unroll_step < num_unroll_steps:
    #                 # no action for last unroll step (since you dont act on that state)
    #                 n_step_actions[unroll_step] = -1  # self absorbing state

    #     return (
    #         n_step_values,  # [initial value, recurrent values]
    #         n_step_policies,  # [initial policy, recurrent policies]
    #         n_step_rewards,  # [initial reward (0), recurrent rewards] initial reward is useless like the first last action, but we ignore it in the learn function
    #         n_step_actions,  # [recurrent actions, extra action]
    #     )  # remove the last actions, as there should be one less action than other stuff

    def _get_n_step_info(
        self,
        index: int,
        values: list,
        policies: list,
        rewards: list,
        actions: list,
        infos: list,
        num_unroll_steps: int,
        n_step: int,
        lstm_horizon_len: int = 5,
    ):
        """
        Returns:
            n_step_values: tensor shape (num_unroll_steps+1,)
            n_step_policies: tensor shape (num_unroll_steps+1, num_actions)
            n_step_rewards: tensor shape (num_unroll_steps+1,)  # n_step_rewards[0] == 0
            n_step_actions: tensor shape (num_unroll_steps,)
            n_step_to_plays: tensor shape (num_unroll_steps+1,)  # NEW: player id at each unroll step (or -1 if OOB)
        Conventions:
            - rewards[t] is the reward from taking action at state t (transition t → t+1)
            - infos[t]["player"] is the player who acted at state t
            - n_step_rewards[0] = 0 (no reward leading into root)
        """
        n_step_values = torch.zeros(num_unroll_steps + 1, dtype=torch.float32)
        n_step_rewards = torch.zeros(num_unroll_steps + 1, dtype=torch.float32)
        n_step_policies = torch.zeros(
            (num_unroll_steps + 1, self.num_actions), dtype=torch.float32
        )
        n_step_actions = torch.zeros(num_unroll_steps, dtype=torch.float16)
        n_step_to_plays = torch.zeros(
            (num_unroll_steps + 1, self.num_players), dtype=torch.int16
        )
        value_prefix = 0.0
        horizon_id = 0
        root_player = (
            infos[index].get("player", None)
            if index < len(infos) and "player" in infos[index]
            else None
        )

        for u in range(0, num_unroll_steps + 1):
            current_index = index + u

            # 1. discounted n-step value from current_index (same logic as before)
            value = 0.0
            for k in range(n_step):
                r_idx = current_index + k
                if r_idx < len(rewards):
                    r = rewards[r_idx]
                    node_player = (
                        infos[current_index].get("player", None)
                        if current_index < len(infos)
                        else None
                    )
                    acting_player = (
                        infos[r_idx].get("player", None) if r_idx < len(infos) else None
                    )
                    sign = (
                        1.0
                        if (
                            node_player is None
                            or acting_player is None
                            or node_player == acting_player
                        )
                        else -1.0
                    )
                    value += (self.gamma**k) * (sign * r)
                else:
                    break

            boot_idx = current_index + n_step
            if boot_idx < len(values):
                v_boot = values[boot_idx]
                node_player = (
                    infos[current_index].get("player", None)
                    if current_index < len(infos)
                    else None
                )
                boot_player = (
                    infos[boot_idx].get("player", None)
                    if boot_idx < len(infos)
                    else None
                )
                sign_leaf = (
                    1.0
                    if (
                        node_player is None
                        or boot_player is None
                        or node_player == boot_player
                    )
                    else -1.0
                )
                value += (self.gamma**n_step) * (sign_leaf * v_boot)

            n_step_values[u] = value

            # 2. reward target with first cell zeroed
            if self.value_prefix:
                # 2. Value Prefix (EfficientZero Logic)
                # Reset accumulation periodically based on LSTM horizon
                if horizon_id % lstm_horizon_len == 0:
                    value_prefix = 0.0
                horizon_id += 1

                # Get immediate reward (root u=0 has 0 reward)
                current_reward = 0.0
                if u > 0:
                    reward_idx = current_index - 1
                    if reward_idx < len(rewards):
                        # <-- **use raw reward directly** (no sign flipping)
                        current_reward = rewards[reward_idx]

                # Accumulate into prefix
                value_prefix += current_reward
                n_step_rewards[u] = value_prefix
            else:
                if u == 0:
                    n_step_rewards[u] = 0.0  # root has no preceding reward
                else:
                    reward_idx = current_index - 1
                    n_step_rewards[u] = (
                        rewards[reward_idx] if reward_idx < len(rewards) else 0.0
                    )

            # 3. policy
            if current_index < len(policies):
                n_step_policies[u] = policies[current_index]
            else:
                n_step_policies[u] = torch.ones(self.num_actions) / self.num_actions

            # 4. action
            if u < num_unroll_steps:
                # n_step_actions[u] = (
                # actions[current_index]
                # if current_index < len(actions)
                # else torch.nan
                # )
                if current_index < len(actions):
                    n_step_actions[u] = actions[current_index]
                else:
                    n_step_actions[u] = int(torch.randint(0, self.num_actions, (1,)))
            # 5. to_play (NEW): store the player id for this state (or -1 if OOB)
            if current_index < len(infos):
                if "player" in infos[current_index]:
                    n_step_to_plays[u][infos[current_index]["player"]] = 1

        return (
            n_step_values,
            n_step_policies,
            n_step_rewards,
            n_step_actions,
            n_step_to_plays,
        )

    def set_beta(self, beta: float):
        self.beta = beta

    @property
    def size(self):
        # return self._size.value
        return int(self._size.item())

    @size.setter
    def size(self, val):
        # self._size.value = val
        self._size[0] = val

    def clear(self):
        with self.write_lock:
            with self.priority_lock:
                self._size = torch.zeros(1, dtype=torch.int32).share_memory_()
                self.max_priority = self.initial_max_priority
                self.pointer = torch.zeros(1, dtype=torch.int64).share_memory_()
                self.tree_pointer = torch.zeros(1, dtype=torch.int64).share_memory_()

                self.observation_buffer = torch.zeros(
                    (self.max_size,) + self.observation_dimensions,
                    dtype=numpy_dtype_to_torch_dtype(self.observation_dtype),
                ).share_memory_()

                self.n_step_rewards_buffer = torch.zeros(
                    (self.max_size, self.unroll_steps + 1),
                    dtype=torch.float32,
                ).share_memory_()
                self.n_step_policies_buffer = torch.zeros(
                    (self.max_size, self.unroll_steps + 1, self.num_actions),
                    dtype=torch.float32,
                ).share_memory_()
                self.n_step_values_buffer = torch.zeros(
                    (self.max_size, self.unroll_steps + 1), dtype=torch.float32
                ).share_memory_()
                self.n_step_actions_buffer = torch.zeros(
                    (self.max_size, self.unroll_steps),
                    dtype=torch.float16,
                ).share_memory_()

                # NEW: buffer for to_play IDs (one per unroll step)
                self.n_step_to_plays_buffer = torch.zeros(
                    (self.max_size, self.unroll_steps + 1, self.num_players),
                    dtype=torch.int16,
                ).share_memory_()

                # add after n_step_to_plays_buffer creation
                self.id_buffer = torch.zeros(
                    (self.max_size,), dtype=torch.int64
                ).share_memory_()
                self.game_id_buffer = torch.zeros(
                    (self.max_size,), dtype=torch.int64
                ).share_memory_()

                self._next_id = torch.zeros(1, dtype=torch.int64).share_memory_()
                self._next_game_id = torch.zeros(1, dtype=torch.int64).share_memory_()

                self.legal_moves_mask_buffer = torch.zeros(
                    (self.max_size, self.num_actions), dtype=torch.bool
                ).share_memory_()

                self.training_step_buffer = torch.zeros(
                    (self.max_size,), dtype=torch.int64
                ).share_memory_()

                tree_capacity = 1
                while tree_capacity < self.max_size:
                    tree_capacity *= 2

                self.sum_tree = SumSegmentTree(tree_capacity)
                self.min_tree = MinSegmentTree(tree_capacity)

    def sample_from_indices(self, indices: list[int]):
        # --- LOGIC ADDED FOR OBSERVATION ROLLOUTS ---
        # 1. Convert indices to tensor for vectorized operations (respect device if available)
        try:
            device = self.observation_buffer.device
        except Exception:
            device = None
        if device is not None:
            indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        else:
            indices_tensor = torch.tensor(indices, dtype=torch.long)

        # 2. Create offsets for unroll steps [1, 2, ..., K]
        offsets = torch.arange(
            0, self.unroll_steps + 1, dtype=torch.long, device=indices_tensor.device
        ).unsqueeze(0)

        # 3. Compute target indices with circular buffer wrapping
        # Shape: (Batch, UnrollSteps)
        target_indices = (indices_tensor.unsqueeze(1) + offsets) % self.max_size

        # 4. Validate Game IDs to prevent crossing episode boundaries or buffer overwrites
        # Shape: (Batch, 1)
        current_game_ids = self.game_id_buffer[indices_tensor].unsqueeze(1)
        # Shape: (Batch, UnrollSteps)
        target_game_ids = self.game_id_buffer[target_indices]

        # Create a mask where the game ID matches (Valid transitions)
        not_self_absorbing_mask = (
            current_game_ids == target_game_ids
        )  # dtype: bool, shape (B, U)

        assert torch.all(
            not_self_absorbing_mask[:, 0]
        ), "The first step of the unroll (t -> t+1) must be valid and within the same episode."
        # 5. Fetch unrolled observations
        # Shape: (Batch, UnrollSteps, C, H, W)
        unroll_observations = self.observation_buffer[target_indices].clone()

        # base (t=0) observation for each sampled index
        base_observations = self.observation_buffer[
            indices_tensor
        ]  # shape: (B, C, H, W)

        # 6. Replace invalid frames with the last *valid* observation (absorbing-state behavior)
        # We'll iterate along the unroll dimension and for each invalid entry set it to the previous valid frame.
        # This handles consecutive invalid steps naturally.
        output_unroll = unroll_observations  # already cloned above

        for step in range(self.unroll_steps):
            not_self_absorbing = not_self_absorbing_mask[:, step]  # shape: (B,)
            self_absorbing = ~not_self_absorbing  # shape: (B,)
            if step == 0:
                prev_obs = (
                    base_observations  # use t=0 observation for first step's invalids
                )
            else:
                prev_obs = output_unroll[
                    :, step - 1
                ]  # use previously possibly-filled obs

            if self_absorbing.any():
                # Select only invalid batch elements and copy previous obs there
                output_unroll[self_absorbing] = prev_obs[self_absorbing]

        # Unroll observations should be (B, unroll_steps + 1, ...)
        # It must contain the root (0) plus the K future steps
        assert output_unroll.shape[1] == self.unroll_steps + 1, (
            f"Unroll obs shape mismatch. Expected dim 1 to be {self.unroll_steps + 1} "
            f"(Root + {self.unroll_steps} steps), but got {output_unroll.shape[1]}"
        )

        # --- Check 2: Verify Mask Topology (True...True -> False...False) ---
        # We want to ensure we never transition from False (invalid) back to True (valid)
        # 1. Cast mask to float/int for arithmetic: (B, Unroll+1)
        mask_numeric = not_self_absorbing_mask.float()

        # 2. Calculate difference between step t+1 and step t
        # If True(1) -> False(0), result is -1 (Valid: episode ended)
        # If False(0) -> False(0), result is 0 (Valid: still ended)
        # If True(1) -> True(1), result is 0 (Valid: episode ongoing)
        # If False(0) -> True(1), result is +1 (INVALID: Zombie state)
        mask_diff = mask_numeric[:, 1:] - mask_numeric[:, :-1]

        # 3. Assert no values are positive (meaning no 0->1 transitions)
        if (mask_diff > 0).any():
            invalid_indices = (mask_diff > 0).nonzero(as_tuple=False)
            print(
                f"CRITICAL ERROR: Found {len(invalid_indices)} Invalid Mask Sequences!"
            )
            print(
                f"First invalid mask row: {not_self_absorbing_mask[invalid_indices[0][0]]}"
            )
            raise ValueError(
                "Masks contain 'holes' (False -> True transitions). Indexing logic is broken."
            )

        # 4. Check Root Validity
        # The root (step 0) must always be valid for every sample in the batch
        if not not_self_absorbing_mask[:, 0].all():
            invalid_roots = (~not_self_absorbing_mask[:, 0]).sum()
            raise ValueError(
                f"Found {invalid_roots} samples where the Root (t=0) is masked out. "
                "You are sampling garbage/padding data from the buffer."
            )

        # 7. Return everything (including the explicit unroll_masks)
        return dict(
            observations=self.observation_buffer[indices],
            valid_masks=not_self_absorbing_mask,
            unroll_observations=output_unroll,  # filled with last valid obs where needed
            rewards=self.n_step_rewards_buffer[indices],
            policy=self.n_step_policies_buffer[indices],
            values=self.n_step_values_buffer[indices],
            actions=self.n_step_actions_buffer[indices],
            to_plays=self.n_step_to_plays_buffer[indices],
            ids=self.id_buffer[indices].clone(),
            legal_moves_masks=self.legal_moves_mask_buffer[indices],
            indices=indices,
            training_steps=self.training_step_buffer[indices],
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
        traj_rewards,
        traj_actions,
        traj_infos,
        ids,
        current_training_step,
        total_training_steps,
    ):
        first_idx = indices[0]
        t_st = int(self.training_step_buffer[first_idx].item())

        # 2. Calculate Dynamic Horizon l (Eq 12)
        # l = (k - floor((T_current - T_st) / (tau * T_total))).clip(1, k)
        k = self.unroll_steps

        # Avoid div by zero
        denom = self.tau * total_training_steps

        age_factor = (current_training_step - t_st) / denom
        l_calc = k - np.floor(age_factor)

        # Clip l between [1, k]
        l = int(np.clip(l_calc, 1, k))

        # Optional: Print debug info occasionally
        # if np.random.random() < 0.01:
        #     print(f"Reanalyze: Age={current_training_step - t_st}, Horizon l={l}")

        for i, (idx, pid, value) in enumerate(zip(indices, new_policies, new_values)):
            with self.write_lock:
                # check id again
                if int(self.id_buffer[idx].item()) != ids[indices.index(idx)]:
                    continue
                # print(i)
                values, policies, rewards, actions, to_plays = self._get_n_step_info(
                    i,
                    new_values,
                    new_policies,
                    traj_rewards,
                    traj_actions,
                    traj_infos,
                    self.unroll_steps,
                    l,
                    self.lstm_horizon_len,
                )
                # print("BEFORE")
                # print("value", self.n_step_values_buffer[idx])
                # print("policy", self.n_step_policies_buffer[idx])
                # print("reward", self.n_step_rewards_buffer[idx])
                # print("action", self.n_step_actions_buffer[idx])
                # print("to_plays", self.n_step_to_plays_buffer[idx])

                self.n_step_values_buffer[idx] = values
                self.n_step_policies_buffer[idx] = policies
                self.n_step_rewards_buffer[idx] = rewards
                self.n_step_actions_buffer[idx] = actions
                self.n_step_to_plays_buffer[idx] = (
                    to_plays  # NEW: store to_play sequence
                )

                # print("AFTER")
                # print("value", self.n_step_values_buffer[idx])
                # print("policy", self.n_step_policies_buffer[idx])
                # print("reward", self.n_step_rewards_buffer[idx])
                # print("action", self.n_step_actions_buffer[idx])
                # print("to_plays", self.n_step_to_plays_buffer[idx])

    def __getstate__(self):
        state = self.__dict__.copy()

        del state["write_lock"]
        del state["priority_lock"]

        assert "write_lock" not in state
        assert "priority_lock" not in state
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.write_lock = mp.Lock()
        self.priority_lock = mp.Lock()
