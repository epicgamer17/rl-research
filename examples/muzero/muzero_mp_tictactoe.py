"""
MuZero on PettingZoo TicTacToe (torch.multiprocessing Multiprocess Version)
=============================================================================

Paper Reference:
    "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"
    Schrittwieser et al., Nature 2020 (arXiv 2019: https://arxiv.org/abs/1911.08265)

Multiprocessing Architecture (Matching ape_x_mp_cartpole.py style):
    - 3 Actor Processes: Each actor runs batched MuZero self-play across 5 parallel
      environments simultaneously using batched MCTS (root embeddings shape [5, 3, 3, 2]).
    - 1 Learner Process: Continuously samples minibatches from SharedReplayBuffer,
      performs initial inference & K-step recurrent unrolling (K=5), updates joint
      representation/dynamics/prediction parameters, and updates shared weights.
    - 1 Async Evaluator Process: Periodically evaluates shared model against a Random
      baseline in the background, logging metrics to W&B asynchronously.
    - Shared Model Memory: Weights synced via shared_model.share_memory().
"""

import copy
import os
import time
import random
from typing import Tuple, List, Dict, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from tensordict import TensorDict
import wandb

from atomic_rl.search import mcts_search, get_mcts_visit_policy
from atomic_rl.losses import compute_cross_entropy_loss, compute_mse_loss
from atomic_rl.action_selection import argmax_selector, sample_distribution
from atomic_rl.buffers import SharedReplayBuffer
from atomic_rl.buffers.replay import (
    init_buffer,
    circular_write_strategy_,
    uniform_sample,
    BufferState,
)
from atomic_rl.evaluation import evaluate_pettingzoo_vs_random
from atomic_rl.envs.functions.tictactoe import check_tictactoe_winner, get_canonical_obs
from atomic_rl.envs.wrappers import PettingZooChannelFirstWrapper
from pettingzoo.classic import tictactoe_v3


# ---------------------------------------------------------------------------
# Hyperparameters & Constants
# ---------------------------------------------------------------------------
TOTAL_TRAINING_STEPS = 20000
NUM_ACTORS = 3
ENVS_PER_ACTOR = 5  # 5 parallel vectorized environments per actor
UNROLL_STEPS_K = 5  # Number of hypothetical unroll steps K = 5
MIN_BUFFER_SIZE = 64
EVAL_INTERVAL_STEPS = 500
PARAM_SYNC_INTERVAL = 50
NUM_MCTS_SIMULATIONS = 25

# MCTS PUCT Search Constants
C_PUCT_1 = 1.25
C_PUCT_2 = 19652.0
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

# Temperature Schedule Constants
TEMP_THRESHOLD_MOVES = 6
TEMPERATURE_EXPLORATION = 1.0
TEMPERATURE_EXPLOITATION = 0.0
TEMPERATURE_EVAL = 0.0

# Optimization & Network Parameters
BATCH_SIZE = 48
REPLAY_BUFFER_CAPACITY = 10000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_FILTERS = 24
NUM_RES_BLOCKS = 3

NUM_EVAL_GAMES = 20
SEED = 42


# ============================================================================
# 1. MuZero Representation, Dynamics & Prediction Neural Network
# ============================================================================


def encode_action_plane(
    action_idx: int, num_actions: int = 9, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    plane = torch.zeros(1, 1, 3, 3, device=device)
    row, col = action_idx // 3, action_idx % 3
    plane[0, 0, row, col] = 1.0
    return plane


class MuZeroTicTacToeNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_filters: int = NUM_FILTERS,
        num_res_blocks: int = NUM_RES_BLOCKS,
    ):
        super().__init__()
        self.num_filters = num_filters

        # 1. Representation Network h_theta(o_1...o_t) -> s_0
        self.representation_conv = nn.Conv2d(
            in_channels, num_filters, kernel_size=3, padding=1
        )
        self.representation_bn = nn.BatchNorm2d(num_filters)
        self.representation_res = nn.ModuleList(
            [ResNetBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # 2. Dynamics Network g_theta(s_k-1, a_k) -> r_k, s_k
        self.dynamics_conv = nn.Conv2d(
            num_filters + 1, num_filters, kernel_size=3, padding=1
        )
        self.dynamics_bn = nn.BatchNorm2d(num_filters)
        self.dynamics_res = nn.ModuleList(
            [ResNetBlock(num_filters) for _ in range(num_res_blocks)]
        )

        self.reward_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 3 * 3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )

        # 3. Prediction Network f_theta(s_k) -> p_k, v_k
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 3 * 3, 9),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1 * 3 * 3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def initial_inference(
        self, observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s0 = F.relu(self.representation_bn(self.representation_conv(observation)))
        for block in self.representation_res:
            s0 = block(s0)

        min_s0 = (
            s0.view(s0.shape[0], -1).min(dim=-1, keepdim=True).values.view(-1, 1, 1, 1)
        )
        max_s0 = (
            s0.view(s0.shape[0], -1).max(dim=-1, keepdim=True).values.view(-1, 1, 1, 1)
        )
        span = torch.where(
            max_s0 - min_s0 > 1e-5, max_s0 - min_s0, torch.ones_like(max_s0)
        )
        s0_scaled = (s0 - min_s0) / span

        policy_logits = self.policy_head(s0_scaled)
        value = self.value_head(s0_scaled)
        return s0_scaled, policy_logits, value

    def recurrent_inference(
        self, hidden_state: torch.Tensor, action_plane: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([hidden_state, action_plane], dim=1)
        sk = F.relu(self.dynamics_bn(self.dynamics_conv(x)))
        for block in self.dynamics_res:
            sk = block(sk)

        min_sk = (
            sk.view(sk.shape[0], -1).min(dim=-1, keepdim=True).values.view(-1, 1, 1, 1)
        )
        max_sk = (
            sk.view(sk.shape[0], -1).max(dim=-1, keepdim=True).values.view(-1, 1, 1, 1)
        )
        span = torch.where(
            max_sk - min_sk > 1e-5, max_sk - min_sk, torch.ones_like(max_sk)
        )
        sk_scaled = (sk - min_sk) / span

        reward = self.reward_head(sk_scaled)
        policy_logits = self.policy_head(sk_scaled)
        value = self.value_head(sk_scaled)

        return reward, sk_scaled, policy_logits, value


# ============================================================================
# 3. Thread & Process-Safe Shared Replay Buffer
# ============================================================================


# ============================================================================
# 3. Thread & Process-Safe Shared Replay Buffer
# ============================================================================


# SharedReplayBuffer is imported from atomic_rl.buffers


# ============================================================================
# 4. Multiprocessing Workers (Matching ape_x_mp_cartpole.py style)
# ============================================================================


def actor_worker(
    actor_id: int,
    num_actors: int,
    model_creator: Callable[[], nn.Module],
    shared_model: nn.Module,
    buffer: SharedReplayBuffer,
    envs_per_actor: int = ENVS_PER_ACTOR,
    num_simulations: int = NUM_MCTS_SIMULATIONS,
    device_str: str = "cpu",
):
    device = torch.device(device_str)
    local_model = model_creator().to(device)
    local_model.eval()
    local_model.load_state_dict(shared_model.state_dict())

    worker_seed = SEED + actor_id * 1000
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)

    boards = torch.zeros(envs_per_actor, 3, 3, device=device)
    players = torch.zeros(envs_per_actor, dtype=torch.long, device=device)
    move_counts = [0] * envs_per_actor
    trajectories = [[] for _ in range(envs_per_actor)]

    step_counter = 0

    while True:
        step_counter += 1
        if step_counter % PARAM_SYNC_INTERVAL == 0:
            local_model.load_state_dict(shared_model.state_dict())

        # Construct canonical obs batch [5, 3, 3, 3]
        obs_batch = torch.stack(
            [
                get_canonical_obs(boards[b], players[b].item()).squeeze(0)
                for b in range(envs_per_actor)
            ],
            dim=0,
        )

        root_legal_mask = boards.view(envs_per_actor, -1) == 0

        with torch.no_grad():
            s0_batch, _, _ = local_model.initial_inference(obs_batch)

        def muzero_expansion_fn(embeddings):
            with torch.no_grad():
                logits = local_model.policy_head(embeddings)
                value = local_model.value_head(embeddings)
                return logits, value.squeeze(-1)

        def muzero_dynamics_fn(embeddings, actions_taken):
            with torch.no_grad():
                action_planes = torch.cat(
                    [
                        encode_action_plane(a.item(), device=device)
                        for a in actions_taken
                    ],
                    dim=0,
                )
                reward, sk, _, _ = local_model.recurrent_inference(
                    embeddings, action_planes
                )
                return sk, reward.squeeze(-1)

        tree = mcts_search(
            root_embeddings=s0_batch,
            num_simulations=num_simulations,
            num_actions=9,
            expansion_fn=muzero_expansion_fn,
            dynamics_fn=muzero_dynamics_fn,
            root_to_play=players,
            root_legal_mask=root_legal_mask,
            pb_c_base=C_PUCT_2,
            pb_c_init=C_PUCT_1,
            dirichlet_epsilon=DIRICHLET_EPSILON,
            dirichlet_alpha=DIRICHLET_ALPHA,
        )

        for b_idx in range(envs_per_actor):
            move_counts[b_idx] += 1
            curr_player = players[b_idx].item()
            action_mask = (boards[b_idx] == 0).view(-1)
            root_visits = tree["children_visits"][b_idx, 0]

            raw_target_policy = get_mcts_visit_policy(
                root_visits.unsqueeze(0), temperature=1.0
            ).squeeze(0)
            target_policy = torch.where(action_mask, raw_target_policy, 0.0)
            pol_sum = target_policy.sum()
            target_policy = (
                target_policy / pol_sum
                if pol_sum > 0
                else action_mask.float() / action_mask.float().sum()
            )

            temp = (
                TEMPERATURE_EXPLORATION
                if move_counts[b_idx] <= TEMP_THRESHOLD_MOVES
                else TEMPERATURE_EXPLOITATION
            )
            action_policy = get_mcts_visit_policy(
                root_visits.unsqueeze(0), temperature=temp
            ).squeeze(0)
            action_policy = torch.where(action_mask, action_policy, 0.0)
            act_sum = action_policy.sum()
            action_policy = (
                action_policy / act_sum
                if act_sum > 0
                else action_mask.float() / action_mask.float().sum()
            )

            if temp > 0.0:
                dist = torch.distributions.Categorical(probs=action_policy)
                action_idx_t, _ = sample_distribution(dist, explore=True)
                action_idx = action_idx_t.item()
            else:
                action_idx_t, _ = argmax_selector(action_policy.unsqueeze(0))
                action_idx = action_idx_t.squeeze().item()

            canonical_obs = get_canonical_obs(boards[b_idx], curr_player).squeeze(0)

            trajectories[b_idx].append(
                {
                    "state": canonical_obs.cpu(),
                    "action": action_idx,
                    "target_policy": target_policy.cpu(),
                    "player": curr_player,
                }
            )

            row, col = action_idx // 3, action_idx % 3
            rewards_vec, is_term = check_tictactoe_winner(boards[b_idx].unsqueeze(0))

            if is_term.item():
                p0_win_val = rewards_vec[0, 0].item()
                p0_reward = 0.5
                if p0_win_val > 0:
                    p0_reward = 1.0
                elif p0_win_val < 0:
                    p0_reward = 0.0

                p1_reward = 1.0 - p0_reward

                for step in trajectories[b_idx]:
                    pl = step["player"]
                    z = p0_reward if pl == 0 else p1_reward
                    completed_samples.append(
                        {
                            "state": step["state"],
                            "action": torch.tensor(step["action"], dtype=torch.long),
                            "target_policy": step["target_policy"],
                            "target_value": torch.tensor([z], dtype=torch.float32),
                        }
                    )

                boards[b_idx].zero_()
                players[b_idx] = 0
                move_counts[b_idx] = 0
                trajectories[b_idx] = []
            else:
                players[b_idx] = 1 - curr_player

        if len(completed_samples) > 0:
            batch_td = TensorDict(
                {
                    "state": torch.stack([s["state"] for s in completed_samples]),
                    "action": torch.stack([s["action"] for s in completed_samples]),
                    "target_policy": torch.stack(
                        [s["target_policy"] for s in completed_samples]
                    ),
                    "target_value": torch.stack(
                        [s["target_value"] for s in completed_samples]
                    ),
                },
                batch_size=[len(completed_samples)],
            ).to(device)
            buffer.add_samples(batch_td)


def evaluator_worker(
    model_creator: Callable[[], nn.Module],
    shared_model: nn.Module,
    eval_queue: mp.Queue,
    num_eval_games: int = NUM_EVAL_GAMES,
    device_str: str = "cpu",
):

    device = torch.device(device_str)
    local_model = model_creator().to(device)
    local_model.eval()

    def select_eval_action(env, curr_player_idx, obs_t, action_mask_t):
        with torch.no_grad():
            _, init_logits, _ = local_model.initial_inference(obs_t.unsqueeze(0))
        masked_logits = apply_action_mask(init_logits, action_mask_t.unsqueeze(0))
        target_policy = F.softmax(masked_logits, dim=-1).squeeze(0)
        action_idx_tensor, _ = argmax_selector(target_policy.unsqueeze(0))
        return action_idx_tensor.squeeze().item()

    while True:
        local_model.load_state_dict(shared_model.state_dict())
        eval_metrics = evaluate_pettingzoo_vs_random(
            env_fn=lambda: PettingZooChannelFirstWrapper(tictactoe_v3.env()),
            select_action_fn=select_eval_action,
            num_games=num_eval_games,
            device=device,
        )
        eval_queue.put(eval_metrics)




def learner_worker(
    model_creator: Callable[[], nn.Module],
    shared_model: nn.Module,
    buffer: SharedReplayBuffer,
    eval_queue: mp.Queue,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    unroll_steps_k: int = UNROLL_STEPS_K,
    max_steps: int = TOTAL_TRAINING_STEPS,
    device_str: str = "cpu",
):
    device = torch.device(device_str)
    local_model = model_creator().to(device)
    local_model.train()
    local_model.load_state_dict(shared_model.state_dict())

    optimizer = torch.optim.Adam(
        local_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    rng_key = torch.Generator(device=device)
    rng_key.manual_seed(SEED)

    wandb.init(
        project="muzero-tictactoe",
        name=f"muzero_mp_res{NUM_RES_BLOCKS}_f{NUM_FILTERS}_sims{NUM_MCTS_SIMULATIONS}_actors{NUM_ACTORS}",
        config={
            "total_training_steps": max_steps,
            "num_actors": NUM_ACTORS,
            "envs_per_actor": ENVS_PER_ACTOR,
            "unroll_steps_k": unroll_steps_k,
            "num_mcts_simulations": NUM_MCTS_SIMULATIONS,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": SEED,
        },
    )
    wandb.define_metric("*", step_metric="global_step")

    step_count = 0
    latest_eval_metrics = {}

    while step_count < max_steps:
        minibatch = buffer.sample_batch(rng_key, batch_size, MIN_BUFFER_SIZE)
        if minibatch is None:
            time.sleep(0.05)
            continue

        obs_batch = minibatch["state"].to(device)
        target_policies = minibatch["target_policy"].to(device)
        target_values = minibatch["target_value"].to(device)
        actions = minibatch["action"].to(device)

        # Initial inference k=0
        sk, p_logits, v_pred = local_model.initial_inference(obs_batch)

        raw_p_loss, _ = compute_cross_entropy_loss(p_logits, target_policies)
        raw_v_loss, _ = compute_mse_loss(v_pred.view(-1), target_values.view(-1))

        # Loss scaling: 1/K on each unrolled head (Note 51)
        # TODO: confirm that this behaviour for question 51 is correct
        total_loss = (raw_p_loss.mean() + raw_v_loss.mean()) * (1.0 / UNROLL_STEPS_K)

        # Unroll K hypothetical steps recurrently
        for k in range(1, UNROLL_STEPS_K + 1):
            target_policies = minibatch["target_policy_seq"][:, k]
            target_values = minibatch["target_value_seq"][:, k]

            # Use action from sequence at step k-1 (shape [B])
            action = minibatch["action_seq"][:, k - 1]
            action_planes = (
                torch.zeros((action.size(0), 1, 3, 3), device=device) + action.view(-1, 1, 1, 1).float() / 8.0
            )

            # Recurrent Dynamics & Prediction
            sk, p_logits_k, v_pred_k = local_model.recurrent_inference(
                sk, action_planes
            )

            raw_pk_loss, _ = compute_cross_entropy_loss(p_logits_k, target_policies)
            raw_vk_loss, _ = compute_mse_loss(v_pred_k.view(-1), target_values.view(-1))

            step_loss = (raw_pk_loss.mean() + raw_vk_loss.mean()) * (
                1.0 / unroll_steps_k
            )
            total_loss += step_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update shared model memory
        shared_model.load_state_dict(local_model.state_dict())

        step_count += 1

        # Check for fresh evaluation metrics from background evaluator
        while not eval_queue.empty():
            try:
                latest_eval_metrics = eval_queue.get_nowait()
            except Exception:
                break

        log_dict = {
            "global_step": step_count,
            "loss/total": total_loss.item(),
            "buffer/size": buffer.size,
            "search/mcts_simulations": NUM_MCTS_SIMULATIONS,
        }
        log_dict.update(latest_eval_metrics)

        if step_count % 100 == 0:
            print(
                f"[Learner Step {step_count}/{max_steps}] Loss: {total_loss.item():.4f} | Buffer: {buffer.size}"
            )

        wandb.log(log_dict)

    wandb.finish()


# ============================================================================
# 5. Main Multiprocessing Setup
# ============================================================================


def model_creator_fn() -> MuZeroTicTacToeNet:
    return MuZeroTicTacToeNet(
        in_channels=3, num_filters=NUM_FILTERS, num_res_blocks=NUM_RES_BLOCKS
    )


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    device_str = "cpu"
    device = torch.device(device_str)

    shared_model = model_creator_fn().to(device)
    shared_model.share_memory()

    buffer_shapes = {
        "state": (3, 3, 3),
        "action": (),
        "target_policy": (9,),
        "target_value": (1,),
    }
    buffer = SharedReplayBuffer(REPLAY_BUFFER_CAPACITY, buffer_shapes, device=device)
    eval_queue = mp.Queue()

    processes = []

    # Start Learner Process
    learner_p = mp.Process(
        target=learner_worker,
        args=(model_creator_fn, shared_model, buffer, eval_queue),
        kwargs={"device_str": device_str},
    )
    learner_p.start()
    processes.append(learner_p)

    # Start 3 Actor Processes
    for actor_id in range(NUM_ACTORS):
        actor_p = mp.Process(
            target=actor_worker,
            args=(actor_id, NUM_ACTORS, model_creator_fn, shared_model, buffer),
            kwargs={"device_str": device_str},
        )
        actor_p.start()
        processes.append(actor_p)

    # Start Async Evaluator Process
    eval_p = mp.Process(
        target=evaluator_worker,
        args=(model_creator_fn, shared_model, eval_queue),
        kwargs={"device_str": device_str},
    )
    eval_p.daemon = True
    eval_p.start()
    processes.append(eval_p)

    print(
        f"Launched MuZero Multiprocessing Pipeline (1 Learner, {NUM_ACTORS} Actors with {ENVS_PER_ACTOR} Envs per Actor, 1 Evaluator)."
    )

    learner_p.join()

    for p in processes:
        if p.is_alive():
            p.terminate()

    print("MuZero Multiprocessing Training Completed Successfully.")


if __name__ == "__main__":
    main()
