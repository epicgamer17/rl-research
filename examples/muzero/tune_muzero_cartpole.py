"""
Optuna Hyperparameter Optimization for MuZero on CartPole-v1.

This script performs automated Bayesian hyperparameter search (TPE) with early pruning
(MedianPruner) to discover optimal configurations for MuZero on CartPole-v1.

Hyperparameter Search Space:
  - Architecture & Support: network_depth [1, 2, 3], latent_dim [32, 64], support_max [15, 20, 25, 30] (support_min=0 fixed)
  - Search: num_mcts_simulations [15, 35], dirichlet_alpha [0.2, 1.2], unroll_steps [3, 10], td_steps [3, 10]
  - Learning: learning_rate [1e-4, 2e-3] (log), value_loss_weight [0.2, 1.0], batch_size [32, 64, 128], gamma [0.99, 0.995, 0.997]
  - Replay & Freshness: use_per [True, False], buffer_capacity [100, 200, 500, 1000], param_sync_interval [10, 25, 50, 100], num_actors [1, 2, 4], envs_per_actor [1, 2, 4]

Usage:
  python examples/muzero/tune_muzero_cartpole.py --n-trials 20 --steps-per-trial 8000

  PROMISING TRIALS:
  Trial 4 (was constant upwards, not degrading but didnt reach 500 after 8000 steps but worth trying at like 20k steps)
  Trial 7! Does it keep up its good results? Can we tweek it to be more similar to existing params (from tic tac toe and the paper)
  trial 21 looked goated af too but 7 is better
  trial 36 looked super goated!
  trial 42!
"""

import argparse
import math
import os
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from tensordict import TensorDict
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from atomic_rl.action_selection import (
    add_dirichlet_noise,
    argmax_selector,
    sample_distribution,
)
from atomic_rl.buffers import SharedPERBuffer
from atomic_rl.losses import compute_cross_entropy_loss
from atomic_rl.schedules import get_stepwise_schedule
from atomic_rl.search import (
    get_mcts_visit_policy,
    mcts_search,
    puct_selection,
)
from atomic_rl.search.qtransforms import qtransform_by_min_max
from atomic_rl.transforms import (
    compute_contractive_transform,
    compute_expected_value,
    compute_inverse_contractive_transform,
    compute_two_hot_distribution,
)
from atomic_rl.utils import min_max_normalize, scale_gradient


# ---------------------------------------------------------------------------
# Default Constants & Environment Shapes
# ---------------------------------------------------------------------------
OBS_DIM = 4
NUM_ACTIONS = 2
C_PUCT = 1.25  # Fixed per design
DIRICHLET_EPSILON = 0.25
EPSILON_TRANSFORM = 0.001

SUPPORT_MIN = 0
DEFAULT_SUPPORT_MAX = 25

MAX_EPISODE_STEPS = 500
PARAM_SYNC_INTERVAL = 50
EVAL_INTERVAL_STEPS = 1000
NUM_TRAIN_EVAL_EPISODES = 5
NUM_FINAL_EVAL_EPISODES = 20


def make_env():
    """Constructs a Gymnasium CartPole-v1 environment."""
    return gym.make("CartPole-v1")


# ============================================================================
# 1. Parameterized MuZero MLP Architecture (Variable Depth)
# ============================================================================
def _make_mlp(in_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    """Builds a multi-layer MLP with LayerNorm and ReLU activations."""
    layers: List[nn.Module] = []
    curr_dim = in_dim
    for _ in range(depth):
        layers.extend(
            [
                nn.Linear(curr_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ]
        )
        curr_dim = hidden_dim
    return nn.Sequential(*layers)


class ParametricCartPoleMuZeroNet(nn.Module):
    """Depth-parameterized MuZero network for continuous vector environments."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        num_actions: int = NUM_ACTIONS,
        latent_dim: int = 32,
        depth: int = 2,
        support_min: int = SUPPORT_MIN,
        support_max: int = DEFAULT_SUPPORT_MAX,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.depth = depth
        self.support_min = support_min
        self.support_max = support_max
        self.num_atoms = support_max - support_min + 1

        # Discrete support buffer
        support = torch.arange(support_min, support_max + 1, dtype=torch.float32)
        self.register_buffer("support", support)

        # 1-hot action encodings
        self.register_buffer(
            "action_encodings", torch.eye(num_actions, dtype=torch.float32)
        )

        # 1. Representation Network h(o) -> s0
        self.representation_net = _make_mlp(obs_dim, latent_dim, depth)

        # 2. Dynamics Network g(s, a) -> (s', r_logits)
        self.dynamics_net = _make_mlp(latent_dim + num_actions, latent_dim, depth)
        self.reward_head = nn.Linear(latent_dim, self.num_atoms)

        # 3. Prediction Network f(s) -> (policy_logits, value_logits)
        self.prediction_backbone = _make_mlp(latent_dim, latent_dim, max(1, depth - 1))
        self.policy_head = nn.Linear(latent_dim, num_actions)
        self.value_head = nn.Linear(latent_dim, self.num_atoms)

    def representation(self, obs: torch.Tensor) -> torch.Tensor:
        raw_s = self.representation_net(obs)
        return min_max_normalize(raw_s, dim=-1)

    def dynamics(
        self, hidden_state: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if actions.ndim > 1:
            actions = actions.squeeze(-1)
        action_one_hot = self.action_encodings[actions]
        concat_input = torch.cat([hidden_state, action_one_hot], dim=-1)
        raw_next_hidden = self.dynamics_net(concat_input)
        next_hidden = min_max_normalize(raw_next_hidden, dim=-1)
        reward_logits = self.reward_head(next_hidden)
        return next_hidden, reward_logits

    def prediction(
        self, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.prediction_backbone(hidden_state)
        policy_logits = self.policy_head(feat)
        value_logits = self.value_head(feat)
        return policy_logits, value_logits

    def decode_support_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        scaled_val = compute_expected_value(predictions=logits, support=self.support)
        return compute_inverse_contractive_transform(
            scaled_val, epsilon=EPSILON_TRANSFORM
        )

    def initial_inference(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s0 = self.representation(obs)
        policy_logits, value_logits = self.prediction(s0)
        scalar_value = self.decode_support_to_scalar(value_logits)
        return s0, policy_logits, scalar_value, value_logits

    def recurrent_inference(
        self, hidden_state: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        next_hidden, reward_logits = self.dynamics(hidden_state, actions)
        policy_logits, value_logits = self.prediction(next_hidden)
        scalar_reward = self.decode_support_to_scalar(reward_logits)
        scalar_value = self.decode_support_to_scalar(value_logits)
        return (
            next_hidden,
            scalar_reward,
            policy_logits,
            scalar_value,
            reward_logits,
            value_logits,
        )


# ============================================================================
# 2. MCTS Helpers
# ============================================================================
def create_recurrent_fn(model: nn.Module, gamma: float) -> Callable:
    def recurrent_fn(actions_taken, embeddings, to_play):
        with torch.no_grad():
            (
                next_hidden,
                scalar_reward,
                logits,
                scalar_value,
                _,
                _,
            ) = model.recurrent_inference(embeddings, actions_taken)

        next_to_play = torch.zeros_like(to_play)
        value_vec = scalar_value.view(-1, 1)
        reward_vec = scalar_reward.view(-1, 1)
        discount_vec = torch.full_like(reward_vec, gamma)

        return (
            logits,
            value_vec,
            reward_vec,
            discount_vec,
            next_hidden,
            next_to_play,
        )

    return recurrent_fn


def run_batched_mcts(
    model: nn.Module,
    recurrent_fn: Callable,
    root_embed: torch.Tensor,
    num_simulations: int,
    dirichlet_alpha: float,
    dirichlet_epsilon: float = DIRICHLET_EPSILON,
    temperature: float = 1.0,
    explore_dirichlet: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = root_embed.shape[0]
    device = root_embed.device

    with torch.no_grad():
        s0, root_logits, root_scalar_value, _ = model.initial_inference(root_embed)

    root_to_play = torch.zeros((batch_size,), device=device, dtype=torch.long)
    root_vector_value = root_scalar_value.view(-1, 1)

    if explore_dirichlet and dirichlet_epsilon > 0.0:
        root_logits = add_dirichlet_noise(
            root_logits,
            epsilon=dirichlet_epsilon,
            alpha=dirichlet_alpha,
        )

    selection_fn = functools.partial(
        puct_selection,
        pb_c_init=C_PUCT,
        qtransform=qtransform_by_min_max,
    )

    tree = mcts_search(
        root_embeddings=s0,
        root_logits=root_logits,
        root_value=root_vector_value,
        root_to_play=root_to_play,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        num_actions=NUM_ACTIONS,
        num_players=1,
        selection_fn=selection_fn,
        legal_mask=None,
    )
    action_probs = get_mcts_visit_policy(
        tree["children_visits"][:, 0], temperature=temperature
    )
    root_search_value = tree["node_values"][:, 0, 0].detach()
    return action_probs, root_search_value


def evaluate_model(
    model: nn.Module,
    num_eval_episodes: int = NUM_TRAIN_EVAL_EPISODES,
    num_simulations: int = 25,
    gamma: float = 0.99,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
) -> dict:
    model.eval()
    envs = [make_env() for _ in range(num_eval_episodes)]
    recurrent_fn = create_recurrent_fn(model, gamma=gamma)

    obs_list = [env.reset(seed=seed + i * 100)[0] for i, env in enumerate(envs)]
    episode_returns = [0.0] * num_eval_episodes
    episode_lengths = [0] * num_eval_episodes
    dones = [False] * num_eval_episodes

    while not all(dones):
        active_indices = [i for i, d in enumerate(dones) if not d]
        active_obs = np.array([obs_list[i] for i in active_indices], dtype=np.float32)
        root_embed = torch.from_numpy(active_obs).to(device)

        action_probs, _ = run_batched_mcts(
            model,
            recurrent_fn,
            root_embed,
            num_simulations=num_simulations,
            dirichlet_alpha=0.3,
            temperature=0.0,
            explore_dirichlet=False,
        )
        action_idx_tensor, _ = argmax_selector(action_probs)
        actions = action_idx_tensor.squeeze(-1).cpu().numpy()

        for idx, act in zip(active_indices, actions):
            obs, reward, terminated, truncated, _ = envs[idx].step(int(act))
            episode_returns[idx] += reward
            episode_lengths[idx] += 1
            if terminated or truncated:
                dones[idx] = True
            else:
                obs_list[idx] = obs

    for env in envs:
        env.close()

    return {
        "eval/mean_episode_return": sum(episode_returns) / len(episode_returns),
        "eval/min_episode_return": min(episode_returns),
        "eval/max_episode_return": max(episode_returns),
        "eval/mean_episode_length": sum(episode_lengths) / len(episode_lengths),
    }


def make_parametric_model(
    latent_dim: int, depth: int, support_max: int = DEFAULT_SUPPORT_MAX
) -> ParametricCartPoleMuZeroNet:
    return ParametricCartPoleMuZeroNet(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        latent_dim=latent_dim,
        depth=depth,
        support_min=SUPPORT_MIN,
        support_max=support_max,
    )


# ============================================================================
# 3. Multiprocessing Workers
# ============================================================================
def actor_worker(
    actor_id: int,
    latent_dim: int,
    depth: int,
    shared_model: nn.Module,
    buffer: SharedPERBuffer,
    training_step: torch.Tensor,
    stop_event: mp.Event,
    envs_per_actor: int,
    num_simulations: int,
    unroll_steps: int,
    td_steps: int,
    gamma: float,
    dirichlet_alpha: float,
    total_steps: int,
    support_max: int = DEFAULT_SUPPORT_MAX,
    param_sync_interval: int = PARAM_SYNC_INTERVAL,
    device_str: str = "cpu",
    seed: int = 42,
):
    torch.set_num_threads(1)
    device = torch.device(device_str)
    local_model = make_parametric_model(latent_dim, depth, support_max=support_max).to(
        device
    )
    local_model.eval()
    local_model.load_state_dict(shared_model.state_dict())

    torch.manual_seed(seed + actor_id * 1000)
    random.seed(seed + actor_id * 1000)

    envs = [make_env() for _ in range(envs_per_actor)]
    obs_list = [
        env.reset(seed=seed + actor_id * 1000 + i)[0] for i, env in enumerate(envs)
    ]

    trajectories = [[] for _ in range(envs_per_actor)]
    recurrent_fn = create_recurrent_fn(local_model, gamma=gamma)
    loop_counter = 0

    temp_thresholds = [int(total_steps * 0.5), int(total_steps * 0.75)]
    temp_values = [1.0, 0.5, 0.25]

    while not stop_event.is_set():
        loop_counter += 1
        if loop_counter % param_sync_interval == 0:
            local_model.load_state_dict(shared_model.state_dict())

        completed_tds = []
        root_embed = torch.from_numpy(np.array(obs_list, dtype=np.float32)).to(device)

        curr_step = training_step[0].item()
        temp = get_stepwise_schedule(
            step=curr_step,
            thresholds=temp_thresholds,
            values=temp_values,
        )

        action_probs, root_values = run_batched_mcts(
            local_model,
            recurrent_fn,
            root_embed,
            num_simulations=num_simulations,
            dirichlet_alpha=dirichlet_alpha,
            temperature=1.0,
            explore_dirichlet=True,
        )

        for b_idx in range(envs_per_actor):
            target_policy = action_probs[b_idx]
            root_val = root_values[b_idx].item()

            if temp == 0.0:
                action_idx_tensor, _ = argmax_selector(target_policy.unsqueeze(0))
                action = action_idx_tensor.squeeze().item()
            elif temp == 1.0:
                dist = torch.distributions.Categorical(probs=target_policy)
                action_idx_t, _ = sample_distribution(dist, explore=True)
                action = action_idx_t.item()
            else:
                sampling_policy = torch.pow(target_policy, 1.0 / temp)
                sampling_policy = sampling_policy / sampling_policy.sum(
                    dim=-1, keepdim=True
                )
                dist = torch.distributions.Categorical(probs=sampling_policy)
                action_idx_t, _ = sample_distribution(dist, explore=True)
                action = action_idx_t.item()

            next_o, reward, terminated, truncated, _ = envs[b_idx].step(action)
            done = terminated or truncated

            trajectories[b_idx].append(
                (
                    root_embed[b_idx].cpu(),
                    action,
                    target_policy.cpu(),
                    root_val,
                    float(reward),
                )
            )

            if done:
                traj_len = len(trajectories[b_idx])
                obs_t = torch.stack([t[0] for t in trajectories[b_idx]])
                act_t = torch.tensor(
                    [t[1] for t in trajectories[b_idx]], dtype=torch.long
                )
                pol_t = torch.stack([t[2] for t in trajectories[b_idx]])
                val_raw = [t[3] for t in trajectories[b_idx]]
                rew_raw = [t[4] for t in trajectories[b_idx]]

                # Bootstrapped n-step return
                target_vals = []
                for t_idx in range(traj_len):
                    bootstrap_idx = t_idx + td_steps
                    if bootstrap_idx < traj_len:
                        val = val_raw[bootstrap_idx] * (gamma**td_steps)
                        for n in range(td_steps):
                            val += rew_raw[t_idx + n] * (gamma**n)
                    else:
                        val = 0.0
                        for n in range(traj_len - t_idx):
                            val += rew_raw[t_idx + n] * (gamma**n)
                    target_vals.append(val)

                val_t = torch.tensor(target_vals, dtype=torch.float32).unsqueeze(-1)
                rew_t = torch.tensor(rew_raw, dtype=torch.float32).unsqueeze(-1)

                pad_len = MAX_EPISODE_STEPS - traj_len
                if pad_len > 0:
                    state_pad = torch.zeros((pad_len, OBS_DIM), dtype=torch.float32)
                    act_pad = torch.zeros((pad_len,), dtype=torch.long)
                    pol_pad = torch.zeros((pad_len, NUM_ACTIONS), dtype=torch.float32)
                    val_pad = torch.zeros((pad_len, 1), dtype=torch.float32)
                    rew_pad = torch.zeros((pad_len, 1), dtype=torch.float32)
                    valid_pad = torch.zeros((pad_len,), dtype=torch.float32)

                    states = torch.cat([obs_t, state_pad], dim=0)
                    actions = torch.cat([act_t, act_pad], dim=0)
                    policies = torch.cat([pol_t, pol_pad], dim=0)
                    values = torch.cat([val_t, val_pad], dim=0)
                    rewards = torch.cat([rew_t, rew_pad], dim=0)
                    valid = torch.cat(
                        [torch.ones(traj_len, dtype=torch.float32), valid_pad], dim=0
                    )
                else:
                    states = obs_t[:MAX_EPISODE_STEPS]
                    actions = act_t[:MAX_EPISODE_STEPS]
                    policies = pol_t[:MAX_EPISODE_STEPS]
                    values = val_t[:MAX_EPISODE_STEPS]
                    rewards = rew_t[:MAX_EPISODE_STEPS]
                    valid = torch.ones(MAX_EPISODE_STEPS, dtype=torch.float32)

                priorities = torch.ones((MAX_EPISODE_STEPS,), dtype=torch.float32)
                ep_td = TensorDict(
                    {
                        "state": states,
                        "actions": actions,
                        "target_policies": policies,
                        "target_values": values,
                        "target_rewards": rewards,
                        "valid": valid,
                        "priority": priorities,
                    },
                    batch_size=[MAX_EPISODE_STEPS],
                )
                completed_tds.append(ep_td)
                trajectories[b_idx] = []
                next_o, _ = envs[b_idx].reset()

            obs_list[b_idx] = next_o

        if len(completed_tds) > 0:
            batch_td = torch.stack(completed_tds).to(device)
            buffer.add_transitions(batch_td, valid_mask=batch_td["valid"].bool())

    for env in envs:
        env.close()


def evaluator_worker(
    latent_dim: int,
    depth: int,
    shared_model: nn.Module,
    eval_queue: mp.Queue,
    training_step: torch.Tensor,
    stop_event: mp.Event,
    num_simulations: int,
    gamma: float,
    support_max: int = DEFAULT_SUPPORT_MAX,
    eval_interval_steps: int = EVAL_INTERVAL_STEPS,
    device_str: str = "cpu",
    seed: int = 42,
):
    torch.set_num_threads(1)
    device = torch.device(device_str)
    local_model = make_parametric_model(latent_dim, depth, support_max=support_max).to(
        device
    )
    next_eval_step = 0

    while not stop_event.is_set():
        curr_step = int(training_step[0].item())
        if curr_step < next_eval_step:
            time.sleep(0.5)
            continue

        next_eval_step = curr_step + eval_interval_steps
        local_model.load_state_dict(shared_model.state_dict())

        eval_metrics = evaluate_model(
            local_model,
            num_eval_episodes=NUM_TRAIN_EVAL_EPISODES,
            num_simulations=num_simulations,
            gamma=gamma,
            device=device,
            seed=seed,
        )
        eval_metrics["eval/step"] = curr_step
        eval_queue.put(eval_metrics)


def learner_worker(
    latent_dim: int,
    depth: int,
    shared_model: nn.Module,
    buffer: SharedPERBuffer,
    eval_queue: mp.Queue,
    result_queue: mp.Queue,
    training_step: torch.Tensor,
    stop_event: mp.Event,
    batch_size: int,
    learning_rate: float,
    value_loss_weight: float,
    unroll_steps: int,
    max_steps: int,
    use_per: bool,
    num_simulations: int,
    gamma: float,
    support_max: int = DEFAULT_SUPPORT_MAX,
    min_buffer_size: int = 500,
    device_str: str = "cpu",
    seed: int = 42,
):
    torch.set_num_threads(1)
    device = torch.device(device_str)
    local_model = make_parametric_model(latent_dim, depth, support_max=support_max).to(
        device
    )
    local_model.train()
    local_model.load_state_dict(shared_model.state_dict())

    optimizer = torch.optim.Adam(
        local_model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    rng_key = torch.Generator(device=device)
    rng_key.manual_seed(seed)

    support = torch.arange(
        SUPPORT_MIN, support_max + 1, device=device, dtype=torch.float32
    )
    step_count = 0

    while step_count < max_steps and not stop_event.is_set():
        beta_val = 1.0 if use_per else 0.0
        sample_res = buffer.sample_unrolls(
            batch_size, unroll_steps, beta=beta_val, min_size=min_buffer_size
        )
        if sample_res is None or sample_res[0] is None:
            time.sleep(0.01)
            continue

        unroll_data, valid_mask, coords, is_weights = sample_res
        is_weights = is_weights.to(device)

        ep_valid = unroll_data["valid"].bool() & valid_mask
        policy_masks = ep_valid.float().to(device)

        states = unroll_data["state"][:, 0].to(device)
        actions = unroll_data["actions"].to(device)
        target_policies = unroll_data["target_policies"].to(device)
        target_values = unroll_data["target_values"].to(device)
        target_rewards = unroll_data["target_rewards"].to(device)

        curr_s, p0, _, v0_logits = local_model.initial_inference(states)

        z0 = target_values[:, 0].squeeze(-1)
        z0_scaled = compute_contractive_transform(z0, epsilon=EPSILON_TRANSFORM)
        z0_two_hot = compute_two_hot_distribution(z0_scaled, SUPPORT_MIN, support_max)

        loss_p0, _ = compute_cross_entropy_loss(p0, target_policies[:, 0])
        mask0 = policy_masks[:, 0]
        loss_p0 = (loss_p0 * mask0).sum() / torch.clamp(mask0.sum(), min=1.0)
        raw_v0_loss, _ = compute_cross_entropy_loss(v0_logits, z0_two_hot)

        batch_loss = loss_p0 + value_loss_weight * raw_v0_loss

        for k in range(1, unroll_steps + 1):
            act_k = actions[:, k - 1]
            curr_s = scale_gradient(curr_s, 0.5)

            next_s, _, p_k, _, r_k_logits, v_k_logits = local_model.recurrent_inference(
                curr_s, act_k
            )
            curr_s = next_s

            raw_p_loss, _ = compute_cross_entropy_loss(p_k, target_policies[:, k])
            mask_k = policy_masks[:, k]
            l_p = (raw_p_loss * mask_k).sum() / torch.clamp(mask_k.sum(), min=1.0)

            zk = target_values[:, k].squeeze(-1)
            zk_scaled = compute_contractive_transform(zk, epsilon=EPSILON_TRANSFORM)
            zk_two_hot = compute_two_hot_distribution(
                zk_scaled, SUPPORT_MIN, support_max
            )
            raw_v_loss, _ = compute_cross_entropy_loss(v_k_logits, zk_two_hot)

            uk = target_rewards[:, k - 1].squeeze(-1)
            uk_scaled = compute_contractive_transform(uk, epsilon=EPSILON_TRANSFORM)
            uk_two_hot = compute_two_hot_distribution(
                uk_scaled, SUPPORT_MIN, support_max
            )
            raw_r_loss, _ = compute_cross_entropy_loss(r_k_logits, uk_two_hot)

            step_loss = l_p + value_loss_weight * raw_v_loss + raw_r_loss
            batch_loss = batch_loss + (step_loss / float(unroll_steps))

        if use_per:
            total_loss = (batch_loss * is_weights).mean()
        else:
            total_loss = batch_loss.mean()

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
        optimizer.step()

        if use_per:
            with torch.no_grad():
                v0_pred_scaled = compute_expected_value(v0_logits, support)
                v0_pred = compute_inverse_contractive_transform(
                    v0_pred_scaled, epsilon=EPSILON_TRANSFORM
                )
                new_priorities = torch.abs(v0_pred - z0).detach().cpu()
                buffer.update_priorities(coords, new_priorities, alpha=1.0)

        shared_model.load_state_dict(local_model.state_dict())
        step_count += 1
        training_step[0] = step_count

    # Final evaluation at the end of training
    if not stop_event.is_set():
        final_metrics = evaluate_model(
            local_model,
            num_eval_episodes=NUM_FINAL_EVAL_EPISODES,
            num_simulations=num_simulations,
            gamma=gamma,
            device=device,
            seed=seed,
        )
        result_queue.put(final_metrics["eval/mean_episode_return"])


# ============================================================================
# 4. Optuna Objective & Trial Runner
# ============================================================================
def run_trial(trial: optuna.Trial, config: dict, total_steps: int = 8000) -> float:
    """Runs a single MuZero training trial with Optuna intermediate reporting and pruning."""
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    device_str = "cpu"
    device = torch.device(device_str)

    latent_dim = config["latent_dim"]
    depth = config["network_depth"]
    support_max = config.get("support_max", DEFAULT_SUPPORT_MAX)
    buffer_capacity = config.get("buffer_capacity", 200)
    param_sync_interval = config.get("param_sync_interval", PARAM_SYNC_INTERVAL)

    shared_model = make_parametric_model(latent_dim, depth, support_max=support_max).to(
        device
    )
    shared_model.share_memory()

    training_step = torch.zeros(1, dtype=torch.long).share_memory_()
    stop_event = mp.Event()

    buffer_shapes = {
        "state": (OBS_DIM,),
        "actions": (),
        "target_policies": (NUM_ACTIONS,),
        "target_values": (1,),
        "target_rewards": (1,),
        "valid": (),
        "priority": (),
    }
    buffer = SharedPERBuffer(
        (buffer_capacity, MAX_EPISODE_STEPS),
        buffer_shapes,
        device=device,
        epsilon=1e-4,
    )
    eval_queue = mp.Queue()
    result_queue = mp.Queue()

    processes: List[mp.Process] = []

    learner_p = mp.Process(
        target=learner_worker,
        args=(
            latent_dim,
            depth,
            shared_model,
            buffer,
            eval_queue,
            result_queue,
            training_step,
            stop_event,
        ),
        kwargs={
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "value_loss_weight": config["value_loss_weight"],
            "unroll_steps": config["unroll_steps"],
            "max_steps": total_steps,
            "use_per": config["use_per"],
            "num_simulations": config["num_mcts_simulations"],
            "gamma": config["gamma"],
            "support_max": support_max,
            "min_buffer_size": min(500, max(20, total_steps // 10)),
            "device_str": device_str,
        },
    )
    learner_p.start()
    processes.append(learner_p)

    for actor_id in range(config["num_actors"]):
        actor_p = mp.Process(
            target=actor_worker,
            args=(
                actor_id,
                latent_dim,
                depth,
                shared_model,
                buffer,
                training_step,
                stop_event,
            ),
            kwargs={
                "envs_per_actor": config["envs_per_actor"],
                "num_simulations": config["num_mcts_simulations"],
                "unroll_steps": config["unroll_steps"],
                "td_steps": config["td_steps"],
                "gamma": config["gamma"],
                "dirichlet_alpha": config["dirichlet_alpha"],
                "total_steps": total_steps,
                "support_max": support_max,
                "param_sync_interval": param_sync_interval,
                "device_str": device_str,
            },
        )
        actor_p.start()
        processes.append(actor_p)

    eval_p = mp.Process(
        target=evaluator_worker,
        args=(latent_dim, depth, shared_model, eval_queue, training_step, stop_event),
        kwargs={
            "num_simulations": config["num_mcts_simulations"],
            "gamma": config["gamma"],
            "support_max": support_max,
            "eval_interval_steps": min(EVAL_INTERVAL_STEPS, max(50, total_steps // 5)),
            "device_str": device_str,
        },
    )
    eval_p.daemon = True
    eval_p.start()
    processes.append(eval_p)

    last_eval_score = 0.0
    pruned = False

    try:
        while learner_p.is_alive():
            # Check for evaluation checkpoints to report to Optuna
            try:
                metrics = eval_queue.get(timeout=1.0)
                mean_return = metrics["eval/mean_episode_return"]
                step = metrics["eval/step"]
                last_eval_score = mean_return

                trial.report(mean_return, step=step)
                print(
                    f"[Trial {trial.number} | Step {step}/{total_steps}] "
                    f"Eval Return: {mean_return:.1f} (min: {metrics['eval/min_episode_return']:.1f}, "
                    f"max: {metrics['eval/max_episode_return']:.1f})",
                    flush=True,
                )

                if trial.should_prune():
                    print(
                        f"[Trial {trial.number}] Pruned early at step {step} with return {mean_return:.1f}",
                        flush=True,
                    )
                    pruned = True
                    stop_event.set()
                    break
            except Exception:
                pass

        if not pruned:
            learner_p.join(timeout=30.0)
            try:
                final_return = result_queue.get(timeout=10.0)
                last_eval_score = final_return
            except Exception:
                pass

    finally:
        stop_event.set()
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)

    if pruned:
        raise optuna.TrialPruned()

    return float(last_eval_score)


def objective(
    trial: optuna.Trial, total_steps: int = 20000, force_per: bool = False
) -> float:
    """Optuna objective function sampling parameters and executing trial."""
    config = {
        # Search & Credit Assignment
        "unroll_steps": trial.suggest_int("unroll_steps", 3, 10),
        "td_steps": trial.suggest_int("td_steps", 3, 10),
        "num_mcts_simulations": trial.suggest_int(
            "num_mcts_simulations", 15, 35, step=5
        ),
        "dirichlet_alpha": trial.suggest_float("dirichlet_alpha", 0.2, 1.2),
        # Dynamics & Optimization
        "gamma": trial.suggest_categorical("gamma", [0.99, 0.995, 0.997]),
        "value_loss_weight": trial.suggest_float("value_loss_weight", 0.2, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        # Capacity, Support & Architecture
        "network_depth": trial.suggest_categorical("network_depth", [1, 2, 3]),
        "latent_dim": trial.suggest_categorical("latent_dim", [32, 64]),
        "support_max": trial.suggest_categorical("support_max", [15, 20, 25, 30]),
        # Replay, Freshness & Throughput (Worker Ratio)
        "use_per": True if force_per else trial.suggest_categorical("use_per", [True, False]),
        "buffer_capacity": trial.suggest_categorical(
            "buffer_capacity", [100, 200, 500, 1000]
        ),
        "param_sync_interval": trial.suggest_categorical(
            "param_sync_interval", [10, 25, 50, 100]
        ),
        "num_actors": trial.suggest_categorical("num_actors", [1, 2, 4]),
        "envs_per_actor": trial.suggest_categorical("envs_per_actor", [1, 2, 4]),
    }

    print(f"\n{'=' * 70}")
    print(f"Starting Trial {trial.number} with configuration:")
    for k, v in config.items():
        print(f"  {k:22s}: {v}")
    print(f"{'=' * 70}\n", flush=True)

    return run_trial(trial, config, total_steps=total_steps)


# ============================================================================
# 5. CLI & Main Study Runner
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Optimization for MuZero CartPole"
    )
    parser.add_argument(
        "--n-trials", type=int, default=20, help="Number of optimization trials to run"
    )
    parser.add_argument(
        "--steps-per-trial",
        type=int,
        default=8000,
        help="Training step budget per trial",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_muzero_cartpole.db",
        help="Optuna database storage URI (enables resuming)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="muzero_cartpole_optimization",
        help="Optuna study name",
    )
    parser.add_argument(
        "--force-per",
        action="store_true",
        help="Force Prioritized Experience Replay (use_per=True) and exclude it from the search space",
    )
    args = parser.parse_args()

    # If force-per is set and default study-name was not changed, use a dedicated PER study name
    study_name = args.study_name
    if args.force_per and args.study_name == "muzero_cartpole_optimization":
        study_name = "muzero_cartpole_per_optimization"

    # Median pruner discards trials performing worse than median at comparable steps
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=2000, interval_steps=1000)
    sampler = TPESampler(seed=42)

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    print(
        f"Starting Optuna Study '{study_name}' ({args.n_trials} trials, "
        f"{args.steps_per_trial} steps/trial, force_per={args.force_per})..."
    )
    print(f"Storage: {args.storage}\n", flush=True)

    study.optimize(
        lambda trial: objective(
            trial, total_steps=args.steps_per_trial, force_per=args.force_per
        ),
        n_trials=args.n_trials,
    )

    print("\n" + "=" * 70)
    print("HYPERPARAMETER OPTIMIZATION COMPLETED")
    print("=" * 70)
    print(f"Best Trial #{study.best_trial.number}")
    print(f"Best Mean Return: {study.best_value:.2f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k:22s}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
