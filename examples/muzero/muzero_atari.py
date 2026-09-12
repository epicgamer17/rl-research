"""MuZero on Gymnasium CartPole-v1 (torch.multiprocessing Multiprocess Version)

================================================================================

Paper Reference:
    "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"
    Schrittwieser et al., Nature 2020 (arXiv 2019: https://arxiv.org/abs/1911.08265)

Goal & Domain:
    This script demonstrates MuZero in the Single-Player / Atari regime (Section 2,
    Appendix B, Appendix F, and Table S1 of the paper) applied to Gymnasium CartPole-v1.
    Unlike board games (which use undiscounted terminal outcomes and MSE loss), single-player
    domains with intermediate rewards feature:
      1. Single Player Search: num_players=1, to_play=0, no opponent turn-flipping.
      2. Exponential Discounting: gamma=0.99, intermediate transition rewards r=+1.0.
      3. Categorical Support Softmax Heads: Both value and reward heads output discrete
         probability distributions over an integer support [-20, 20] (41 atoms).
      4. Contractive Invertible Scaling: Targets are compressed using
             h(x) = sign(x) * (sqrt(|x| + 1) - 1) + eps * x
         and decoded during inference using the analytical inverse h^{-1}(y).
      5. Two-Hot Support Projection: Scaled targets are projected onto adjacent atoms
         via `compute_two_hot_distribution` and trained using cross-entropy loss.
      6. n-Step Bootstrapped Returns: Value targets are computed using n-step bootstrapping
             z_t = sum_{tau=0}^{n-1} gamma^tau r_{t+1+tau} + gamma^n nu_{t+n}
         where nu is the MCTS root search value.

Multiprocessing Architecture:
    - Actor Workers: Self-play across parallel CartPole environments using batched single-player
      MuZero MCTS. Completed episodes are converted into n-step returns and sliced into K+1
      unroll blocks stored in SharedReplayBuffer.
    - Learner Worker: Samples minibatches from SharedReplayBuffer, unrolls K steps, computes
      cross-entropy losses on policy, value, and reward heads, updates parameters, and syncs weights.
    - Evaluator Worker: Periodically evaluates greedy MCTS play (temperature=0.0) on CartPole-v1,
      logging metrics to W&B asynchronously.
"""

import functools
import random
import time
from typing import Callable, List, Tuple

import gymnasium as gym
import numpy as np
from tensordict import TensorDict
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import wandb

from atomic_rl.action_selection import (
    argmax_selector,
    sample_distribution,
)
from atomic_rl.buffers import SharedReplayBuffer, slice_unrolls
from atomic_rl.losses import compute_cross_entropy_loss
from atomic_rl.schedules import get_stepwise_schedule
from atomic_rl.search import mcts_search, qtransform_by_min_max
from atomic_rl.transforms import (
    compute_contractive_transform,
    compute_expected_value,
    compute_inverse_contractive_transform,
    compute_two_hot_distribution,
)
from atomic_rl.utils import min_max_normalize, scale_gradient

# ---------------------------------------------------------------------------
# Hyperparameters & Constants
# ---------------------------------------------------------------------------
TOTAL_TRAINING_STEPS = 50000
NUM_ACTORS = 2
ENVS_PER_ACTOR = 4
MIN_BUFFER_SIZE = 500  # Minimum number of valid transitions before training starts
EVAL_INTERVAL_STEPS = 1000
PARAM_SYNC_INTERVAL = 50
NUM_MCTS_SIMULATIONS = 25
UNROLL_STEPS = 5
TD_STEPS = 5
GAMMA = 0.997  # MuZero Appendix C standard discount; ensures G_500 vs G_300 has ~2.1 atom delta
EPSILON_TRANSFORM = 0.001

# Discrete Support Bounds (max discounted return at gamma=0.997 is ~259.1 -> h(259.1) ≈ 15.39)
# Support [-25, 25] (51 atoms) provides fine-grained, non-saturating resolution up to 500 steps.
SUPPORT_MIN = -25
SUPPORT_MAX = 25
NUM_ATOMS = SUPPORT_MAX - SUPPORT_MIN + 1  # 51 atoms

# Environment & Network Shapes
OBS_DIM = 4
NUM_ACTIONS = 2
LATENT_DIM = 64

# MCTS PUCT Search Constants
C_PUCT = 1.25
DIRICHLET_ALPHA = 1.0  # Uniform Dirichlet distribution for 2 discrete actions
DIRICHLET_EPSILON = 0.15  # 15% exploration noise on root prior

# Temperature Schedule Constants (MuZero Atari Appendix B "Action Selection")
# Specifically, for the first 500k training steps a temperature of 1 is used,
# for the next 250k steps a temperature of 0.5 and for the remaining 250k a
# temperature of 0.25. This ensures that the action selection becomes greedier
# as training progresses.
TEMP_THRESHOLDS = [
    int(TOTAL_TRAINING_STEPS * 0.75),  # 50% of training (5,000 steps)
    int(TOTAL_TRAINING_STEPS * 1.0),  # 75% of training (7,500 steps)
]
TEMP_VALUES = [1.0, 0.5, 0.25]

# Optimization & Network Parameters
BATCH_SIZE = 64
MAX_EPISODE_STEPS = 500
BUFFER_NUM_EPISODES = 200  # 200 episodes * 500 steps = 100,000 steps capacity
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VALUE_LOSS_WEIGHT = 0.25  # MuZero Table S1 standard

NUM_EVAL_EPISODES = 10
NUM_TRAIN_EVAL_EPISODES = 5
SEED = 42


def make_env():
    """Constructs a Gymnasium CartPole-v1 environment."""
    return gym.make("CartPole-v1")


# ============================================================================
# 1. MuZero MLP Neural Network Architecture
# ============================================================================
class CartPoleMuZeroNet(nn.Module):
    """MLP-based MuZero network for continuous vector state environments (CartPole).

    Components:
      - Representation Function h(o): maps observation o [B, 4] -> hidden state s [B, 64]
      - Dynamics Function g(s, a): maps hidden state s [B, 64] + 1-hot action a [B, 2]
        -> next hidden state s' [B, 64] and reward categorical logits r [B, 41]
      - Prediction Function f(s): maps hidden state s [B, 64] -> policy logits p [B, 2]
        and value categorical logits v [B, 41]
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        num_actions: int = NUM_ACTIONS,
        latent_dim: int = LATENT_DIM,
        support_min: int = SUPPORT_MIN,
        support_max: int = SUPPORT_MAX,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.support_min = support_min
        self.support_max = support_max
        self.num_atoms = support_max - support_min + 1

        # Register discrete support buffer [-20, ..., 20]
        support = torch.arange(support_min, support_max + 1, dtype=torch.float32)
        self.register_buffer("support", support)

        # Pre-materialized 1-hot action encoding buffer [2, 2]
        self.register_buffer(
            "action_encodings", torch.eye(num_actions, dtype=torch.float32)
        )

        # 1. Representation Network h(o) -> s0
        self.representation_net = nn.Sequential(
            nn.Linear(obs_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

        # 2. Dynamics Network g(s, a) -> (s', r_logits)
        self.dynamics_net = nn.Sequential(
            nn.Linear(latent_dim + num_actions, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )
        self.reward_head = nn.Linear(latent_dim, self.num_atoms)

        # 3. Prediction Network f(s) -> (policy_logits, value_logits)
        self.prediction_backbone = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(latent_dim, num_actions)
        self.value_head = nn.Linear(latent_dim, self.num_atoms)

    def representation(self, obs: torch.Tensor) -> torch.Tensor:
        """h(o) -> initial hidden state s0 [B, latent_dim] scaled to [0, 1]."""
        raw_s = self.representation_net(obs)
        return min_max_normalize(raw_s, dim=-1)

    def dynamics(
        self, hidden_state: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """g(s, a) -> (next_hidden_state [B, latent_dim], reward_logits [B, num_atoms])."""
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
        """f(s) -> (policy_logits [B, num_actions], value_logits [B, num_atoms])."""
        feats = self.prediction_backbone(hidden_state)
        policy_logits = self.policy_head(feats)
        value_logits = self.value_head(feats)
        return policy_logits, value_logits

    def decode_support_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        """Decodes categorical logits into unscaled continuous scalar returns via h^{-1}(E[support])."""
        scaled_val = compute_expected_value(predictions=logits, support=self.support)
        return compute_inverse_contractive_transform(
            scaled_val, epsilon=EPSILON_TRANSFORM
        )

    def initial_inference(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """h(o) -> s0, followed by f(s0) -> (policy_logits, scalar_value, value_logits)."""
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
        """g(s, a) -> (s', r), followed by f(s') -> (policy_logits, scalar_value)."""
        next_hidden, reward_logits = self.dynamics(hidden_state, actions)
        scalar_reward = self.decode_support_to_scalar(reward_logits)
        policy_logits, value_logits = self.prediction(next_hidden)
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
# 2. MCTS Search Wrappers & Helpers
# ============================================================================
def create_recurrent_fn(model: nn.Module) -> Callable:
    """Creates a single-player recurrent_fn for MuZero MCTS simulation."""

    def recurrent_fn(
        actions_taken: torch.Tensor,
        embeddings: torch.Tensor,
        to_play: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        with torch.no_grad():
            (
                next_hidden,
                scalar_reward,
                logits,
                scalar_value,
                _,
                _,
            ) = model.recurrent_inference(embeddings, actions_taken)

        # Single player: player index is always 0
        next_to_play = torch.zeros_like(to_play)
        value_vec = scalar_value.view(-1, 1)
        reward_vec = scalar_reward.view(-1, 1)
        discount_vec = torch.full_like(reward_vec, GAMMA)

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
    num_simulations: int = NUM_MCTS_SIMULATIONS,
    temperature: float = 1.0,
    explore_dirichlet: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs batched MuZero MCTS search over continuous representation states."""
    batch_size = root_embed.shape[0]
    device = root_embed.device

    with torch.no_grad():
        s0, root_logits, root_scalar_value, _ = model.initial_inference(root_embed)

    root_to_play = torch.zeros((batch_size,), device=device, dtype=torch.long)
    root_vector_value = root_scalar_value.view(-1, 1)
    dirichlet_eps = DIRICHLET_EPSILON if explore_dirichlet else 0.0

    # Single-player dynamic min-max normalization matching MuZero paper & mctx
    # (unvisited actions mapped to min_value, normalizing value_score to 0.0)
    qtransform = qtransform_by_min_max

    action_probs, tree = mcts_search(
        root_embeddings=s0,
        root_logits=root_logits,
        root_value=root_vector_value,
        root_to_play=root_to_play,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        num_actions=NUM_ACTIONS,
        num_players=1,
        legal_mask=None,
        qtransform=qtransform,
        dirichlet_epsilon=dirichlet_eps,
        dirichlet_alpha=DIRICHLET_ALPHA,
        pb_c_init=C_PUCT,
        temperature=temperature,
    )
    # Return MCTS root search value (root.value() in MuZero paper), not raw prior network value
    root_search_value = tree["node_values"][:, 0, 0].detach()
    return action_probs, root_search_value


# ============================================================================
# 3. Multiprocessing Workers
# ============================================================================
def actor_worker(
    actor_id: int,
    model_creator: Callable[[], nn.Module],
    shared_model: nn.Module,
    buffer: SharedReplayBuffer,
    training_step: torch.Tensor,
    envs_per_actor: int = ENVS_PER_ACTOR,
    num_simulations: int = NUM_MCTS_SIMULATIONS,
    unroll_steps: int = UNROLL_STEPS,
    td_steps: int = TD_STEPS,
    gamma: float = GAMMA,
    device_str: str = "cpu",
):
    """Worker process generating self-play trajectories via batched MuZero MCTS."""
    torch.set_num_threads(1)
    device = torch.device(device_str)
    local_model = model_creator().to(device)
    local_model.eval()
    local_model.load_state_dict(shared_model.state_dict())

    torch.manual_seed(SEED + actor_id * 1000)
    random.seed(SEED + actor_id * 1000)

    envs = [make_env() for _ in range(envs_per_actor)]
    obs_list = []
    for env in envs:
        o, _ = env.reset(seed=SEED + actor_id * 1000)
        obs_list.append(o)

    # Trajectory entry: (obs [4], action int, target_policy [2], root_value float, reward float)
    trajectories: List[List[Tuple[torch.Tensor, int, torch.Tensor, float, float]]] = [
        [] for _ in range(envs_per_actor)
    ]

    recurrent_fn = create_recurrent_fn(local_model)
    loop_counter = 0

    while True:
        loop_counter += 1
        if loop_counter % PARAM_SYNC_INTERVAL == 0:
            local_model.load_state_dict(shared_model.state_dict())

        completed_tds = []
        root_embed = torch.from_numpy(np.array(obs_list, dtype=np.float32)).to(device)

        # Atari training-step temperature schedule: 1.0 (0-50%) -> 0.5 (50-75%) -> 0.25 (75-100%)
        curr_step = training_step[0].item()
        temp = get_stepwise_schedule(
            step=curr_step,
            thresholds=TEMP_THRESHOLDS,
            values=TEMP_VALUES,
        )

        action_probs, root_values = run_batched_mcts(
            local_model,
            recurrent_fn,
            root_embed,
            num_simulations=num_simulations,
            temperature=temp,
            explore_dirichlet=True,
        )

        for b_idx in range(envs_per_actor):
            target_policy = action_probs[b_idx]
            root_val = root_values[b_idx].item()

            dist = torch.distributions.Categorical(probs=target_policy)
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
                ep_obs, ep_actions, ep_policies, ep_root_vals, ep_rewards = zip(
                    *trajectories[b_idx]
                )
                ep_len = len(ep_obs)

                # Compute n-step bootstrapped return targets z_t
                ep_z_values = []
                for t in range(ep_len):
                    return_t = 0.0
                    discount = 1.0
                    for tau in range(td_steps):
                        idx = t + tau
                        if idx < ep_len:
                            return_t += discount * ep_rewards[idx]
                            discount *= gamma
                        else:
                            break
                    # Add bootstrapped root value at t + n
                    if t + td_steps < ep_len:
                        return_t += discount * ep_root_vals[t + td_steps]
                    elif truncated:
                        # Crucial for truncation: bootstrap from the final valid state value
                        # because the episode did not fail; it was merely cut off by the time limit.
                        return_t += discount * ep_root_vals[-1]
                    ep_z_values.append(return_t)

                # Pack episode into fixed-length sequence [MAX_EPISODE_STEPS, ...]
                seq_obs = torch.zeros((MAX_EPISODE_STEPS, OBS_DIM), dtype=torch.float32)
                seq_actions = torch.zeros((MAX_EPISODE_STEPS,), dtype=torch.long)
                seq_policies = torch.zeros(
                    (MAX_EPISODE_STEPS, NUM_ACTIONS), dtype=torch.float32
                )
                seq_values = torch.zeros((MAX_EPISODE_STEPS, 1), dtype=torch.float32)
                seq_rewards = torch.zeros((MAX_EPISODE_STEPS, 1), dtype=torch.float32)
                seq_valid = torch.zeros((MAX_EPISODE_STEPS,), dtype=torch.float32)

                seq_obs[:ep_len] = torch.stack(ep_obs)
                seq_actions[:ep_len] = torch.tensor(ep_actions, dtype=torch.long)
                seq_policies[:ep_len] = torch.stack(ep_policies)
                seq_values[:ep_len] = torch.tensor(
                    ep_z_values, dtype=torch.float32
                ).unsqueeze(-1)
                seq_rewards[:ep_len] = torch.tensor(
                    ep_rewards, dtype=torch.float32
                ).unsqueeze(-1)
                seq_valid[:ep_len] = 1.0

                completed_tds.append(
                    TensorDict(
                        {
                            "state": seq_obs,
                            "actions": seq_actions,
                            "target_policies": seq_policies,
                            "target_values": seq_values,
                            "target_rewards": seq_rewards,
                            "valid": seq_valid,
                        },
                        batch_size=[MAX_EPISODE_STEPS],
                    )
                )

                trajectories[b_idx] = []
                next_o, _ = envs[b_idx].reset()

            obs_list[b_idx] = next_o

        if len(completed_tds) > 0:
            batch_td = torch.stack(completed_tds).to(device)
            buffer.add_transitions(batch_td, valid_mask=batch_td["valid"])


def evaluator_worker(
    model_creator: Callable[[], nn.Module],
    shared_model: nn.Module,
    eval_queue: mp.Queue,
    training_step: torch.Tensor,
    num_eval_episodes: int = NUM_TRAIN_EVAL_EPISODES,
    eval_interval_steps: int = EVAL_INTERVAL_STEPS,
    device_str: str = "cpu",
):
    """Background process evaluating model performance asynchronously on CartPole-v1."""
    torch.set_num_threads(1)
    device = torch.device(device_str)
    local_model = model_creator().to(device)
    local_model.eval()

    envs = [make_env() for _ in range(num_eval_episodes)]
    recurrent_fn = create_recurrent_fn(local_model)
    next_eval_step = 0

    while True:
        curr_step = int(training_step[0].item())
        if curr_step < next_eval_step:
            time.sleep(1.0)
            continue

        next_eval_step = curr_step + eval_interval_steps
        local_model.load_state_dict(shared_model.state_dict())

        # Reset all evaluation environments
        obs_list = [env.reset(seed=SEED + i * 100)[0] for i, env in enumerate(envs)]
        episode_returns = [0.0] * num_eval_episodes
        episode_lengths = [0] * num_eval_episodes
        dones = [False] * num_eval_episodes

        while not all(dones):
            active_indices = [i for i, d in enumerate(dones) if not d]
            active_obs = np.array(
                [obs_list[i] for i in active_indices], dtype=np.float32
            )
            root_embed = torch.from_numpy(active_obs).to(device)

            action_probs, _ = run_batched_mcts(
                local_model,
                recurrent_fn,
                root_embed,
                num_simulations=NUM_MCTS_SIMULATIONS,
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

        eval_metrics = {
            "eval/mean_episode_return": sum(episode_returns) / len(episode_returns),
            "eval/min_episode_return": min(episode_returns),
            "eval/max_episode_return": max(episode_returns),
            "eval/mean_episode_length": sum(episode_lengths) / len(episode_lengths),
            "eval/min_episode_length": min(episode_lengths),
            "eval/max_episode_length": max(episode_lengths),
            "eval/step": curr_step,
        }
        eval_queue.put(eval_metrics)


def learner_worker(
    model_creator: Callable[[], nn.Module],
    shared_model: nn.Module,
    buffer: SharedReplayBuffer,
    eval_queue: mp.Queue,
    training_step: torch.Tensor,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    value_loss_weight: float = VALUE_LOSS_WEIGHT,
    unroll_steps: int = UNROLL_STEPS,
    max_steps: int = TOTAL_TRAINING_STEPS,
    device_str: str = "cpu",
):
    """Learner process training model weights on sampled minibatches using two-hot cross-entropy."""
    torch.set_num_threads(1)
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
        project="muzero-cartpole",
        name=f"muzero_cartpole_sims{NUM_MCTS_SIMULATIONS}_td{TD_STEPS}_actors{NUM_ACTORS}",
        config={
            "total_training_steps": max_steps,
            "num_actors": NUM_ACTORS,
            "envs_per_actor": ENVS_PER_ACTOR,
            "num_mcts_simulations": NUM_MCTS_SIMULATIONS,
            "unroll_steps": unroll_steps,
            "td_steps": TD_STEPS,
            "gamma": GAMMA,
            "c_puct": C_PUCT,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "value_loss_weight": value_loss_weight,
            "support_min": SUPPORT_MIN,
            "support_max": SUPPORT_MAX,
        },
    )

    step_count = 0
    latest_eval_metrics = {}

    while step_count < max_steps:
        sample_res = buffer.sample_unrolls(
            rng_key, batch_size, unroll_steps, min_size=MIN_BUFFER_SIZE
        )
        if sample_res is None:
            time.sleep(0.01)
            continue

        unroll_data, valid_mask, coords = sample_res

        # A step is valid only if within tensor boundary AND was part of the actual episode
        ep_valid = unroll_data["valid"].bool() & valid_mask
        policy_masks = ep_valid.float().to(device)

        states = unroll_data["state"][:, 0].to(device)
        actions = unroll_data["actions"].to(device)
        target_policies = unroll_data["target_policies"].to(device)
        target_values = unroll_data["target_values"].to(device)
        target_rewards = unroll_data["target_rewards"].to(device)

        # 1. Initial Step k=0 Prediction
        s0, p0, _, v0_logits = local_model.initial_inference(states)

        raw_p_loss_0, _ = compute_cross_entropy_loss(p0, target_policies[:, 0])
        mask_0 = policy_masks[:, 0]
        loss_p0 = (raw_p_loss_0 * mask_0).sum() / torch.clamp(mask_0.sum(), min=1.0)

        # Value target: contractive scaling + two-hot projection
        z0 = target_values[:, 0].squeeze(-1)
        z0_scaled = compute_contractive_transform(z0, epsilon=EPSILON_TRANSFORM)
        z0_two_hot = compute_two_hot_distribution(z0_scaled, SUPPORT_MIN, SUPPORT_MAX)
        raw_v_loss_0, _ = compute_cross_entropy_loss(v0_logits, z0_two_hot)
        loss_v0 = raw_v_loss_0.mean()
        # Initial step k=0 uses full weight 1.0 (paper: predictions = [(1.0, value, reward, policy_logits)]), not 1/K
        total_loss = loss_p0 + value_loss_weight * loss_v0

        curr_s = s0
        step_p_losses = [loss_p0.item()]
        step_v_losses = [loss_v0.item()]
        step_r_losses = []

        # 2. Unroll Steps k=1..K Dynamics & Prediction
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

            # Value target at step k
            zk = target_values[:, k].squeeze(-1)
            zk_scaled = compute_contractive_transform(zk, epsilon=EPSILON_TRANSFORM)
            zk_two_hot = compute_two_hot_distribution(
                zk_scaled, SUPPORT_MIN, SUPPORT_MAX
            )
            raw_v_loss, _ = compute_cross_entropy_loss(v_k_logits, zk_two_hot)

            # Reward target at step k
            uk = target_rewards[:, k - 1].squeeze(-1)
            uk_scaled = compute_contractive_transform(uk, epsilon=EPSILON_TRANSFORM)
            uk_two_hot = compute_two_hot_distribution(
                uk_scaled, SUPPORT_MIN, SUPPORT_MAX
            )
            raw_r_loss, _ = compute_cross_entropy_loss(r_k_logits, uk_two_hot)

            l_v = raw_v_loss.mean()
            l_r = raw_r_loss.mean()

            step_p_losses.append(l_p.item())
            step_v_losses.append(l_v.item())
            step_r_losses.append(l_r.item())

            step_loss = l_p + value_loss_weight * l_v + l_r
            total_loss = total_loss + (step_loss / float(unroll_steps))

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
        optimizer.step()

        shared_model.load_state_dict(local_model.state_dict())
        step_count += 1
        training_step[0] = step_count

        while not eval_queue.empty():
            try:
                latest_eval_metrics = eval_queue.get_nowait()
            except Exception:
                break

        log_dict = {
            "global_step": step_count,
            "loss/total": total_loss.item(),
            "loss/policy": sum(step_p_losses) / len(step_p_losses),
            "loss/value": sum(step_v_losses) / len(step_v_losses),
            "loss/reward": (
                sum(step_r_losses) / len(step_r_losses) if step_r_losses else 0.0
            ),
            "buffer/size": buffer.size,
        }
        log_dict.update(latest_eval_metrics)

        if step_count % 100 == 0:
            eval_info = ""
            if "eval/mean_episode_return" in latest_eval_metrics:
                mean_r = latest_eval_metrics["eval/mean_episode_return"]
                min_r = latest_eval_metrics.get("eval/min_episode_return", 0.0)
                max_r = latest_eval_metrics.get("eval/max_episode_return", 0.0)
                mean_l = latest_eval_metrics.get("eval/mean_episode_length", 0.0)
                eval_info = f" | Eval Return: {mean_r:.1f} [min: {min_r:.0f}, max: {max_r:.0f}] | Eval Len: {mean_l:.1f}"
            print(
                f"[Learner Step {step_count}/{max_steps}] Loss: {total_loss.item():.4f} | "
                f"Policy: {log_dict['loss/policy']:.4f} | Value: {log_dict['loss/value']:.4f} | "
                f"Reward: {log_dict['loss/reward']:.4f} | Buffer: {buffer.size}{eval_info}"
            )

        wandb.log(log_dict)

    wandb.finish()


# ============================================================================
# 5. Main Multiprocessing Pipeline Setup
# ============================================================================
def model_creator_fn() -> CartPoleMuZeroNet:
    return CartPoleMuZeroNet(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        latent_dim=LATENT_DIM,
        support_min=SUPPORT_MIN,
        support_max=SUPPORT_MAX,
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

    training_step = torch.zeros(1, dtype=torch.long).share_memory_()

    buffer_shapes = {
        "state": (OBS_DIM,),
        "actions": (),
        "target_policies": (NUM_ACTIONS,),
        "target_values": (1,),
        "target_rewards": (1,),
        "valid": (),
    }
    buffer = SharedReplayBuffer(
        (BUFFER_NUM_EPISODES, MAX_EPISODE_STEPS),
        buffer_shapes,
        device=device,
    )
    eval_queue = mp.Queue()

    processes = []

    learner_p = mp.Process(
        target=learner_worker,
        args=(model_creator_fn, shared_model, buffer, eval_queue, training_step),
        kwargs={"device_str": device_str},
    )
    learner_p.start()
    processes.append(learner_p)

    for actor_id in range(NUM_ACTORS):
        actor_p = mp.Process(
            target=actor_worker,
            args=(actor_id, model_creator_fn, shared_model, buffer, training_step),
            kwargs={"device_str": device_str},
        )
        actor_p.start()
        processes.append(actor_p)

    eval_p = mp.Process(
        target=evaluator_worker,
        args=(model_creator_fn, shared_model, eval_queue, training_step),
        kwargs={"device_str": device_str},
    )
    eval_p.daemon = True
    eval_p.start()
    processes.append(eval_p)

    print(
        f"Launched MuZero CartPole Multiprocessing Pipeline ({NUM_ACTORS} Actors, 1 Learner, 1 Evaluator) | Support [{SUPPORT_MIN}, {SUPPORT_MAX}] ({NUM_ATOMS} Atoms).",
        flush=True,
    )
    learner_p.join()

    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join()

    print("MuZero CartPole Training Completed Successfully.", flush=True)


if __name__ == "__main__":
    main()
