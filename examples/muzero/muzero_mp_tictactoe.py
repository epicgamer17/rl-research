"""MuZero on PettingZoo TicTacToe (torch.multiprocessing Multiprocess Version)

================================================================================

Paper Reference:
    "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"
    Schrittwieser et al., Nature 2020 (arXiv 2019: https://arxiv.org/abs/1911.08265)

Goal & Key Ideas:
    MuZero is a model-based reinforcement learning algorithm that plans using a learned model
    without requiring any prior knowledge of environment dynamics or game rules. Unlike AlphaZero,
    which relies on a simulator for Monte-Carlo Tree Search (MCTS) transitions, MuZero learns an
    abstract model composed of three connected neural network components:
      1. Representation Function h(o_t) -> s_t^0: maps input observations to an initial hidden state.
      2. Dynamics Function g(s_t^{k-1}, a_t^k) -> (s_t^k, r_t^k): predicts the next hidden state
         and immediate reward given a candidate action.
      3. Prediction Function f(s_t^k) -> (p_t^k, v_t^k): predicts policy action priors and state
         value estimates directly from a hidden state.

Algorithm Dynamics & Loss Function:
    - Planning: MCTS traverses hypothetical trajectories inside the latent state space using the
      learned dynamics function g and prediction function f.
    - Training Objective: The model parameters theta are trained end-to-end by unrolling K hypothetical
      steps along trajectories sampled from experience replay:
          l_t(theta) = 1/K * sum_{k=0}^K [ l^p(pi_{t+k}, p_t^k) + l^v(z_{t+k}, v_t^k) + l^r(u_{t+k}, r_t^k) ] + c||theta||^2
      where:
          - l^p is policy cross-entropy loss against MCTS search target pi_{t+k}.
          - l^v is value MSE against target outcome z_{t+k}.
          - l^r is reward MSE against observed reward u_{t+k}. Note: in board games like TicTacToe
            without intermediate rewards, all intermediate transition rewards u_{t+k} are 0.0,
            so reward loss rapidly converges to 0.0 as documented in MuZero Appendix G / Figure S2.

Gradient Scaling & Hidden State Normalization:
    - Per Appendix G of the MuZero paper:
        1. Loss for each head is scaled by 1/K to keep total gradient magnitude constant.
        2. Gradients passing through the dynamics function input are scaled by 1/2.
        3. Hidden states are scaled using Min-Max normalization to maintain activation stability.

Multiprocessing Architecture:
    - Actor Workers: Each actor runs batched self-play across parallel PettingZoo TicTacToe environments
      using batched MuZero MCTS. Completed episodes are sliced/padded into length K+1 unroll blocks and
      written to SharedReplayBuffer.
    - Learner Worker: Samples minibatches of unrolled sequences from SharedReplayBuffer, unrolls the
      representation, dynamics, and prediction networks for K steps, updates parameters, and syncs weights.
    - Evaluator Worker: Periodically evaluates the shared model against a Random baseline in
      PettingZoo tictactoe_v3, logging metrics to W&B asynchronously.

    NOTE: We use vector values for each player for rewards and values. This allows the MCTS to work
    on n-player games (cooperative or competitive) and aligns with PettingZoo multi-agent reward vectors.
"""

import random
import time
from typing import Callable, Dict, List, Optional, Tuple

from atomic_rl.action_selection import (
    apply_action_mask,
    argmax_selector,
    sample_distribution,
)
from atomic_rl.buffers import SharedReplayBuffer
from atomic_rl.envs import get_agent_index, get_terminal_rewards_vec
from atomic_rl.envs.wrappers import PettingZooChannelFirstWrapper
from atomic_rl.evaluation import evaluate_pettingzoo_vs_random
from atomic_rl.losses import compute_cross_entropy_loss, compute_mse_loss
from atomic_rl.networks.architectures import ResNetBackbone
from atomic_rl.schedules import get_stepwise_schedule
from atomic_rl.search import mcts_search
from pettingzoo.classic import tictactoe_v3

from tensordict import TensorDict
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import wandb

# ---------------------------------------------------------------------------
# Hyperparameters & Constants
# ---------------------------------------------------------------------------
TOTAL_TRAINING_STEPS = 20000
NUM_ACTORS = 1
ENVS_PER_ACTOR = 3
MIN_BUFFER_SIZE = 64
EVAL_INTERVAL_STEPS = 250
PARAM_SYNC_INTERVAL = 100
NUM_MCTS_SIMULATIONS = 25
UNROLL_STEPS = 5

# MCTS PUCT Search Constants
C_PUCT = 1.25
DIRICHLET_ALPHA = 2.0
DIRICHLET_EPSILON = 0.25

# Temperature Schedule Constants
TEMP_THRESHOLD_MOVES = 5
TEMPERATURE_EXPLORATION = 1.0
TEMPERATURE_EXPLOITATION = 0.0

# Optimization & Network Parameters
BATCH_SIZE = 32
REPLAY_BUFFER_CAPACITY = 10000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_FILTERS = 24
NUM_RES_BLOCKS = 3

NUM_EVAL_GAMES = 100
SEED = 42


# ============================================================================
# 1. Pure Functional Helpers
# ============================================================================
def encode_action_plane(
    actions: torch.Tensor, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Encodes 1D action indices [B] into 2D spatial 1-hot feature planes [B, 1, 3, 3]."""
    planes = torch.zeros((batch_size, 1, 3, 3), device=device, dtype=torch.float32)
    rows = actions // 3
    cols = actions % 3
    batch_idx = torch.arange(batch_size, device=device)
    planes[batch_idx, 0, rows, cols] = 1.0
    return planes


def scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Scales gradients passing through tensor during backpropagation (MuZero Appendix G)."""
    return tensor * scale + tensor.detach() * (1.0 - scale)


def min_max_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Scales hidden state tensor to range [0, 1] per sample (MuZero Appendix G)."""
    flat = tensor.view(tensor.size(0), -1)
    min_val = flat.min(dim=-1, keepdim=True)[0].view(-1, 1, 1, 1)
    max_val = flat.max(dim=-1, keepdim=True)[0].view(-1, 1, 1, 1)
    denom = max_val - min_val
    denom = torch.where(denom == 0.0, torch.ones_like(denom), denom)
    return (tensor - min_val) / denom


# ============================================================================
# 2. MuZero Neural Network Architecture
# ============================================================================
class TicTacToeMuZeroNet(nn.Module):
    """MuZero Neural Network for TicTacToe comprising:

    - Representation Function h(o): maps observation o [B, 2, 3, 3] -> hidden state s [B, F, 3, 3]
    - Dynamics Function g(s, a): maps hidden state s [B, F, 3, 3] + action plane a [B, 1, 3, 3]
      -> next hidden state s' [B, F, 3, 3] and predicted reward r [B, 1]
    - Prediction Function f(s): maps hidden state s [B, F, 3, 3] -> policy logits p [B, 9]
      and scalar value v [B, 1]
    """

    def __init__(
        self, num_filters: int = NUM_FILTERS, num_res_blocks: int = NUM_RES_BLOCKS
    ):
        super().__init__()
        self.num_filters = num_filters

        # 1. Representation Network h(o) -> s^0
        self.representation_stem = nn.Sequential(
            nn.Conv2d(2, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )
        self.representation_backbone = ResNetBackbone(
            in_channels=num_filters,
            num_filters=num_filters,
            num_blocks=num_res_blocks,
            norm_type="batch",
        )

        # 2. Dynamics Network g(s, a) -> (s', r)
        self.dynamics_stem = nn.Sequential(
            nn.Conv2d(num_filters + 1, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )
        self.dynamics_backbone = ResNetBackbone(
            in_channels=num_filters,
            num_filters=num_filters,
            num_blocks=num_res_blocks,
            norm_type="batch",
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

        # 3. Prediction Network f(s) -> (policy_logits, value)
        self.prediction_backbone = ResNetBackbone(
            in_channels=num_filters,
            num_filters=num_filters,
            num_blocks=num_res_blocks,
            norm_type="batch",
        )
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
            nn.Tanh(),
        )

    def representation(self, obs: torch.Tensor) -> torch.Tensor:
        """h(o) -> initial hidden state s0 [B, num_filters, 3, 3]."""
        x = self.representation_stem(obs)
        return self.representation_backbone(x)

    def dynamics(
        self, hidden_state: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """g(s, a) -> (next_hidden_state [B, num_filters, 3, 3], reward [B, 1])."""
        batch_size = hidden_state.size(0)
        device = hidden_state.device
        if actions.ndim == 1:
            action_plane = encode_action_plane(actions, batch_size, device)
        else:
            action_plane = actions

        concat_input = torch.cat([hidden_state, action_plane], dim=1)
        x = self.dynamics_stem(concat_input)
        next_hidden = self.dynamics_backbone(x)
        reward = self.reward_head(next_hidden)
        return next_hidden, reward

    def prediction(
        self, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """f(s) -> (policy_logits [B, 9], scalar_value [B, 1])."""
        feats = self.prediction_backbone(hidden_state)
        policy_logits = self.policy_head(feats)
        scalar_value = self.value_head(feats).squeeze(-1)
        return policy_logits, scalar_value

    def initial_inference(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h(obs) -> s0, followed by f(s0) -> (policy_logits, scalar_value)."""
        s0 = self.representation(obs)
        policy_logits, scalar_value = self.prediction(s0)
        return s0, policy_logits, scalar_value

    def recurrent_inference(
        self, hidden_state: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """g(s, a) -> (s', r), followed by f(s') -> (policy_logits, scalar_value)."""
        next_hidden, reward = self.dynamics(hidden_state, actions)
        policy_logits, scalar_value = self.prediction(next_hidden)
        return next_hidden, reward, policy_logits, scalar_value


# ============================================================================
# 3. MCTS Search Wrappers & Helpers
# ============================================================================
def create_recurrent_fn(model: nn.Module) -> Callable:
    """Creates an N-player compliant recurrent_fn for MuZero MCTS simulation."""

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
            ) = model.recurrent_inference(embeddings, actions_taken)

        next_hidden = min_max_normalize(next_hidden)
        next_to_play = 1 - to_play

        # In zero-sum 2-player games, player 0 and player 1 values are inverted.
        # scalar_value predicted at next_hidden is the value for active player at next_hidden (next_to_play).
        v_p0 = torch.where(next_to_play == 0, scalar_value, -scalar_value)
        value_vec = torch.stack([v_p0, -v_p0], dim=-1)

        r_p0 = torch.where(
            next_to_play == 0, scalar_reward.squeeze(-1), -scalar_reward.squeeze(-1)
        )
        reward_vec = torch.stack([r_p0, -r_p0], dim=-1)

        # Undiscounted board game transitions inside search
        discount_vec = torch.ones_like(reward_vec)

        # Per MuZero Appendix A: legal action masking is applied ONLY at root, not inside tree
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
    curr_players: torch.Tensor,
    legal_masks: torch.Tensor,
    num_simulations: int = NUM_MCTS_SIMULATIONS,
    temperature: float = 1.0,
    explore_dirichlet: bool = True,
) -> torch.Tensor:
    """Runs batched MuZero MCTS search over learned representation states."""
    with torch.no_grad():
        s0, root_logits, root_scalar_value = model.initial_inference(root_embed)

    s0 = min_max_normalize(s0)

    v_p0_root = torch.where(curr_players == 0, root_scalar_value, -root_scalar_value)
    root_vector_value = torch.stack([v_p0_root, -v_p0_root], dim=-1)
    dirichlet_eps = DIRICHLET_EPSILON if explore_dirichlet else 0.0

    action_probs, _ = mcts_search(
        root_embeddings=s0,
        root_logits=root_logits,
        root_value=root_vector_value,
        root_to_play=curr_players,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        num_actions=9,
        num_players=2,
        legal_mask=legal_masks,
        dirichlet_epsilon=dirichlet_eps,
        dirichlet_alpha=DIRICHLET_ALPHA,
        pb_c_init=C_PUCT,
        temperature=temperature,
    )
    return action_probs


# ============================================================================
# 4. Multiprocessing Workers
# ============================================================================
def actor_worker(
    actor_id: int,
    model_creator: Callable[[], nn.Module],
    shared_model: nn.Module,
    buffer: SharedReplayBuffer,
    envs_per_actor: int = ENVS_PER_ACTOR,
    num_simulations: int = NUM_MCTS_SIMULATIONS,
    unroll_steps: int = UNROLL_STEPS,
    device_str: str = "cpu",
):
    """Worker process generating self-play trajectories via batched MuZero MCTS."""
    device = torch.device(device_str)
    local_model = model_creator().to(device)
    local_model.eval()
    local_model.load_state_dict(shared_model.state_dict())

    torch.manual_seed(SEED + actor_id * 1000)
    random.seed(SEED + actor_id * 1000)

    envs = [
        PettingZooChannelFirstWrapper(tictactoe_v3.env()) for _ in range(envs_per_actor)
    ]
    for env in envs:
        env.reset()

    move_counts = [0] * envs_per_actor
    # Trajectory entry: (obs [2,3,3], action int, target_policy [9], player int)
    trajectories: List[List[Tuple[torch.Tensor, int, torch.Tensor, int]]] = [
        [] for _ in range(envs_per_actor)
    ]

    step_counter = 0
    recurrent_fn = create_recurrent_fn(local_model)

    while True:
        step_counter += 1
        if step_counter % PARAM_SYNC_INTERVAL == 0:
            local_model.load_state_dict(shared_model.state_dict())

        completed_tds = []
        root_embed = torch.empty(
            (envs_per_actor, 2, 3, 3), device=device, dtype=torch.float32
        )
        legal_masks = torch.empty((envs_per_actor, 9), device=device, dtype=torch.bool)
        curr_players = torch.empty((envs_per_actor,), device=device, dtype=torch.long)

        for b_idx, env in enumerate(envs):
            obs, reward, term, trunc, _ = env.last()
            if term or trunc:
                rewards_vec = get_terminal_rewards_vec(env)
                ep_states, ep_actions, ep_policies, ep_players = zip(
                    *trajectories[b_idx]
                )
                ep_len = len(ep_states)

                # Process episode steps into unrolled K+1 sequence blocks
                for t in range(ep_len):
                    unroll_obs = ep_states[t]
                    unroll_actions = []
                    unroll_policies = []
                    unroll_values = []
                    unroll_rewards = []

                    for k in range(unroll_steps + 1):
                        step_idx = t + k
                        if step_idx < ep_len:
                            player_k = ep_players[step_idx]
                            val_k = rewards_vec[player_k].item()
                            unroll_policies.append(ep_policies[step_idx])
                            unroll_values.append(
                                torch.tensor([val_k], dtype=torch.float32)
                            )

                            if k > 0:
                                act_prev = ep_actions[step_idx - 1]
                                unroll_actions.append(act_prev)
                                unroll_rewards.append(
                                    torch.tensor([0.0], dtype=torch.float32)
                                )
                        else:
                            # Absorbing state padding beyond terminal step:
                            # In alternating 2-player game, active player flips every step,
                            # so value target flips sign for alternating players.
                            last_val = (
                                unroll_values[-1].item() if unroll_values else 0.0
                            )
                            padded_val = -last_val if (k > 0) else last_val
                            unroll_policies.append(
                                torch.full((9,), 1.0 / 9.0, dtype=torch.float32)
                            )
                            unroll_values.append(
                                torch.tensor([padded_val], dtype=torch.float32)
                            )

                            if k > 0:
                                unroll_actions.append(0)
                                unroll_rewards.append(
                                    torch.tensor([0.0], dtype=torch.float32)
                                )

                    completed_tds.append(
                        TensorDict(
                            {
                                "state": unroll_obs,
                                "actions": torch.tensor(
                                    unroll_actions, dtype=torch.long
                                ),
                                "target_policies": torch.stack(unroll_policies),
                                "target_values": torch.stack(unroll_values),
                                "target_rewards": torch.stack(unroll_rewards),
                            },
                            batch_size=[],
                        )
                    )

                trajectories[b_idx] = []
                move_counts[b_idx] = 0
                env.reset()
                obs, reward, term, trunc, _ = env.last()

            curr_players[b_idx] = get_agent_index(env)
            root_embed[b_idx].copy_(
                torch.from_numpy(obs["observation"]).to(
                    device=device, dtype=torch.float32
                )
            )
            legal_masks[b_idx].copy_(
                torch.from_numpy(obs["action_mask"]).to(device=device, dtype=torch.bool)
            )

        action_probs = run_batched_mcts(
            local_model,
            recurrent_fn,
            root_embed,
            curr_players,
            legal_masks,
            num_simulations=num_simulations,
            temperature=1.0,
            explore_dirichlet=True,
        )

        for b_idx, env in enumerate(envs):
            move_counts[b_idx] += 1
            curr_player = curr_players[b_idx].item()
            target_policy = action_probs[b_idx]

            temp = get_stepwise_schedule(
                step=move_counts[b_idx],
                thresholds=[TEMP_THRESHOLD_MOVES],
                values=[TEMPERATURE_EXPLORATION, TEMPERATURE_EXPLOITATION],
            )

            if temp == 0.0:
                action_idx_tensor, _ = argmax_selector(target_policy.unsqueeze(0))
                action_idx = action_idx_tensor.squeeze().item()
            else:
                dist = torch.distributions.Categorical(probs=target_policy)
                action_idx_t, _ = sample_distribution(dist, explore=True)
                action_idx = action_idx_t.item()

            trajectories[b_idx].append(
                (
                    root_embed[b_idx].clone().cpu(),
                    action_idx,
                    target_policy.cpu(),
                    curr_player,
                )
            )
            env.step(action_idx)

        if len(completed_tds) > 0:
            batch_td = torch.stack(completed_tds).to(device)
            buffer.add_samples(batch_td)


def evaluator_worker(
    model_creator: Callable[[], nn.Module],
    shared_model: nn.Module,
    eval_queue: mp.Queue,
    num_eval_games: int = NUM_EVAL_GAMES,
    device_str: str = "cpu",
):
    """Background process evaluating model performance asynchronously."""
    device = torch.device(device_str)
    local_model = model_creator().to(device)
    local_model.eval()

    def select_eval_action(env, curr_player_idx, obs_t, action_mask_t):
        recurrent_fn = create_recurrent_fn(local_model)
        root_embed = obs_t.unsqueeze(0)
        legal_masks = action_mask_t.unsqueeze(0)
        players_t = torch.tensor([curr_player_idx], device=device, dtype=torch.long)

        action_probs = run_batched_mcts(
            local_model,
            recurrent_fn,
            root_embed,
            players_t,
            legal_masks,
            num_simulations=NUM_MCTS_SIMULATIONS,
            temperature=0.0,
            explore_dirichlet=False,
        )
        action_idx_tensor, _ = argmax_selector(action_probs)
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
        time.sleep(5)


def learner_worker(
    model_creator: Callable[[], nn.Module],
    shared_model: nn.Module,
    buffer: SharedReplayBuffer,
    eval_queue: mp.Queue,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    unroll_steps: int = UNROLL_STEPS,
    max_steps: int = TOTAL_TRAINING_STEPS,
    device_str: str = "cpu",
):
    """Learner process training model weights on sampled minibatches."""
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
            "num_mcts_simulations": NUM_MCTS_SIMULATIONS,
            "unroll_steps": unroll_steps,
            "c_puct": C_PUCT,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
    )

    step_count = 0
    latest_eval_metrics = {}

    while step_count < max_steps:
        minibatch = buffer.sample_batch(rng_key, batch_size, min_size=MIN_BUFFER_SIZE)
        if minibatch is None:
            time.sleep(0.01)
            continue

        states = minibatch["state"].to(device)
        actions = minibatch["actions"].to(device)
        target_policies = minibatch["target_policies"].to(device)
        target_values = minibatch["target_values"].to(device)
        target_rewards = minibatch["target_rewards"].to(device)

        # 1. Initial Step k=0 Prediction
        s0, p0, v0 = local_model.initial_inference(states)
        s0 = min_max_normalize(s0)

        raw_p_loss_0, _ = compute_cross_entropy_loss(p0, target_policies[:, 0])
        raw_v_loss_0, _ = compute_mse_loss(v0.view(-1), target_values[:, 0].view(-1))

        loss_p0 = raw_p_loss_0.mean()
        loss_v0 = raw_v_loss_0.mean()
        total_loss = (loss_p0 + loss_v0) / float(unroll_steps)

        curr_s = s0
        step_p_losses = [loss_p0.item()]
        step_v_losses = [loss_v0.item()]
        step_r_losses = []

        # 2. Unroll Steps k=1..K Dynamics & Prediction
        for k in range(1, unroll_steps + 1):
            act_k = actions[:, k - 1]
            curr_s = scale_gradient(curr_s, 0.5)

            next_s, r_k, p_k, v_k = local_model.recurrent_inference(curr_s, act_k)
            curr_s = min_max_normalize(next_s)

            raw_p_loss, _ = compute_cross_entropy_loss(p_k, target_policies[:, k])
            raw_v_loss, _ = compute_mse_loss(v_k.view(-1), target_values[:, k].view(-1))
            raw_r_loss, _ = compute_mse_loss(
                r_k.view(-1), target_rewards[:, k - 1].view(-1)
            )

            l_p = raw_p_loss.mean()
            l_v = raw_v_loss.mean()
            l_r = raw_r_loss.mean()

            step_p_losses.append(l_p.item())
            step_v_losses.append(l_v.item())
            step_r_losses.append(l_r.item())

            step_loss = l_p + l_v + l_r
            total_loss = total_loss + (step_loss / float(unroll_steps))

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
        optimizer.step()

        shared_model.load_state_dict(local_model.state_dict())
        step_count += 1

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
            print(
                f"[Learner Step {step_count}/{max_steps}] Loss: {total_loss.item():.4f} | Buffer: {buffer.size}"
            )

        wandb.log(log_dict)

    wandb.finish()


# ============================================================================
# 5. Main Multiprocessing Pipeline Setup
# ============================================================================
def model_creator_fn() -> TicTacToeMuZeroNet:
    return TicTacToeMuZeroNet(num_filters=NUM_FILTERS, num_res_blocks=NUM_RES_BLOCKS)


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
        "state": (2, 3, 3),
        "actions": (UNROLL_STEPS,),
        "target_policies": (UNROLL_STEPS + 1, 9),
        "target_values": (UNROLL_STEPS + 1, 1),
        "target_rewards": (UNROLL_STEPS, 1),
    }
    buffer = SharedReplayBuffer(REPLAY_BUFFER_CAPACITY, buffer_shapes, device=device)
    eval_queue = mp.Queue()

    processes = []

    learner_p = mp.Process(
        target=learner_worker,
        args=(model_creator_fn, shared_model, buffer, eval_queue),
        kwargs={"device_str": device_str},
    )
    learner_p.start()
    processes.append(learner_p)

    for actor_id in range(NUM_ACTORS):
        actor_p = mp.Process(
            target=actor_worker,
            args=(actor_id, model_creator_fn, shared_model, buffer),
            kwargs={"device_str": device_str},
        )
        actor_p.start()
        processes.append(actor_p)

    eval_p = mp.Process(
        target=evaluator_worker,
        args=(model_creator_fn, shared_model, eval_queue),
        kwargs={"device_str": device_str},
    )
    eval_p.daemon = True
    eval_p.start()
    processes.append(eval_p)

    print(
        f"Launched MuZero Multiprocessing Pipeline ({NUM_ACTORS} Actors, 1 Learner, 1 Evaluator)."
    )
    learner_p.join()

    for p in processes:
        if p.is_alive():
            p.terminate()

    print("MuZero Multiprocessing Training Completed Successfully.")


if __name__ == "__main__":
    main()
