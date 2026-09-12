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

Value Transformations (MuZero Appendix F / Paper p. 4):
    - Board Games (TicTacToe, Chess, Go): Outcomes are strictly bounded in [-1, 1] with zero
      intermediate rewards. As documented in the paper, board games use continuous scalar values
      trained directly with MSE loss without non-linear value transformation or discrete support.
    - Non-Board Games (e.g. Atari): In environments with unbounded returns, the contractive
      value transform h(x) = sign(x)(sqrt(|x|+1) - 1) + eps*x and discrete two-hot support projection
      are used (available via `atomic_rl.transforms.compute_contractive_transform` and
      `atomic_rl.transforms.compute_two_hot_distribution`).

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

import functools
import random
import time
from typing import Callable, List, Tuple

from atomic_rl.action_selection import (
    apply_action_mask,
    argmax_selector,
    sample_distribution,
)
from atomic_rl.buffers import SharedReplayBuffer, slice_unrolls
from atomic_rl.envs import get_agent_index, get_terminal_rewards_vec
from atomic_rl.envs.wrappers import (
    FrameStackWrapper,
    PettingZooChannelFirstWrapper,
    TwoPlayerPlayerPlaneWrapper,
)
from atomic_rl.evaluation import evaluate_pettingzoo_vs_random
from atomic_rl.losses import compute_cross_entropy_loss, compute_mse_loss
from atomic_rl.networks.architectures import ResNetBackbone
from atomic_rl.schedules import get_stepwise_schedule
from atomic_rl.search import fpu_zero, mcts_search, qtransform_by_min_max
from atomic_rl.transforms import build_action_encodings
from atomic_rl.utils import min_max_normalize, scale_gradient
from pettingzoo.classic import tictactoe_v3

from tensordict import TensorDict
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import wandb

# ---------------------------------------------------------------------------
# Hyperparameters & Constants
# ---------------------------------------------------------------------------
TOTAL_TRAINING_STEPS = 20000
NUM_ACTORS = 2
ENVS_PER_ACTOR = 3
MIN_BUFFER_SIZE = 64  # Minimum number of valid transitions before learner starts
EVAL_INTERVAL_STEPS = 1000
PARAM_SYNC_INTERVAL = 100
NUM_MCTS_SIMULATIONS = 25
UNROLL_STEPS = 5

# Frame Stacking & Player Plane Constants
FRAME_STACK = 3
INCLUDE_PLAYER_PLANE = True
NUM_CHANNELS = 2 * FRAME_STACK + (1 if INCLUDE_PLAYER_PLANE else 0)  # 7 channels

# MCTS PUCT Search Constants
C_PUCT = 1.25
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

# Temperature Schedule Constants
TEMP_THRESHOLD_MOVES = 5
TEMPERATURE_EXPLORATION = 1.0
TEMPERATURE_EXPLOITATION = 0.0

# Optimization & Network Parameters
BATCH_SIZE = 48
MAX_GAME_STEPS = 9  # TicTacToe cannot exceed 9 moves
BUFFER_NUM_GAMES = 2000  # 2000 games capacity
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VALUE_LOSS_WEIGHT = 1.0
NUM_FILTERS = 24
NUM_RES_BLOCKS = 3

NUM_EVAL_GAMES = 100
NUM_TRAIN_EVAL_GAMES = 100
SEED = 42


def make_env():
    """Constructs a PettingZoo TicTacToe env wrapped with frame stacking and player plane."""
    env = PettingZooChannelFirstWrapper(tictactoe_v3.env())
    if FRAME_STACK > 1:
        env = FrameStackWrapper(env, k=FRAME_STACK, channel_first=True)
    if INCLUDE_PLAYER_PLANE:
        env = TwoPlayerPlayerPlaneWrapper(env, channel_first=True)
    return env


# ============================================================================
# 1. Pure Functional Helpers
# ============================================================================
def tictactoe_action_to_encoding(action: int, target: torch.Tensor) -> None:
    """Maps 1D action index 0..8 to a 2D spatial 1-hot pixel at (action // 3, action % 3)."""
    target[0, action // 3, action % 3] = 1.0


# Pre-materialized [9, 1, 3, 3] action encodings lookup table:
TICTACTOE_ACTION_ENCODINGS = build_action_encodings(
    num_actions=9,
    encoding_shape=(1, 3, 3),
    mapping_fn=tictactoe_action_to_encoding,
)


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
        self,
        in_channels: int = NUM_CHANNELS,
        num_filters: int = NUM_FILTERS,
        num_res_blocks: int = NUM_RES_BLOCKS,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters

        # 1. Representation Network h(o) -> s^0
        self.representation_stem = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
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

        # Pre-materialized spatial action encodings buffer [9, 1, 3, 3]
        self.register_buffer("action_encodings", TICTACTOE_ACTION_ENCODINGS.clone())

    def representation(self, obs: torch.Tensor) -> torch.Tensor:
        """h(o) -> initial hidden state s0 [B, num_filters, 3, 3] scaled to [0, 1]."""
        x = self.representation_stem(obs)
        raw_s = self.representation_backbone(x)
        return min_max_normalize(raw_s, dim=(-3, -2, -1))

    def dynamics(
        self, hidden_state: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """g(s, a) -> (next_hidden_state [B, num_filters, 3, 3] scaled to [0, 1], reward [B, 1])."""
        if actions.ndim > 1:
            actions = actions.squeeze(-1)
        action_plane = self.action_encodings[actions]

        concat_input = torch.cat([hidden_state, action_plane], dim=1)
        x = self.dynamics_stem(concat_input)
        raw_next_hidden = self.dynamics_backbone(x)
        next_hidden = min_max_normalize(raw_next_hidden, dim=(-3, -2, -1))
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

        next_to_play = 1 - to_play

        # In zero-sum 2-player games, player 0 and player 1 values are inverted.
        # scalar_value predicted at next_hidden is the value for active player at next_hidden (next_to_play).
        v_p0 = torch.where(next_to_play == 0, scalar_value, -scalar_value)
        value_vec = torch.stack([v_p0, -v_p0], dim=-1)

        r_p0 = torch.where(
            to_play == 0, scalar_reward.squeeze(-1), -scalar_reward.squeeze(-1)
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Runs batched MuZero MCTS search over learned representation states."""
    with torch.no_grad():
        s0, root_logits, root_scalar_value = model.initial_inference(root_embed)

    v_p0_root = torch.where(curr_players == 0, root_scalar_value, -root_scalar_value)
    root_vector_value = torch.stack([v_p0_root, -v_p0_root], dim=-1)
    dirichlet_eps = DIRICHLET_EPSILON if explore_dirichlet else 0.0

    qtransform = functools.partial(
        qtransform_by_min_max,
        fpu_fn=fpu_zero,
        min_value=torch.tensor(-1.0),
        max_value=torch.tensor(1.0),
    )

    action_probs, tree = mcts_search(
        root_embeddings=s0,
        root_logits=root_logits,
        root_value=root_vector_value,
        root_to_play=curr_players,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        num_actions=9,
        num_players=2,
        legal_mask=legal_masks,
        qtransform=qtransform,
        dirichlet_epsilon=dirichlet_eps,
        dirichlet_alpha=DIRICHLET_ALPHA,
        pb_c_init=C_PUCT,
        temperature=temperature,
    )
    # Return MCTS root search value from perspective of current player (root.value() in MuZero paper)
    root_search_values = (
        tree["node_values"][:, 0]
        .gather(1, curr_players.unsqueeze(-1))
        .squeeze(-1)
        .detach()
    )
    return action_probs, root_search_values


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
    torch.set_num_threads(1)
    device = torch.device(device_str)
    local_model = model_creator().to(device)
    local_model.eval()
    local_model.load_state_dict(shared_model.state_dict())

    torch.manual_seed(SEED + actor_id * 1000)
    random.seed(SEED + actor_id * 1000)

    envs = [make_env() for _ in range(envs_per_actor)]
    for env in envs:
        env.reset()

    move_counts = [0] * envs_per_actor
    # Trajectory entry: (obs [NUM_CHANNELS,3,3], action int, target_policy [9], player int)
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
            (envs_per_actor, NUM_CHANNELS, 3, 3), device=device, dtype=torch.float32
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

                # Pack completed game into fixed-length sequence [MAX_GAME_STEPS, ...]
                seq_obs = torch.zeros(
                    (MAX_GAME_STEPS, NUM_CHANNELS, 3, 3), dtype=torch.float32
                )
                seq_actions = torch.zeros((MAX_GAME_STEPS,), dtype=torch.long)
                seq_policies = torch.zeros((MAX_GAME_STEPS, 9), dtype=torch.float32)
                seq_values = torch.zeros((MAX_GAME_STEPS, 1), dtype=torch.float32)
                seq_rewards = torch.zeros((MAX_GAME_STEPS, 1), dtype=torch.float32)
                seq_valid = torch.zeros((MAX_GAME_STEPS,), dtype=torch.float32)

                seq_obs[:ep_len] = torch.stack(ep_states)
                seq_actions[:ep_len] = torch.tensor(ep_actions, dtype=torch.long)
                seq_policies[:ep_len] = torch.stack(ep_policies)
                seq_values[:ep_len] = torch.tensor(
                    [rewards_vec[p].item() for p in ep_players], dtype=torch.float32
                ).unsqueeze(-1)
                seq_rewards[:ep_len] = 0.0
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
                        batch_size=[MAX_GAME_STEPS],
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

        action_probs, root_values = run_batched_mcts(
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
            buffer.add_transitions(batch_td, valid_mask=batch_td["valid"])


def evaluator_worker(
    model_creator: Callable[[], nn.Module],
    shared_model: nn.Module,
    eval_queue: mp.Queue,
    training_step: torch.Tensor,
    num_eval_games: int = NUM_TRAIN_EVAL_GAMES,
    eval_interval_steps: int = EVAL_INTERVAL_STEPS,
    device_str: str = "cpu",
):
    """Background process evaluating model performance asynchronously."""
    torch.set_num_threads(1)
    device = torch.device(device_str)
    local_model = model_creator().to(device)
    local_model.eval()

    def select_eval_action(env, curr_player_idx, obs_t, action_mask_t):
        recurrent_fn = create_recurrent_fn(local_model)
        root_embed = obs_t.unsqueeze(0)
        legal_masks = action_mask_t.unsqueeze(0)
        players_t = torch.tensor([curr_player_idx], device=device, dtype=torch.long)

        action_probs, _ = run_batched_mcts(
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

    next_eval_step = 0

    while True:
        curr_step = int(training_step[0].item())
        if curr_step < next_eval_step:
            time.sleep(1.0)
            continue

        next_eval_step = curr_step + eval_interval_steps
        local_model.load_state_dict(shared_model.state_dict())
        eval_metrics = evaluate_pettingzoo_vs_random(
            env_fn=make_env,
            select_action_fn=select_eval_action,
            num_games=num_eval_games,
            device=device,
        )
        eval_metrics["eval/step"] = curr_step
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
    """Learner process training model weights on sampled minibatches."""
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
            "value_loss_weight": value_loss_weight,
            "frame_stack": FRAME_STACK,
            "include_player_plane": INCLUDE_PLAYER_PLANE,
            "num_channels": NUM_CHANNELS,
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

        seq_lens = buffer.buffer_state.valid_lengths[coords[0]].to(device)
        terminal_idx = (seq_lens - 1).clamp(min=0)
        term_vals = (
            buffer.buffer_state.data["target_values"][coords[0], terminal_idx]
            .squeeze(-1)
            .to(device)
        )

        start_steps = coords[1].to(device)

        # A step is valid only if within tensor boundary AND was part of the actual episode
        ep_valid = unroll_data["valid"].bool() & valid_mask
        policy_masks = ep_valid.float().to(device)

        states = unroll_data["state"][:, 0].to(device)
        actions = unroll_data["actions"].to(device)
        target_policies = unroll_data["target_policies"].to(device)
        target_values = unroll_data["target_values"].to(device)
        target_rewards = unroll_data["target_rewards"].to(device)

        # 1. Initial Step k=0 Prediction (guaranteed valid in-game state via prefix-sum sampling)
        s0, p0, v0 = local_model.initial_inference(states)

        raw_p_loss_0, _ = compute_cross_entropy_loss(p0, target_policies[:, 0])
        raw_v_loss_0, _ = compute_mse_loss(v0.view(-1), target_values[:, 0].view(-1))

        loss_p0 = raw_p_loss_0.mean()
        loss_v0 = raw_v_loss_0.mean()
        total_loss = loss_p0 + value_loss_weight * loss_v0

        curr_s = s0
        step_p_losses = [loss_p0.item()]
        step_v_losses = [loss_v0.item()]
        step_r_losses = []

        # 2. Unroll Steps k=1..K Dynamics & Prediction
        for k in range(1, unroll_steps + 1):
            act_k = actions[:, k - 1]
            curr_s = scale_gradient(curr_s, 0.5)

            next_s, r_k, p_k, v_k = local_model.recurrent_inference(curr_s, act_k)
            curr_s = next_s

            raw_p_loss, _ = compute_cross_entropy_loss(p_k, target_policies[:, k])
            mask_k = policy_masks[:, k]
            l_p = (raw_p_loss * mask_k).sum() / torch.clamp(mask_k.sum(), min=1.0)

            # Absorbing post-terminal states (pure value formulation):
            # NOTE: Under the pure value formulation (where transition rewards are 0.0 everywhere), the leaf value alone determines the entire trajectory return during MCTS tree backprop. Setting post-terminal values to 0.0 would dilute winning lines toward 0.0 (a draw) when MCTS simulations explore past terminal states. Therefore, absorbing states must preserve the terminal value from the perspective of the acting player at that step. Since players strictly alternate turns in TicTacToe:
            # - At even steps relative to the terminal state (tau = 0, 2, 4...), it is the winning/losing player's turn -> value target = terminal_value
            # - At odd steps relative to the terminal state (tau = 1, 3...), it is the opponent's turn -> value target = -terminal_value
            step_k = start_steps + k
            tau = step_k - (seq_lens - 1)
            is_post_terminal = step_k >= seq_lens
            alternating_sign = torch.where(tau % 2 == 0, 1.0, -1.0)
            target_v_k = torch.where(
                is_post_terminal,
                alternating_sign * term_vals,
                target_values[:, k].view(-1),
            )

            raw_v_loss, _ = compute_mse_loss(v_k.view(-1), target_v_k)
            raw_r_loss, _ = compute_mse_loss(
                r_k.view(-1), target_rewards[:, k - 1].view(-1)
            )

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
            print(
                f"[Learner Step {step_count}/{max_steps}] Loss: {total_loss.item():.4f} | Buffer: {buffer.size}"
            )

        wandb.log(log_dict)

    wandb.finish()


# ============================================================================
# 5. Main Multiprocessing Pipeline Setup
# ============================================================================
def model_creator_fn() -> TicTacToeMuZeroNet:
    return TicTacToeMuZeroNet(
        in_channels=NUM_CHANNELS,
        num_filters=NUM_FILTERS,
        num_res_blocks=NUM_RES_BLOCKS,
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
        "state": (NUM_CHANNELS, 3, 3),
        "actions": (),
        "target_policies": (9,),
        "target_values": (1,),
        "target_rewards": (1,),
        "valid": (),
    }
    buffer = SharedReplayBuffer(
        (BUFFER_NUM_GAMES, MAX_GAME_STEPS),
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
            args=(actor_id, model_creator_fn, shared_model, buffer),
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
        f"Launched MuZero Multiprocessing Pipeline ({NUM_ACTORS} Actors, 1 Learner, 1 Evaluator) | {NUM_CHANNELS} Channels ({FRAME_STACK} Frames + Player Plane)."
    )
    learner_p.join()

    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join()

    print("MuZero Multiprocessing Training Completed Successfully.")


if __name__ == "__main__":
    main()
