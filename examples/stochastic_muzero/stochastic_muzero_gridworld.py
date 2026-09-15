"""
Stochstic MuZero Notes:
Some environments can be so large or complex compared to the agent that they can appear stochastic to a finite agent.
There are also Stochastic environments and partially observable environments where deterministic models can be insufficient.

Stochastic MuZero uses after states. Essentially, it makes MuZero learn the distribution from these afterstates to the hidden states. After states represent chance events during the environment transition, they don't represent a real observation. They are more like an state between observations. The value predicted by the agent at these states is the value of the hidden state conditioned on the action selected (before randomness has occurred), then after sampling (or quasi sampling) from the possible chance events, we get the next hidden state. (TODO: Is the POMDP to Stochastic valid? Is there a jumping between beliefs that might occur? ie at one node assuming your opponent has one card, and the very next assuming they have a different card)

The major downside of this is that tree search depth for similar performance is essentially doubled. Additionally, the branching factor is significantly increased meaning that in practice its possible that required search compute is more than twice as much.

Additionally, in practice the learning of the chance distributions can be prone to chance code collapse in the VAE leading to essentially a deterministic model that requires twice as much search depth.

---

Alright lets make a plan for implementing Stochastic AlphaZero based on [stochastic_muzero.pdf](file;file:///Users/jonathanlamontange-kratz/Documents/GitHub/rl-stuff/papers/stochastic_muzero.pdf)

I want to implement it myself. So the plan should be rough. I'll need a run down of existing components and related code.

I will give the feature list. Otherwise I want to make the plan on my own, maybe implement it (even if its partially flawed) and then review it with you. After I implement it and you review it you will probably reimplement it yourself. But the goal is for me to get back into coding and used to that process. If I get stuck I might ask for hints etc. If I get really stuck try and recommend a simple or easy individual part of it i can implement on my own.

Alright here are my implementation notes:

I think we will need to add chance nodes to our tree. I think in practice we might do this via a flag indicating the node type. And maybe overloading other existing stuff like the values and policies. Though it may be less readable and then maybe its better to actually have seperate tensors for the chance codes and distributions and stuff.

Stochastic MuZero Implementation Scaffold.

================================================================================
Paper Reference:
    "Planning in Stochastic Environments with a Learned Model"
    Antonoglou et al., ICLR 2022 (https://arxiv.org/abs/2105.14324)

Summary & Core Concepts:
    In stochastic environments (e.g. 2048, Backgammon, or complex partially observable MDPs),
    standard deterministic MuZero fails because a state-action pair (s, a) does not uniquely
    determine the next state s' or reward r.

    Stochastic MuZero resolves this by factoring the transition into two stages:
      1. Deterministic Action Effect:
         Afterstate Dynamics: phi(s, a) -> as (afterstate embedding)
         Afterstate Prediction: psi(as) -> (sigma, q)
           - sigma: categorical prior over M discrete chance outcomes
           - q: afterstate value Q(s, a)
      2. Stochastic Environment Outcome:
         Chance Dynamics: g(as, c) -> (next_s, r)
         Prediction: f(next_s) -> (p, v)
           - p: policy logits over actions
           - v: state value V(s')
           - r: transition reward

    Representation Quantization:
      - Chance Encoder: enc(o_{t+1}) -> c^e (continuous latent code in R^M)
      - Straight-Through Discretization: c = argmax(c^e), c_ST = c^e + stop_gradient(c - c^e)
      - Commitment Loss: beta * || stop_gradient(c) - c^e ||_2^2

================================================================================
Notes & Architecture:
    - Tree Search: Alternating Decision Nodes (actions a in 0..A-1) and Chance Nodes
      (chance codes c in 0..M-1).
    - In-tree Chance Selection: Quasi-sampling argmax_c [ sigma(c|as) / (N(as, c) + 1) ]
    - Backpropagation: Uses the exact same G_t = R_t + gamma * G_{t+1} recursion.
      Afterstate edges have R=0, gamma=1. Chance edges have R=r, gamma=gamma.
================================================================================
"""

import time
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from tensordict import TensorDict
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import wandb

# atomic_rl primitives
from atomic_rl.action_selection import (
    argmax_selector,
    sample_distribution,
    apply_action_mask,
    add_dirichlet_noise,
    compute_dynamic_dirichlet_alpha,
)
from atomic_rl.buffers import SharedReplayBuffer, slice_unrolls
from atomic_rl.losses import (
    compute_cross_entropy_loss,
    compute_commitment_loss,
    compute_mse_loss,
)
from atomic_rl.metrics import (
    compute_codebook_entropy,
    compute_codebook_perplexity,
    compute_codebook_usage,
    plot_codebook_distribution,
    create_wandb_codebook_metrics,
)
from atomic_rl.returns import (
    compute_td_lambda_returns,
    compute_n_step_td_lambda_returns,
)
from atomic_rl.schedules import get_stepwise_schedule
from atomic_rl.search import (
    backpropagate_,
    expand_node_,
    get_mcts_visit_policy,
    init_mcts_tree,
    make_stochastic_selection_fn,
    puct_selection,
    qtransform_by_parent_and_siblings,
    quasi_sampling_selection,
    select_leaf,
)
from atomic_rl.transforms import (
    build_action_encodings,
    compute_contractive_transform,
    compute_inverse_contractive_transform,
    compute_two_hot_distribution,
    compute_expected_value,
    compute_straight_through_one_hot,
)
from atomic_rl.utils import min_max_normalize, scale_gradient


# ===========================================================================
# 0. ENVIRONMENT: SLIPPERY GRID WORLD
# ===========================================================================
from atomic_rl.envs.mdp import SlipperyGridWorldEnv


# ---------------------------------------------------------------------------
# Hyperparameters & Constants
# ---------------------------------------------------------------------------
ENV_NAME = "SlipperyGridWorld"
GRID_SIZE = 4
OBS_DIM = 16  # 4x4 one-hot position
NUM_ACTIONS = 4  # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
CODEBOOK_SIZE = (
    4  # M: discrete chance codes (0: intended, 1: slip_cw, 2: slip_ccw, 3: spare)
)
LATENT_DIM = 64  # Dimension of hidden state s and afterstate as
NUM_SIMULATIONS = 25  # MCTS simulations per decision
UNROLL_STEPS = 5  # K: recurrent unroll steps for BPTT
TD_STEPS = 5  # n-step bootstrap horizon
TD_LAMBDA = 0.95  # TD(lambda) parameter for n-step bootstrapping

GAMMA = 0.99  # Standard discount for gridworld
BETA_COMMITMENT = 0.25  # Commitment loss coefficient beta
BATCH_SIZE = 64
LR = 1e-3
MAX_STEPS = 30
MAX_EPISODE_STEPS = 35  # Maximum action steps in a game
BUFFER_NUM_GAMES = 1500
MIN_BUFFER_SIZE = 50
PARAM_SYNC_INTERVAL = 100
TOTAL_TRAINING_STEPS = 15000
NUM_ACTORS = 2

# Temperature Schedule Constants (matching MuZero / AlphaZero standard)
TEMP_THRESHOLD_MOVES = 30
TEMPERATURE_EXPLORATION = 1.0
TEMPERATURE_EXPLOITATION = 0.0

# Parallel Evaluation Constants
NUM_EVAL_EPISODES = 20
EVAL_INTERVAL_STEPS = 200


# ===========================================================================
# 1. NETWORKS
# ===========================================================================


# Step 1 - Representation Network
# Computes the initial hidden state from an observation:
#   h: o_0 -> s_0
class RepresentationNetwork(nn.Module):
    """
    Computes initial decision hidden state s_0 from observation o_0:
        h: o_0 -> s_0
    """

    def __init__(self, obs_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw_s = self.net(obs)
        return min_max_normalize(raw_s, dim=-1)


# Step 2 - Prediction Network
# Predicts policy logits and scalar state value in [-1, 1] from a decision hidden state:
#   f: s -> (policy_logits, scalar_value)
class PredictionNetwork(nn.Module):
    """
    Predicts policy logits and scalar state value in [-1, 1] from a decision hidden state:
        f: s -> (policy_logits, scalar_value)
    """

    def __init__(self, latent_dim: int, num_actions: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(latent_dim, num_actions)
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(state)
        policy_logits = self.policy_head(h)
        scalar_value = self.value_head(h)
        return policy_logits, scalar_value


# ---------------------------------------------------------------------------
# Pre-materialized Action & Chance Encodings
# ---------------------------------------------------------------------------
def grid_action_to_encoding(action: int, target: torch.Tensor) -> None:
    """1-hot vector encoding for discrete action 0..3."""
    target[action] = 1.0


def grid_chance_to_encoding(chance_code: int, target: torch.Tensor) -> None:
    """1-hot vector encoding for chance code 0..3."""
    target[chance_code] = 1.0


# Pre-materialized [NUM_ACTIONS, NUM_ACTIONS] action encodings lookup table
GRID_ACTION_ENCODINGS = build_action_encodings(
    num_actions=NUM_ACTIONS,
    encoding_shape=(NUM_ACTIONS,),
    mapping_fn=grid_action_to_encoding,
)

# Pre-materialized [CODEBOOK_SIZE, CODEBOOK_SIZE] chance encodings lookup table
GRID_CHANCE_ENCODINGS = build_action_encodings(
    num_actions=CODEBOOK_SIZE,
    encoding_shape=(CODEBOOK_SIZE,),
    mapping_fn=grid_chance_to_encoding,
)


# Step 3 - Afterstate Dynamics Network
# Computes afterstate representation given state and chosen action:
#   phi: (s, a) -> as (afterstate)
class AfterstateDynamicsNetwork(nn.Module):
    """
    Computes afterstate representation from current state and action:
        phi: (s, a) -> as
    """

    def __init__(self, latent_dim: int, num_actions: int):
        super().__init__()
        self.num_actions = num_actions
        self.register_buffer("action_encodings", GRID_ACTION_ENCODINGS.clone())
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_actions, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.ndim > 1:
            action = action.squeeze(-1)
        action_vector = self.action_encodings[action.long()]
        x = torch.cat([state, action_vector], dim=-1)
        raw_as = self.net(x)
        return min_max_normalize(raw_as, dim=-1)


# Step 4 - Afterstate Prediction Network
# Predicts chance distribution and afterstate value (Q-value):
#   psi: as -> (chance_logits, afterstate_value)
class AfterstatePredictionNetwork(nn.Module):
    """
    Predicts chance distribution logits and afterstate value (Q-value):
        psi: as -> (chance_logits, afterstate_value)
    """

    def __init__(self, latent_dim: int, codebook_size: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )
        self.chance_head = nn.Linear(latent_dim, codebook_size)
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Tanh(),
        )

    def forward(self, afterstate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(afterstate)
        chance_logits = self.chance_head(h)
        afterstate_value = self.value_head(h)
        return chance_logits, afterstate_value


# Step 5 - Chance Dynamics Network
# Given afterstate and chance outcome, produces next state and transition reward:
#   g: (as, c) -> (next_s, scalar_reward)
class ChanceDynamicsNetwork(nn.Module):
    """
    Given afterstate and chance code, resolves stochastic environment transition:
        g: (as, c) -> (next_s, scalar_reward)
    """

    def __init__(self, latent_dim: int, codebook_size: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.register_buffer("chance_encodings", GRID_CHANCE_ENCODINGS.clone())
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim + codebook_size, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Tanh(),
        )

    def forward(
        self, afterstate: torch.Tensor, chance_code: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # If chance_code is already 1-hot (e.g. from straight-through c_ST [B, codebook_size])
        if chance_code.ndim == 2 and chance_code.shape[-1] == self.codebook_size:
            c_vector = chance_code.float()
        else:
            if chance_code.ndim > 1:
                chance_code = chance_code.squeeze(-1)
            c_vector = self.chance_encodings[chance_code.long()]

        x = torch.cat([afterstate, c_vector], dim=-1)
        h = self.backbone(x)
        next_state = min_max_normalize(h, dim=-1)
        scalar_reward = self.reward_head(h)
        return next_state, scalar_reward


# Step 6 - Chance Encoder Network
# Computes the continuous latent representation of the real transition observation:
#   enc: o_{t+1} -> c^e (continuous code in R^M)
# Uses compute_straight_through_one_hot to obtain discrete one-hot c_ST for training dynamics.
class ChanceEncoderNetwork(nn.Module):
    """
    Encodes next observation o_{t+1} into continuous chance logits,
    straight-through one-hot code, and discrete indices:
        enc: o_{t+1} -> (c^e, c_ST, indices)
    """

    def __init__(self, obs_dim: int, codebook_size: int, hidden_dim: int = 64):
        super().__init__()
        # TODO: is this complicated enough/based on the paper?
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, codebook_size),
        )

    def forward(
        self, next_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        continuous_logits = self.net(next_obs)
        one_hot_st, indices = compute_straight_through_one_hot(
            continuous_logits, dim=-1
        )
        return continuous_logits, one_hot_st, indices


# Step 7 - Full Stochastic MuZero Model Wrapper
class StochasticMuZeroModel(nn.Module):
    """
    Unified Stochastic MuZero model packaging:
    - Representation: h(o) -> s0
    - Prediction: f(s) -> (policy_logits, value)
    - Afterstate Dynamics: phi(s, a) -> as
    - Afterstate Prediction: psi(as) -> (chance_logits, afterstate_value)
    - Chance Dynamics: g(as, c) -> (next_s, reward)
    - Chance Encoder: enc(o') -> (c^e, c_ST, indices)
    """

    def __init__(
        self, obs_dim: int, num_actions: int, codebook_size: int, latent_dim: int = 64
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_actions = num_actions
        self.codebook_size = codebook_size

        self.representation = RepresentationNetwork(obs_dim, latent_dim)
        self.prediction = PredictionNetwork(latent_dim, num_actions)
        self.afterstate_dynamics = AfterstateDynamicsNetwork(latent_dim, num_actions)
        self.afterstate_prediction = AfterstatePredictionNetwork(
            latent_dim, codebook_size
        )
        self.chance_dynamics = ChanceDynamicsNetwork(latent_dim, codebook_size)
        self.chance_encoder = ChanceEncoderNetwork(obs_dim, codebook_size)

    def initial_inference(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """h(o) -> s0, f(s0) -> (policy_logits, value)."""
        s0 = self.representation(obs)
        policy_logits, value = self.prediction(s0)
        return s0, policy_logits, value

    def afterstate_inference(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """phi(s, a) -> as, psi(as) -> (chance_logits, afterstate_value)."""
        afterstate = self.afterstate_dynamics(state, action)
        chance_logits, afterstate_value = self.afterstate_prediction(afterstate)
        return afterstate, chance_logits, afterstate_value

    def chance_inference(
        self, afterstate: torch.Tensor, chance_code: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """g(as, c) -> (next_s, reward), f(next_s) -> (policy_logits, value)."""
        next_state, reward = self.chance_dynamics(afterstate, chance_code)
        policy_logits, value = self.prediction(next_state)
        return next_state, reward, policy_logits, value


# ===========================================================================
# 2. SEARCH & SELECTION
# ===========================================================================


# Step 8 - Stochastic Selection Function
# Uses generic `make_stochastic_selection_fn` from `atomic_rl.search` routing between
# decision nodes (`puct_selection`) and chance nodes (`quasi_sampling_selection`):
stochastic_selection_fn = make_stochastic_selection_fn(
    decision_selection_fn=puct_selection,
    chance_selection_fn=quasi_sampling_selection,
)


# Step 9 - Implement Stochastic MCTS Simulation Loop
def stochastic_mcts_search(
    model: StochasticMuZeroModel,
    root_observations: torch.Tensor,
    num_simulations: int,
    num_actions: int,
    codebook_size: int,
    legal_mask: Optional[torch.Tensor] = None,
    max_depth: Optional[int] = None,
    explore_dirichlet: bool = False,
    dirichlet_epsilon: float = 0.25,
    dirichlet_alpha: Optional[float] = None,
) -> TensorDict:
    """
    Runs Stochastic MCTS with alternating Decision and Chance expansions.

    Args:
        model: StochasticMuZeroModel
        root_observations: [B, obs_dim]
        num_simulations: number of simulation steps
        num_actions: |A|
        codebook_size: M
        legal_mask: Optional [B, num_actions] boolean tensor of legal actions at root
        max_depth: Optional maximum search depth
        explore_dirichlet: Whether to add Dirichlet exploration noise to root policy prior
        dirichlet_epsilon: Exploration noise weight (default: 0.25)
        dirichlet_alpha: Dirichlet concentration parameter (default: dynamic 1/sqrt(|A|))

    Returns:
        tree: final MCTS tree TensorDict (caller extracts visit policy via get_mcts_visit_policy)
    """
    assert (
        root_observations.ndim >= 2
    ), f"root_observations must be at least 2D [B, obs_dim], got shape {tuple(root_observations.shape)}"
    batch_size = root_observations.shape[0]
    device = root_observations.device
    max_branching = max(num_actions, codebook_size)
    batch_range = torch.arange(batch_size, device=device)

    # 1. Initial representation and prediction at root (Decision node)
    root_s, root_p_logits, root_v = model.initial_inference(root_observations)
    assert root_p_logits.shape == (batch_size, num_actions)
    assert root_v.shape == (batch_size, 1)

    if explore_dirichlet and dirichlet_epsilon > 0.0:
        alpha = (
            dirichlet_alpha
            if dirichlet_alpha is not None
            else compute_dynamic_dirichlet_alpha(num_actions)
        )
        root_p_logits = add_dirichlet_noise(
            root_p_logits,
            epsilon=dirichlet_epsilon,
            alpha=alpha,
            mask=legal_mask,
        )

    # Construct root prior logits over max_branching actions (pad with -1e9 for illegal branches)
    root_prior_logits = root_observations.new_full((batch_size, max_branching), -1e9)
    root_prior_logits[:, :num_actions] = root_p_logits

    root_branch_mask = torch.zeros(
        (batch_size, max_branching), dtype=torch.bool, device=device
    )
    if legal_mask is not None:
        assert legal_mask.shape == (batch_size, num_actions)
        root_branch_mask[:, :num_actions] = legal_mask
    else:
        root_branch_mask[:, :num_actions] = True

    root_to_play = torch.zeros(batch_size, dtype=torch.long, device=device)

    # 2. Initialize tree state
    tree = init_mcts_tree(
        root_embeddings=root_s,
        root_logits=root_prior_logits,
        root_value=root_v,
        root_to_play=root_to_play,
        num_simulations=num_simulations,
        num_actions=max_branching,
        num_players=1,
        legal_mask=root_branch_mask,
    )

    # 3. Simulation Loop
    for _ in range(num_simulations):
        leaf_parents, leaf_actions, expansion_mask, trajectory = select_leaf(
            tree,
            selection_fn=stochastic_selection_fn,
            max_depth=max_depth,
        )

        if not trajectory:
            break

        b_idx = batch_range[expansion_mask]
        leaf_value = tree["node_values"].new_zeros(batch_size, 1)

        if b_idx.numel() > 0:
            p_nodes = leaf_parents[b_idx]
            actions_taken = leaf_actions[b_idx]
            parent_is_chance = tree["is_chance"][b_idx, p_nodes]

            prior_logits_full = root_prior_logits.new_full(
                (batch_size, max_branching), -1e9
            )
            value_full = tree["node_values"].new_zeros(batch_size, 1)
            reward_full = tree["children_rewards"].new_zeros(batch_size, 1)
            discount_full = tree["children_discounts"].new_zeros(batch_size, 1)
            next_embeddings_full = root_s.new_zeros((batch_size, model.latent_dim))
            players_full = tree["to_play"].new_zeros(batch_size)
            is_chance_full = tree["is_chance"].new_zeros(batch_size, dtype=torch.bool)
            branch_mask_full = torch.zeros(
                (batch_size, max_branching), dtype=torch.bool, device=device
            )

            # Sub-case A: Decision -> Chance (parent is decision, expanding afterstate node)
            dec_mask = ~parent_is_chance
            if dec_mask.any():
                dec_b = b_idx[dec_mask]
                dec_parents = p_nodes[dec_mask]
                dec_actions = actions_taken[dec_mask]
                parent_states = tree["embeddings"][dec_b, dec_parents]

                afterstate, chance_logits, afterstate_val = model.afterstate_inference(
                    parent_states, dec_actions
                )

                prior_logits_full[dec_b, :codebook_size] = chance_logits
                branch_mask_full[dec_b, :codebook_size] = True
                value_full[dec_b] = afterstate_val
                reward_full[dec_b] = 0.0
                discount_full[dec_b] = 1.0
                next_embeddings_full[dec_b] = afterstate
                is_chance_full[dec_b] = True
                leaf_value[dec_b] = afterstate_val

            # Sub-case B: Chance -> Decision (parent is chance/afterstate, expanding decision state)
            chance_mask = parent_is_chance
            if chance_mask.any():
                ch_b = b_idx[chance_mask]
                ch_parents = p_nodes[chance_mask]
                ch_codes = actions_taken[chance_mask]
                parent_afterstates = tree["embeddings"][ch_b, ch_parents]

                next_state, reward, p_logits, state_val = model.chance_inference(
                    parent_afterstates, ch_codes
                )

                prior_logits_full[ch_b, :num_actions] = p_logits
                branch_mask_full[ch_b, :num_actions] = True
                value_full[ch_b] = state_val
                reward_full[ch_b] = reward
                discount_full[ch_b] = GAMMA
                next_embeddings_full[ch_b] = next_state
                is_chance_full[ch_b] = False
                leaf_value[ch_b] = state_val

            expand_node_(
                tree=tree,
                parent_nodes=leaf_parents,
                actions_taken=leaf_actions,
                policy_logits=prior_logits_full,
                value=value_full,
                rewards=reward_full,
                discounts=discount_full,
                next_embeddings=next_embeddings_full,
                players=players_full,
                legal_mask=branch_mask_full,
                masks=expansion_mask,
                is_chance=is_chance_full,
            )

        backpropagate_(
            tree=tree,
            trajectory=trajectory,
            leaf_value=leaf_value,
        )

    return tree


# ===========================================================================
# 3. LEARNER LOSS & UNROLL
# ===========================================================================


# Step 10 - Implement the K-Step Recurrent Unroll Loss
def compute_stochastic_muzero_loss(
    model: StochasticMuZeroModel,
    batch: TensorDict,
    valid_mask: Optional[torch.Tensor] = None,
    num_actions: int = NUM_ACTIONS,
    codebook_size: int = CODEBOOK_SIZE,
    beta: float = BETA_COMMITMENT,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the joint Stochastic MuZero loss across K recurrent unroll steps.

    Loss components:
      1. Policy loss: cross-entropy between p_k and search policy pi_{t+k}
      2. Value loss: MSE between v_k and target return z_{t+k}
      3. Reward loss: MSE between r_k and transition reward u_{t+k}
      4. Afterstate Value (Q) loss: MSE between q_k and afterstate target (u_{t+k} + gamma * z_{t+k})
      5. Chance Distribution loss: cross-entropy between sigma_k and quantized code c_{t+k}
      6. Commitment loss: beta * || c^e - stop_gradient(c) ||_2^2

    Args:
        model: StochasticMuZeroModel
        batch: TensorDict with trajectory slices of length K+1:
               'state': [B, K+1, obs_dim]
               'actions': [B, K+1]
               'target_policies': [B, K+1, num_actions]
               'target_values': [B, K+1, 1]
               'target_rewards': [B, K+1, 1]
               'valid': [B, K+1]
        valid_mask: Optional boolean tensor [B, K+1] from buffer sample_unrolls
        num_actions: |A|
        codebook_size: M
        beta: commitment loss weight

    Returns:
        total_loss: scalar tensor
        info: logging dictionary
    """
    device = batch["state"].device
    unroll_steps = batch["state"].shape[1] - 1  # K steps

    # Compute step mask: valid within trajectory AND within tensor boundary
    mask = batch["valid"].bool()
    if valid_mask is not None:
        mask = mask & valid_mask.bool()
    mask_float = mask.float()

    # 1. Step 0 (Initial observation)
    s_0 = model.representation(batch["state"][:, 0])
    p_0, v_0 = model.prediction(s_0)

    loss_p0 = (
        compute_cross_entropy_loss(p_0, batch["target_policies"][:, 0])[0]
        * mask_float[:, 0]
    ).sum()
    loss_v0 = (
        compute_mse_loss(v_0.squeeze(-1), batch["target_values"][:, 0].squeeze(-1))[0]
        * mask_float[:, 0]
    ).sum()

    total_p_loss = loss_p0
    total_v_loss = loss_v0
    total_r_loss = torch.tensor(0.0, device=device)
    total_q_loss = torch.tensor(0.0, device=device)
    total_sigma_loss = torch.tensor(0.0, device=device)
    total_commit_loss = torch.tensor(0.0, device=device)

    # 2. Recurrent Steps k in 1..K
    s_prev = s_0
    all_chance_indices = []

    for k in range(1, unroll_steps + 1):
        step_mask = mask_float[:, k]

        # A. Encode real next observation o_{t+k}
        c_e, c_st, c_indices = model.chance_encoder(batch["state"][:, k])
        all_chance_indices.append(c_indices)

        loss_commit_k = (
            compute_commitment_loss(c_e, c_st, beta=beta)[0] * step_mask
        ).sum()

        # B. Afterstate transition
        actions_prev = batch["actions"][:, k - 1]
        as_prev = model.afterstate_dynamics(s_prev, actions_prev)
        sigma_prev, q_prev = model.afterstate_prediction(as_prev)

        loss_sigma_k = (
            F.cross_entropy(sigma_prev, c_indices, reduction="none") * step_mask
        ).sum()

        target_q = (
            batch["target_rewards"][:, k - 1] + GAMMA * batch["target_values"][:, k]
        ).squeeze(-1)
        loss_q_k = (compute_mse_loss(q_prev.squeeze(-1), target_q)[0] * step_mask).sum()

        # C. Chance transition to next state
        s_k, r_k = model.chance_dynamics(as_prev, c_st)
        s_k = scale_gradient(s_k, 0.5)
        p_k, v_k = model.prediction(s_k)

        loss_p_k = (
            compute_cross_entropy_loss(p_k, batch["target_policies"][:, k])[0]
            * step_mask
        ).sum()
        loss_v_k = (
            compute_mse_loss(v_k.squeeze(-1), batch["target_values"][:, k].squeeze(-1))[
                0
            ]
            * step_mask
        ).sum()
        loss_r_k = (
            compute_mse_loss(
                r_k.squeeze(-1), batch["target_rewards"][:, k - 1].squeeze(-1)
            )[0]
            * step_mask
        ).sum()

        # Accumulate
        total_p_loss = total_p_loss + loss_p_k
        total_v_loss = total_v_loss + loss_v_k
        total_r_loss = total_r_loss + loss_r_k
        total_q_loss = total_q_loss + loss_q_k
        total_sigma_loss = total_sigma_loss + loss_sigma_k
        total_commit_loss = total_commit_loss + loss_commit_k

        s_prev = s_k

    # Normalize by total valid steps
    total_valid = mask_float.sum().clamp(min=1.0)
    loss_p_mean = total_p_loss / total_valid
    loss_v_mean = total_v_loss / total_valid
    loss_r_mean = total_r_loss / total_valid
    loss_q_mean = total_q_loss / total_valid
    loss_sigma_mean = total_sigma_loss / total_valid
    loss_commit_mean = total_commit_loss / total_valid

    total_loss = (
        loss_p_mean
        + loss_v_mean
        + loss_r_mean
        + loss_q_mean
        + loss_sigma_mean
        + loss_commit_mean
    )

    # 3. Diagnostic Metrics
    if all_chance_indices:
        flat_indices = torch.cat(all_chance_indices, dim=0)
        codebook_entropy = compute_codebook_entropy(flat_indices, codebook_size)
        codebook_perplexity = compute_codebook_perplexity(flat_indices, codebook_size)
        _, active_codes = compute_codebook_usage(flat_indices, codebook_size)
    else:
        flat_indices = torch.zeros(1, dtype=torch.long, device=device)
        codebook_entropy = torch.tensor(0.0, device=device)
        codebook_perplexity = torch.tensor(1.0, device=device)
        active_codes = torch.tensor(0, device=device)

    info = {
        "loss/total": total_loss.detach(),
        "loss/policy": loss_p_mean.detach(),
        "loss/value": loss_v_mean.detach(),
        "loss/reward": loss_r_mean.detach(),
        "loss/q": loss_q_mean.detach(),
        "loss/sigma": loss_sigma_mean.detach(),
        "loss/commitment": loss_commit_mean.detach(),
        "metrics/codebook_entropy": codebook_entropy.detach(),
        "metrics/codebook_perplexity": codebook_perplexity.detach(),
        "metrics/codebook_active_codes": active_codes.detach(),
        "chance_indices": flat_indices.detach(),
    }
    return total_loss, info


# ===========================================================================
# 4. MULTIPROCESSING ORCHESTRATION SKELETON
# ===========================================================================


# ===========================================================================
# 4. MULTIPROCESSING ORCHESTRATION PIPELINE
# ===========================================================================


# Step 11 - Actor Self-Play Worker
def actor_worker(
    actor_id: int,
    shared_model: nn.Module,
    buffer: SharedReplayBuffer,
    training_step: torch.Tensor,
    num_simulations: int = NUM_SIMULATIONS,
    unroll_steps: int = UNROLL_STEPS,
    td_steps: int = TD_STEPS,
    gamma: float = GAMMA,
    td_lambda: float = TD_LAMBDA,
    device_str: str = "cpu",
):
    """
    Self-play actor worker:
    1. Runs PassThePig games.
    2. Runs Stochastic MCTS to generate search policy pi and value v.
    3. Selects action, steps environment.
    4. Computes n-step TD(lambda) bootstrapped targets via `compute_n_step_td_lambda_returns`.
    5. Pushes game trajectories into SharedReplayBuffer.
    """
    torch.set_num_threads(1)
    device = torch.device(device_str)
    local_model = StochasticMuZeroModel(
        OBS_DIM, NUM_ACTIONS, CODEBOOK_SIZE, LATENT_DIM
    ).to(device)
    local_model.eval()

    env = SlipperyGridWorldEnv(slip_prob=0.30, max_steps=MAX_STEPS)
    seed = 42 + actor_id * 1000
    torch.manual_seed(seed)
    np.random.seed(seed)

    loop_counter = 0

    while True:
        loop_counter += 1
        if loop_counter % PARAM_SYNC_INTERVAL == 0:
            local_model.load_state_dict(shared_model.state_dict())

        trajectory_obs = []
        trajectory_actions = []
        trajectory_policies = []
        trajectory_values = []
        trajectory_rewards = []
        trajectory_chances = []

        obs, info = env.reset(seed=int(actor_id * 10000 + loop_counter))
        done = False

        while not done:
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            legal_mask = torch.from_numpy(info["action_mask"]).unsqueeze(0).to(device)

            # Step 9 - Run stochastic_mcts_search
            tree = stochastic_mcts_search(
                model=local_model,
                root_observations=obs_tensor,
                num_simulations=num_simulations,
                num_actions=NUM_ACTIONS,
                codebook_size=CODEBOOK_SIZE,
                legal_mask=legal_mask,
                explore_dirichlet=True,
            )
            # Raw MCTS visit policy target for training (always temperature=1.0 / raw visits)
            raw_target_policy = get_mcts_visit_policy(
                tree["children_visits"][:, 0, :NUM_ACTIONS], temperature=1.0
            )[0]
            root_val = tree["node_values"][0, 0].item()

            step_idx = len(trajectory_obs)
            temp = get_stepwise_schedule(
                step=step_idx,
                thresholds=[TEMP_THRESHOLD_MOVES],
                values=[TEMPERATURE_EXPLORATION, TEMPERATURE_EXPLOITATION],
            )

            if temp == 0.0:
                action_tensor, _ = argmax_selector(raw_target_policy.unsqueeze(0))
                action = action_tensor.squeeze().item()
            else:
                action_policy = (
                    raw_target_policy
                    if temp == 1.0
                    else get_mcts_visit_policy(
                        tree["children_visits"][:, 0, :NUM_ACTIONS], temperature=temp
                    )[0]
                )
                dist = torch.distributions.Categorical(probs=action_policy)
                action_tensor, _ = sample_distribution(dist, explore=True)
                action = action_tensor.item()

            next_obs, reward, terminated, truncated, info = env.step(action)

            trajectory_obs.append(torch.from_numpy(obs).float())
            trajectory_actions.append(action)
            trajectory_policies.append(raw_target_policy.cpu())
            trajectory_values.append(root_val)
            trajectory_rewards.append(reward)
            trajectory_chances.append(info.get("chance", 0))

            done = terminated or truncated or (len(trajectory_obs) >= MAX_EPISODE_STEPS)
            obs = next_obs

        ep_len = min(len(trajectory_obs), MAX_EPISODE_STEPS)
        trajectory_obs = trajectory_obs[:ep_len]
        trajectory_actions = trajectory_actions[:ep_len]
        trajectory_policies = trajectory_policies[:ep_len]
        trajectory_values = trajectory_values[:ep_len]
        trajectory_rewards = trajectory_rewards[:ep_len]

        # Compute n-step TD(lambda) bootstrapped return targets z_t
        rewards_t = torch.tensor(trajectory_rewards, dtype=torch.float32).unsqueeze(-1)
        values_t = torch.tensor(trajectory_values, dtype=torch.float32).unsqueeze(-1)
        next_values_t = torch.zeros_like(values_t)
        if ep_len > 1:
            next_values_t[:-1] = values_t[1:]
        terminated_t = torch.zeros_like(rewards_t)
        terminated_t[-1] = 1.0  # Terminal state at episode end
        truncated_t = torch.zeros_like(rewards_t)

        z_values = compute_n_step_td_lambda_returns(
            rewards=rewards_t.unsqueeze(0),
            terminated=terminated_t.unsqueeze(0),
            truncated=truncated_t.unsqueeze(0),
            values=values_t.unsqueeze(0),
            next_values=next_values_t.unsqueeze(0),
            gamma=gamma,
            td_lambda=td_lambda,
            n=td_steps,
            time_dim=1,
        ).squeeze(0)

        # Pack episode into fixed-length TensorDict for SharedReplayBuffer
        seq_obs = torch.zeros((MAX_EPISODE_STEPS, OBS_DIM), dtype=torch.float32)
        seq_actions = torch.zeros((MAX_EPISODE_STEPS,), dtype=torch.long)
        seq_policies = torch.zeros(
            (MAX_EPISODE_STEPS, NUM_ACTIONS), dtype=torch.float32
        )
        seq_values = torch.zeros((MAX_EPISODE_STEPS, 1), dtype=torch.float32)
        seq_rewards = torch.zeros((MAX_EPISODE_STEPS, 1), dtype=torch.float32)
        seq_valid = torch.zeros((MAX_EPISODE_STEPS,), dtype=torch.float32)

        seq_obs[:ep_len] = torch.stack(trajectory_obs)
        seq_actions[:ep_len] = torch.tensor(trajectory_actions, dtype=torch.long)
        seq_policies[:ep_len] = torch.stack(trajectory_policies)
        seq_values[:ep_len] = z_values
        seq_rewards[:ep_len] = rewards_t
        seq_valid[:ep_len] = 1.0

        game_td = TensorDict(
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
        buffer.add_transitions(
            game_td.unsqueeze(0), valid_mask=seq_valid.unsqueeze(0).bool()
        )


# Step 12 - Learner Worker
def learner_worker(
    shared_model: nn.Module,
    buffer: SharedReplayBuffer,
    training_step: torch.Tensor,
    eval_queue: Optional[mp.Queue] = None,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LR,
    unroll_steps: int = UNROLL_STEPS,
    max_steps: int = TOTAL_TRAINING_STEPS,
    device_str: str = "cpu",
    use_wandb: bool = True,
):
    """
    Learner training worker:
    1. Samples minibatches of unroll slices from SharedReplayBuffer via sample_unrolls.
    2. Calls compute_stochastic_muzero_loss.
    3. Backprops and steps optimizer.
    4. Periodically broadcasts weights to shared_model.
    5. Logs codebook collapse diagnostics and evaluation win rates to Weights & Biases.
    """
    torch.set_num_threads(1)
    device = torch.device(device_str)
    local_model = StochasticMuZeroModel(
        OBS_DIM, NUM_ACTIONS, CODEBOOK_SIZE, LATENT_DIM
    ).to(device)
    local_model.train()
    local_model.load_state_dict(shared_model.state_dict())

    optimizer = torch.optim.Adam(local_model.parameters(), lr=learning_rate)
    rng_key = torch.Generator(device=device)
    rng_key.manual_seed(42)

    if use_wandb:
        try:
            wandb.init(
                project="stochastic-muzero-gridworld",
                name=f"stochastic_muzero_grid_c{CODEBOOK_SIZE}_sims{NUM_SIMULATIONS}",
                config={
                    "env": ENV_NAME,
                    "grid_size": GRID_SIZE,
                    "obs_dim": OBS_DIM,
                    "num_actions": NUM_ACTIONS,
                    "codebook_size": CODEBOOK_SIZE,
                    "latent_dim": LATENT_DIM,
                    "num_simulations": NUM_SIMULATIONS,
                    "unroll_steps": unroll_steps,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "beta_commitment": BETA_COMMITMENT,
                },
            )
        except Exception as e:
            print(
                f"[Learner] wandb.init note: {e}. Continuing with console logging only."
            )
            use_wandb = False

    step_count = 0
    waiting_logged = False
    while step_count < max_steps:
        if buffer.size < MIN_BUFFER_SIZE:
            if not waiting_logged:
                print(
                    f"[Learner] Self-play active. Waiting for replay buffer to reach {MIN_BUFFER_SIZE} games (currently {buffer.size})..."
                )
                waiting_logged = True
            time.sleep(0.5)
            continue

        if waiting_logged:
            print(
                f"[Learner] Replay buffer ready with {buffer.size} games! Starting gradient updates..."
            )
            waiting_logged = False

        sample_res = buffer.sample_unrolls(
            rng_key, batch_size, unroll_steps, min_size=MIN_BUFFER_SIZE
        )
        if sample_res is None:
            time.sleep(0.02)
            continue

        unroll_batch, valid_mask, coords = sample_res
        unroll_batch = unroll_batch.to(device)
        valid_mask = valid_mask.to(device)

        # Step 10 - Compute Stochastic MuZero Loss
        loss, loss_info = compute_stochastic_muzero_loss(
            model=local_model,
            batch=unroll_batch,
            valid_mask=valid_mask,
            num_actions=NUM_ACTIONS,
            codebook_size=CODEBOOK_SIZE,
            beta=BETA_COMMITMENT,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=5.0)
        optimizer.step()

        step_count += 1
        training_step[0] = step_count

        if step_count % PARAM_SYNC_INTERVAL == 0:
            shared_model.load_state_dict(local_model.state_dict())
            entropy_val = loss_info["metrics/codebook_entropy"].item()
            perp_val = loss_info["metrics/codebook_perplexity"].item()
            active_val = loss_info["metrics/codebook_active_codes"].item()

            print(
                f"[Learner] Step {step_count}/{max_steps} | Loss: {loss.item():.4f} | "
                f"Policy: {loss_info['loss/policy'].item():.4f} | "
                f"Value: {loss_info['loss/value'].item():.4f} | "
                f"Reward: {loss_info['loss/reward'].item():.4f} | "
                f"Q: {loss_info['loss/q'].item():.4f} | "
                f"Sigma: {loss_info['loss/sigma'].item():.4f} | "
                f"Commit: {loss_info['loss/commitment'].item():.4f} | "
                f"CB Entropy: {entropy_val:.3f} | "
                f"Perplexity: {perp_val:.2f}/{CODEBOOK_SIZE} | "
                f"Active: {active_val}/{CODEBOOK_SIZE}"
            )

            if use_wandb:
                cb_metrics = create_wandb_codebook_metrics(
                    chance_codes=loss_info["chance_indices"],
                    codebook_size=CODEBOOK_SIZE,
                    step=step_count,
                    log_histogram=True,
                    log_bar_chart=True,
                )
                log_dict = {
                    "step": step_count,
                    "loss/total": loss.item(),
                    "loss/policy": loss_info["loss/policy"].item(),
                    "loss/value": loss_info["loss/value"].item(),
                    "loss/reward": loss_info["loss/reward"].item(),
                    "loss/q": loss_info["loss/q"].item(),
                    "loss/sigma": loss_info["loss/sigma"].item(),
                    "loss/commitment": loss_info["loss/commitment"].item(),
                    **cb_metrics,
                }
                wandb.log(log_dict, step=step_count)

        # Drain asynchronous evaluation results from parallel evaluator worker
        if eval_queue is not None:
            while not eval_queue.empty():
                try:
                    eval_metrics = eval_queue.get_nowait()
                    eval_step = eval_metrics.get("step", step_count)
                    print(
                        f"[Evaluation @ Step {eval_step}] Win Rate: {eval_metrics['eval/win_rate']:.2f} | "
                        f"Avg Turns: {eval_metrics['eval/avg_turns']:.1f}"
                    )
                    if use_wandb:
                        wandb.log(eval_metrics, step=eval_step)
                except Exception:
                    break

    if use_wandb:
        wandb.finish()


def evaluate_model(
    model: nn.Module,
    num_episodes: int = 50,
    max_steps: int = MAX_STEPS,
    device_str: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluates the model greedily on SlipperyGridWorld (temperature=0.0).
    """
    device = torch.device(device_str)
    model.eval()
    env = SlipperyGridWorldEnv(slip_prob=0.30, max_steps=max_steps)

    wins = 0
    total_steps = 0

    for ep in range(num_episodes):
        obs, info = env.reset(seed=10000 + ep)
        done = False
        while not done:
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            legal_mask = torch.from_numpy(info["action_mask"]).unsqueeze(0).to(device)
            # Run Stochastic MCTS search greedily (temperature=0.0, no Dirichlet noise)
            tree = stochastic_mcts_search(
                model=model,
                root_observations=obs_tensor,
                num_simulations=NUM_SIMULATIONS,
                num_actions=NUM_ACTIONS,
                codebook_size=CODEBOOK_SIZE,
                legal_mask=legal_mask,
            )
            eval_policy = get_mcts_visit_policy(
                tree["children_visits"][:, 0, :NUM_ACTIONS], temperature=0.0
            )[0]
            action_tensor, _ = argmax_selector(eval_policy.unsqueeze(0))
            action = action_tensor.squeeze().item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            obs = next_obs

        if info.get("cell") == "G":
            wins += 1
        total_steps += info.get("step_count", 0)

    win_rate = wins / num_episodes
    avg_steps = total_steps / num_episodes
    return {"eval/win_rate": win_rate, "eval/avg_turns": avg_steps}


def evaluator_worker(
    shared_model: nn.Module,
    eval_queue: mp.Queue,
    training_step: torch.Tensor,
    num_eval_episodes: int = NUM_EVAL_EPISODES,
    eval_interval_steps: int = EVAL_INTERVAL_STEPS,
    max_steps: int = TOTAL_TRAINING_STEPS,
    device_str: str = "cpu",
):
    """
    Background worker evaluating shared_model asynchronously in parallel.
    Runs completely decoupled from the learner to eliminate training freezes.
    """
    torch.set_num_threads(1)
    device = torch.device(device_str)
    local_model = StochasticMuZeroModel(
        OBS_DIM, NUM_ACTIONS, CODEBOOK_SIZE, LATENT_DIM
    ).to(device)
    local_model.eval()

    next_eval_step = eval_interval_steps

    while True:
        curr_step = int(training_step[0].item())
        if curr_step >= max_steps:
            break
        if curr_step < next_eval_step:
            time.sleep(0.5)
            continue

        next_eval_step = curr_step + eval_interval_steps
        local_model.load_state_dict(shared_model.state_dict())

        eval_metrics = evaluate_model(
            local_model, num_episodes=num_eval_episodes, device_str=device_str
        )
        eval_metrics["step"] = curr_step
        eval_queue.put(eval_metrics)


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    device_str = "cpu"
    device = torch.device(device_str)

    shared_model = StochasticMuZeroModel(
        OBS_DIM, NUM_ACTIONS, CODEBOOK_SIZE, LATENT_DIM
    ).to(device)
    shared_model.share_memory()

    training_step = torch.zeros(1, dtype=torch.long).share_memory_()
    eval_queue = mp.Queue()

    buffer_shapes = {
        "state": (OBS_DIM,),
        "actions": (),
        "target_policies": (NUM_ACTIONS,),
        "target_values": (1,),
        "target_rewards": (1,),
        "valid": (),
    }
    buffer = SharedReplayBuffer(
        (BUFFER_NUM_GAMES, MAX_EPISODE_STEPS),
        buffer_shapes,
        device=device,
    )

    print(
        f"Initialized Stochastic MuZero for {ENV_NAME} ({GRID_SIZE}x{GRID_SIZE} Grid, {NUM_ACTIONS} actions, {CODEBOOK_SIZE} chance codes)."
    )

    processes = []
    learner_p = mp.Process(
        target=learner_worker,
        args=(shared_model, buffer, training_step, eval_queue),
        kwargs={"device_str": device_str},
    )
    learner_p.start()
    processes.append(learner_p)

    evaluator_p = mp.Process(
        target=evaluator_worker,
        args=(shared_model, eval_queue, training_step),
        kwargs={
            "num_eval_episodes": NUM_EVAL_EPISODES,
            "eval_interval_steps": EVAL_INTERVAL_STEPS,
            "max_steps": TOTAL_TRAINING_STEPS,
            "device_str": device_str,
        },
    )
    evaluator_p.start()
    processes.append(evaluator_p)

    for actor_id in range(NUM_ACTORS):
        actor_p = mp.Process(
            target=actor_worker,
            args=(actor_id, shared_model, buffer, training_step),
            kwargs={"device_str": device_str},
        )
        actor_p.start()
        processes.append(actor_p)

    print(
        f"Launched Stochastic MuZero Multiprocessing Pipeline ({NUM_ACTORS} Actors, 1 Learner, 1 Parallel Evaluator)."
    )
    learner_p.join()

    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join()

    print("Training finished! Running final evaluation...")
    eval_results = evaluate_model(shared_model, num_episodes=20, device_str=device_str)
    print(f"Final Evaluation Results: {eval_results}")


if __name__ == "__main__":
    main()
