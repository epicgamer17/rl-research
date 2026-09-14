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

# atomic_rl primitives
from atomic_rl.buffers import SharedReplayBuffer, slice_unrolls
from atomic_rl.losses import (
    compute_cross_entropy_loss,
    compute_commitment_loss,
)
from atomic_rl.metrics import compute_codebook_entropy
from atomic_rl.returns import compute_td_lambda_returns
from atomic_rl.search import (
    backpropagate_,
    expand_node_,
    init_mcts_tree,
    puct_selection,
    select_leaf,
    qtransform_by_parent_and_siblings,
    get_mcts_visit_policy,
)
from atomic_rl.transforms import (
    compute_contractive_transform,
    compute_inverse_contractive_transform,
    compute_two_hot_distribution,
    compute_expected_value,
    compute_straight_through_one_hot,
)
from atomic_rl.utils import min_max_normalize, scale_gradient


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
ENV_NAME = "CartPole-v1"
CODEBOOK_SIZE = 16  # M: number of discrete chance outcomes / codes
LATENT_DIM = 64  # Dimension of hidden state s and afterstate as
NUM_SIMULATIONS = 25  # MCTS simulations per search
UNROLL_STEPS = 5  # K: recurrent unroll steps for BPTT
TD_STEPS = 10  # n-step bootstrap horizon
GAMMA = 0.99  # Discount factor
BETA_COMMITMENT = 0.25  # Commitment loss coefficient beta
BATCH_SIZE = 64
LR = 1e-3


# ===========================================================================
# 1. NETWORKS
# ===========================================================================


# TODO: Step 1 - Define the Representation Network
# Computes the initial hidden state from an observation:
#   h: o_0 -> s_0
class RepresentationNetwork(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int):
        super().__init__()
        # TODO: Define MLP/Conv layers mapping obs_dim -> latent_dim
        # Recommended: nn.Linear -> ReLU/LayerNorm -> nn.Linear
        pass

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: obs: [B, obs_dim] -> hidden_state: [B, latent_dim]
        # Recommended: normalize hidden state (e.g. min_max_normalize or LayerNorm)
        pass


# TODO: Step 2 - Define the Prediction Network
# Predicts policy logits and state value from a decision hidden state:
#   f: s -> (policy_logits, value_logits)
class PredictionNetwork(nn.Module):
    def __init__(self, latent_dim: int, num_actions: int, num_atoms: int = 41):
        super().__init__()
        # TODO: Define policy head (latent_dim -> num_actions)
        # TODO: Define value head (latent_dim -> num_atoms support distribution)
        pass

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: state: [B, latent_dim] -> policy_logits: [B, num_actions], value_logits: [B, num_atoms]
        pass


# TODO: Step 3 - Define Afterstate Dynamics Network
# Computes afterstate representation given state and chosen action:
#   phi: (s, a) -> as (afterstate)
class AfterstateDynamicsNetwork(nn.Module):
    def __init__(self, latent_dim: int, num_actions: int):
        super().__init__()
        # TODO: Combine state and action (e.g. one-hot action concatenated with state)
        # and pass through MLP -> latent_dim
        pass

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # TODO: state: [B, latent_dim], action: [B] -> afterstate: [B, latent_dim]
        pass


# TODO: Step 4 - Define Afterstate Prediction Network
# Predicts chance distribution and afterstate value (Q-value):
#   psi: as -> (chance_logits, q_logits)
class AfterstatePredictionNetwork(nn.Module):
    def __init__(self, latent_dim: int, codebook_size: int, num_atoms: int = 41):
        super().__init__()
        # TODO: Define chance distribution head: latent_dim -> codebook_size (sigma)
        # TODO: Define afterstate value head: latent_dim -> num_atoms (Q-value support)
        pass

    def forward(self, afterstate: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: afterstate: [B, latent_dim] -> chance_logits: [B, codebook_size], q_logits: [B, num_atoms]
        pass


# TODO: Step 5 - Define Chance Dynamics Network
# Given afterstate and chance outcome, produces next state and transition reward:
#   g: (as, c) -> (next_s, reward_logits)
class ChanceDynamicsNetwork(nn.Module):
    def __init__(self, latent_dim: int, codebook_size: int, num_atoms: int = 41):
        super().__init__()
        # TODO: Combine afterstate and one-hot chance code c -> next_s
        # TODO: Define reward head: -> num_atoms
        pass

    def forward(
        self, afterstate: torch.Tensor, chance_code: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: afterstate: [B, latent_dim], chance_code: [B, codebook_size] (one-hot)
        # -> next_state: [B, latent_dim], reward_logits: [B, num_atoms]
        pass


# TODO: Step 6 - Define Chance Encoder Network
# Computes the continuous latent representation of the real transition observation:
#   enc: o_{t+1} -> c^e (continuous code in R^M)
# Uses compute_straight_through_one_hot to obtain discrete one-hot c_ST for training dynamics.
class ChanceEncoderNetwork(nn.Module):
    def __init__(self, obs_dim: int, codebook_size: int):
        super().__init__()
        # TODO: MLP mapping next_obs -> codebook_size
        pass

    def forward(
        self, next_obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: next_obs: [B, obs_dim]
        # 1. continuous_logits = self.net(next_obs)
        # 2. one_hot_st, indices = compute_straight_through_one_hot(continuous_logits, dim=-1)
        # 3. return continuous_logits, one_hot_st, indices
        pass


# TODO: Step 7 - Full Stochastic MuZero Model Wrapper
class StochasticMuZeroModel(nn.Module):
    def __init__(
        self, obs_dim: int, num_actions: int, codebook_size: int, latent_dim: int = 64
    ):
        super().__init__()
        self.representation = RepresentationNetwork(obs_dim, latent_dim)
        self.prediction = PredictionNetwork(latent_dim, num_actions)
        self.afterstate_dynamics = AfterstateDynamicsNetwork(latent_dim, num_actions)
        self.afterstate_prediction = AfterstatePredictionNetwork(
            latent_dim, codebook_size
        )
        self.chance_dynamics = ChanceDynamicsNetwork(latent_dim, codebook_size)
        self.chance_encoder = ChanceEncoderNetwork(obs_dim, codebook_size)


# ===========================================================================
# 2. SEARCH & SELECTION
# ===========================================================================


# TODO: Step 8 - Build Stochastic Selection Function
# Higher-order selection function to pass to `select_leaf(tree, selection_fn=...)`:
def make_stochastic_selection_fn(num_actions: int, codebook_size: int) -> Callable:
    """
    Creates a selection function that branches between decision nodes (PUCT)
    and chance nodes (quasi-sampling or categorical sampling).
    """

    def selection_fn(
        tree: TensorDict, parent_nodes: torch.Tensor, depth: int, **kwargs
    ) -> torch.Tensor:
        # TODO:
        # 1. Check if parent_nodes are chance nodes:
        #    batch_range = torch.arange(tree.batch_size[0], device=tree.device)
        #    is_chance = tree["is_chance"][batch_range, parent_nodes]  # [B] (bool)
        #
        # 2. For Decision nodes (~is_chance):
        #    Use PUCT selection:
        #    decision_action = puct_selection(tree, parent_nodes, depth, **kwargs)
        #
        # 3. For Chance nodes (is_chance):
        #    Quasi-sampling (matching paper Eq. 4):
        #    visits = tree["children_visits"][batch_range, parent_nodes]  # [B, C]
        #    prior = F.softmax(tree["children_prior_logits"][batch_range, parent_nodes], dim=-1)
        #    chance_code = (prior / (visits + 1.0)).argmax(dim=-1)
        #
        # 4. Return combined actions:
        #    return torch.where(is_chance, chance_code, decision_action)
        raise NotImplementedError("TODO: Implement make_stochastic_selection_fn")

    return selection_fn


# TODO: Step 9 - Implement Stochastic MCTS Simulation Loop
def stochastic_mcts_search(
    model: StochasticMuZeroModel,
    root_observations: torch.Tensor,
    num_simulations: int,
    num_actions: int,
    codebook_size: int,
) -> TensorDict:
    """
    Runs Stochastic MCTS with alternating Decision and Chance expansions.

    Args:
        model: StochasticMuZeroModel
        root_observations: [B, obs_dim]
        num_simulations: number of simulation steps
        num_actions: |A|
        codebook_size: M

    Returns:
        tree: final MCTS tree TensorDict (caller extracts visit policy via get_mcts_visit_policy)
    """
    batch_size = root_observations.shape[0]
    device = root_observations.device
    max_branching = max(num_actions, codebook_size)

    # TODO:
    # 1. Initial representation:
    #    root_s = model.representation(root_observations)
    #    root_p_logits, root_v_logits = model.prediction(root_s)
    #    root_v = compute_expected_value(root_v_logits)  # [B, 1]
    #
    # 2. Allocate MCTS tree via `init_mcts_tree`:
    #    tree = init_mcts_tree(..., num_actions=max_branching, num_players=1)
    #    tree["is_chance"] = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)
    #
    # 3. Simulation Loop (for sim in range(num_simulations)):
    #    A. select_leaf(tree, selection_fn=stochastic_selector)
    #    B. For active leaves:
    #       - If parent is Decision node:
    #         Expand Chance node: afterstate = phi(parent_s, action)
    #         predictions: chance_logits, q_logits = psi(afterstate)
    #         expand_node_(tree, ..., policy_logits=chance_logits, value=Q, rewards=0, discounts=1)
    #         tree["is_chance"][leaf_idx] = True
    #       - If parent is Chance node:
    #         Expand Decision node: next_s, r_logits = g(afterstate, chance_code)
    #         predictions: p_logits, v_logits = f(next_s)
    #         expand_node_(tree, ..., policy_logits=p_logits, value=V, rewards=r, discounts=GAMMA)
    #         tree["is_chance"][leaf_idx] = False
    #    C. backpropagate_(tree, trajectory, leaf_value)
    #
    # 4. Return completed tree TensorDict
    raise NotImplementedError("TODO: Implement stochastic_mcts_search")


# ===========================================================================
# 3. LEARNER LOSS & UNROLL
# ===========================================================================


# TODO: Step 10 - Implement the K-Step Recurrent Unroll Loss
def compute_stochastic_muzero_loss(
    model: StochasticMuZeroModel,
    batch: TensorDict,
    num_actions: int,
    codebook_size: int,
    beta: float = BETA_COMMITMENT,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Computes the joint Stochastic MuZero loss across K recurrent unroll steps.

    Loss components:
      1. Policy loss: cross-entropy between p_k and search policy pi_{t+k}
      2. Value loss: two-hot cross-entropy between v_k and target z_{t+k}
      3. Reward loss: two-hot cross-entropy between r_k and real reward u_{t+k}
      4. Afterstate Value (Q) loss: two-hot cross-entropy between q_k and afterstate target
      5. Chance Distribution loss: cross-entropy between sigma_k and quantized code c_{t+k}
      6. Commitment loss: compute_commitment_loss(c^e, stop_gradient(c))

    Args:
        model: StochasticMuZeroModel
        batch: TensorDict with trajectory slices of length K+1:
               'observations': [B, K+1, obs_dim]
               'actions': [B, K]
               'rewards': [B, K]
               'target_values': [B, K+1]
               'target_policies': [B, K+1, num_actions]
        num_actions: |A|
        codebook_size: M
        beta: commitment loss weight

    Returns:
        total_loss: scalar tensor
        info: logging dictionary
    """
    # TODO:
    # 1. Step 0 (Initial observation):
    #    s_0 = model.representation(batch["observations"][:, 0])
    #    p_0, v_0 = model.prediction(s_0)
    #    loss_p0 = compute_cross_entropy_loss(p_0, batch["target_policies"][:, 0])[0]
    #    loss_v0 = compute_cross_entropy_loss(v_0, batch["target_values"][:, 0])[0]
    #
    # 2. Recurrent Steps k in 1..K:
    #    For each step k:
    #      A. Encode real next observation o_{t+k}:
    #         c_e, c_st, c_indices = model.chance_encoder(batch["observations"][:, k])
    #         loss_commit = compute_commitment_loss(c_e, c_st.detach(), beta=beta)[0]
    #
    #      B. Afterstate transition:
    #         as_{k-1} = model.afterstate_dynamics(s_{k-1}, batch["actions"][:, k-1])
    #         sigma_{k-1}, q_{k-1} = model.afterstate_prediction(as_{k-1})
    #         loss_sigma = F.cross_entropy(sigma_{k-1}, c_indices)
    #         loss_q = compute_cross_entropy_loss(q_{k-1}, target_q)[0]
    #
    #      C. Chance transition to next state:
    #         s_k, r_k = model.chance_dynamics(as_{k-1}, c_st)
    #         s_k = scale_gradient(s_k, 0.5)
    #         p_k, v_k = model.prediction(s_k)
    #         loss_p = compute_cross_entropy_loss(p_k, batch["target_policies"][:, k])[0]
    #         loss_v = compute_cross_entropy_loss(v_k, batch["target_values"][:, k])[0]
    #         loss_r = compute_cross_entropy_loss(r_k, batch["rewards"][:, k-1])[0]
    #
    # 3. Diagnostic Metrics:
    #    codebook_entropy = compute_codebook_entropy(c_indices, codebook_size)
    #
    # 4. Sum all losses:
    #    total_loss = (loss_p + loss_v + loss_r + loss_q + loss_sigma + loss_commit).mean()
    #    return total_loss, info_dict
    raise NotImplementedError("TODO: Implement compute_stochastic_muzero_loss")


# ===========================================================================
# 4. MULTIPROCESSING ORCHESTRATION SKELETON
# ===========================================================================


# TODO: Step 11 - Actor Self-Play Worker
def actor_worker(
    actor_id: int, model: StochasticMuZeroModel, buffer: SharedReplayBuffer
):
    """
    Self-play actor worker:
    1. Runs environment steps.
    2. Calls stochastic_mcts_search to obtain action policy pi.
    3. Samples action from pi, steps environment.
    4. Computes n-step bootstrapped targets via `compute_td_lambda_returns`.
    5. Pushes (obs, action, reward, target_value, target_policy) into buffer.
    """
    pass


# TODO: Step 12 - Learner Worker
def learner_worker(
    model: StochasticMuZeroModel,
    buffer: SharedReplayBuffer,
    optimizer: torch.optim.Optimizer,
):
    """
    Learner training worker:
    1. Samples minibatches of unroll slices from buffer via `slice_unrolls`.
    2. Calls compute_stochastic_muzero_loss.
    3. Backprops and steps optimizer.
    4. Periodically broadcasts weights to actors.
    """
    pass


def main():
    print("Stochastic MuZero Scaffold initialized.")
    print("Follow the TODO comments in steps 1-12 to complete the implementation.")


if __name__ == "__main__":
    main()
