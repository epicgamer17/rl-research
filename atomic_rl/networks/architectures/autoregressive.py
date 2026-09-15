"""
Autoregressive Policy Heads for Combinatorial and Multi-Part Action Spaces.

Origin & Primary Reference:
- "Learning and Planning in Complex Action Spaces" (Sampled MuZero)
  Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Mohammadamin Barekatain,
  Simon Schmitt, David Silver, ICML 2021.
  arXiv: https://arxiv.org/abs/2104.06303
  Section 4.2 & Appendix: Autoregressive policy representation for multi-step / micro-action spaces.

Secondary Reference & Extensions:
- "Planning in Stochastic Environments with a Learned Model" (Stochastic MuZero)
  Ioannis Antonoglou, Julian Schrittwieser, Sherjil Ozair, Thomas K. Hubert, David Silver, ICLR 2022.
  arXiv: https://arxiv.org/abs/2105.14324
  Appendix A.3: Backgammon architecture unrolled for 4 micro-action steps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Sequence, Tuple, Union


# NOTE (Origin: Sampled MuZero, Hubert et al. 2021):
# This autoregressive policy head formulation was introduced in Sampled MuZero
# to tractably handle large or continuous/combinatorial action spaces without
# enumerating the exponential cartesian product of actions:
#
# Context & Motivation:
# In combinatorial action spaces (e.g. Backgammon with up to 4 dice micro-actions,
# multi-agent joint action spaces, or multi-discrete control), the joint action space
# |A| = prod_{k=1}^K |A_k| is far too large to represent with a flat categorical softmax.
#
# Mathematical Factorization (Sampled MuZero §4.2):
# Using the chain rule of probability, the joint policy is factorized autoregressively:
#     P(a_1, a_2, ..., a_K | s) = prod_{k=1}^K P(a_k | s, a_{<k})
#
# Architecture & Lifecycle (Sampled MuZero / Stochastic MuZero Backgammon):
# 1. State Conditioning:
#    The root latent state s in R^D initializes the recurrent hidden state:
#        h_0 = W_init * s + b_init
#
# 2. Sequential Micro-Action Step (k = 1 ... K):
#    Given previous micro-action a_{k-1} (or a learned start token for k=1):
#        e_{k-1} = Embedding(a_{k-1})
#        h_k = RNNCell(e_{k-1}, h_{k-1})  [e.g. nn.GRUCell or small MLP]
#        logits_k = Linear_k(h_k) in R^{|A_k|}
#
# 3. Training Mode (Teacher Forcing):
#    Given ground-truth micro-actions a = [a_1, ..., a_K], all logits are computed
#    sequentially using ground-truth past actions:
#        log P(a | s) = sum_{k=1}^K log P(a_k | s, a_{<k})
#    Cross-entropy loss against MCTS search policy targets pi_k is summed across steps.
#
# 4. Inference / Generation Mode (Autoregressive Sampling):
#    For step k = 1 ... K:
#        a. Predict logits_k from h_{k-1} and a_{k-1}.
#        b. Mask invalid micro-actions: logits_k = torch.where(mask_k, logits_k, -inf).
#        c. Sample a_k ~ Softmax(logits_k / temperature).
#        d. Feed a_k as input to step k+1.
#    Returns tuple of sampled micro-actions (a_1, ..., a_K).
class AutoregressivePolicyHead(nn.Module):
    """
    Autoregressive policy head modeling joint probability of K sequential micro-actions.

    Note:
        Originally proposed in **Sampled MuZero** (Hubert et al., ICML 2021) to address
        combinatorial and complex action spaces, and subsequently used in **Stochastic MuZero**
        (Antonoglou et al., ICLR 2022, Appendix A.3) for games like Backgammon (4-step micro-actions).

    Args:
        latent_dim (int): Dimensionality of the conditioning hidden state s.
        action_dims (Sequence[int]): Number of discrete choices for each micro-action step [|A_1|, ..., |A_K|].
        hidden_dim (int, optional): Dimensionality of the recurrent hidden state. Defaults to latent_dim.
        embed_dim (int, optional): Dimensionality of micro-action embeddings. Defaults to 64.
        rnn_type (str): Recurrent cell type: 'gru' or 'mlp'. Defaults to 'gru'.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dims: Sequence[int],
        hidden_dim: Optional[int] = None,
        embed_dim: int = 64,
        rnn_type: str = "gru",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dims = list(action_dims)
        self.num_steps = len(action_dims)
        self.hidden_dim = hidden_dim if hidden_dim is not None else latent_dim
        self.embed_dim = embed_dim
        self.rnn_type = rnn_type

        # TODO: Define initial state projection:
        # self.init_proj = nn.Linear(latent_dim, self.hidden_dim)

        # TODO: Define learned start token embedding for step 0:
        # self.start_token = nn.Parameter(torch.zeros(1, embed_dim))

        # TODO: Define action embeddings for each step:
        # self.action_embeddings = nn.ModuleList([
        #     nn.Embedding(num_actions, embed_dim) for num_actions in self.action_dims
        # ])

        # TODO: Define transition cell:
        # if rnn_type == "gru":
        #     self.cell = nn.GRUCell(embed_dim, self.hidden_dim)

        # TODO: Define output prediction heads for each micro-action step:
        # self.heads = nn.ModuleList([
        #     nn.Linear(self.hidden_dim, num_actions) for num_actions in self.action_dims
        # ])

    def forward(
        self,
        state: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes micro-action logits conditioned on state and optionally past actions.

        Args:
            state (torch.Tensor): Conditioning state representation [*Batch, latent_dim].
            actions (torch.Tensor, optional): Ground-truth micro-actions [*Batch, num_steps]
                for teacher forcing during training. If None, indicates sequential sampling mode.

        Returns:
            List[torch.Tensor]: List of length num_steps containing logits [*Batch, |A_k|]
                for each micro-action step.
        """
        # TODO: Implement teacher-forced forward pass:
        # 1. h = self.init_proj(state)
        # 2. logits_list = []
        # 3. for k in range(self.num_steps):
        #        embed = self.start_token.expand(B, -1) if k == 0 else self.action_embeddings[k-1](actions[:, k-1])
        #        h = self.cell(embed, h)
        #        logits_k = self.heads[k](h)
        #        logits_list.append(logits_k)
        # 4. return logits_list
        raise NotImplementedError("TODO: Implement AutoregressivePolicyHead.forward")

    @torch.no_grad()
    def sample(
        self,
        state: torch.Tensor,
        masks: Optional[Sequence[torch.Tensor]] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sequentially samples a complete action tuple (a_1, ..., a_K) autoregressively.

        Args:
            state (torch.Tensor): Conditioning state representation [*Batch, latent_dim].
            masks (Sequence[torch.Tensor], optional): Optional list of boolean masks [*Batch, |A_k|]
                specifying legal micro-actions at each step.
            temperature (float): Softmax sampling temperature.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - sampled_actions: Integer tensor [*Batch, num_steps] containing the sampled micro-actions.
                - joint_log_probs: Log probabilities [*Batch] of the complete sampled actions.
        """
        # TODO: Implement autoregressive sequential sampling loop:
        # 1. h = self.init_proj(state)
        # 2. sampled_actions = []
        # 3. joint_log_probs = 0.0
        # 4. for k in range(self.num_steps):
        #        embed = self.start_token if k == 0 else self.action_embeddings[k-1](prev_action)
        #        h = self.cell(embed, h)
        #        logits = self.heads[k](h)
        #        if masks is not None and masks[k] is not None:
        #            logits = torch.where(masks[k], logits, -1e9)
        #        action_k = sample_distribution(logits, temperature)
        #        prev_action = action_k
        #        sampled_actions.append(action_k)
        # 5. return torch.stack(sampled_actions, dim=-1), joint_log_probs
        raise NotImplementedError("TODO: Implement AutoregressivePolicyHead.sample")
