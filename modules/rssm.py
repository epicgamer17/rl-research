from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from modules.dense import DenseStack
from modules.distributions import SampleDist


class RSSM(nn.Module):
    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        stoch=30,
        deter=200,
        hidden=200,
        activation=nn.ELU(),
        discrete=False,
        noisy_sigma=0.0,
    ):
        super().__init__()
        self._stoch_size = stoch
        self._deter_size = deter
        self._hidden_size = hidden
        self._discrete = discrete
        self._activation = activation

        # GRU Cell for deterministic path
        # Input to GRU is (stoch + action)
        gru_input_size = stoch + action_dim
        self._cell = nn.GRUCell(gru_input_size, deter)

        # Image/Prior Step MLPs (img1, img2, img3)
        # img1 processes input before GRU
        self._img1 = DenseStack(
            gru_input_size, [hidden], activation=activation, noisy_sigma=noisy_sigma
        )

        # img2, img3 process determ state to produce stoch state stats
        self._img_out = DenseStack(
            deter, [hidden], activation=activation, noisy_sigma=noisy_sigma
        )
        # Project to 2 * stoch (mean, std) or logits
        self._img_dist = nn.Linear(hidden, 2 * stoch if not discrete else stoch)

        # Observation Step MLPs (obs1, obs2)
        # Processes (deter + embed) -> stoch state stats
        self._obs_out = DenseStack(
            deter + embed_dim, [hidden], activation=activation, noisy_sigma=noisy_sigma
        )
        self._obs_dist = nn.Linear(hidden, 2 * stoch if not discrete else stoch)

    def initial(self, batch_size, device):
        dtype = torch.float32
        return dict(
            mean=torch.zeros(batch_size, self._stoch_size, device=device, dtype=dtype),
            std=torch.zeros(batch_size, self._stoch_size, device=device, dtype=dtype),
            stoch=torch.zeros(batch_size, self._stoch_size, device=device, dtype=dtype),
            deter=torch.zeros(batch_size, self._deter_size, device=device, dtype=dtype),
        )

    def get_feat(self, state):
        return torch.cat([state["stoch"], state["deter"]], -1)

    def get_dist(self, mean, std=None):
        if self._discrete:
            # For discrete, mean is actually logits
            return td.Independent(td.OneHotCategorical(logits=mean), 1)
        else:
            dist = td.Normal(mean, std)
            return td.Independent(dist, 1)

    def obs_step(self, prev_state, prev_action, embed):
        # 1. Run Imagine Step (Prior) to get deterministic state update
        prior = self.img_step(prev_state, prev_action)

        # 2. Compute Posterior (Observed) using new deterministic state + embedding
        x = torch.cat([prior["deter"], embed], -1)
        x = self._obs_out(x)
        x = self._obs_dist(x)

        if self._discrete:
            logits = x
            dist = self.get_dist(logits)
            stoch = dist.rsample()  # Straight-through if OneHotDist supports it
            post = {
                "mean": logits,
                "std": None,
                "stoch": stoch,
                "deter": prior["deter"],
            }
        else:
            mean, std = torch.chunk(x, 2, dim=-1)
            std = F.softplus(std) + 0.1
            dist = self.get_dist(mean, std)
            stoch = dist.rsample()
            post = {"mean": mean, "std": std, "stoch": stoch, "deter": prior["deter"]}

        return post, prior

    def img_step(self, prev_state, prev_action):
        # 1. Prepare inputs for GRU
        x = torch.cat([prev_state["stoch"], prev_action], -1)
        x = self._img1(x)

        # 2. GRU Step
        deter = self._cell(x, prev_state["deter"])

        # 3. Compute Prior Distribution from new deterministic state
        x = self._img_out(deter)
        x = self._img_dist(x)

        if self._discrete:
            logits = x
            dist = self.get_dist(logits)
            stoch = dist.rsample()
            prior = {"mean": logits, "std": None, "stoch": stoch, "deter": deter}
        else:
            mean, std = torch.chunk(x, 2, dim=-1)
            std = F.softplus(std) + 0.1
            dist = self.get_dist(mean, std)
            stoch = dist.rsample()
            prior = {"mean": mean, "std": std, "stoch": stoch, "deter": deter}

        return prior

    def observe(self, embed, action, state=None):
        # embed: [B, T, D]
        # action: [B, T, A]
        if state is None:
            state = self.initial(action.shape[0], action.device)

        # Unbind time dimension for loop
        embeds = embed.unbind(1)
        actions = action.unbind(1)

        posts, priors = [], []
        current = state

        for emb, act in zip(embeds, actions):
            post, prior = self.obs_step(current, act, emb)
            posts.append(post)
            priors.append(prior)
            current = post

        # Stack results
        return self._stack_states(posts), self._stack_states(priors)

    def imagine(self, action, state=None):
        # action: [B, T, A]
        if state is None:
            state = self.initial(action.shape[0], action.device)

        actions = action.unbind(1)
        priors = []
        current = state

        for act in actions:
            prior = self.img_step(current, act)
            priors.append(prior)
            current = prior

        return self._stack_states(priors)

    def _stack_states(self, states_list):
        keys = states_list[0].keys()
        return {k: torch.stack([s[k] for s in states_list], dim=1) for k in keys}
