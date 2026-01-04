"""
World Models Paper Implementation - World Model Component
Based on: https://arxiv.org/abs/1803.10122

This implements the V (VAE) and M (MDN-RNN) components as a unified world model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, List, Tuple, Dict, Optional
from torch import Tensor

from agent_configs.base_config import Config
from modules.encoder_decoder import VAE
from modules.heads import CategoricalHead, ScalarHead
from modules.mdrnn import MDNRNN
from modules.world_models.world_model import WorldModelInterface, WorldModelOutput


class MDRNNWorldModel(WorldModelInterface, nn.Module):
    """
    Complete World Model from the World Models paper.
    Combines VAE (V) and MDN-RNN (M) into a unified interface.
    """

    def __init__(
        self,
        config: Config,
        input_shape: Tuple[int],
        num_actions: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        num_mixtures: int = 5,
    ):
        nn.Module.__init__(self)
        self.config = config
        self.latent_dim = latent_dim
        self.num_actions = num_actions

        # V Model: VAE for encoding observations
        input_channels = input_shape[1] if len(input_shape) == 4 else 3
        self.vae = VAE(input_channels=input_channels, latent_dim=latent_dim)

        # M Model: MDN-RNN for predicting future
        self.mdn_rnn = MDNRNN(
            latent_dim=latent_dim,
            action_dim=num_actions,
            hidden_dim=hidden_dim,
            num_mixtures=num_mixtures,
        )

        self.reward_head = ScalarHead(
            config,
            input_shape=(config.minibatch_size, hidden_dim),
            layer_prefix="reward",
        )

        self.to_play_head = CategoricalHead(
            config,
            input_shape=(config.minibatch_size, hidden_dim),
            output_size=config.game.num_players,
            layer_prefix="to_play",
        )

        self.hidden_dim = hidden_dim

    def initial_inference(self, observation: Tensor) -> Tensor:
        """
        Encode observation into latent state z.

        Args:
            observation: Raw observation (B, C, H, W)

        Returns:
            z: Latent state (B, latent_dim)
        """
        with torch.no_grad():
            z = self.vae.sample_latent(observation)
        return WorldModelOutput(feature_vector=z)

    def recurrent_inference(
        self,
        hidden_state: Tensor,
        action: Tensor,
        rnn_hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tuple[Tensor, Tensor]]:
        """
        Predict next state given current state and action.

        Args:
            hidden_state: Current latent state z_t (B, latent_dim)
            action: Action taken (B, action_dim)
            rnn_hidden: LSTM hidden state (h, c)
        """
        # Get MDN parameters
        pi, mu, sigma, done_logit, rnn_hidden = self.mdn_rnn(
            hidden_state, action, rnn_hidden
        )

        # Sample next z
        next_z = self.mdn_rnn.sample(pi, mu, sigma, temperature=1.0)

        # Done prediction
        done = torch.sigmoid(done_logit)

        lstm_h = rnn_hidden[0].squeeze(0)  # (B, hidden_dim)
        reward = self.reward_head(lstm_h)
        to_play = self.to_play_head(lstm_h)

        return WorldModelOutput(
            feature_vector=next_z,
            reward=reward,
            to_play=to_play,
            done=done,
            rnn_hidden=rnn_hidden,
        )

    def unroll_sequence(
        self,
        initial_hidden_state: Tensor,
        initial_rnn_hidden: Tuple[Tensor, Tensor],
        actions: Tensor,
        target_observations: Optional[Tensor] = None,
    ) -> Dict[str, List[Tensor]]:
        """
        Performs the unrolling loop using the MDN-RNN.

        Args:
            initial_hidden_state: The starting latent z (B, latent_dim)
            initial_rnn_hidden: The starting LSTM hidden state (h, c)
            actions: Sequence of actions (B, unroll_steps, ...)
            target_observations: Optional ground truth observations (unused for
                                 inference logic here, but kept for API consistency)
        """
        # --- Initialize Storage ---
        hidden_states = initial_hidden_state  # z_t
        rnn_hidden = initial_rnn_hidden  # (h_t, c_t)

        latent_states = [hidden_states]
        rewards = []
        to_plays = []
        dones = []

        # We also want to capture the feature vector sequences for loss calculation later
        # In World Models, prediction is autoregressive on z

        unroll_steps = actions.shape[1]

        # --- Unroll Loop ---
        for k in range(unroll_steps):
            action_k = actions[:, k]

            # Predict next step
            output = self.recurrent_inference(
                hidden_state=hidden_states, action=action_k, rnn_hidden=rnn_hidden
            )

            # Unpack predictions
            next_z = output.feature_vector
            reward = output.reward
            to_play = output.to_play
            done = output.done
            rnn_hidden = output.rnn_hidden  # Update LSTM state

            # Update current state for next iteration
            hidden_states = next_z

            # Store results
            latent_states.append(hidden_states)
            rewards.append(reward)
            to_plays.append(to_play)
            dones.append(done)

        return {
            "latent_states": latent_states,
            "rewards": rewards,
            "to_plays": to_plays,
            "dones": dones,
        }

    def get_networks(self) -> Dict[str, nn.Module]:
        """Return trainable networks."""
        return {
            "vae": self.vae,
            "mdn_rnn": self.mdn_rnn,
        }
