from torch import nn, Tensor
import torch
from typing import Tuple, Optional
import torch.nn.functional as F


class MDNRNN(nn.Module):
    """
    Mixture Density Network RNN (M Model) from World Models.
    Predicts P(z_t+1, done_t+1 | z_t, a_t, h_t) as mixture of Gaussians.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 3,
        hidden_dim: int = 256,
        num_mixtures: int = 5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures

        # LSTM core
        self.lstm = nn.LSTM(
            input_size=latent_dim + action_dim, hidden_size=hidden_dim, batch_first=True
        )

        # MDN output heads
        # For each mixture: pi (weight), mu (mean), sigma (std) for each latent dim
        self.pi_head = nn.Linear(hidden_dim, num_mixtures)
        self.mu_head = nn.Linear(hidden_dim, num_mixtures * latent_dim)
        self.sigma_head = nn.Linear(hidden_dim, num_mixtures * latent_dim)

        # Done prediction head
        self.done_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, z: Tensor, action: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass through MDN-RNN.

        Args:
            z: Latent vectors (B, latent_dim) or (B, T, latent_dim)
            action: Actions (B, action_dim) or (B, T, action_dim)
            hidden: LSTM hidden state (h, c)

        Returns:
            pi: Mixture weights (B, num_mixtures)
            mu: Mixture means (B, num_mixtures, latent_dim)
            sigma: Mixture stds (B, num_mixtures, latent_dim)
            done_logit: Done prediction logit (B, 1)
            hidden: Updated LSTM hidden state
        """
        # Ensure 3D input for LSTM
        if z.dim() == 2:
            z = z.unsqueeze(1)  # (B, 1, latent_dim)
        if action.dim() == 2:
            action = action.unsqueeze(1)  # (B, 1, action_dim)

        # Concatenate z and action
        lstm_input = torch.cat([z, action], dim=-1)  # (B, T, latent_dim + action_dim)

        # LSTM forward
        lstm_out, hidden = self.lstm(lstm_input, hidden)

        # For single timestep, squeeze
        if lstm_out.size(1) == 1:
            lstm_out = lstm_out.squeeze(1)  # (B, hidden_dim)

        # MDN outputs
        pi = F.softmax(self.pi_head(lstm_out), dim=-1)  # (B, num_mixtures)
        mu = self.mu_head(lstm_out).view(-1, self.num_mixtures, self.latent_dim)
        sigma = F.softplus(self.sigma_head(lstm_out)).view(
            -1, self.num_mixtures, self.latent_dim
        )

        # Done prediction
        done_logit = self.done_head(lstm_out)

        return pi, mu, sigma, done_logit, hidden

    def sample(
        self, pi: Tensor, mu: Tensor, sigma: Tensor, temperature: float = 1.0
    ) -> Tensor:
        """
        Sample next z from mixture of Gaussians.

        Args:
            pi: Mixture weights (B, num_mixtures)
            mu: Mixture means (B, num_mixtures, latent_dim)
            sigma: Mixture stds (B, num_mixtures, latent_dim)
            temperature: Temperature for sampling

        Returns:
            Sampled z (B, latent_dim)
        """
        # Apply temperature to mixture weights
        pi_temp = pi / temperature
        pi_temp = F.softmax(pi_temp, dim=-1)

        # Sample mixture component
        component = torch.multinomial(pi_temp, 1).squeeze(-1)  # (B,)

        # Select corresponding mu and sigma
        mu_selected = mu[torch.arange(mu.size(0)), component]  # (B, latent_dim)
        sigma_selected = sigma[torch.arange(sigma.size(0)), component] * temperature

        # Sample from Gaussian
        eps = torch.randn_like(mu_selected)
        z_next = mu_selected + sigma_selected * eps

        return z_next
