from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributions as td

from agent_configs.base_config import Config
from modules.conv import Conv2dStack, ConvTranspose2dStack
from modules.dense import DenseStack
from modules.rssm import RSSM
from modules.distributions import TanhBijector, SampleDist


class VAE(nn.Module):
    """
    Variational Autoencoder (V Model) from World Models.
    Encodes observations into latent vectors z.
    """

    def __init__(self, input_channels: int = 3, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 64x64x3 -> latent_dim
        self.encoder = nn.Sequential(
            # 64x64x3 -> 32x32x32
            nn.Conv2d(input_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            # 32x32x32 -> 16x16x64
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            # 16x16x64 -> 8x8x128
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            # 8x8x128 -> 4x4x256
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        # Decoder: latent_dim -> 64x64x3
        self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            # 4x4x256 -> 8x8x128
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            # 8x8x128 -> 16x16x64
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            # 16x16x64 -> 32x32x32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            # 32x32x32 -> 64x64x3
            nn.ConvTranspose2d(32, input_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode input to mu and logvar."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vector to image."""
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.decoder(h)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through VAE.
        Returns: (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def sample_latent(self, x: Tensor) -> Tensor:
        """Sample latent vector z from input."""
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar)


class ConvEncoder(nn.Module):
    def __init__(self, input_shape: Tuple[int], depth=32, activation=nn.ReLU()):
        super().__init__()
        self.activation = activation
        self.depth = depth

        # Hardcoded architecture from reference: 4 layers, stride 2
        # Input: (B, C, H, W) or (C, H, W)
        # We use Conv2dStack for consistency
        self.net = Conv2dStack(
            input_shape=input_shape,
            filters=[1 * depth, 2 * depth, 4 * depth, 8 * depth],
            kernel_sizes=[4, 4, 4, 4],
            strides=[2, 2, 2, 2],
            activation=activation,
            norm_type="none",  # Dreamer often doesn't use BN in encoder
        )
        self.output_dim = 32 * depth  # Flattened dimension assumption for 64x64 input

    def forward(self, obs: Tensor) -> Tensor:
        # obs input might need reshaping if it comes as dict or different format
        # Assuming obs is tensor (B, C, H, W)
        x = self.net(obs)
        return x.reshape(x.shape[0], -1)


class ConvDecoder(nn.Module):
    def __init__(self, feat_size, depth=32, shape=(3, 64, 64), activation=nn.ReLU()):
        super().__init__()
        self.activation = activation
        self.depth = depth
        self.shape = shape  # (C, H, W)

        self.linear = nn.Linear(feat_size, 32 * depth)

        # Transpose Conv Stack
        # Logic to upscale 1x1 -> 64x64
        # We need to manually define this to match the TF implementation logic

        self.decoder_net = ConvTranspose2dStack(
            input_shape=(32 * depth, 1, 1),  # Dummy shape for initialization
            filters=[4 * depth, 2 * depth, 1 * depth, shape[0]],
            kernel_sizes=[5, 5, 6, 6],
            strides=[2, 2, 2, 2],
            activation=activation,
            norm_type="none",
            # The last layer usually shouldn't have activation if it outputs mean
            # but Stack applies to all. We handle last layer activation manually/identity.
        )

    def forward(self, features: Tensor) -> td.Distribution:
        x = self.linear(features)
        x = x.reshape(x.shape[0], 32 * self.depth, 1, 1)

        # Manual forward through stack to control final activation
        for i, layer in enumerate(self.decoder_net._layers):
            x = layer(x)
            # Apply activation to all except the very last one (Image Mean)
            if i < len(self.decoder_net._layers) - 1:
                x = self.activation(x)

        mean = x
        return td.Independent(td.Normal(mean, 1.0), 3)


class ActionDecoder(nn.Module):
    def __init__(
        self,
        feat_size: int,
        action_size: int,
        layers: int = 4,
        units: int = 400,
        dist: str = "tanh_normal",
        activation=nn.ELU(),
        min_std=1e-4,
        init_std=5.0,
        mean_scale=5.0,
    ):
        super().__init__()
        self._size = action_size
        self._dist = dist
        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

        # Use DenseStack
        widths = [units] * layers
        self.net = DenseStack(
            initial_width=feat_size,
            widths=widths,
            activation=activation,
            norm_type="none",
        )

        if dist == "tanh_normal":
            self.head = nn.Linear(self.net.output_width, 2 * action_size)
        elif dist == "onehot":
            self.head = nn.Linear(self.net.output_width, action_size)

    def forward(self, features: Tensor):
        raw_init_std = np.log(np.exp(self._init_std) - 1)
        x = self.net(features)
        x = self.head(x)

        if self._dist == "tanh_normal":
            mean, std = torch.chunk(x, 2, dim=-1)
            mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
            std = F.softplus(std + raw_init_std) + self._min_std

            dist = td.Normal(mean, std)
            dist = td.TransformedDistribution(dist, TanhBijector())
            dist = td.Independent(dist, 1)
            return SampleDist(dist)  # Custom wrapper for mode/sample

        elif self._dist == "onehot":
            return td.OneHotCategorical(logits=x)


class DenseDecoder(nn.Module):
    def __init__(
        self, feat_size, shape, layers, units, dist="normal", activation=nn.ELU()
    ):
        super().__init__()
        self._shape = shape
        self._dist = dist

        widths = [units] * layers
        self.net = DenseStack(
            initial_width=feat_size, widths=widths, activation=activation
        )
        self.head = nn.Linear(self.net.output_width, int(np.prod(shape)))

    def forward(self, features):
        x = self.net(features)
        x = self.head(x)
        x = x.reshape(x.shape[0], *self._shape)

        if self._dist == "normal":
            return td.Independent(td.Normal(x, 1.0), len(self._shape))
        if self._dist == "binary":
            return td.Independent(td.Bernoulli(logits=x), len(self._shape))
