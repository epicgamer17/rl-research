from turtle import st
from typing import Callable, Tuple

from torch import channels_last, nn, Tensor
import torch.nn.functional as F
import torch
import numpy as np

from modules.conv import Conv2dStack
from modules.dense import DenseStack, build_dense
from utils import to_lists
from modules.residual import ResidualStack


class SequenceModelNetwork(nn.Module):
    def __init__():
        pass


class Encoder(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        is_image: bool,
        embedding_dim: int,
        layers: int = 3,  # for vector
        units: int = 1024,  # for vector
        norm: Callable = nn.BatchNorm2d,  # nn.RMSNorm,
        activation: Callable = nn.SiLU(),  #
    ):
        super(Encoder, self).__init__()
        self.activation = activation
        self.is_image = is_image
        self.embedding_dim = embedding_dim
        if is_image:
            assert len(input_shape) == 3
            assert input_shape[0] == input_shape[1], input_shape
            num_layers = 0
            width = input_shape[1]  # should be square image
            while width > 6:  # go to images of size 6x6 or 4x4
                width = width // 2
                num_layers += 1

            filters = [32, 64, 128, 256]
            # filters = [embedding_dim // 4] * num_layers
            # filters[0] = embedding_dim // 16
            # filters[1] = embedding_dim // 8
            # code uses embedding_dim // 16 * [2, 3, 4, 4]
            # paper says first layer has embedding_dim // 16 filters

            self.conv_layers = nn.ModuleList()
            self.norm_layers = nn.ModuleList()
            input_channels = input_shape[2]
            for f in filters:
                self.conv_layers.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=f,
                        kernel_size=5,  # param used in DreamerV3 code
                        stride=2,
                        padding=1,  # add functionality for padding same
                        # padding="same",
                    )
                )

                # self.norm_layers.append(nn.BatchNorm2d(f))
                self.norm_layers.append(norm(f))
                # self.norm_layers.append(nn.RMSNorm(f))
                input_channels = f
            self.output_layer = nn.Linear(
                self._compute_conv_output_shape(input_shape), embedding_dim
            )
        else:
            self.layers = nn.ModuleList()
            self.norm_layers = nn.ModuleList()
            for i in range(layers):
                if i == 0:
                    self.layers.append(nn.Linear(input_shape[0], units))
                else:
                    self.layers.append(nn.Linear(units, units))
                self.norm_layers.append(norm(units))

            self.output_layer = nn.Linear(units, embedding_dim)

    def _compute_conv_output_shape(self, input_shape):
        with torch.no_grad():
            x = torch.randn(1, input_shape[2], input_shape[0], input_shape[1])
            for conv in self.conv_layers:
                x = conv(x)
            return x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x: Tensor) -> Tensor:
        if self.is_image:
            # go from channels last to channels first
            for conv, norm in zip(self.conv_layers, self.norm_layers):
                # print(x.shape)
                x = self.activation(norm(conv(x)))
            x = x.flatten(start_dim=1)
        else:
            for layer, norm in zip(self.layers, self.norm_layers):
                x = self.activation(norm(layer(x)))
        return self.activation(self.output_layer(x))


class Decoder(nn.Module):
    def __init__(
        self,
        output_shape: Tuple[int, ...],
        is_image: bool,
        hidden_dim,
        state_dim,
        embedding_dim,
        min_resolution: int = 16,  # 4
        layers: int = 3,  # for vector encoder
        units: int = 1024,  # for vector encoder
        norm: Callable = nn.BatchNorm2d,  # nn.RMSNorm,
        activation: Callable = nn.SiLU(),
        output_activation: Callable = nn.Sigmoid(),  # None,  #
    ):
        super(Decoder, self).__init__()
        self.activation = activation
        self.is_image = is_image
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.min_resolution = min_resolution
        self.output_shape = output_shape
        if is_image:
            # (C, W, H)
            assert len(output_shape) == 3
            assert output_shape[1] == output_shape[2], output_shape

            assert 3 <= min_resolution <= 16, min_resolution
            self.num_layers = int(np.log2(output_shape[1] / min_resolution))
            # min_resolution * 2^num_layers = output_shape[1]

            # embedding_dim is the formula below
            # output_size = min_resolution**2 * depth * 2 ** (num_layers - 1)
            # conv_transpose_input_size = (
            #     min_resolution**2 * 3 * 2 ** (self.num_layers - 1)
            # )
            conv_transpose_input_size = 256

            self.input_layer = nn.Linear(hidden_dim + state_dim, embedding_dim)
            self.linear_layer = nn.Linear(
                embedding_dim, 256 * (output_shape[1] // 2**self.num_layers) ** 2
            )

            self.conv_transpose_layers = nn.ModuleList()
            self.norm_layers = nn.ModuleList()
            for i in range(self.num_layers):
                # (B, C_in, W, H) -> (B, C_out, W, H)
                if i != self.num_layers - 1:
                    self.conv_transpose_layers.append(
                        nn.ConvTranspose2d(
                            in_channels=conv_transpose_input_size,
                            out_channels=conv_transpose_input_size // 2,
                            kernel_size=3,  # param used in DreamerV3 code
                            stride=2,
                            # padding="same",
                            padding=1,
                            output_padding=1,  # to make sure output size is correct
                        )
                    )

                    conv_transpose_input_size = conv_transpose_input_size // 2

                    self.norm_layers.append(norm(conv_transpose_input_size))
                else:
                    self.output_layer = nn.ConvTranspose2d(
                        in_channels=conv_transpose_input_size,
                        out_channels=output_shape[0],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        # padding="same",
                        output_padding=1,  # to make sure output size is correct
                    )
                self.output_activation = output_activation

            # self.output_layer = nn.ConvTranspose2d(
            #     in_channels=conv_transpose_input_size,
            #     out_channels=output_shape[0],
            #     kernel_size=3,
            #     stride=2,
            #     padding=1,
            #     # padding="same",
            #     output_padding=1,  # to make sure output size is correct
            # )
            # self.output_activation = output_activation
        else:
            self.layers = nn.ModuleList()
            self.norm_layers = nn.ModuleList()
            for i in range(layers):
                if i == 0:
                    self.layers.append(nn.Linear(hidden_dim + state_dim, units))
                else:
                    self.layers.append(nn.Linear(units, units))
                self.norm_layers.append(norm(units))

            self.output_layer = nn.Linear(units, output_shape[0])
            self.output_activation = output_activation

    def forward(self, hidden: Tensor, state: Tensor) -> Tensor:
        if self.is_image:
            x = torch.cat((hidden, state), dim=-1)
            x = self.input_layer(x)
            x = self.linear_layer(x)

            x = x.view(
                -1,
                256,
                self.output_shape[1] // 2**self.num_layers,
                self.output_shape[2] // 2**self.num_layers,
            )

            for conv_transpose, norm in zip(
                self.conv_transpose_layers, self.norm_layers
            ):
                # print(x.shape)
                x = self.activation(norm(conv_transpose(x)))

            x = self.output_layer(x)
            if self.output_activation is not None:
                x = self.output_activation(x)
        else:
            x = torch.cat((hidden, state), dim=-1)
            for layer, norm in zip(self.layers, self.norm_layers):
                x = self.activation(norm(layer(x)))
            x = self.output_layer(x)
            if self.output_activation is not None:
                x = self.output_activation(x)
        return x


class DynamicsPredictor(nn.Module):
    def __init__(
        self,
        hidden_dim,
        state_dim,
        embedding_dim,
        action_dim,
        rnn_layers=1,
        norm: Callable = nn.BatchNorm1d,
        activation: Callable = nn.SiLU(),
    ):
        super(DynamicsPredictor, self).__init__()
        self.rnn_layers = nn.ModuleList(
            [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(rnn_layers)]
        )
        self.activation = activation

        self.project_state_action = nn.Linear(action_dim + state_dim, hidden_dim)

        self.prior = nn.Linear(hidden_dim, state_dim * 2)
        self.project_hidden_action = nn.Linear(action_dim + hidden_dim, hidden_dim)

        self.posterior = nn.Linear(hidden_dim, state_dim * 2)
        self.project_hidden_observation = nn.Linear(
            hidden_dim + embedding_dim, hidden_dim
        )

    def forward(self, prev_hidden, prev_state, actions, observations, dones):
        # Batch size, time steps, action dim
        # assert actions are one hot
        assert actions.sum(dim=-1).max() == 1
        assert len(actions.shape) == 3, f"actions shape: {actions.shape}"
        print(actions.shape)
        B, T, _ = actions.size()  # not sure what this does

        hiddens_list = []
        posterior_means_list = []
        posterior_log_vars_list = []
        prior_means_list = []
        prior_log_vars_list = []
        prior_states_list = []
        posterior_states_list = []

        # (B, 1, hidden_dim)
        hiddens_list.append(prev_hidden.unsqueeze(1))
        prior_states_list.append(prev_state.unsqueeze(1))
        posterior_states_list.append(prev_state.unsqueeze(1))

        for t in range(T - 1):
            action_t = actions[:, t, :]
            observation_t = (
                observations[:, t, :]
                if observations is not None
                else torch.zeros((B, self.embedding_dim), device=actions.device)
            )
            # print(observation_t.shape)
            state_t = (
                posterior_states_list[-1][:, 0, :]
                if observations is not None
                else prior_states_list[-1][:, 0, :]
            )
            # print(state_t.shape)
            # print(dones.shape)
            state_t = (
                state_t if dones is None else state_t * (~dones[:, t, :])
            )  # was 1 - dones
            hidden_t = hiddens_list[-1][:, 0, :]

            state_action = torch.cat([state_t, action_t], dim=-1)
            state_action = self.activation(
                # self.norm(self.project_state_action(state_action))
                self.project_state_action(state_action)
            )

            ### Update the deterministic hidden state ###
            for i in range(len(self.rnn_layers)):
                hidden_t = self.rnn_layers[i](state_action, hidden_t)

            ### Determine the prior distribution ###
            hidden_action = torch.cat([hidden_t, action_t], dim=-1)
            hidden_action = self.activation(self.project_hidden_action(hidden_action))
            prior_params = self.prior(hidden_action)
            prior_mean, prior_log_var = torch.chunk(prior_params, 2, dim=-1)

            ### Sample from the prior distribution ###
            prior_dist = torch.distributions.Normal(
                prior_mean, torch.exp(F.softplus(prior_log_var))
            )
            prior_state_t = prior_dist.rsample()

            ### Determine the posterior distribution ###
            # If observations are not available, we just use the prior
            if observations is None:
                posterior_mean = prior_mean
                posterior_log_var = prior_log_var
            else:
                hidden_observation = torch.cat([hidden_t, observation_t], dim=-1)
                hidden_observation = self.activation(
                    self.project_hidden_observation(hidden_observation)
                )
                posterior_params = self.posterior(hidden_observation)
                posterior_mean, posterior_log_var = torch.chunk(
                    posterior_params, 2, dim=-1
                )

            ### Sample from the posterior distribution ###
            posterior_dist = torch.distributions.Normal(
                posterior_mean, torch.exp(F.softplus(posterior_log_var))
            )

            # Make sure to use rsample to enable the gradient flow
            # Otherwise you could also use code the reparameterization trick by hand
            posterior_state_t = posterior_dist.rsample()

            ### Store results in lists (instead of in-place modification) ###
            posterior_means_list.append(posterior_mean.unsqueeze(1))
            posterior_log_vars_list.append(posterior_log_var.unsqueeze(1))
            prior_means_list.append(prior_mean.unsqueeze(1))
            prior_log_vars_list.append(prior_log_var.unsqueeze(1))
            prior_states_list.append(prior_state_t.unsqueeze(1))
            posterior_states_list.append(posterior_state_t.unsqueeze(1))
            hiddens_list.append(hidden_t.unsqueeze(1))

        # Convert lists to tensors using torch.cat()
        hiddens = torch.cat(hiddens_list, dim=1)
        prior_states = torch.cat(prior_states_list, dim=1)
        posterior_states = torch.cat(posterior_states_list, dim=1)
        prior_means = torch.cat(prior_means_list, dim=1)
        prior_log_vars = torch.cat(prior_log_vars_list, dim=1)
        posterior_means = torch.cat(posterior_means_list, dim=1)
        posterior_log_vars = torch.cat(posterior_log_vars_list, dim=1)

        return (
            hiddens,
            prior_states,
            posterior_states,
            prior_means,
            prior_log_vars,
            posterior_means,
            posterior_log_vars,
        )


class RewardPredictor(nn.Module):
    def __init__(
        self,
        hidden_dim,
        state_dim,
        layers: int = 2,
        activation: Callable = nn.SiLU(),
        norm: Callable = nn.BatchNorm1d,
    ):
        super(RewardPredictor, self).__init__()

        self.dense_layers = DenseStack(
            initial_width=hidden_dim + state_dim,
            widths=[hidden_dim] * layers,
            activation=activation,
            noisy_sigma=0.0,
        )

        # no relu on output
        self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(self, hidden, state):
        x = torch.cat((hidden, state), dim=-1)
        # print(x.shape)
        x = self.dense_layers(x)
        x = self.output_layer(x)
        return x


class ContinuePredictor(nn.Module):
    def __init__(
        self,
        hidden_dim,
        state_dim,
        layers: int = 2,
        activation: Callable = nn.SiLU(),
    ):
        super(ContinuePredictor, self).__init__()

        self.dense_layers = DenseStack(
            initial_width=hidden_dim + state_dim,
            widths=[hidden_dim] * layers,
            activation_final=activation,
            noisy_sigma=0.0,
        )

        # bernouli output
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, hidden, state):
        x = torch.cat((hidden, state), dim=-1)
        x = self.dense_layers(x)
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x
