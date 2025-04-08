from typing import Callable, Tuple

from torch import nn, Tensor
import torch.nn.functional as F
import torch

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
        norm: Callable = nn.RMSNorm,
        activation: Callable = nn.SELU,
    ):
        self.activation = activation
        self.is_image = is_image
        self.embedding_dim = embedding_dim
        if is_image:
            assert len(input_shape) == 3
            assert input_shape[1] == input_shape[2]
            num_layers = 0
            width = input_shape[0]  # should be square image
            while width > 6:  # go to images of size 6x6 or 4x4
                width = width // 2
                num_layers += 1

            filters = [embedding_dim // 4] * num_layers
            filters[0] = embedding_dim // 16
            filters[1] = embedding_dim // 8
            # code uses embedding_dim // 16 * [2, 3, 4, 4]
            # paper says first layer has embedding_dim // 16 filters

            self.conv_layers = nn.ModuleList()
            self.norm_layers = nn.ModuleList()
            for i in range(filters):
                self.conv_layers.append(
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=filters[i],
                        kernel_size=5,  # param used in DreamerV3 code
                        stride=2,
                        padding="same",
                    )
                )

                self.norm_layers.append(norm(filters[i]))

        else:
            raise NotImplementedError("Only image inputs are supported")

        self.output_layer = nn.Linear(filters[-1], embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.is_image:
            for conv, norm in zip(self.conv_layers, self.norm_layers):
                x = self.activation(norm(conv(x)))
            x = x.flatten(start_dim=1)
        else:
            raise NotImplementedError("Only image inputs are supported")
        return self.activation(self.output_layer(x))


class Decoder(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        is_image: bool,
        hidden_dim,
        state_dim,
        norm: Callable = nn.RMSNorm,
        activation: Callable = nn.SELU,
        output_activation: Callable = nn.Sigmoid,
        output_size: int = 3,
    ):
        pass


class DynamicsPredictor(nn.Module):
    def __init__(
        self,
        hidden_dim,
        state_dim,
        embedding_dim,
        action_dim,
        rnn_layers=1,
    ):

        self.rnn_layers = nn.ModuleList(
            [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(rnn_layers)]
        )

        self.project_state_action = nn.Linear(action_dim + state_dim, hidden_dim)

        self.prior = nn.Linear(hidden_dim, state_dim * 2)
        self.project_hidden_action = nn.Linear(action_dim + hidden_dim, hidden_dim)

        self.posterior = nn.Linear(hidden_dim, state_dim * 2)
        self.project_hidden_observation = nn.Linear(
            hidden_dim + embedding_dim, hidden_dim
        )

    def forward(self, prev_hidden, prev_state, actions, observations, dones):
        # Batch size, time steps, action dim
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

        for t in range(T):
            action_t = actions[:, t, :]
            observation_t = (
                observations[:, t, :]
                if observations is not None
                else torch.zeros((B, self.embedding_dim), device=actions.device)
            )
            state_t = (
                posterior_states_list[-1][:, 0, :]
                if observations is not None
                else prior_states_list[-1][:, 0, :]
            )
            state_t = state_t if dones is None else state_t * (1 - dones[:, t, :])
            hidden_t = hiddens_list[-1][:, 0, :]

            state_action = torch.cat([state_t, action_t], dim=-1)
            state_action = self.act_fn(self.project_state_action(state_action))

            ### Update the deterministic hidden state ###
            for i in range(len(self.rnn)):
                hidden_t = self.rnn[i](state_action, hidden_t)

            ### Determine the prior distribution ###
            hidden_action = torch.cat([hidden_t, action_t], dim=-1)
            hidden_action = self.act_fn(self.project_hidden_action(hidden_action))
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
                hidden_observation = self.act_fn(
                    self.project_hidden_obs(hidden_observation)
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
            posterior_logvars_list.append(posterior_logvar.unsqueeze(1))
            prior_means_list.append(prior_mean.unsqueeze(1))
            prior_logvars_list.append(prior_logvar.unsqueeze(1))
            prior_states_list.append(prior_state_t.unsqueeze(1))
            posterior_states_list.append(posterior_state_t.unsqueeze(1))
            hiddens_list.append(hidden_t.unsqueeze(1))

        # Convert lists to tensors using torch.cat()
        hiddens = torch.cat(hiddens_list, dim=1)
        prior_states = torch.cat(prior_states_list, dim=1)
        posterior_states = torch.cat(posterior_states_list, dim=1)
        prior_means = torch.cat(prior_means_list, dim=1)
        prior_logvars = torch.cat(prior_logvars_list, dim=1)
        posterior_means = torch.cat(posterior_means_list, dim=1)
        posterior_logvars = torch.cat(posterior_logvars_list, dim=1)

        return (
            hiddens,
            prior_states,
            posterior_states,
            prior_means,
            prior_logvars,
            posterior_means,
            posterior_logvars,
        )


class RewardPredictor(nn.Module):
    def __init__(
        self,
        hidden_dim,
        state_dim,
    ):
        super(RewardPredictor, self).__init__()

        self.dense_layers = DenseStack(
            initial_width=hidden_dim + state_dim,
            widths=[hidden_dim, hidden_dim],
            activation_final=nn.ReLU,
            noisy_sigma=0.0,
        )

        # no relu on output
        self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(self, hidden, state):
        x = torch.cat((hidden, state), dim=-1)
        x = self.dense_layers(x)
        x = self.output_layer(x)
        return x


class ContinuePredictor(nn.Module):
    def __init__():
        pass
