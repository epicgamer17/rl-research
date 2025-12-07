from typing import Callable, Tuple

from agent_configs.muzero_config import MuZeroConfig
from torch import nn, Tensor
from modules.action_encoder import ActionEncoder
from modules.actor import ActorNetwork
from modules.critic import CriticNetwork
from modules.network_block import NetworkBlock
from modules.sim_siam_projector_predictor import Projector
from modules.utils import _normalize_hidden_state, zero_weights_initializer
from muzero.muzero_world_model import MuzeroWorldModel
from utils.utils import to_lists

from modules.conv import Conv2dStack
from modules.dense import DenseStack, build_dense
from modules.residual import ResidualStack
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


# class Representation(nn.Module):
#     def __init__(
#         self,
#         config: MuZeroConfig,
#         input_shape: Tuple[int],
#     ):
#         assert (
#             config.game.is_discrete
#         ), "AlphaZero only works for discrete action space games (board games)"

#         self.config = config

#         super(Representation, self).__init__()

#         self.has_residual_layers = len(config.representation_residual_layers) > 0
#         self.has_conv_layers = len(config.representation_conv_layers) > 0
#         self.has_dense_layers = len(config.representation_dense_layer_widths) > 0
#         assert (
#             self.has_conv_layers or self.has_dense_layers or self.has_residual_layers
#         ), "At least one of the layers should be present."

#         current_shape = input_shape
#         B = current_shape[0]

#         # INPUTS = CONV + BATCHNORM + maybe RELU? into residual etc

#         if self.has_residual_layers:
#             assert (
#                 len(current_shape) == 4
#             ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
#             filters, kernel_sizes, strides = to_lists(
#                 config.representation_residual_layers
#             )

#             # (B, C_in, H, W) -> (B, C_out H, W)
#             self.residual_layers = ResidualStack(
#                 input_shape=current_shape,
#                 filters=filters,
#                 kernel_sizes=kernel_sizes,
#                 strides=strides,
#                 activation=self.config.activation,
#                 noisy_sigma=config.noisy_sigma,
#             )
#             current_shape = (
#                 B,
#                 self.residual_layers.output_channels,
#                 current_shape[2],
#                 current_shape[3],
#             )

#         if self.has_conv_layers:
#             assert (
#                 len(current_shape) == 4
#             ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
#             filters, kernel_sizes, strides = to_lists(config.representation_conv_layers)

#             # (B, C_in, H, W) -> (B, C_out H, W)
#             self.conv_layers = Conv2dStack(
#                 input_shape=current_shape,
#                 filters=filters,
#                 kernel_sizes=kernel_sizes,
#                 strides=strides,
#                 activation=self.config.activation,
#                 noisy_sigma=config.noisy_sigma,
#             )
#             current_shape = (
#                 B,
#                 self.conv_layers.output_channels,
#                 current_shape[2],
#                 current_shape[3],
#             )

#         if self.has_dense_layers:
#             if len(current_shape) == 4:
#                 initial_width = current_shape[1] * current_shape[2] * current_shape[3]
#             else:
#                 assert len(current_shape) == 2
#                 initial_width = current_shape[1]
#             # print(initial_width)
#             # (B, width_in) -> (B, width_out)
#             self.dense_layers = DenseStack(
#                 initial_width=initial_width,
#                 widths=self.config.representation_dense_layer_widths,
#                 activation=self.config.activation,
#                 noisy_sigma=self.config.noisy_sigma,
#             )
#             current_shape = (
#                 B,
#                 self.dense_layers.output_width,
#             )
#         self.output_shape = current_shape

#     def initialize(self, initializer: Callable[[Tensor], None]) -> None:
#         if self.has_residual_layers:
#             self.residual_layers.initialize(initializer)
#         if self.has_conv_layers:
#             self.conv_layers.initialize(initializer)
#         if self.has_dense_layers:
#             self.dense_layers.initialize(initializer)

#     def forward(self, inputs: Tensor):
#         if self.has_conv_layers:
#             assert inputs.dim() == 4

#         # (B, *)
#         S = inputs
#         # INPUT CONV LAYERS???
#         # input batch norm
#         # relu?

#         # (B, C_in, H, W) -> (B, C_out, H, W)
#         if self.has_residual_layers:
#             S = self.residual_layers(S)

#         # (B, C_in, H, W) -> (B, C_out, H, W)
#         if self.has_conv_layers:
#             S = self.conv_layers(S)

#         # (B, *) -> (B, dense_features_in)

#         # (B, dense_features_in) -> (B, dense_features_out)
#         if self.has_dense_layers:
#             # print("S", S.shape)
#             S = S.flatten(1, -1)
#             # print("Flattened S", S.shape)
#             S = self.dense_layers(S)

#             # normalize inputs as per paper
#             min_hidden_state = S.min(1, keepdim=True)[0]
#             max_hidden_state = S.max(1, keepdim=True)[0]
#             scale_hidden_state = max_hidden_state - min_hidden_state
#             scale_hidden_state[scale_hidden_state < 1e-5] += 1e-5
#             hidden_state = (S - min_hidden_state) / scale_hidden_state
#         else:
#             # normalize inputs as per paper
#             min_hidden_state = (
#                 S.view(
#                     -1,
#                     S.shape[1],
#                     S.shape[2] * S.shape[3],
#                 )
#                 .min(2, keepdim=True)[0]
#                 .unsqueeze(-1)
#             )
#             max_hidden_state = (
#                 S.view(
#                     -1,
#                     S.shape[1],
#                     S.shape[2] * S.shape[3],
#                 )
#                 .max(2, keepdim=True)[0]
#                 .unsqueeze(-1)
#             )
#             scale_hidden_state = max_hidden_state - min_hidden_state
#             scale_hidden_state[scale_hidden_state < 1e-5] += 1e-5
#             hidden_state = (S - min_hidden_state) / scale_hidden_state

#         return hidden_state


# class Dynamics(nn.Module):
#     def __init__(
#         self,
#         config: MuZeroConfig,
#         input_shape: Tuple[int],
#         num_actions,
#         action_embedding_dim,
#     ):
#         print("dynamics input shape", input_shape)
#         self.config = config

#         super(Dynamics, self).__init__()

#         is_continuous = not self.config.game.is_discrete

#         self.action_embedding_dim = action_embedding_dim
#         self.action_encoder = ActionEncoder(
#             action_space_size=num_actions,
#             embedding_dim=self.action_embedding_dim,
#             is_continuous=is_continuous,  # Main dynamics handles discrete integer actions
#             single_action_plane=(True),  # self.config.single_action_plane
#         )

#         # TODO: Move fusion into the ActionEncoder
#         if len(input_shape) == 4:
#             self.num_channels = input_shape[1]
#             self.fusion = nn.Conv2d(
#                 self.num_channels + self.action_embedding_dim,
#                 self.num_channels,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             )
#             self.fusion_bn = nn.BatchNorm2d(self.num_channels)
#         else:
#             self.input_size = input_shape[1]
#             self.fusion = nn.Linear(
#                 self.input_size + self.action_embedding_dim, self.input_size, bias=False
#             )
#             self.fusion_bn = nn.BatchNorm1d(self.input_size)
#         # -----------------------------------------------

#         self.has_residual_layers = len(config.dynamics_residual_layers) > 0
#         self.has_conv_layers = len(config.dynamics_conv_layers) > 0
#         self.has_dense_layers = len(config.dynamics_dense_layer_widths) > 0

#         assert (
#             self.has_conv_layers or self.has_dense_layers or self.has_residual_layers
#         ), "At least one of the layers should be present."

#         current_shape = input_shape
#         B = current_shape[0]

#         # INPUTS = CONV + BATCHNORM + maybe RELU? into residual etc

#         if self.has_residual_layers:
#             assert (
#                 len(current_shape) == 4
#             ), "dynamics residual layers expected an input shape should be (B, C, H, W), got {}".format(
#                 current_shape
#             )
#             filters, kernel_sizes, strides = to_lists(config.dynamics_residual_layers)

#             # (B, C_in, H, W) -> (B, C_out H, W)
#             self.residual_layers = ResidualStack(
#                 input_shape=current_shape,
#                 filters=filters,
#                 kernel_sizes=kernel_sizes,
#                 strides=strides,
#                 activation=self.config.activation,
#                 noisy_sigma=config.noisy_sigma,
#             )
#             current_shape = (
#                 B,
#                 self.residual_layers.output_channels,
#                 current_shape[2],
#                 current_shape[3],
#             )

#         if self.has_conv_layers:
#             assert (
#                 len(current_shape) == 4
#             ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
#             filters, kernel_sizes, strides = to_lists(config.dynamics_conv_layers)

#             # (B, C_in, H, W) -> (B, C_out H, W)
#             self.conv_layers = Conv2dStack(
#                 input_shape=current_shape,
#                 filters=filters,
#                 kernel_sizes=kernel_sizes,
#                 strides=strides,
#                 activation=self.config.activation,
#                 noisy_sigma=config.noisy_sigma,
#             )
#             current_shape = (
#                 B,
#                 self.conv_layers.output_channels,
#                 current_shape[2],
#                 current_shape[3],
#             )

#         if self.has_dense_layers:
#             if len(current_shape) == 4:
#                 initial_width = current_shape[1] * current_shape[2] * current_shape[3]
#             else:
#                 assert len(current_shape) == 2
#                 initial_width = current_shape[1]

#             # (B, width_in) -> (B, width_out)
#             # print("initial_width", initial_width)
#             self.dense_layers = DenseStack(
#                 initial_width=initial_width,
#                 widths=self.config.dynamics_dense_layer_widths,
#                 activation=self.config.activation,
#                 noisy_sigma=self.config.noisy_sigma,
#             )
#             current_shape = (
#                 B,
#                 self.dense_layers.output_width,
#             )

#         self.output_shape = current_shape

#         self.has_reward_conv_layers = len(config.reward_conv_layers) > 0
#         self.has_reward_dense_layers = len(config.reward_dense_layer_widths) > 0

#         if self.has_reward_conv_layers:
#             assert (
#                 len(current_shape) == 4
#             ), "reward layers expected an input shape should be (B, C, H, W), got {}".format(
#                 current_shape
#             )
#             filters, kernel_sizes, strides = to_lists(config.reward_conv_layers)

#             self.reward_conv_layers = Conv2dStack(
#                 input_shape=current_shape,
#                 filters=filters,
#                 kernel_sizes=kernel_sizes,
#                 strides=strides,
#                 activation=self.config.activation,
#                 noisy_sigma=config.noisy_sigma,
#             )

#             current_shape = (
#                 B,
#                 self.reward_conv_layers.output_channels,
#                 current_shape[2],
#                 current_shape[3],
#             )

#         if self.has_reward_dense_layers:
#             if len(current_shape) == 4:
#                 initial_width = current_shape[1] * current_shape[2] * current_shape[3]
#             else:
#                 assert len(current_shape) == 2
#                 initial_width = current_shape[1]

#             self.reward_dense_layers = DenseStack(
#                 initial_width=initial_width,
#                 widths=self.config.reward_dense_layer_widths,
#                 activation=self.config.activation,
#                 noisy_sigma=self.config.noisy_sigma,
#             )

#             current_shape = (
#                 B,
#                 self.reward_dense_layers.output_width,
#             )

#         if len(current_shape) == 4:
#             initial_width = current_shape[1] * current_shape[2] * current_shape[3]
#         else:
#             assert len(current_shape) == 2
#             initial_width = current_shape[1]

#         if self.config.value_prefix:
#             self.lstm = nn.LSTM(
#                 input_size=initial_width, hidden_size=self.config.lstm_hidden_size
#             )
#             initial_width = self.config.lstm_hidden_size

#         if config.support_range is not None:
#             self.full_support_size = 2 * config.support_range + 1
#             self.reward = build_dense(
#                 in_features=initial_width,
#                 out_features=self.full_support_size,
#                 sigma=0,
#             )
#         else:
#             self.reward = build_dense(
#                 in_features=initial_width,
#                 out_features=1,
#                 sigma=0,
#             )

#         current_shape = self.output_shape

#         self.has_to_play_conv_layers = len(config.to_play_conv_layers) > 0
#         self.has_to_play_dense_layers = len(config.to_play_dense_layer_widths) > 0

#         if self.has_to_play_conv_layers:
#             assert (
#                 len(current_shape) == 4
#             ), "to_play layers expected an input shape should be (B, C, H, W), got {}".format(
#                 current_shape
#             )
#             filters, kernel_sizes, strides = to_lists(config.to_play_conv_layers)

#             self.to_play_conv_layers = Conv2dStack(
#                 input_shape=current_shape,
#                 filters=filters,
#                 kernel_sizes=kernel_sizes,
#                 strides=strides,
#                 activation=self.config.activation,
#                 noisy_sigma=config.noisy_sigma,
#             )

#             current_shape = (
#                 B,
#                 self.to_play_conv_layers.output_channels,
#                 current_shape[2],
#                 current_shape[3],
#             )

#         if self.has_to_play_dense_layers:
#             if len(current_shape) == 4:
#                 initial_width = current_shape[1] * current_shape[2] * current_shape[3]
#             else:
#                 assert len(current_shape) == 2
#                 initial_width = current_shape[1]

#             self.to_play_dense_layers = DenseStack(
#                 initial_width=initial_width,
#                 widths=self.config.to_play_dense_layer_widths,
#                 activation=self.config.activation,
#                 noisy_sigma=self.config.noisy_sigma,
#             )

#             current_shape = (
#                 B,
#                 self.to_play_dense_layers.output_width,
#             )

#         if len(current_shape) == 4:
#             initial_width = current_shape[1] * current_shape[2] * current_shape[3]
#         else:
#             assert len(current_shape) == 2
#             initial_width = current_shape[1]

#         self.to_play = build_dense(
#             in_features=initial_width,
#             out_features=self.config.game.num_players,
#             sigma=0,
#         )

#     def initialize(self, initializer: Callable[[Tensor], None]) -> None:
#         if self.has_residual_layers:
#             self.residual_layers.initialize(initializer)
#         if self.has_conv_layers:
#             self.conv_layers.initialize(initializer)
#         if self.has_dense_layers:
#             self.dense_layers.initialize(initializer)
#         if self.has_reward_conv_layers:
#             self.reward_conv_layers.initialize(initializer)
#         if self.has_reward_dense_layers:
#             self.reward_dense_layers.initialize(initializer)

#         if self.config.support_range is not None:
#             self.reward.apply(zero_weights_initializer)
#         else:
#             self.reward.initialize(
#                 initializer
#             )  # Standard initialization for scalar output

#         # To Play (always a probability distribution)
#         self.to_play.apply(zero_weights_initializer)

#     def forward(self, hidden_state: Tensor, action, reward_hidden: Tensor):
#         if self.has_conv_layers:
#             assert hidden_state.dim() == 4

#         # --- 1. Embed Action & Fuse (Reference Logic) ---
#         # spatial_shape = (
#         #     hidden_state.shape[0],
#         #     hidden_state.shape[2],
#         #     hidden_state.shape[3],
#         # )

#         # Embed action to (B, EmbedDim, H, W)
#         action_embedded = self.action_encoder(action, hidden_state.shape)

#         # Concatenate (B, C+EmbedDim, H, W)
#         x = torch.cat((hidden_state, action_embedded), dim=1)

#         # Reduce channels (B, C, H, W)
#         x = self.fusion(x)
#         # x = self.fusion_bn(x)

#         # Residual Connection from Input State
#         x = x + hidden_state
#         S = F.relu(x)
#         # ------------------------------------------------

#         # INPUT CONV LAYERS???
#         # input batch norm
#         # relu?

#         # SHOULD I HAVE AN INPUT HERE THAT REDUCES THE CHANNELS BY 1 SO THAT THE NETWORK IS "the same" as the representation?

#         # (B, C_in, H, W) -> (B, C_out, H, W)
#         if self.has_residual_layers:
#             S = self.residual_layers(S)

#         # (B, C_in, H, W) -> (B, C_out, H, W)
#         if self.has_conv_layers:
#             S = self.conv_layers(S)

#         # (B, *) -> (B, dense_features_in)
#         if self.has_dense_layers:
#             # print("S Shape", S.shape)
#             flattened_hidden_state = S.flatten(1, -1)
#             # print("flattened shape", flattened_hidden_state.shape)
#             S = self.dense_layers(flattened_hidden_state)

#             # normalize inputs as per paper
#             min_hidden_state = S.min(1, keepdim=True)[0]
#             max_hidden_state = S.max(1, keepdim=True)[0]
#             scale_encoded_state = max_hidden_state - min_hidden_state
#             scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
#             hidden_state = (S - min_hidden_state) / scale_encoded_state
#         else:
#             # normalize inputs as per paper
#             min_hidden_state = (
#                 S.view(
#                     -1,
#                     S.shape[1],
#                     S.shape[2] * S.shape[3],
#                 )
#                 .min(2, keepdim=True)[0]
#                 .unsqueeze(-1)
#             )
#             max_hidden_state = (
#                 S.view(
#                     -1,
#                     S.shape[1],
#                     S.shape[2] * S.shape[3],
#                 )
#                 .max(2, keepdim=True)[0]
#                 .unsqueeze(-1)
#             )
#             scale_hidden_state = max_hidden_state - min_hidden_state
#             scale_hidden_state[scale_hidden_state < 1e-5] += 1e-5
#             hidden_state = (S - min_hidden_state) / scale_hidden_state

#         # (B, dense_features_in) -> (B, dense_features_out)
#         if self.has_reward_conv_layers:
#             reward_vector = self.reward_conv_layers(S)
#             flattened_reward_vector = reward_vector.flatten(1, -1)
#         else:
#             flattened_reward_vector = hidden_state.flatten(1, -1)

#         if self.has_reward_dense_layers:
#             flattened_reward_vector = self.reward_dense_layers(flattened_reward_vector)

#         if self.config.value_prefix:
#             flattened_reward_vector = flattened_reward_vector.unsqueeze(0)
#             flattened_reward_vector, reward_hidden = self.lstm(
#                 flattened_reward_vector, reward_hidden
#             )
#             flattened_reward_vector = flattened_reward_vector.squeeze(0)
#             # flattened_reward_vector = self.bn_value_prefix(flattened_reward_vector)
#         if self.config.support_range is None:
#             reward = self.reward(flattened_reward_vector)
#         else:
#             # TODO: should this be turned into an expected value and then in the loss function into a two hot?
#             reward = self.reward(flattened_reward_vector).softmax(dim=-1)

#         # (B, dense_features_in) -> (B, dense_features_out)
#         if self.has_to_play_conv_layers:
#             to_play_vector = self.to_play_conv_layers(S)
#             flattened_to_play_vector = to_play_vector.flatten(1, -1)
#         else:
#             flattened_to_play_vector = hidden_state.flatten(1, -1)

#         if self.has_to_play_dense_layers:
#             flattened_to_play_vector = self.to_play_dense_layers(
#                 flattened_to_play_vector
#             )

#         to_play = self.to_play(flattened_to_play_vector).softmax(dim=-1)

#         return reward, hidden_state, to_play, reward_hidden


# class Prediction(nn.Module):
#     def __init__(
#         self,
#         config: MuZeroConfig,
#         output_size: int,
#         input_shape: Tuple[int],
#     ):
#         assert (
#             config.game.is_discrete
#         ), "AlphaZero only works for discrete action space games (board games)"

#         self.config = config

#         super(Prediction, self).__init__()

#         self.has_residual_layers = len(config.residual_layers) > 0
#         self.has_conv_layers = len(config.conv_layers) > 0
#         self.has_dense_layers = len(config.dense_layer_widths) > 0
#         if not (
#             self.has_conv_layers or self.has_dense_layers or self.has_residual_layers
#         ):
#             print("Warning no layers set for prediction network.")

#         current_shape = input_shape
#         B = current_shape[0]

#         # INPUTS = CONV + BATCHNORM + maybe RELU?

#         if self.has_residual_layers:
#             assert (
#                 len(current_shape) == 4
#             ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
#             filters, kernel_sizes, strides = to_lists(config.residual_layers)

#             # (B, C_in, H, W) -> (B, C_out H, W)
#             self.residual_layers = ResidualStack(
#                 input_shape=current_shape,
#                 filters=filters,
#                 kernel_sizes=kernel_sizes,
#                 strides=strides,
#                 activation=self.config.activation,
#                 noisy_sigma=config.noisy_sigma,
#             )
#             current_shape = (
#                 B,
#                 self.residual_layers.output_channels,
#                 current_shape[2],
#                 current_shape[3],
#             )

#         if self.has_conv_layers:
#             assert (
#                 len(current_shape) == 4
#             ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
#             filters, kernel_sizes, strides = to_lists(config.conv_layers)

#             # (B, C_in, H, W) -> (B, C_out H, W)
#             self.conv_layers = Conv2dStack(
#                 input_shape=current_shape,
#                 filters=filters,
#                 kernel_sizes=kernel_sizes,
#                 strides=strides,
#                 activation=self.config.activation,
#                 noisy_sigma=config.noisy_sigma,
#             )
#             current_shape = (
#                 B,
#                 self.conv_layers.output_channels,
#                 current_shape[2],
#                 current_shape[3],
#             )

#         if self.has_dense_layers:
#             if len(current_shape) == 4:
#                 initial_width = current_shape[1] * current_shape[2] * current_shape[3]
#             else:
#                 assert len(current_shape) == 2
#                 initial_width = current_shape[1]

#             # (B, width_in) -> (B, width_out)
#             self.dense_layers = DenseStack(
#                 initial_width=initial_width,
#                 widths=self.config.dense_layer_widths,
#                 activation=self.config.activation,
#                 noisy_sigma=self.config.noisy_sigma,
#             )
#             current_shape = (
#                 B,
#                 self.dense_layers.output_width,
#             )

#         self.critic = CriticNetwork(config, current_shape)
#         self.actor = ActorNetwork(config, current_shape, output_size)

#     def initialize(self, initializer: Callable[[Tensor], None]) -> None:
#         if self.has_residual_layers:
#             self.residual_layers.initialize(initializer)
#         if self.has_conv_layers:
#             self.conv_layers.initialize(initializer)
#         if self.has_dense_layers:
#             self.dense_layers.initialize(initializer)

#         self.actor.initialize(initializer)
#         self.critic.initialize(initializer)

#     def forward(self, inputs: Tensor):
#         if self.has_conv_layers:
#             assert inputs.dim() == 4

#         # (B, *)
#         S = inputs
#         # INPUT CONV LAYERS???

#         # (B, C_in, H, W) -> (B, C_out, H, W)
#         if self.has_residual_layers:
#             S = self.residual_layers(S)

#         # (B, C_in, H, W) -> (B, C_out, H, W)
#         if self.has_conv_layers:
#             S = self.conv_layers(S)

#         # (B, *) -> (B, dense_features_in)

#         # (B, dense_features_in) -> (B, dense_features_out)
#         if self.has_dense_layers:
#             S = S.flatten(1, -1)
#             S = self.dense_layers(S)

#         return self.critic(S), self.actor(S)


# class AfterstateDynamics(nn.Module):
#     def __init__(
#         self,
#         config: MuZeroConfig,
#         input_shape: Tuple[int],
#         num_actions,
#         action_embedding_dim,
#     ):
#         print("afterstate dynamics input shape", input_shape)
#         self.config = config

#         super(AfterstateDynamics, self).__init__()

#         is_continuous = not self.config.game.is_discrete

#         self.action_embedding_dim = action_embedding_dim
#         self.action_encoder = ActionEncoder(
#             action_space_size=num_actions,
#             embedding_dim=self.action_embedding_dim,
#             is_continuous=is_continuous,  # Main dynamics handles discrete integer actions
#             single_action_plane=True,
#         )

#         if len(input_shape) == 4:
#             self.num_channels = input_shape[1]
#             self.fusion = nn.Conv2d(
#                 self.num_channels + self.action_embedding_dim,
#                 self.num_channels,
#                 kernel_size=3,
#                 padding=1,
#                 bias=False,
#             )
#             self.fusion_bn = nn.BatchNorm2d(self.num_channels)
#         else:
#             self.input_size = input_shape[1]
#             self.fusion = nn.Linear(
#                 self.input_size + self.action_embedding_dim, self.input_size, bias=False
#             )
#             self.fusion_bn = nn.BatchNorm1d(self.input_size)
#         # -----------------------------------------------

#         self.has_residual_layers = len(config.dynamics_residual_layers) > 0
#         self.has_conv_layers = len(config.dynamics_conv_layers) > 0
#         self.has_dense_layers = len(config.dynamics_dense_layer_widths) > 0

#         assert (
#             self.has_conv_layers or self.has_dense_layers or self.has_residual_layers
#         ), "At least one of the layers should be present."

#         current_shape = input_shape
#         B = current_shape[0]

#         # INPUTS = CONV + BATCHNORM + maybe RELU? into residual etc

#         if self.has_residual_layers:
#             assert (
#                 len(current_shape) == 4
#             ), "dynamics residual layers expected an input shape should be (B, C, H, W), got {}".format(
#                 current_shape
#             )
#             filters, kernel_sizes, strides = to_lists(config.dynamics_residual_layers)

#             # (B, C_in, H, W) -> (B, C_out H, W)
#             self.residual_layers = ResidualStack(
#                 input_shape=current_shape,
#                 filters=filters,
#                 kernel_sizes=kernel_sizes,
#                 strides=strides,
#                 activation=self.config.activation,
#                 noisy_sigma=config.noisy_sigma,
#             )
#             current_shape = (
#                 B,
#                 self.residual_layers.output_channels,
#                 current_shape[2],
#                 current_shape[3],
#             )

#         if self.has_conv_layers:
#             assert (
#                 len(current_shape) == 4
#             ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
#             filters, kernel_sizes, strides = to_lists(config.dynamics_conv_layers)

#             # (B, C_in, H, W) -> (B, C_out H, W)
#             self.conv_layers = Conv2dStack(
#                 input_shape=current_shape,
#                 filters=filters,
#                 kernel_sizes=kernel_sizes,
#                 strides=strides,
#                 activation=self.config.activation,
#                 noisy_sigma=config.noisy_sigma,
#             )
#             current_shape = (
#                 B,
#                 self.conv_layers.output_channels,
#                 current_shape[2],
#                 current_shape[3],
#             )

#         if self.has_dense_layers:
#             if len(current_shape) == 4:
#                 initial_width = current_shape[1] * current_shape[2] * current_shape[3]
#             else:
#                 assert len(current_shape) == 2
#                 initial_width = current_shape[1]

#             # (B, width_in) -> (B, width_out)
#             # print("initial_width", initial_width)
#             self.dense_layers = DenseStack(
#                 initial_width=initial_width,
#                 widths=self.config.dynamics_dense_layer_widths,
#                 activation=self.config.activation,
#                 noisy_sigma=self.config.noisy_sigma,
#             )
#             current_shape = (
#                 B,
#                 self.dense_layers.output_width,
#             )

#         self.output_shape = current_shape

#     def initialize(self, initializer: Callable[[Tensor], None]) -> None:
#         if self.has_residual_layers:
#             self.residual_layers.initialize(initializer)
#         if self.has_conv_layers:
#             self.conv_layers.initialize(initializer)
#         if self.has_dense_layers:
#             self.dense_layers.initialize(initializer)

#     def forward(self, hidden_state: Tensor, action):
#         if self.has_conv_layers:
#             assert hidden_state.dim() == 4

#         # --- 1. Embed Action & Fuse (Reference Logic) ---
#         # spatial_shape = (
#         #     hidden_state.shape[0],
#         #     hidden_state.shape[2],
#         #     hidden_state.shape[3],
#         # )

#         # Embed action to (B, EmbedDim, H, W)
#         action_embedded = self.action_encoder(action, hidden_state.shape)

#         # Concatenate (B, C+EmbedDim, H, W)
#         x = torch.cat((hidden_state, action_embedded), dim=1)

#         # Reduce channels (B, C, H, W)
#         x = self.fusion(x)
#         # x = self.fusion_bn(x)
#         # TODO: FIX BATCH NORMS IN DYNAMICS
#         # TODO: ADD BATCH NORMS TO DENSE LAYER NETWORKS OF MUZERO

#         # Residual Connection from Input State
#         x = x + hidden_state
#         S = F.relu(x)
#         # ------------------------------------------------

#         # INPUT CONV LAYERS???
#         # input batch norm
#         # relu?

#         # SHOULD I HAVE AN INPUT HERE THAT REDUCES THE CHANNELS BY 1 SO THAT THE NETWORK IS "the same" as the representation?

#         # (B, C_in, H, W) -> (B, C_out, H, W)
#         if self.has_residual_layers:
#             S = self.residual_layers(S)

#         # (B, C_in, H, W) -> (B, C_out, H, W)
#         if self.has_conv_layers:
#             S = self.conv_layers(S)

#         # (B, *) -> (B, dense_features_in)
#         if self.has_dense_layers:
#             # print("S Shape", S.shape)
#             flattened_hidden_state = S.flatten(1, -1)
#             # print("flattened shape", flattened_hidden_state.shape)
#             S = self.dense_layers(flattened_hidden_state)

#             # normalize inputs as per paper
#             min_hidden_state = S.min(1, keepdim=True)[0]
#             max_hidden_state = S.max(1, keepdim=True)[0]
#             scale_encoded_state = max_hidden_state - min_hidden_state
#             scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
#             hidden_state = (S - min_hidden_state) / scale_encoded_state
#         else:
#             # normalize inputs as per paper
#             min_hidden_state = (
#                 S.view(
#                     -1,
#                     S.shape[1],
#                     S.shape[2] * S.shape[3],
#                 )
#                 .min(2, keepdim=True)[0]
#                 .unsqueeze(-1)
#             )
#             max_hidden_state = (
#                 S.view(
#                     -1,
#                     S.shape[1],
#                     S.shape[2] * S.shape[3],
#                 )
#                 .max(2, keepdim=True)[0]
#                 .unsqueeze(-1)
#             )
#             scale_hidden_state = max_hidden_state - min_hidden_state
#             scale_hidden_state[scale_hidden_state < 1e-5] += 1e-5
#             hidden_state = (S - min_hidden_state) / scale_hidden_state

#         return hidden_state


class PredictionHead(nn.Module):
    """
    Combines the final network block output with the Actor and Critic heads.
    This replaces the 'prediction' logic from the old Prediction class forward.
    """

    def __init__(self, config: MuZeroConfig, input_shape: Tuple[int], output_size: int):
        super().__init__()
        self.critic = CriticNetwork(config, input_shape)
        self.actor = ActorNetwork(config, input_shape, output_size)

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        self.actor.initialize(initializer)
        self.critic.initialize(initializer)

    def forward(self, S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.critic(S), self.actor(S)


class Prediction(nn.Module):
    def __init__(self, config: MuZeroConfig, output_size: int, input_shape: Tuple[int]):
        super().__init__()
        assert (
            config.game.is_discrete
        ), "AlphaZero only works for discrete action space games (board games)"
        self.config = config

        self.net = NetworkBlock(
            config, input_shape, ""
        )  # Uses default layers (config.residual_layers etc.)
        self.head = PredictionHead(config, self.net.output_shape, output_size)

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        self.net.initialize(initializer)
        self.head.initialize(initializer)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        S = self.net(inputs)
        return self.head(S)


class Encoder(nn.Module):
    def __init__(
        self,
        input_shape,
        num_codes: int = 32,
        channel_first: bool = True,
    ):
        """
        Args:
            input_shape: tuple, e.g. (C, H, W) or (B, C, H, W).
            num_codes: embedding size output by encoder.
            channel_first: if True, treats input as (..., C, H, W).
        """
        super().__init__()
        self.channel_first = channel_first
        self.num_codes = num_codes

        # --- Image / 4D Path ---
        if len(input_shape) == 4:
            # Heuristic: treating len=4 or len=3 (C,H,W) as image data
            self.use_conv = True

            # Determine input channels based on flag
            # Assuming input_shape excludes Batch dim if len==3, or includes it if len==4
            if len(input_shape) == 4:
                c = input_shape[1] if self.channel_first else input_shape[3]
            else:
                c = input_shape[0] if self.channel_first else input_shape[2]

            self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1)
            self.act = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

            # --- REMOVED POOLING HERE ---
            # previously: self.pool = nn.AdaptiveAvgPool2d(1)

            # We must calculate the flattened size dynamically because
            # we no longer force the output to 1x1.
            with torch.no_grad():
                # Create a dummy input to trace shape
                # We assume input_shape provided matches the user's tensor setup
                if len(input_shape) == 3:
                    dummy_in = torch.zeros(1, *input_shape)
                else:
                    dummy_in = torch.zeros(*input_shape)

                # Check for channel last input and permute for PyTorch convs if necessary
                if not self.channel_first and dummy_in.shape[-1] == c:
                    dummy_in = dummy_in.permute(0, 3, 1, 2)

                d = self.conv1(dummy_in)
                d = self.conv2(d)
                d = self.conv3(d)
                flat_dim = d.flatten(1).shape[1]

            # Linear head: maps (64 * H' * W') -> num_codes
            self.fc = nn.Linear(flat_dim, num_codes)

        # --- Vector / 2D Path ---
        else:
            self.use_conv = False
            # Assuming input_shape is (Batch, Features) or just (Features,)
            self.fc1 = nn.Linear(input_shape[-1], 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, num_codes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            probs: (B, num_codes) - Softmax probabilities
            one_hot_st: (B, num_codes) - Straight-Through gradient flow
        """
        # 1. Processing to Logits
        if self.use_conv:
            x = self.act(self.conv1(x))
            x = self.act(self.conv2(x))
            x = self.act(self.conv3(x))

            # Flatten directly (preserving spatial info in the flat vector)
            x = x.flatten(1, -1)
            x = self.fc(x)  # (B, num_codes)
        else:
            # Vector path returns LOGITS (removed .softmax inside here to avoid double softmax)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)

        # 2. Softmax
        probs = x.softmax(dim=-1)

        # Convert to one-hot (B, num_codes)
        one_hot = torch.zeros_like(probs).scatter_(
            -1, torch.argmax(probs, dim=-1, keepdim=True), 1.0
        )

        # # 4. Straight-Through Estimator
        # # Forward: use one_hot
        # # Backward: use gradients of probs
        one_hot_st = (one_hot - probs).detach() + probs

        # TODO: LIKE LIGHT ZERO NO SOFTMAX?
        # probs = x
        # one_hot_st = OnehotArgmax.apply(probs)
        return probs, one_hot_st


class OnehotArgmax(torch.autograd.Function):
    """
    Overview:
        Custom PyTorch function for one-hot argmax. This function transforms the input tensor \
        into a one-hot tensor where the index with the maximum value in the original tensor is \
        set to 1 and all other indices are set to 0. It allows gradients to flow to the encoder \
        during backpropagation.

        For more information, refer to: \
        https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input):
        """
        Overview:
            Forward method for the one-hot argmax function. This method transforms the input \
            tensor into a one-hot tensor.
        Arguments:
            - ctx (:obj:`context`): A context object that can be used to stash information for
            backward computation.
            - input (:obj:`torch.Tensor`): Input tensor.
        Returns:
            - (:obj:`torch.Tensor`): One-hot tensor.
        """
        # Transform the input tensor to a one-hot tensor
        return torch.zeros_like(input).scatter_(
            -1, torch.argmax(input, dim=-1, keepdim=True), 1.0
        )

    @staticmethod
    def backward(ctx, grad_output):
        """
        Overview:
            Backward method for the one-hot argmax function. This method allows gradients \
            to flow to the encoder during backpropagation.
        Arguments:
            - ctx (:obj:`context`):  A context object that was stashed in the forward pass.
            - grad_output (:obj:`torch.Tensor`): The gradient of the output tensor.
        Returns:
            - (:obj:`torch.Tensor`): The gradient of the input tensor.
        """
        return grad_output


class Network(nn.Module):
    def __init__(
        self,
        config: MuZeroConfig,
        num_actions: int,
        input_shape: Tuple[int],
        channel_first: bool = True,
        world_model_cls=MuzeroWorldModel,
    ):
        super(Network, self).__init__()
        self.config = config
        self.channel_first = channel_first

        self.world_model = world_model_cls(config, input_shape, num_actions)

        hidden_state_shape = self.world_model.representation.output_shape
        print("Hidden state shape:", hidden_state_shape)

        self.prediction = Prediction(config, num_actions, hidden_state_shape)
        if self.config.stochastic:
            self.afterstate_prediction = Prediction(
                config, config.num_chance, hidden_state_shape
            )

        # --- 4. EFFICIENT ZERO Projector ---
        # The flat hidden dimension is simply the total size of the hidden state
        self.flat_hidden_dim = torch.Size(hidden_state_shape[1:]).numel()
        self.projector = Projector(self.flat_hidden_dim, config)

        encoder_input_shape = list(input_shape)
        encoder_input_shape[1] = input_shape[1] * 2
        encoder_input_shape = tuple(encoder_input_shape)
        print("encoder input shape", encoder_input_shape)
        self.encoder = Encoder(
            encoder_input_shape,
            num_codes=self.config.num_chance,
            channel_first=channel_first,
        )

    def initial_inference(self, obs):
        hidden_state = self.world_model.initial_inference(obs)
        value, policy = self.prediction(hidden_state)
        return value, policy, hidden_state

    def recurrent_inference(
        self,
        hidden_state,
        action,
        reward_h_states,
        reward_c_states,
    ):
        reward, next_hidden_state, to_play, reward_hidden = (
            self.world_model.recurrent_inference(
                hidden_state, action, reward_h_states, reward_c_states
            )
        )

        value, policy = self.prediction(next_hidden_state)
        return reward, next_hidden_state, value, policy, to_play, reward_hidden

    def afterstate_recurrent_inference(
        self,
        hidden_state,
        action,
    ):
        afterstate = self.world_model.afterstate_recurrent_inference(
            hidden_state, action
        )
        value, sigma = self.afterstate_prediction(afterstate)
        return afterstate, value, sigma

    def project(self, hidden_state: Tensor, grad=True) -> Tensor:
        """
        Projects the hidden state (s_t) into the embedding space.
        Used for both the 'real' target observation and the 'predicted' latent.
        """
        # Flatten the spatial dimensions (B, C, H, W) -> (B, C*H*W)
        flat_hidden = hidden_state.flatten(1, -1)
        proj = self.projector.projection(flat_hidden)

        # with grad, use proj_head
        if grad:
            proj = self.projector.projection_head(proj)
            return proj
        else:
            return proj.detach()


# class Network(nn.Module):
#     def __init__(
#         self,
#         config: MuZeroConfig,
#         output_size: int,
#         input_shape: Tuple[int],
#         channel_first: bool = True,
#     ):
#         super(Network, self).__init__()
#         self.config = config
#         self.channel_first = channel_first
#         self.num_chance = config.num_chance

#         # --- 1. Representation Network ---
#         self.representation = Representation(config, input_shape)
#         self.num_actions = output_size

#         hidden_state_shape = self.representation.output_shape
#         print("Hidden state shape:", hidden_state_shape)

#         # --- 3. Dynamics and Prediction Networks ---
#         if config.stochastic:
#             self.afterstate_dynamics = AfterstateDynamics(
#                 config,
#                 self.representation.output_shape,
#                 num_actions=self.num_actions,
#                 action_embedding_dim=self.config.action_embedding_dim,
#             )
#             # self.afterstate_prediction = AfterstatePrediction(
#             #     config, config.num_chance, self.representation.output_shape
#             # )
#             self.afterstate_prediction = Prediction(
#                 config, config.num_chance, self.representation.output_shape
#             )

#         if self.config.stochastic:
#             self.dynamics = Dynamics(
#                 config,
#                 self.representation.output_shape,
#                 num_actions=self.num_chance,
#                 action_embedding_dim=self.config.action_embedding_dim,
#             )
#         else:
#             self.dynamics = Dynamics(
#                 config,
#                 self.representation.output_shape,
#                 num_actions=self.num_actions,
#                 action_embedding_dim=self.config.action_embedding_dim,
#             )

#         # Dynamics output must match Representation output shape
#         assert (
#             self.dynamics.output_shape == self.representation.output_shape
#         ), f"{self.dynamics.output_shape} = {self.representation.output_shape}"
#         self.prediction = Prediction(config, output_size, hidden_state_shape)

#         # --- 4. EFFICIENT ZERO Projector ---
#         # The flat hidden dimension is simply the total size of the hidden state
#         self.flat_hidden_dim = torch.Size(hidden_state_shape[1:]).numel()
#         self.projector = Projector(self.flat_hidden_dim, config)

#         encoder_input_shape = list(input_shape)
#         encoder_input_shape[1] = input_shape[1] * 2
#         encoder_input_shape = tuple(encoder_input_shape)
#         print("encoder input shape", encoder_input_shape)
#         self.encoder = Encoder(
#             encoder_input_shape,
#             num_codes=self.config.num_chance,
#             channel_first=channel_first,
#         )

#     def initial_inference(self, x):
#         # print("Initial inference")
#         hidden_state = self.representation(x)
#         value, policy = self.prediction(hidden_state)
#         # print("Hidden state:", hidden_state.shape)
#         return value, policy, hidden_state

#     def recurrent_inference(
#         self,
#         hidden_state,
#         action,
#         reward_h_states,
#         reward_c_states,
#     ):
#         # hidden_state: (B, C, H, W)
#         # action: (B,) Int Tensor

#         # We no longer do the manual concatenation/reshaping here.
#         # The Dynamics class now handles (State, Action) -> Fusion -> NextState internally.
#         if not self.config.stochastic:
#             action = action.view(-1).to(hidden_state.device)
#             # one-hot the action -> (B, num_actions)
#             action = (
#                 F.one_hot(action.long(), num_classes=self.num_actions)
#                 .float()
#                 .to(hidden_state.device)
#             )
#         # print("hidden_state", hidden_state.shape)
#         # print("action", action.shape)

#         reward, next_hidden_state, to_play, reward_hidden = self.dynamics(
#             hidden_state, action, (reward_h_states, reward_c_states)
#         )

#         value, policy = self.prediction(next_hidden_state)
#         return reward, next_hidden_state, value, policy, to_play, reward_hidden

#     def afterstate_recurrent_inference(
#         self,
#         hidden_state,
#         action,
#     ):
#         # The AfterstateDynamics class handles (State, Action) -> Fusion -> Afterstate internally.
#         action = action.view(-1).to(hidden_state.device)
#         # one-hot the action -> (B, num_actions)
#         action = (
#             F.one_hot(action.long(), num_classes=self.num_actions)
#             .float()
#             .to(hidden_state.device)
#         )

#         afterstate = self.afterstate_dynamics(hidden_state, action)
#         value, sigma = self.afterstate_prediction(afterstate)
#         return afterstate, value, sigma

#     def project(self, hidden_state: Tensor, grad=True) -> Tensor:
#         """
#         Projects the hidden state (s_t) into the embedding space.
#         Used for both the 'real' target observation and the 'predicted' latent.
#         """
#         # Flatten the spatial dimensions (B, C, H, W) -> (B, C*H*W)
#         flat_hidden = hidden_state.flatten(1, -1)
#         proj = self.projector.projection(flat_hidden)

#         # with grad, use proj_head
#         if grad:
#             proj = self.projector.projection_head(proj)
#             return proj
#         else:
#             return proj.detach()
