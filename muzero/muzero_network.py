from typing import Callable, Tuple
from agent_configs.muzero_config import MuZeroConfig
from torch import nn, Tensor
from utils.utils import to_lists

from modules.conv import Conv2dStack
from modules.dense import DenseStack, build_dense
from modules.residual import ResidualStack
import torch


class Representation(nn.Module):
    def __init__(
        self,
        config: MuZeroConfig,
        input_shape: Tuple[int],
    ):
        assert (
            config.game.is_discrete
        ), "AlphaZero only works for discrete action space games (board games)"

        self.config = config

        super(Representation, self).__init__()

        self.has_residual_layers = len(config.representation_residual_layers) > 0
        self.has_conv_layers = len(config.representation_conv_layers) > 0
        self.has_dense_layers = len(config.representation_dense_layer_widths) > 0
        assert (
            self.has_conv_layers or self.has_dense_layers or self.has_residual_layers
        ), "At least one of the layers should be present."

        current_shape = input_shape
        B = current_shape[0]

        # INPUTS = CONV + BATCHNORM + maybe RELU? into residual etc

        if self.has_residual_layers:
            assert (
                len(current_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
            filters, kernel_sizes, strides = to_lists(
                config.representation_residual_layers
            )

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.residual_layers = ResidualStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.residual_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_conv_layers:
            assert (
                len(current_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
            filters, kernel_sizes, strides = to_lists(config.representation_conv_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.conv_layers = Conv2dStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert len(current_shape) == 2
                initial_width = current_shape[1]

            # (B, width_in) -> (B, width_out)
            self.dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.representation_dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )
            current_shape = (
                B,
                self.dense_layers.output_width,
            )
        self.output_shape = current_shape

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if self.has_residual_layers:
            self.residual_layers.initialize(initializer)
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)

    def forward(self, inputs: Tensor):
        if self.has_conv_layers:
            assert inputs.dim() == 4

        # (B, *)
        S = inputs
        # INPUT CONV LAYERS???
        # input batch norm
        # relu?

        # (B, C_in, H, W) -> (B, C_out, H, W)
        if self.has_residual_layers:
            S = self.residual_layers(S)

        # (B, C_in, H, W) -> (B, C_out, H, W)
        if self.has_conv_layers:
            S = self.conv_layers(S)

        # (B, *) -> (B, dense_features_in)

        # (B, dense_features_in) -> (B, dense_features_out)
        if self.has_dense_layers:
            S = S.flatten(1, -1)
            S = self.dense_layers(S)

            # normalize inputs as per paper
            min_hidden_state = S.min(1, keepdim=True)[0]
            max_hidden_state = S.max(1, keepdim=True)[0]
            scale_hidden_state = max_hidden_state - min_hidden_state
            scale_hidden_state[scale_hidden_state < 1e-5] += 1e-5
            hidden_state = (S - min_hidden_state) / scale_hidden_state
        else:
            # normalize inputs as per paper
            min_hidden_state = (
                S.view(
                    -1,
                    S.shape[1],
                    S.shape[2] * S.shape[3],
                )
                .min(2, keepdim=True)[0]
                .unsqueeze(-1)
            )
            max_hidden_state = (
                S.view(
                    -1,
                    S.shape[1],
                    S.shape[2] * S.shape[3],
                )
                .max(2, keepdim=True)[0]
                .unsqueeze(-1)
            )
            scale_hidden_state = max_hidden_state - min_hidden_state
            scale_hidden_state[scale_hidden_state < 1e-5] += 1e-5
            hidden_state = (S - min_hidden_state) / scale_hidden_state

        return hidden_state


class Dynamics(nn.Module):
    def __init__(
        self,
        config: MuZeroConfig,
        input_shape: Tuple[int],
    ):
        assert (
            config.game.is_discrete
        ), "AlphaZero only works for discrete action space games (board games)"

        self.config = config

        super(Dynamics, self).__init__()

        self.has_residual_layers = len(config.dynamics_residual_layers) > 0
        self.has_conv_layers = len(config.dynamics_conv_layers) > 0
        self.has_dense_layers = len(config.dynamics_dense_layer_widths) > 0
        assert (
            self.has_conv_layers or self.has_dense_layers or self.has_residual_layers
        ), "At least one of the layers should be present."

        current_shape = input_shape
        B = current_shape[0]

        # INPUTS = CONV + BATCHNORM + maybe RELU? into residual etc

        if self.has_residual_layers:
            assert (
                len(current_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
            filters, kernel_sizes, strides = to_lists(config.dynamics_residual_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.residual_layers = ResidualStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.residual_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_conv_layers:
            assert (
                len(current_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
            filters, kernel_sizes, strides = to_lists(config.dynamics_conv_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.conv_layers = Conv2dStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert len(current_shape) == 2
                initial_width = current_shape[1]

            # (B, width_in) -> (B, width_out)
            self.dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.dynamics_dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )
            current_shape = (
                B,
                self.dense_layers.output_width,
            )

        self.output_shape = current_shape

        self.has_reward_conv_layers = len(config.reward_conv_layers) > 0
        self.has_reward_dense_layers = len(config.reward_dense_layer_widths) > 0

        if self.has_reward_conv_layers:
            assert (
                len(current_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
            filters, kernel_sizes, strides = to_lists(config.reward_conv_layers)

            self.reward_conv_layers = Conv2dStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )

            current_shape = (
                B,
                self.reward_conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_reward_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert len(current_shape) == 2
                initial_width = current_shape[1]

            self.reward_dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.reward_dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )

            current_shape = (
                B,
                self.reward_dense_layers.output_width,
            )

        if len(current_shape) == 4:
            initial_width = current_shape[1] * current_shape[2] * current_shape[3]
        else:
            assert len(current_shape) == 2
            initial_width = current_shape[1]

        self.reward = build_dense(
            in_features=initial_width,
            out_features=1,
            sigma=0,
        )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if self.has_residual_layers:
            self.residual_layers.initialize(initializer)
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)
        if self.has_reward_conv_layers:
            self.reward_conv_layers.initialize(initializer)
        if self.has_reward_dense_layers:
            self.reward_dense_layers.initialize(initializer)
        self.reward.initialize(initializer)

    def forward(self, inputs: Tensor):
        if self.has_conv_layers:
            assert inputs.dim() == 4

        # (B, *)
        S = inputs
        # INPUT CONV LAYERS???
        # input batch norm
        # relu?

        # SHOULD I HAVE AN INPUT HERE THAT REDUCES THE CHANNELS BY 1 SO THAT THE NETWORK IS "the same" as the representation?

        # (B, C_in, H, W) -> (B, C_out, H, W)
        if self.has_residual_layers:
            S = self.residual_layers(S)

        # (B, C_in, H, W) -> (B, C_out, H, W)
        if self.has_conv_layers:
            S = self.conv_layers(S)

        # (B, *) -> (B, dense_features_in)

        # (B, dense_features_in) -> (B, dense_features_out)
        if self.has_reward_conv_layers:
            reward_vector = self.reward_conv_layers(S)
            flattened_reward_vector = reward_vector.flatten(1, -1)
        else:
            flattened_reward_vector = S.flatten(1, -1)

        if self.has_reward_dense_layers:
            flattened_reward_vector = self.reward_dense_layers(flattened_reward_vector)

        reward = self.reward(flattened_reward_vector)

        if self.has_dense_layers:
            flattened_hidden_state = S.flatten(1, -1)
            S = self.dense_layers(flattened_hidden_state)

            # normalize inputs as per paper
            min_hidden_state = S.min(1, keepdim=True)[0]
            max_hidden_state = S.max(1, keepdim=True)[0]
            scale_encoded_state = max_hidden_state - min_hidden_state
            scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
            hidden_state = (S - min_hidden_state) / scale_encoded_state
        else:
            # normalize inputs as per paper
            min_hidden_state = (
                S.view(
                    -1,
                    S.shape[1],
                    S.shape[2] * S.shape[3],
                )
                .min(2, keepdim=True)[0]
                .unsqueeze(-1)
            )
            max_hidden_state = (
                S.view(
                    -1,
                    S.shape[1],
                    S.shape[2] * S.shape[3],
                )
                .max(2, keepdim=True)[0]
                .unsqueeze(-1)
            )
            scale_hidden_state = max_hidden_state - min_hidden_state
            scale_hidden_state[scale_hidden_state < 1e-5] += 1e-5
            hidden_state = (S - min_hidden_state) / scale_hidden_state

        return reward, hidden_state


class Prediction(nn.Module):
    def __init__(
        self,
        config: MuZeroConfig,
        output_size: int,
        input_shape: Tuple[int],
    ):
        assert (
            config.game.is_discrete
        ), "AlphaZero only works for discrete action space games (board games)"

        self.config = config

        super(Prediction, self).__init__()

        self.has_residual_layers = len(config.residual_layers) > 0
        self.has_conv_layers = len(config.conv_layers) > 0
        self.has_dense_layers = len(config.dense_layer_widths) > 0
        assert (
            self.has_conv_layers or self.has_dense_layers or self.has_residual_layers
        ), "At least one of the layers should be present."

        current_shape = input_shape
        B = current_shape[0]

        # INPUTS = CONV + BATCHNORM + maybe RELU?

        if self.has_residual_layers:
            assert (
                len(current_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
            filters, kernel_sizes, strides = to_lists(config.residual_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.residual_layers = ResidualStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.residual_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_conv_layers:
            assert (
                len(current_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
            filters, kernel_sizes, strides = to_lists(config.conv_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.conv_layers = Conv2dStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert len(current_shape) == 2
                initial_width = current_shape[1]

            # (B, width_in) -> (B, width_out)
            self.dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )
            current_shape = (
                B,
                self.dense_layers.output_width,
            )

        self.critic = CriticNetwork(config, current_shape)
        self.actor = ActorNetwork(config, current_shape, output_size)

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if self.has_residual_layers:
            self.residual_layers.initialize(initializer)
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)

        self.actor.initialize(initializer)
        self.critic.initialize(initializer)

    def forward(self, inputs: Tensor):
        if self.has_conv_layers:
            assert inputs.dim() == 4

        # (B, *)
        S = inputs
        # INPUT CONV LAYERS???

        # (B, C_in, H, W) -> (B, C_out, H, W)
        if self.has_residual_layers:
            S = self.residual_layers(S)

        # (B, C_in, H, W) -> (B, C_out, H, W)
        if self.has_conv_layers:
            S = self.conv_layers(S)

        # (B, *) -> (B, dense_features_in)

        # (B, dense_features_in) -> (B, dense_features_out)
        if self.has_dense_layers:
            S = S.flatten(1, -1)
            S = self.dense_layers(S)

        return self.critic(S), self.actor(S)


class CriticNetwork(nn.Module):
    def __init__(self, config: MuZeroConfig, input_shape: Tuple[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.has_conv_layers = len(config.critic_conv_layers) > 0
        self.has_dense_layers = len(config.critic_dense_layer_widths) > 0

        current_shape = input_shape
        B = current_shape[0]
        if self.has_conv_layers:
            # WITH BATCHNORM FOR EVERY CONV LAYER
            assert len(current_shape) == 4
            filters, kernel_sizes, strides = to_lists(config.critic_conv_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.conv_layers = Conv2dStack(
                input_shape=input_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert len(current_shape) == 2
                initial_width = current_shape[1]

            # (B, width_in) -> (B, width_out)
            self.dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.critic_dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )
            current_shape = (
                B,
                self.dense_layers.output_width,
            )

        if len(current_shape) == 4:
            initial_width = current_shape[1] * current_shape[2] * current_shape[3]
        else:
            assert len(current_shape) == 2
            initial_width = current_shape[1]

        self.value = build_dense(
            in_features=initial_width,
            out_features=1,
            sigma=0,
        )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)
        self.value.initialize(initializer)  # OUTPUT LAYER

    def forward(self, inputs: Tensor):
        if self.has_conv_layers:
            assert (
                inputs.dim() == 4
            ), "Input shape should be (B, C, H, W), got {}".format(inputs.shape)

        x = inputs
        if self.has_conv_layers:
            x = self.conv_layers(x)
        x = x.flatten(1, -1)
        if self.has_dense_layers:
            x = self.dense_layers(x)
        value = self.value(x)
        return value.tanh()

    def reset_noise(self):
        if self.has_conv_layers:
            self.conv_layers.reset_noise()
        if self.has_dense_layers:
            self.dense_layers.reset_noise()
        self.value.reset_noise()


class ActorNetwork(nn.Module):
    def __init__(
        self,
        config: MuZeroConfig,
        input_shape: Tuple[int],
        output_size: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.has_conv_layers = len(config.actor_conv_layers) > 0
        self.has_dense_layers = len(config.actor_dense_layer_widths) > 0

        current_shape = input_shape
        B = current_shape[0]
        if self.has_conv_layers:
            # WITH BATCHNORM FOR EVERY CONV LAYER
            assert len(current_shape) == 4
            filters, kernel_sizes, strides = to_lists(config.actor_conv_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.conv_layers = Conv2dStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert (
                    len(current_shape) == 2
                ), "Input shape should be (B, width), got {}".format(current_shape)
                initial_width = current_shape[1]

            # (B, width_in) -> (B, width_out)
            self.dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.actor_dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )

            current_shape = (
                B,
                self.dense_layers.output_width,
            )

        if len(current_shape) == 4:
            initial_width = current_shape[1] * current_shape[2] * current_shape[3]
        else:
            assert len(current_shape) == 2
            initial_width = current_shape[1]

        self.actions = build_dense(
            in_features=initial_width,
            out_features=output_size,
            sigma=self.config.noisy_sigma,
        )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)
        self.actions.initialize(
            initializer
        )  # OUTPUT LAYER TO IMPLIMENT INTIALIZING WITH CONSTANT OF 0.01

    def forward(self, inputs: Tensor):
        if self.has_conv_layers:
            assert (
                inputs.dim() == 4
            ), "Input shape should be (B, C, H, W), got {}".format(inputs.shape)

        x = inputs
        if self.has_conv_layers:
            x = self.conv_layers(x)

        x = x.flatten(1, -1)
        if self.has_dense_layers:
            x = self.dense_layers(x)
        actions = self.actions(x)
        return actions.softmax(dim=-1)

    def reset_noise(self):
        if self.has_conv_layers:
            self.conv_layers.reset_noise()
        if self.has_dense_layers:
            self.dense_layers.reset_noise()
        self.actions.reset_noise()


class Network(nn.Module):
    def __init__(
        self,
        config: MuZeroConfig,
        output_size: int,
        input_shape: Tuple[int],
        action_function: Callable,
    ):
        super(Network, self).__init__()
        self.config = config
        self.representation = Representation(
            config,
            input_shape,
        )

        # Board planes (116, 8, 8) + action planes (8, 8, 8)
        # observation vector + 1-hot action vector, shape = (4,) + (2,)
        self.action_function = action_function
        print("Hidden state shape:", self.representation.output_shape)
        print("Action function output shape:", self.action_function(0).shape)
        dynamics_input_shape = (
            torch.Size([self.representation.output_shape[0]])
            + torch.concat(
                [
                    torch.zeros(self.representation.output_shape[1:]),
                    self.action_function(0),
                ],
            ).shape
        )
        print(dynamics_input_shape)
        self.dynamics = Dynamics(config, dynamics_input_shape)
        assert self.dynamics.output_shape == self.representation.output_shape
        self.prediction = Prediction(
            config, output_size, self.representation.output_shape
        )

    def initial_inference(self, x):
        hidden_state = self.representation(x)
        value, policy = self.prediction(hidden_state)
        # print("Hidden state:", hidden_state)
        return value, policy, hidden_state

    def recurrent_inference(self, hidden_state, action):
        if len(hidden_state.shape) > len(self.action_function(action).shape):
            assert hidden_state.shape[0] == 1, "does not work with batches"
            hidden_state = hidden_state.squeeze(0)
        # print("hidden state shape:", hidden_state.shape)
        # print("action shape:", self.action_function(action).shape)
        nn_input = torch.concat((hidden_state, self.action_function(action)))
        nn_input = nn_input.unsqueeze(0)
        reward, hidden_state = self.dynamics(nn_input)
        value, policy = self.prediction(hidden_state)
        # print("Hidden state:", hidden_state)
        return reward, hidden_state, value, policy
