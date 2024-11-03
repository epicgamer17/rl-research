from typing import Callable, Tuple
from agent_configs.alphazero_config import AlphaZeroConfig
from torch import nn, Tensor
from utils.utils import to_lists

from modules.conv import Conv2dStack
from modules.dense import DenseStack, build_dense
from modules.residual import ResidualStack


class Network(nn.Module):
    def __init__(
        self,
        config: AlphaZeroConfig,
        output_size: int,
        input_shape: Tuple[int],
    ):
        assert (
            config.game.is_discrete
        ), "AlphaZero only works for discrete action space games (board games)"

        self.config = config

        super(Network, self).__init__()

        self.has_residual_layers = len(config.residual_layers) > 0
        self.has_conv_layers = len(config.conv_layers) > 0
        self.has_dense_layers = len(config.dense_layer_widths) > 0
        assert (
            self.has_conv_layers or self.has_dense_layers or self.has_residual_layers
        ), "At least one of the layers should be present."

        self.output_size = output_size

        current_shape = input_shape
        B = current_shape[0]

        # INPUTS = CONV + BATCHNORM + maybe RELU?

        if self.has_residual_layers:
            assert (
                len(input_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(input_shape)
            filters, kernel_sizes, strides = to_lists(config.residual_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.residual_layers = ResidualStack(
                input_shape=input_shape,
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
                len(input_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(input_shape)
            filters, kernel_sizes, strides = to_lists(config.conv_layers)

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
    def __init__(
        self, config: AlphaZeroConfig, input_shape: Tuple[int], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.has_conv_layers = len(config.critic_conv_layers) > 0
        self.has_dense_layers = len(config.critic_dense_layer_widths) > 0

        current_shape = input_shape
        B = current_shape[0]
        if self.has_conv_layers:
            # WITH BATCHNORM FOR EVERY CONV LAYER
            assert len(input_shape) == 4
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
            sigma=config.noisy_sigma,
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
        if self.has_dense_layers:
            x = x.flatten(1, -1)  # should this be batch, -1?
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
        config: AlphaZeroConfig,
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
            assert len(input_shape) == 4
            filters, kernel_sizes, strides = to_lists(config.actor_conv_layers)

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
        if self.has_dense_layers:
            x = x.flatten(1, -1)  # should this be batch, -1?
            x = self.dense_layers(x)
        actions = self.actions(x)
        return actions.softmax(dim=-1)

    def reset_noise(self):
        if self.has_conv_layers:
            self.conv_layers.reset_noise()
        if self.has_dense_layers:
            self.dense_layers.reset_noise()
        self.actions.reset_noise()
