from typing import Callable, Tuple
from agent_configs.ppo_config import PPOConfig
from torch import Tensor
from utils.utils import to_lists

from modules.conv import Conv2dStack
from modules.dense import DenseStack, build_dense
import torch.nn as nn


class Network(nn.Module):
    def __init__(
        self, config: PPOConfig, input_shape: Tuple[int], output_size: int, discrete
    ):
        if discrete:
            assert output_size > 0

        print(input_shape)

        super(Network, self).__init__()
        self.critic = CriticNetwork(config, input_shape)
        self.actor = ActorNetwork(config, input_shape, output_size, discrete)

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        self.actor.initialize(initializer)
        self.critic.initialize(initializer)

    def forward(self, inputs: Tensor):
        return self.actor(inputs), self.critic(inputs)


class CriticNetwork(nn.Module):
    def __init__(self, config: PPOConfig, input_shape: Tuple[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.has_conv_layers = len(config.critic_conv_layers) > 0
        self.has_dense_layers = len(config.critic_dense_layer_widths) > 0

        current_shape = input_shape
        B = current_shape[0]
        if self.has_conv_layers:
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
            assert inputs.dim() == 4

        x = inputs
        if self.has_conv_layers:
            x = self.conv_layers(x)
        if self.has_dense_layers:
            x = self.dense_layers(x)
        value = self.value(x)
        return value

    def reset_noise(self):
        if self.has_conv_layers:
            self.conv_layers.reset_noise()
        if self.has_dense_layers:
            self.dense_layers.reset_noise()
        self.value.reset_noise()


class ActorNetwork(nn.Module):
    def __init__(
        self,
        config: PPOConfig,
        input_shape: Tuple[int],
        output_size: int,
        discrete: bool,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.has_conv_layers = len(config.actor_conv_layers) > 0
        self.has_dense_layers = len(config.actor_dense_layer_widths) > 0
        self.discrete = discrete

        current_shape = input_shape
        B = current_shape[0]
        if self.has_conv_layers:
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
                assert len(current_shape) == 2
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

        if self.discrete:
            self.actions = build_dense(
                in_features=initial_width,
                out_features=output_size,
                sigma=self.config.noisy_sigma,
            )
        else:
            self.mean = build_dense(
                in_features=initial_width,
                out_features=output_size,
                sigma=self.config.noisy_sigma,
            )

            self.std = build_dense(
                in_features=initial_width,
                out_features=output_size,
                sigma=self.config.noisy_sigma,
            )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)
        if self.discrete:
            self.actions.initialize(
                initializer
            )  # OUTPUT LAYER TO IMPLIMENT INTIALIZING WITH CONSTANT OF 0.01
        else:
            self.mean.initialize(initializer)  # OUTPUT LAYER
            self.std.initialize(initializer)  # OUTPUT LAYER

    def forward(self, inputs: Tensor):
        if self.has_conv_layers:
            assert inputs.dim() == 4

        x = inputs
        if self.has_conv_layers:
            x = self.conv_layers(x)
        if self.has_dense_layers:
            x = self.dense_layers(x)
        if self.discrete:
            actions = self.actions(x)
            return actions.softmax(dim=-1)
        else:
            mean = self.mean(x).tanh(dim=-1)
            std = self.std(x).softplus(dim=-1)
            return mean, std

    def reset_noise(self):
        if self.has_conv_layers:
            self.conv_layers.reset_noise()
        if self.has_dense_layers:
            self.dense_layers.reset_noise()
        if self.discrete:
            self.actions.reset_noise()
        else:
            self.mean.reset_noise()
            self.std.reset_noise()
