import torch
from torch import nn

from agent_configs import RainbowConfig
from modules.conv_stack import Conv2dStack
from modules.dense_stack import DenseStack
from modules.noisy_dense import NoisyDense
from utils.utils import to_lists


class RainbowNetwork(nn.Module):
    def __init__(
        self, config: RainbowConfig, output_size: int, input_shape, *args, **kwargs
    ):
        super().__init__()
        B = current_shape[0]
        self.config = config

        self.has_conv_layers = len(config.conv_layers) > 0
        self.has_dense_layers = len(config.dense_layers_widths) > 0
        self.has_value_hidden_layers = len(config.value_hidden_layers_widths) > 0
        self.has_advantage_hidden_layers = (
            len(config.advantage_hidden_layers_widths) > 0
        )

        current_shape = input_shape
        if self.has_conv_layers:
            assert len(input_shape) == 4
            filters, kernel_sizes, strides = to_lists(self.conv_layers)

            # (B, H, W, C_in) -> (B, H, W, C_out)
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
                current_shape[1],
                current_shape[2],
                self.conv_layers.output_channels,
            )

        if self.has_dense_layers:
            if len(current_shape == 4):
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert len(current_shape) == 2
                initial_width = current_shape[1]

            # (B, width_in) -> (B, width_out)
            self.dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.dense_layers_widths,
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

        if self.has_value_hidden_layers:
            # (B, width_in) -> (B, value_width_out) -> (B, atom_size)
            self.value_hidden_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.value_hidden_layers_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )
            value_output_width = self.value_hidden_layers.output_width
            if self.config.noisy_sigma != 0:
                self.value = NoisyDense(
                    in_features=value_output_width,
                    out_features=config.atom_size,
                    sigma=config.noisy_sigma,
                )
            else:
                self.value = nn.Linear(
                    in_features=value_output_width,
                    out_features=config.atom_size,
                )
        else:
            if self.config.noisy_sigma != 0:
                self.value = NoisyDense(
                    in_features=self.dense_layers.output_width,
                    out_features=config.atom_size,
                    sigma=config.noisy_sigma,
                )
            else:
                self.value = nn.Linear(
                    in_features=self.dense_layers.output_width,
                    out_features=config.atom_size,
                )

        if self.has_advantage_hidden_layers:
            # (B, width_in) -> (B, advantage_width_out) -> (B, atom_size * output_size)
            self.advantage_hidden_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.advantage_hidden_layers_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )
            advantage_output_width = self.advantage_hidden_layers.output_width
            if self.config.noisy_sigma != 0:
                self.advantage = NoisyDense(
                    in_features=advantage_output_width,
                    out_features=config.atom_size * output_size,
                    sigma=config.noisy_sigma,
                )
            else:
                self.advantage = nn.Linear(
                    in_features=advantage_output_width,
                    out_features=config.atom_size * output_size,
                )
        else:
            if self.config.noisy_sigma != 0:
                self.advantage = NoisyDense(
                    in_features=self.dense_layers.output_width,
                    out_features=config.atom_size * output_size,
                    sigma=config.noisy_sigma,
                )
            else:
                self.advantage = nn.Linear(
                    in_features=self.dense_layers.output_width,
                    out_features=config.atom_size * output_size,
                )

        self.softmax = nn.Softmax(1)

        # self.outputs = tf.keras.layers.Lambda(
        #     lambda q: tf.reduce_sum(q * config.support, axis=2), name="Q"
        # )
        # ??? config.support not found anywhere

        if config.kernel_initializer != None:
            self.apply(config.kernel_initializer)

    def call(self, inputs: torch.Tensor, training=False):
        assert inputs.dim() == 4
        x = inputs
        if self.has_conv_layers:
            x = self.conv_layers(x)
        x = self.flatten(x)
        if self.has_dense_layers:
            x = self.dense_layers(x)

        if self.has_value_hidden_layers:
            x = self.value_hidden_layers(x)
        value = self.value(x)

        if self.has_advantage_hidden_layers:
            x = self.advantage_hidden_layers(x)
        advantage: torch.Tensor = self.advantage(x)
        advantage = advantage.mean(2)
        q = value + advantage
        q: torch.Tensor = self.softmax(q)

        # ONLY CLIP FOR CATEGORICAL CROSS ENTROPY LOSS TO PREVENT NAN
        # MIGHT BE ABLE TO REMOVE CLIPPING ENTIRELY SINCE I DONT THINK THE TENSORFLOW LOSSES CAN RETURN NaN
        # q.clip(1e-3, 1)
        # q = self.outputs(q)

        # print(q.shape)
        return q

    def reset_noise(self):
        if self.config.noisy_sigma != 0:
            if self.has_conv_layers:
                self.conv_layers.reset_noise()
            if self.has_dense_layers:
                self.dense_layers.reset_noise()
            if self.has_value_hidden_layers:
                self.value_hidden_layers.reset_noise()
            if self.has_advantage_hidden_layers:
                self.advantage_hidden_layers.reset_noise()
            self.value.reset_noise()
            self.advantage.reset_noise()
