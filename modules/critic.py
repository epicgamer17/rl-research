from typing import Callable, Tuple
from torch import Tensor

from modules.conv import Conv2dStack
from modules.dense import DenseStack, build_dense
from modules.utils import zero_weights_initializer
from packages.agent_configs.agent_configs.base_config import Config
from packages.utils.utils.utils import to_lists

from torch import nn


class CriticNetwork(nn.Module):
    def __init__(self, config: Config, input_shape: Tuple[int], *args, **kwargs):
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
                norm_type=self.config.norm_type,
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
                norm_type=self.config.norm_type,
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

        if config.support_range is not None:
            self.full_support_size = 2 * config.support_range + 1
            self.value = build_dense(
                in_features=initial_width,
                out_features=self.full_support_size,
                sigma=0,
            )
        else:
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
        # Initialize categorical layers with zero weights
        # Value (if support_range is not None, it's a probability distribution)
        if self.config.support_range is not None:
            self.value.apply(zero_weights_initializer)
        else:
            self.value.initialize(
                initializer
            )  # Standard initialization for scalar output

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
        if self.config.support_range is None:
            return value
        else:
            # TODO: should this be turned into an expected value and then in the loss function into a two hot?
            return value.softmax(dim=-1)

    def reset_noise(self):
        if self.has_conv_layers:
            self.conv_layers.reset_noise()
        if self.has_dense_layers:
            self.dense_layers.reset_noise()
        self.value.reset_noise()
