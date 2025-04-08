from typing import Callable, Tuple

from torch import Tensor

from modules.conv import Conv2dStack
from modules.dense import DenseStack
from modules.residual import ResidualStack
from packages.utils.utils.utils import to_lists


class InputHead(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_size: int,
        config,
    ):
        super().__init__()

        self.config = config

        self.has_residual_layers = len(config.residual_layers) > 0
        self.has_conv_layers = len(config.conv_layers) > 0
        self.has_dense_layers = len(config.dense_layer_widths) > 0

        self.output_size = output_size

        current_shape = input_shape
        B = current_shape[0]

        if self.has_residual_layers:
            assert (
                len(input_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(input_shape)
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
                len(input_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(input_shape)
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

        assert (
            self.output_size == current_shape[1]
        ), "Output size {} does not match the final layer size {}".format(
            self.output_size, current_shape[1]
        )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if self.has_residual_layers:
            self.residual_layers.initialize(initializer)
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)

    def forward(self, x: Tensor) -> Tensor:
        if self.has_residual_layers:
            x = self.residual_layers(x)
        if self.has_conv_layers:
            x = self.conv_layers(x)

        x = x.flatten(1, -1)

        if self.has_dense_layers:
            x = self.dense_layers(x)
        return x
