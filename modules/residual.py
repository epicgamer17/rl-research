from typing import Callable, Tuple

from torch import nn, Tensor
from utils import calculate_padding


class ResidualStack(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int],
        filters: list[int],
        kernel_sizes: list[int | Tuple[int, int]],
        strides: list[int | Tuple[int, int]],
        activation: nn.Module = nn.ReLU(),
        noisy_sigma: float = 0,
    ):
        """A sequence of residual layers with the activation function applied after each layer.
        Always applies the minimum zero-padding that ensures the output shape is equal to the input shape.
        Input shape in "BCHW" form, i.e. (batch_size, input_channels, height, width)
        """
        super(ResidualStack, self).__init__()
        self.residual_layers = nn.ModuleList()

        self.activation = activation

        # [B, C_in, H, W]
        assert (
            len(input_shape) == 4
            and len(filters) == len(kernel_sizes) == len(strides)
            and len(filters) > 0
        )

        self.noisy = noisy_sigma != 0
        if self.noisy:
            print("warning: Noisy convolutions not implemented yet")
            # raise NotImplementedError("")

        current_input_channels = input_shape[1]

        for i in range(len(filters)):
            print(current_input_channels)
            layer = Residual(
                in_channels=current_input_channels,
                out_channels=filters[i],
                kernel_size=kernel_sizes[i],
                stride=strides[i],
            )
            self.residual_layers.append(layer)
            current_input_channels = filters[i]

        self._output_len = current_input_channels

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        def initialize_if_conv(m: nn.Module):
            if isinstance(m, nn.Conv2d):
                initializer(m.weight)

        self.apply(initialize_if_conv)

    def forward(self, inputs):
        x = inputs
        for layer in self.residual_layers:
            x = self.activation(layer(x))
        return x

    def reset_noise(self):
        assert self.noisy

        # noisy not implemented

        # for layer in self.conv_layers:
        #     # layer.reset_noise()
        # return

    def remove_noise(self):
        assert self.noisy

        # noisy not implemented

        # for layer in self.conv_layers:
        #     # layer.reset_noise()
        # return

    @property
    def output_channels(self):
        return self._output_len


class Residual(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
    ):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            bias=False,
        )

        # REGULARIZATION?
        self.bn1 = nn.BatchNorm2d(
            num_features=out_channels,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
            bias=False,
        )

        # REGULARIZATION?
        self.bn2 = nn.BatchNorm2d(
            num_features=out_channels,
        )

        self.relu = nn.ReLU()
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        def initialize_if_conv(m: nn.Module):
            if isinstance(m, nn.Conv2d):
                initializer(m.weight)

        self.apply(initialize_if_conv)

    def forward(self, inputs):
        residual = self.downsample(inputs) if self.downsample else inputs

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + residual)
        return x
