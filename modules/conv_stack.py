from typing import Callable

from torch import nn


class Conv2dStack(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int],
        filters: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        activation: nn.Module = nn.ReLU(),
        kernel_initializer: Callable[[nn.Module], None] | None = None,
        kernel_regularizer: Callable[[nn.Module], None] | None = None,
        noisy_sigma: float = None,
    ):
        super(Conv2dStack, self).__init__()
        self.conv_layers: list[nn.Module] = []
        self.activation = activation

        assert len(input_shape) == 3
        assert len(filters) == len(kernel_sizes) == len(strides)
        assert len(filters) > 0

        self.noisy = noisy_sigma != 0
        if self.noisy:
            raise NotImplementedError("Noisy convolutions not implemented yet")
        else:
            current_input_channels = input_shape[2]

            for i in range(len(filters)):
                layer = nn.Conv2d(
                    in_channels=current_input_channels,
                    out_channels=filters[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding="same",
                )

                if kernel_initializer != None:
                    kernel_initializer(layer)

                if kernel_regularizer != None:
                    kernel_regularizer(layer)

                self.conv_layers.append(layer)

                current_input_channels *= filters[i]

            self._output_len = current_input_channels

    def forward(self, inputs):
        x = inputs
        for layer in self.conv_layers:
            x = self.activation.forward(layer.forward(x))
        return x

    def reset_noise(self):
        assert self.noisy

        for layer in self.conv_layers:
            layer.reset_noise()
        return

    @property
    def output_channels(self):
        return self._output_len
