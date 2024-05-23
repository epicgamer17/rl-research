from typing import Callable

from torch import nn

from noisy_dense import NoisyDense


class DenseStack(nn.Module):
    def __init__(
        self,
        initial_width: int,
        widths: list[int],
        activation: nn.Module = nn.ReLU(),
        kernel_initializer: Callable[[nn.Module], None] | None = None,
        noisy_sigma: float = None,
    ):
        super(DenseStack, self).__init__()
        self.dense_layers: list[nn.Module] = []
        self.activation = activation

        assert len(widths) > 0
        self.dense_layers = []
        self.noisy = noisy_sigma != 0

        current_input_width = initial_width
        if self.noisy:
            for i in range(len(widths)):
                layer = NoisyDense(
                    in_features=current_input_width,
                    out_features=widths[i],
                    sigma=noisy_sigma,
                )
                self.dense_layers.append(layer)
                current_input_width *= widths[i]
        else:
            for i in range(len(widths)):
                layer = nn.Linear(
                    in_features=current_input_width,
                    out_features=widths[i],
                )
                self.dense_layers.append(layer)
                current_input_width *= widths[i]

        if kernel_initializer != None:
            self.apply(kernel_initializer)

        self._output_len = current_input_width

    def forward(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = self.activation(layer(x))
        return x

    def reset_noise(self):
        assert self.noisy

        for layer in self.dense_layers:
            layer.reset_noise()
        return

    @property
    def output_width(self):
        return self._output_len
