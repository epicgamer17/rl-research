from typing import Callable

from torch import nn, Tensor

from noisy_dense import NoisyDense


class DenseStack(nn.Module):
    def __init__(
        self,
        initial_width: int,
        widths: list[int],
        activation: nn.Module = nn.ReLU(),
        kernel_initializer: Callable[[Tensor], None] | None = None,
        noisy_sigma: float = 0,
    ):
        super(DenseStack, self).__init__()
        self.dense_layers: list[nn.Linear] = []
        self.activation = activation

        assert len(widths) > 0
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
                current_input_width = widths[i]
        else:
            for i in range(len(widths)):
                layer = nn.Linear(
                    in_features=current_input_width,
                    out_features=widths[i],
                )
                self.dense_layers.append(layer)
                current_input_width = widths[i]

        if kernel_initializer != None:
            for layer in self.dense_layers:
                kernel_initializer(layer.weight)

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
