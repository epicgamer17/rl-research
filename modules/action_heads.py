from collections.abc import Callable
from typing import Tuple
from torch import Tensor
from modules.dense import build_dense
from torch import nn


class DiscreteActionHead(nn.Module):
    def __init__(self, config, input_width: int, output_size: int):
        super().__init__()
        self.output_layer = build_dense(
            in_features=input_width,
            out_features=output_size,
            sigma=config.noisy_sigma,
        )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if hasattr(self.output_layer, "initialize"):
            self.output_layer.initialize(initializer)
        else:
            self.output_layer.apply(initializer)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_layer(x).softmax(dim=-1)


class ContinuousActionHead(nn.Module):
    def __init__(self, config, input_width: int, output_size: int):
        super().__init__()
        self.mean = build_dense(input_width, output_size, sigma=config.noisy_sigma)
        self.std = build_dense(input_width, output_size, sigma=config.noisy_sigma)

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        self.mean.initialize(initializer)
        self.std.initialize(initializer)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        mean = self.mean(x).tanh()
        std = self.std(x).softplus()
        return mean, std

    def reset_noise(self):
        self.mean.reset_noise()
        self.std.reset_noise()
