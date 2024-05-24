from typing import Callable

import torch
from torch import nn, Tensor, functional


class Dense(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super(Dense, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        initializer(self.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        self(inputs)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class NoisyDense(nn.Module):
    """See https://arxiv.org/pdf/1706.10295."""

    @staticmethod
    def f(x: Tensor):
        return x.sgn() * x.abs().sqrt()

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        initial_sigma: float = 0.5,
        use_factorized: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initial_sigma = initial_sigma
        self.use_factorized = use_factorized
        self.bias = bias

        self.mu_w = nn.Parameter(
            torch.empty(out_features, in_features), **factory_kwargs
        )
        self.sigma_w = nn.Parameter(
            torch.empty(out_features, in_features), **factory_kwargs
        )
        self.eps_w = self.register_buffer(
            "eps_w", torch.empty(out_features, in_features)
        )
        if self.bias:
            self.mu_b = nn.Parameter(torch.empty(out_features))
            self.sigma_b = nn.Parameter(torch.empty(out_features))
            self.eps_b = self.register_buffer("eps_b", torch.empty(out_features))
        else:
            self.register_parameter("mu_b", None)
            self.register_parameter("sigma_b", None)
            self.eps_b = self.register_buffer("eps_b", None)

        self.reset_parameters()
        self.reset_noise()

    def reset_noise(self) -> None:
        if self.use_factorized:
            eps_i = torch.randn(1, self.in_features)
            eps_j = torch.randn(self.out_features, 1)
            self.eps_w = self.f(eps_j) @ self.f(eps_i)
            self.eps_b = eps_j.reshape(self.out_features)
        else:
            self.eps_w = torch.randn(self.mu_w.shape)
            if self.bias:
                self.eps_b = torch.randn(size=self.mu_b.shape)

    def remove_noise(self) -> None:
        self.eps_w = torch.zeros_like(self.mu_w)
        if self.bias:
            self.eps_b = torch.zeros_like(self.mu_b)

    def reset_parameters(self) -> None:
        p = self.in_features
        if self.use_factorized:
            mu_init = 1.0 / (p**0.5)
            sigma_init = self.initial_sigma / (p**0.5)
        else:
            mu_init = (3.0 / p) ** 0.5
            sigma_init = 0.017

        nn.init.constant_(self.sigma_w, sigma_init)
        nn.init.uniform_(self.mu_w, -mu_init, mu_init)
        if self.bias:
            nn.init.constant_(self.sigma_b, sigma_init)
            nn.init.uniform_(self.mu_b, -mu_init, mu_init)

    @property
    def weight(self):
        return self.mu_w + self.sigma_w * self.eps_w

    @property
    def bias(self):
        if self.bias:
            return self.mu_b + self.sigma_b * self.eps_b
        else:
            return None

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        pass

    def forward(self, input: Tensor) -> Tensor:
        return functional.F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, initial_sigma={self.initial_sigma}, use_factorized={self.use_factorized}"


def build_dense(in_features: int, out_features: int, sigma: float = 0):
    if sigma == 0:
        return Dense(in_features, out_features)
    else:
        return NoisyDense(in_features, out_features)


class DenseStack(nn.Module):
    def __init__(
        self,
        initial_width: int,
        widths: list[int],
        activation: nn.Module = nn.ReLU(),
        noisy_sigma: float = 0,
    ):
        super(DenseStack, self).__init__()
        self.dense_layers: list[Dense | NoisyDense] = []
        self.activation = activation

        assert len(widths) > 0
        self.noisy = noisy_sigma != 0

        current_input_width = initial_width
        for i in range(len(widths)):
            layer = build_dense(
                in_features=current_input_width,
                out_features=widths[i],
                sigma=noisy_sigma,
            )
            self.dense_layers.append(layer)
            current_input_width = widths[i]

        self._output_len = current_input_width

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        for layer in self.dense_layers:
            layer.initialize(initializer)

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for layer in self.dense_layers:
            x = self.activation(layer(x))
        return x

    def reset_noise(self) -> None:
        assert self.noisy

        for layer in self.dense_layers:
            layer.reset_noise()
        return

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, initial_sigma={self.initial_sigma}, use_factorized={self.use_factorized}"

    @property
    def output_width(self):
        return self._output_len
