from typing import Callable

import torch
from torch import nn, Tensor, functional


class Dense(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, *args, **kwargs):
        super(Dense, self).__init__(*args, **kwargs)
        self.layer = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        initializer(self.layer.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layer(inputs)

    def extra_repr(self) -> str:
        return self.layer.extra_repr()


class NoisyDense(nn.Module):
    """See https://arxiv.org/pdf/1706.10295."""

    @staticmethod
    def f(x: Tensor):
        return x.sgn() * (x.abs().sqrt())

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        initial_sigma: float = 0.5,
        use_factorized: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initial_sigma = initial_sigma
        self.use_factorized = use_factorized
        self.use_bias = bias

        self.mu_w = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_w = nn.Parameter(torch.empty(out_features, in_features))
        self.eps_w = self.register_buffer("eps_w", torch.empty(out_features, in_features))
        if self.use_bias:
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
            eps_i = torch.randn(1, self.in_features).to(self.mu_w.device)
            eps_j = torch.randn(self.out_features, 1).to(self.mu_w.device)
            self.eps_w = self.f(eps_j) @ self.f(eps_i)
            self.eps_b = self.f(eps_j).reshape(self.out_features)
        else:
            self.eps_w = self.f(torch.randn(self.mu_w.shape)).to(self.mu_w.device)
            if self.use_bias:
                self.eps_b = self.f(torch.randn(size=self.mu_b.shape)).to(self.mu_w.device)

    def remove_noise(self) -> None:
        self.eps_w = torch.zeros_like(self.mu_w).to(self.mu_w.device)
        if self.use_bias:
            self.eps_b = torch.zeros_like(self.mu_b).to(self.mu_w.device)

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
        if self.use_bias:
            nn.init.constant_(self.sigma_b, sigma_init)
            nn.init.uniform_(self.mu_b, -mu_init, mu_init)

    @property
    def weight(self):
        return self.mu_w + self.sigma_w * self.eps_w

    @property
    def bias(self):
        if self.use_bias:
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
        self.dense_layers: nn.ModuleList = nn.ModuleList()
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

        self.initial_width = initial_width
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
        return f"in_features={self.initial_width}, out_width={self.output_width}, noisy={self.noisy}"

    @property
    def output_width(self):
        return self._output_len
