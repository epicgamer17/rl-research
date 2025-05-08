import torch

from competetive_attractor_dynamics import (
    generate_epsilon,
    update_internal_P_jk_batched,
    generate_delta,
    update_inter_layer_P_ijk_batched,
    global_inhibition_batched,
    batch_rescale,
)

_epsilon = 1e-8


class Smoothing:
    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth a batch of tensors.

        Input shape: (B, ...)

        """
        pass

    def __str__(self):
        return self.__class__.__name__


class SoftmaxSmoothing(Smoothing):
    def __init__(self, T=1e-3):
        super().__init__()
        assert T > 0
        self.T = T

    def __call__(self, x):
        y = x.flatten(1).T
        maxes = torch.max(y, dim=0).values
        y = y - maxes
        exp = torch.exp(y / self.T)
        sums = torch.sum(exp, dim=0) + _epsilon
        out = (exp / sums).T
        return out.reshape(*x.shape)

    def __str__(self):
        return super().__str__() + f" (T={self.T})"


class PolynomialSmoothing(Smoothing):
    def __init__(self, k):
        super().__init__()
        assert k > 0
        self.k = k
        pass

    def __call__(self, x):
        y = x.flatten(1).T
        y = y**self.k
        sums = torch.sum(y, dim=0) + _epsilon
        out = (y / sums).T
        return out.reshape(*x.shape)

    def __str__(self):
        return super().__str__() + f" (k={self.k})"


class ArgmaxSmoothing(Smoothing):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        y = x.flatten(1).T
        maxes = torch.max(y, dim=0).values
        y = torch.where(y == maxes, torch.ones_like(y), torch.zeros_like(y))
        scaled = (y / torch.sum(y, dim=0, keepdim=True)).T
        return scaled

    def __str__(self):
        return super().__str__()


class IdentitySmoothing(Smoothing):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x.detach().clone()

    def __str__(self):
        return super().__str__()


class RatSLAMSmoothing(Smoothing):
    def __init__(
        self,
        sigma_xy=0.3,
        sigma_theta=0.3,
        inhibition_constant=0.004,
        delta_gamma=1,
        device=None,
    ):
        super().__init__()
        self.sigma_xy = sigma_xy
        self.sigma_theta = sigma_theta
        self.inhibition_constant = inhibition_constant
        self.delta_gamma = delta_gamma
        self.device = device

    def __call__(self, x):
        assert (
            len(x.shape) == 4
        ), f"x should be a 4D tensor (B, x, y, theta), instead got {x.shape}"
        B, N_x, N_y, N_theta = x.shape
        # Implement the RatSLAM smoothing logic here
        eps = generate_epsilon(N_x, N_y, sigma=self.sigma_xy, device=self.device)
        delta = generate_delta(
            N_theta,
            sigma=self.sigma_theta,
            gamma=self.delta_gamma,
            device=self.device,
        )

        P = update_internal_P_jk_batched(x, eps)
        P = update_inter_layer_P_ijk_batched(P, delta)
        P = global_inhibition_batched(P, inhibition_constant=self.inhibition_constant)
        P = batch_rescale(P)
        return P

    def __str__(self):
        return (
            super().__str__()
            + f" (sigma_xy={self.sigma_xy}, sigma_theta={self.sigma_theta}, inhibition_constant={self.inhibition_constant}, delta_gamma={self.delta_gamma})"
        )


class SequentialSmoothing(Smoothing):
    def __init__(self, methods: list[Smoothing]):
        self.methods = methods

    def __call__(self, x):
        for method in self.methods:
            x = method(x)

        return x
