import torch
import math
import itertools
from vectorhash_functions import generate_1d_gaussian_kernel, outer


def random_fourier_features(n: int, D: int, device=None):
    """Generate a random fourier vector

    Args:
        n (int): periodicity
        D (int): length
    """
    w = 2 * math.pi / n
    k = -w * torch.randint(0, n, (D,), device=device)
    x = torch.exp(k * torch.complex(torch.zeros_like(k), torch.ones_like(k)))
    return x


class FourierScaffold:
    def __init__(
        self,
        shapes: torch.Tensor,
        D: int,
        calculate_g_method="fast",
        device=None,
        limits=None,
    ):
        self.device = device
        self.shapes = torch.tensor(shapes).int()
        """(M, d) where M is the number of grid modules and d is the dimensionality of the grid modules."""
        self.M = self.shapes.shape[0]
        """The number of modules"""
        self.d = self.shapes.shape[1]
        """The dimensionality of each module"""
        self.N_patts = shapes.prod().item()
        """The number of unique states the scaffold has"""
        self.N_g = D
        """N_g = D"""

        self.D = D
        self.C = D ** (-1 / (self.M * self.d))
        """The scale factor when geneerating features"""

        self.features = torch.zeros(
            D,
            self.shapes.shape[0],
            self.shapes.shape[1],
            dtype=torch.complex64,
            device=self.device,
        )
        """The tensor of base features in the scaffold. It has shape (D, M, d)"""

        for module in range(self.M):
            for dim in range(self.d):
                self.features[:, module, dim] = random_fourier_features(
                    n=self.shapes[module, dim], D=self.D, device=self.device
                )

        print("module shapes: ", shapes)
        print("N_g (D) : ", self.N_g)
        print("M       : ", self.M)
        print("d       : ", self.d)
        print("N_patts : ", self.N_patts)

        self.G = self._G(method=calculate_g_method)
        """The matrix of all possible grid states. Shape: `(N_patts, N_g)`"""

        self.K = self._K()
        """Gaussian smoothing kernel"""

        self.g = self._g()
        """The current grid coding state tensor. Shape: `(N_g)`"""

        self.scale_factor = torch.ones(len(self.shapes[0]), device=self.device)
        """ `scale_factor[d]` is the amount to multiply by to convert "world units" into "grid units" """

        self.grid_limits = torch.ones(len(self.shapes[0]), device=self.device)
        for dim in range(self.d):
            self.grid_limits[dim] = torch.prod(self.shapes[:, dim]).item()

        if limits != None:
            for dim in range(self.d):
                self.scale_factor[dim] = self.grid_limits[dim] / limits[dim]

    @torch.no_grad()
    def _G(self, method) -> torch.Tensor:
        """Calculates the matrix of all possible grid states. Shape: `(N_patts, N_g)`"""
        return torch.zeros(1)

    @torch.no_grad()
    def _g(self) -> torch.Tensor:
        """Calculates the current grid coding state tensor. Shape: `(N_g) = (D)`"""
        return torch.complex(
            torch.ones(self.D, device=self.device),
            torch.zeros(self.D, device=self.device),
        ) * (self.C ** (self.M * self.d))

    @torch.no_grad()
    def _K(self) -> torch.Tensor:
        kernel_radii = [10] * self.d
        length = torch.tensor(kernel_radii).int().to(self.device)
        kernel_sigmas = [0.4] * self.d

        kernels = []
        for d in range(self.d):
            kernels.append(
                generate_1d_gaussian_kernel(
                    radius=kernel_radii[d], device=self.device, sigma=kernel_sigmas[d]
                )
            )
        kernel = outer(kernels)
        """Calculates the smoothing kernel. Shape: `(D)`"""
        K = torch.complex(
            torch.zeros(self.D, device=self.device),
            torch.zeros(self.D, device=self.device),
        )
        for k in itertools.product(
            *[
                torch.arange(
                    -kernel_radii[dim], kernel_radii[dim] + 1, device=self.device
                )
                for dim in range(self.d)
            ]
        ):
            kernel_index = torch.tensor(k, device=self.device) + length
            K += kernel[tuple(kernel_index)] * (
                self.features ** -torch.tensor(k, device=self.device)
            ).prod(1).prod(1)
        return K

    def shift(self, v: torch.Tensor):
        """Shift by a certain velocity.
        Shape of v: (d)
        """

        # self.shapes   (M, d)
        #           v   (d)
        # self.features (D, M, d)
        V = (self.features ** -v.to(self.device)).prod(1).prod(1)
        self.g *= V

    def smooth(self):
        self.g *= self.K

    def sharpen(self):
        pass

    def get_probability(self, k: torch.Tensor):
        """Obtain the probability mass located in cell k

        Shape of k: (d)

        Args:
            k (_type_): _description_
        """
        Pk = (self.features**k).prod(1).prod(1)
        return Pk.T @ self.g
