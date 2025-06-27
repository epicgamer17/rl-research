import torch
import math
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


class FourierSharpening:
    def __init__(self):
        pass

    def __call__(self, P: torch.Tensor, features: torch.Tensor) -> torch.Tensor:  # type: ignore
        pass


class HammardSharpening(FourierSharpening):
    def __init__(self, a):
        super().__init__()
        self.a = a

    def __call__(self, P: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """P is a vector, not a matrix"""
        D, M, d = features.shape
        return (
            P.abs() ** self.a
            * torch.exp(
                torch.complex(torch.zeros(len(P), device=P.device), self.a * P.angle())
            )
            * math.sqrt(D)
        )


class ContractionSharpening(FourierSharpening):
    def __init__(self, a):
        super().__init__()

    def __call__(self, P: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """P must be a matrix, not a vector"""
        assert len(P.shape) == 2, "P must be a matrix"
        D, M, d = features.shape

        sharpened_P = P @ P.T
        return sharpened_P


class FourierShift:
    def __init__(self):
        pass

    def __call__(self, P: torch.Tensor, features: torch.Tensor, v: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Shape of P        : (D)
        Shape of features : (D, M, d)
        Shape of v        : (d)

        """
        pass


class HammardShift(FourierShift):
    def __init__(self):
        super().__init__()

    def __call__(
        self, P: torch.Tensor, features: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        V = (features ** -v.to(features.device)).prod(1).prod(1)
        return P * V


class HammardShiftMatrix(FourierShift):
    def __init__(self):
        super().__init__()

    def __call__(
        self, P: torch.Tensor, features: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        V1 = (features ** -v.to(features.device)).prod(1).prod(1)
        V = torch.einsum("i,j->ij", V1, V1)
        return P * V


class FourierSmoothing:
    def __init__(self):
        pass

    def build_K(self, features: torch.Tensor) -> torch.Tensor:  # type: ignore
        """This function must be called before calling the shift function as it
        precalculates the smoothing matrix/vector to multiply by before shifting.

        Returns the calculated smoothing matrix/vector

        Args:
            features (torch.Tensor): Shape (D, M, d). Tensor of all features as
            initialized by a FourierScaffold.
        """
        pass

    def __call__(self, P) -> torch.Tensor:  # type: ignore
        pass


class GaussianFourierSmoothing(FourierSmoothing):
    def __init__(self, kernel_radii: list, kernel_sigmas: list) -> None:
        self.kernel_radii = kernel_radii
        self.kernel_sigmas = kernel_sigmas
        pass

    def build_K(self, features) -> torch.Tensor:
        D, M, d = features.shape

        assert len(self.kernel_radii) == d
        assert len(self.kernel_sigmas) == d
        length = torch.tensor(self.kernel_radii).int().to(features.device)

        kernels = []
        for dim in range(d):
            kernels.append(
                generate_1d_gaussian_kernel(
                    radius=self.kernel_radii[dim],
                    device=features.device,
                    sigma=self.kernel_sigmas[dim],
                )
            )
        kernel = outer(kernels)
        """Calculates the smoothing kernel. Shape: `(D)`"""

        K = torch.complex(
            torch.zeros(D, device=features.device),
            torch.zeros(D, device=features.device),
        )

        for k in torch.cartesian_prod(
            *[
                torch.arange(
                    -self.kernel_radii[dim],
                    self.kernel_radii[dim] + 1,
                    device=features.device,
                )
                for dim in range(d)
            ]
        ):
            kernel_index = k + length
            g = (features**-k).prod(1).prod(1)

            K += kernel[tuple(kernel_index)] * g

        self.K = K
        return K

    def __call__(self, P: torch.Tensor) -> torch.Tensor:
        return P * self.K


class GuassianFourierSmoothingMatrix(FourierSmoothing):
    def __init__(self, kernel_radii: list, kernel_sigmas: list) -> None:
        # kernel_radii = [10] * d
        # kernel_sigmas = [0.4] * d
        self.kernel_radii = kernel_radii
        self.kernel_sigmas = kernel_sigmas
        pass

    def build_K(self, features) -> torch.Tensor:
        D, M, d = features.shape

        assert len(self.kernel_radii) == d
        assert len(self.kernel_sigmas) == d

        length = torch.tensor(self.kernel_radii).int().to(features.device)

        kernels = []
        for dim in range(d):
            kernels.append(
                generate_1d_gaussian_kernel(
                    radius=self.kernel_radii[dim],
                    device=features.device,
                    sigma=self.kernel_sigmas[dim],
                )
            )
        kernel = outer(kernels)
        """Calculates the smoothing kernel. Shape: `(D)`"""

        K = torch.complex(
            torch.zeros(D, D, device=features.device),
            torch.zeros(D, D, device=features.device),
        )

        for k in torch.cartesian_prod(
            *[
                torch.arange(
                    -self.kernel_radii[dim],
                    self.kernel_radii[dim] + 1,
                    device=features.device,
                )
                for dim in range(d)
            ]
        ):
            kernel_index = k + length
            g1 = (features**-k).prod(1).prod(1)
            g = torch.einsum("i,j->ij", g1, g1)

            K += kernel[tuple(kernel_index)] * g

        self.K = K
        return K

    def __call__(self, P: torch.Tensor) -> torch.Tensor:
        return P * self.K


class FourierScaffold:
    def __init__(
        self,
        shapes: torch.Tensor,
        D: int,
        calculate_g_method="fast",
        shift: FourierShift = HammardShift(),
        smoothing: FourierSmoothing = GaussianFourierSmoothing(
            kernel_radii=[10] * 3, kernel_sigmas=[0.4] * 3
        ),
        sharpening: FourierSharpening = HammardSharpening(2),
        representation="vector",
        device=None,
        limits=None,
        debug=False,
    ):
        assert representation in ["matrix", "vector"]
        self.representation = representation
        self.device = device
        self.shift = shift
        self.smoothing = smoothing
        self.sharpening = sharpening
        self.debug = debug
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
        self.C = D ** (-1 / (2 * self.M * self.d))
        """The scale factor when geneerating features"""

        self.features = torch.zeros(
            (self.D, self.M, self.d),
            dtype=torch.complex64,
            device=self.device,
        )
        """The tensor of base features in the scaffold. It has shape (D, M, d)"""

        for module in range(self.M):
            for dim in range(self.d):
                self.features[:, module, dim] = random_fourier_features(
                    n=self.shapes[module, dim], D=self.D, device=self.device
                )

        self.g = self.zero()
        """The current grid coding state tensor. Shape: `(N_g)`"""

        self.psum_feature = torch.zeros_like(self.g)
        for k in torch.cartesian_prod(
            *[
                torch.arange(0, self.shapes[:, i].prod().item(), device=self.device)
                for i in range(self.d)
            ]
        ):
            self.psum_feature += self.encode(k)

        print("module shapes: ", shapes)
        print("N_g (D) : ", self.N_g)
        print("M       : ", self.M)
        print("d       : ", self.d)
        print("N_patts : ", self.N_patts)

        self.G = self._G(method=calculate_g_method)
        """The matrix of all possible grid states. Shape: `(N_patts, N_g)`"""

        self.smoothing.build_K(self.features)
        """Gaussian smoothing kernel"""

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
    def encode(self, k: torch.Tensor) -> torch.Tensor:
        """Generate encoding of position k.
        Shape of k: (d)
        """

        base = self.C ** (self.M * self.d) * (self.features**k).prod(1).prod(1)

        if self.representation == "vector":
            return base
        else:
            return torch.einsum("i,j->ij", base, base)

    @torch.no_grad()
    def zero(self) -> torch.Tensor:
        """Reset the grid coding state tensor to the 'zero' position. Shape: `(N_g) = (D)`"""
        return self.encode(torch.zeros(self.d, device=self.device))

    def smooth(self):
        self.g = self.smoothing(self.g)

    def velocity_shift(self, v: torch.Tensor):
        self.g = self.shift(self.g, self.features, v)

    def sharpen(self):
        self.g = self.sharpening(self.g, self.features)
        if isinstance(self.sharpening, ContractionSharpening):
            scaling = (self.g * self.psum_feature).sum().real
            self.g /= scaling

    def get_probability(self, k: torch.Tensor):
        """Obtain the probability mass located in cell k

        Shape of k: (d)

        Args:
            k (_type_): _description_
        """
        return (self.encode(k) * self.g).sum()
