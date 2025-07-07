import torch
import torch.linalg
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
    return x / x.abs()


class FourierSharpening:
    def __init__(self):
        pass

    def __call__(self, P: torch.Tensor, features: torch.Tensor) -> torch.Tensor:  # type: ignore
        pass

    def sharpen_batch(
        self, P: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:  # type:ignore
        """Input shape of P: (B, D, D)

        Output shape: (B, D, D)
        """
        pass


class HadamardSharpening(FourierSharpening):
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

    def sharpen_batch(self, P: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class ContractionSharpening(FourierSharpening):
    def __init__(self, a):
        super().__init__()

    def __call__(self, P: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """P must be a matrix, not a vector"""
        assert len(P.shape) == 2, "P must be a matrix"
        D, M, d = features.shape

        scaling = P.norm() ** 2
        sharpened_P = P @ P.conj().T / scaling
        return sharpened_P

    def sharpen_batch(self, P: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Input shape of P: (B, D, D)

        Output shape: (B, D, D)

        """
        scaling = torch.linalg.vector_norm(P, dim=(1, 2)) ** 2
        sharpened_P = torch.einsum("bij,bjk->bik", P, P.conj().T)
        return sharpened_P / scaling


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


class HadamardShift(FourierShift):
    def __init__(self):
        super().__init__()

    def __call__(
        self, P: torch.Tensor, features: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        V = (features ** v.to(features.device)).prod(1).prod(1)
        return P * V


class HadamardShiftRat(FourierShift):
    def __init__(self, shapes: torch.Tensor):
        self.shapes = shapes
        super().__init__()

    def calculate_alpha(self, v: torch.Tensor):
        # (D, 1)
        delta_v = v.floor().unsqueeze(-1)
        delta_f_v = v.unsqueeze(-1) - delta_v

        # (D, 2)
        stacked = torch.concat([1 - delta_f_v, delta_f_v], dim=1)

        # (2 x 2 x ... x 2) (D times)
        alpha = outer([stacked[i] for i in range(len(stacked))])
        return alpha

    def __call__(
        self, P: torch.Tensor, features: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        D, M, d = features.shape

        delta_v = v.floor()
        alpha = self.calculate_alpha(v)
        V = torch.complex(
            torch.zeros(D, device=features.device),
            torch.zeros(D, device=features.device),
        )

        for k in torch.cartesian_prod(
            *[torch.arange(2, device=features.device)] * len(v)
        ):
            kernel_index = k
            shift = delta_v + k
            g = (features**shift).prod(1).prod(1)
            V += alpha[tuple(kernel_index)] * g

        return P * V


class HadamardShiftMatrix(FourierShift):
    def __init__(self):
        super().__init__()

    def __call__(
        self, P: torch.Tensor, features: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        V1 = (features ** v.to(features.device)).prod(1).prod(1)
        V2 = V1.conj()
        V = torch.einsum("i,j->ij", V1, V2)
        return P * V


class HadamardShiftMatrixRat(FourierShift):
    def __init__(self, shapes: torch.Tensor):
        self.shapes = shapes
        super().__init__()

    def calculate_alpha(self, v: torch.Tensor):
        # (D, 1)
        delta_v = v.floor().unsqueeze(-1)
        delta_f_v = v.unsqueeze(-1) - delta_v

        # (D, 2)
        stacked = torch.concat([1 - delta_f_v, delta_f_v], dim=1)

        # (2 x 2 x ... x @) (D times)
        alpha = outer([stacked[i] for i in range(len(stacked))])
        return alpha

    def __call__(
        self, P: torch.Tensor, features: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        D, M, d = features.shape

        delta_v = v.floor()
        alpha = self.calculate_alpha(v)
        V = torch.complex(
            torch.zeros(D, D, device=features.device),
            torch.zeros(D, D, device=features.device),
        )

        for k in torch.cartesian_prod(
            *[torch.arange(2, device=features.device)] * len(v)
        ):
            kernel_index = k
            shift = delta_v + k
            g1 = (features**shift).prod(1).prod(1)
            g = torch.einsum("i,j->ij", g1, g1.conj())
            V += alpha[tuple(kernel_index)] * g

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
            g = (features**k).prod(1).prod(1)

            K += kernel[tuple(kernel_index)] * g

        self.K = K
        return K

    def __call__(self, P: torch.Tensor) -> torch.Tensor:
        return P * self.K


class GuassianFourierSmoothingMatrix(FourierSmoothing):
    def __init__(self, kernel_radii: list, kernel_sigmas: list) -> None:

        # ex.  kernel_radii = [10] * d
        #      kernel_sigmas = [0.4] * d
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
            g1 = (features**k).prod(1).prod(1)
            g2 = g1.conj()
            g = torch.einsum("i,j->ij", g1, g2)

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
        shift: FourierShift = HadamardShiftMatrix(),
        smoothing: FourierSmoothing = GuassianFourierSmoothingMatrix(
            kernel_radii=[10] * 3, kernel_sigmas=[0.4] * 3
        ),
        sharpening: FourierSharpening = ContractionSharpening(2),
        representation="matrix",
        device=None,
        limits=None,
        debug=False,
        rescaling=True,
        _skip_K_calc=False,
        _skip_gs_calc=False,
        features: None | torch.Tensor = None,
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

        if features == None:
            for module in range(self.M):
                for dim in range(self.d):
                    self.features[:, module, dim] = random_fourier_features(
                        n=self.shapes[module, dim].item(), D=self.D, device=self.device
                    )
        else:
            self.features = features

        self.P = self.zero()
        """The current grid coding state tensor. Shape: `(N_g)`"""

        print("module shapes: ", shapes)
        print("N_g (D) : ", self.N_g)
        print("M       : ", self.M)
        print("d       : ", self.d)
        print("N_patts : ", self.N_patts)

        if not _skip_K_calc:
            self.smoothing.build_K(self.features)

        self._gbook = None
        if not _skip_gs_calc:
            self.g_s = torch.sum(self.gbook().sum(dim=1))

        self.scale_factor = torch.ones(len(self.shapes[0]), device=self.device)
        """ `scale_factor[d]` is the amount to multiply by to convert "world units" into "grid units" """

        self.grid_limits = torch.ones(len(self.shapes[0]), device=self.device)
        for dim in range(self.d):
            self.grid_limits[dim] = torch.prod(self.shapes[:, dim]).item()

        if limits != None:
            for dim in range(self.d):
                self.scale_factor[dim] = self.grid_limits[dim] / limits[dim]

        self.rescaling = rescaling

    @torch.no_grad()
    def encode(self, k: torch.Tensor) -> torch.Tensor:
        """Generate encoding of position k.
        Shape of k: (d)
        """

        base = self.C ** (self.M * self.d) * (self.features**k).prod(1).prod(1)

        if self.representation == "vector":
            return base
        else:
            base2 = base.conj()
            return torch.einsum("i,j->ij", base, base2)

    @torch.no_grad()
    def encode_batch(
        self, ks: torch.Tensor, representation: str | None = None
    ) -> torch.Tensor:
        """Generate encoding of position k.
        Shape of k: (d, ...)
        """
        d = ks.shape[0]
        B = ks.shape[1:]
        base = self.C ** (self.M * self.d) * (
            self.features.unsqueeze(-1).tile(B) ** ks
        ).prod(1).prod(1)

        if representation == None:
            representation = self.representation

        if representation == "vector":
            return base
        else:
            base2 = base.conj()
            return torch.einsum("i...,j...->ij...", base, base2)

    @torch.no_grad()
    def gbook(self) -> torch.Tensor:
        """Get the tensor of all possible grid states

        Output shape: (D, N_patts)
        """
        # Get the tensor of all possible states
        #
        dim_sizes = [int(self.shapes[:, dim].prod().item()) for dim in range(self.d)]
        if self._gbook == None:
            self._gbook = self.encode_batch(
                torch.cartesian_prod(
                    *[torch.arange(dim_sizes[dim]) for dim in range(self.d)]
                ).T,
                representation="vector",
            )
        return self._gbook

    @torch.no_grad()
    def encode_probability(self, distribution) -> torch.Tensor:
        """Generate encoding of probability distribution"""
        encoding = torch.zeros_like(self.P)
        for k in torch.cartesian_prod(
            *[torch.arange(int(self.shapes[:, i].prod().item())) for i in range(self.d)]
        ):
            encoding += distribution[tuple(k)] * self.encode(k.to(self.device))

        return encoding

    @torch.no_grad()
    def zero(self) -> torch.Tensor:
        """Reset the grid coding state tensor to the 'zero' position. Shape: `(N_g) = (D)`"""
        return self.encode(torch.zeros(self.d, device=self.device))

    def smooth(self):
        self.P = self.smoothing(self.P)

    def velocity_shift(self, v: torch.Tensor):
        self.P = self.shift(self.P, self.features, v)

    def sharpen(self):
        self.P = self.sharpening(self.P, self.features)

    def get_probability(self, k: torch.Tensor):
        """Obtain the probability mass located in cell k

        Shape of k: (d)

        Args:
            k (_type_): _description_
        """
        return (self.P * self.encode(k).conj()).sum()

    def get_all_probabilities(self):
        dim_sizes = [int(self.shapes[:, dim].prod().item()) for dim in range(self.d)]
        ptensor = torch.zeros(*dim_sizes, device=self.device)
        for k in torch.cartesian_prod(
            *[torch.arange(dim_sizes[dim]) for dim in range(self.d)]
        ):
            p = self.get_probability(k.clone().to(self.device))
            ptensor[tuple(k)] = p
        return ptensor

    def g_avg(self):
        return self.P @ self.g_s

    def g_avg_batch(self, P: torch.Tensor):
        """P shape: (B, D, D)

        Output shape: (B, D)
        """
        return torch.einsum("bij,j->bi", P, self.g_s)


def calculate_alpha(delta_f_x, delta_f_y, device=None):
    alpha = torch.zeros(2, 2, device=device)
    for i in range(0, 2):
        for j in range(0, 2):
            alpha[i][j] = g(delta_f_x, i) * g(delta_f_y, j)
    return alpha


def g(a, b):
    if b == 0:
        return 1 - a
    else:
        return a


def calculate_padding(i: int, k: int, s: int):
    """Calculate both padding sizes along 1 dimension for a given input length, kernel length, and stride

    Args:
        i (int): input length
        k (int): kernel length
        s (int): stride

    Returns:
        (p_1, p_2): where p_1 = p_2 - 1 for uneven padding and p_1 == p_2 for even padding
    """

    p = (i - 1) * s - i + k
    p_1 = p // 2
    p_2 = (p + 1) // 2
    return (p_1, p_2)


class FourierScaffoldDebug:
    def __init__(self, shapes: torch.Tensor, device=None, rescale=True):
        self.shapes = torch.tensor(shapes).int()
        self.device = device
        self.M, self.d = self.shapes.shape

        dim_sizes = [int(self.shapes[:, dim].prod().item()) for dim in range(self.d)]
        self.ptensor = torch.zeros(*dim_sizes, device=self.device)
        self.ptensor[tuple([0] * self.d)] = 1
        self.rescale = rescale

    def velocity_shift2d(self, v):
        # ratslam-style velocity shift
        v_x, v_y = v[0], v[1]
        delta_x, delta_y = math.floor(v_x), math.floor(v_y)
        delta_f_x, delta_f_y = v_x - delta_x, v_y - delta_y
        alpha_x_length, alpha_y_length = 2, 2

        x_padding, y_padding = ((alpha_x_length // 2, 0), (alpha_y_length // 2, 0))
        alpha = calculate_alpha(
            delta_f_x,
            delta_f_y,
            device=self.ptensor.device,
        )
        alpha_flipped = torch.flip(alpha, (0, 1))
        shifted = torch.roll(self.ptensor, shifts=(delta_x, delta_y), dims=(0, 1))
        padded = torch.nn.functional.pad(
            shifted.unsqueeze(0).unsqueeze(0),
            y_padding + x_padding,
            mode="circular",
        )
        updated_P = torch.nn.functional.conv2d(
            padded,
            alpha_flipped.unsqueeze(0).unsqueeze(0),
            padding=0,
        )
        self.ptensor = updated_P.squeeze(0).squeeze(0)

    def smooth2d(self):
        kernel_size = 9
        sigma = 0.4
        x = torch.arange(kernel_size, device=self.device) - kernel_size // 2
        y = torch.arange(kernel_size, device=self.device) - kernel_size // 2
        x, y = torch.meshgrid(x, y)
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        x_padding = calculate_padding(kernel_size, kernel.shape[0], 1)
        y_padding = calculate_padding(kernel_size, kernel.shape[1], 1)

        padded = torch.nn.functional.pad(
            self.ptensor.unsqueeze(0).unsqueeze(0),
            y_padding + x_padding,
            mode="circular",
        )

        convoluted = torch.nn.functional.conv2d(
            input=padded, weight=kernel.unsqueeze(0).unsqueeze(0)
        )

        self.ptensor = convoluted.squeeze(0).squeeze(0)

    def sharpen(self):
        sharpened = self.ptensor**2
        if self.rescale:
            scaling = sharpened.sum()
            self.ptensor = sharpened / scaling
        else:
            self.ptensor = sharpened
