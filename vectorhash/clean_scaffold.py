import math
import torch
import numpy as np
from matrix_initializers import SparseMatrixBySparsityInitializer
from ratslam_velocity_shift import inject_activity
from vectorhash_functions import chinese_remainder_theorem, circular_mean_2
from tqdm import tqdm


class Smoothing:
    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth a batch of tensors.

        Input shape: (B, ...)

        """
        pass


class SoftmaxSmoothing(Smoothing):
    def __init__(self, T=1e-3):
        super().__init__()
        assert T > 0
        self.T = T

    def __call__(self, x):
        y = x.flatten(1).T
        maxes = torch.max(y, dim=0).values
        y = y - maxes
        exp = torch.exp(y)
        out = (exp / torch.sum(exp, dim=0)).T
        return out.reshape(*x.shape)


class PolynomialSmoothing(Smoothing):
    def __init__(self, k):
        super().__init__()
        assert k > 0
        self.k = k
        pass

    def __call__(self, x):
        y = x.flatten(1).T
        y = y**self.k
        out = (y / torch.sum(y, dim=0)).T
        return out.reshape(*x.shape)


class ArgmaxSmoothing(Smoothing):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        y = x.flatten(1).T
        maxes = torch.max(y, dim=0).values
        y = torch.where(y == maxes, torch.ones_like(y), torch.zeros_like(y))
        scaled = (y / torch.sum(y, dim=0, keepdim=True)).T
        return scaled


class GridModule:
    def __init__(
        self,
        shape: tuple[int],
        device=None,
        T=1,
        ratshift=True,
        smoothing=SoftmaxSmoothing(T=1e-3),
    ) -> None:
        """Initializes a grid module with a given shape and a temperature.

        Args:
            shape (tuple[int]): The shape of the grid.
            T (int, optional): The temperature of the grid used with softmax when computing the onehot encoded state probabilities. Defaults to 1.
        """
        self.shape = shape
        self.smoothing = smoothing
        # self.state = torch.rand(shape, device=device)
        self.state = torch.zeros(shape, device=device)
        self.ratshift = ratshift
        i = tuple(0 for _ in range(len(shape)))
        self.state[i] = 1
        self.l = torch.prod(torch.tensor(shape)).item()
        """The number of elements in the grid, i.e. the product of each element in the shape. Ex. a `3x3x4` grid has `l=36`."""
        self.T = T
        self.device = device

    @torch.no_grad()
    def onehot(self) -> torch.Tensor:
        """Get the current state as a flattened onehot vector (normalized)"""
        return self.state.flatten() / self.state.sum()
        #return self.state.flatten()

    @torch.no_grad()
    def denoise_onehot(self, onehot: torch.Tensor) -> torch.Tensor:
        """Denoise a batch of one-hot encoded states.

        Input shape: `(B, l)` where l is the product of the shape of the grid.

        Args:
            onehot: The tensor of one-hot encoded states.

        Output shape: `(B, l)`
        """
        # assert sum(onehot.sum(dim=1) == 1) == onehot.shape[0], "Not one-hot encoded"
        if onehot.ndim == 1:
            onehot = onehot.unsqueeze(0)

        state = onehot.view((onehot.shape[0], *self.shape))
        return self.denoise(state).flatten(1)

    @torch.no_grad()
    def denoise(self, state: torch.Tensor) -> torch.Tensor:
        """Denoise a batch of grid states. This finds the maximum value in the grid and sets it to 1, and all other values to 0.
        If there are multiple maximum values, pick a random. (not all set to 1 / number of maximum values)

        Input shape: `(B, *shape)` where `shape` is the shape of the grid.

        Args:
            state: The tensor of grid states.

        Output shape: `(B, *shape)` where `shape` is the shape of the grid.

        """
        if state.ndim == len(self.shape):
            state = state.unsqueeze(0)

        return self.smoothing(state)

    @torch.no_grad()
    def shift(self, v: torch.Tensor):
        """Shifts the state of the grid module by a given velocity.

        Input shape: `(len(shape))`

        Args:
            v: The velocity by which to shift the grid state.
            (x, y, angular velocity)
        """
        v_ = v.int()
        if self.ratshift:
            assert len(v) == 3  # x, y, angular velocity
            speed = (v[0].item() ** 2 + v[1].item() ** 2) ** 0.5
            theta = math.atan2(v[1].item(), v[0].item())
            self.state = inject_activity(self.state, speed, theta, v[2].item())
        else:
            self.state = torch.roll(
                self.state,
                tuple([v_[i].item() for i in range(len(v_))]),
                dims=tuple(i for i in range(len(self.shape))),
            )

    @torch.no_grad()
    def grid_state_from_cartesian_coordinates(
        self, coordinates: torch.Tensor
    ) -> torch.Tensor:
        """Convert integer-valued cartesian coordinates into a onehot-encoded grid state

        Args:
            coordinates (torch.Tensor): Tensor of integers, length equal to the dimensionality of the grid module

        Returns:
            torch.Tensor: A one-hot encoded tensor of the grid state
        """
        phis = torch.remainder(coordinates, torch.Tensor(self.shape)).int()
        gpattern = torch.zeros_like(self.state)
        gpattern[tuple(phis)] = 1
        gpattern = gpattern.flatten()
        return gpattern

    @torch.no_grad()
    def cartesian_coordinates_from_grid_state(self, g: torch.Tensor) -> torch.Tensor:
        """Convert a onehot-encoded grid state into integer-valued cartesian coordinates

        Args:
            g (torch.Tensor): One-hot encoded grid state

        Returns:
            torch.Tensor: Tensor of coordinates, length equal to the dimensionality of the grid module in
            `{0, 1, ..., \lambda_1 - 1} x {0, 1, ..., \lambda_2 - 1} x ... x {0, 1, ..., \lambda_d - 1}  `
        """
        reshaped = g.view(*self.shape).nonzero()
        return reshaped

    @torch.no_grad()
    def grid_state_from_cartesian_coordinates_extended(
        self, coordinates: torch.Tensor
    ) -> torch.Tensor:
        """Convert real-valued cartesian coordinates into a continuous grid state

        Args:
            coordinates (torch.Tensor): Coordinates in the range [0, \lambda_1) x ... x [0, \lambda_d) \subseteq \mathbb{R}^d

        Returns:
            torch.Tensor: A real grid state g in [0,1]^{\lambda_1 \cdot ... \cdot \lambda_d} where \sum_{i=1}^{\lambda_1 \cdot ... \cdot \lambda_d} g_i = 1
        """
        coordinates = torch.remainder(coordinates, torch.Tensor(self.shape).int())
        floored = torch.floor(coordinates).int()
        next_floored = torch.remainder(
            coordinates + 1, torch.Tensor(self.shape).int()
        ).int()
        remainder = coordinates - floored
        alpha = torch.einsum(
            self.einsum_str, *torch.vstack([1 - remainder, remainder]).T
        ).flatten()
        indices = torch.cartesian_prod(*torch.vstack([floored, next_floored]).T)
        # print("s:", indices)
        # print(alpha)

        state = torch.zeros_like(self.state)
        for i, a in enumerate(alpha):
            # print(indices[i])
            state[tuple(indices[i])] = a
        return state.flatten()

    @torch.no_grad()
    def cartesian_coordinates_from_grid_state_extended(
        self, g: torch.Tensor
    ) -> torch.Tensor:
        """Convert a continuous grid state into real-valued cartesian coordinates. NOT WORKING YET

        Args:
            g (torch.Tensor): A continuous grid state in [0,1]^{\lambda_1 \cdot ... \cdot \lambda_d} where \sum_{i=1}^{\lambda_1 \cdot ... \cdot \lambda_d} g_i = 1

        Returns:
            torch.Tensor: Coordinates in the range [0, \lambda_1) x ... x [0, \lambda_d) \subseteq \mathbb{R}^d
        """
        sums = list()
        dims = range(len(self.shape))
        for i in range(len(self.shape)):
            dim = [j for j in dims if j != i]
            pdf = torch.sum(g.reshape(*self.shape), dim=dim)
            sums.append(pdf)

        coordinates = torch.zeros(len(self.shape))
        for i, l in enumerate(self.shape):
            w = sums[i]
            # print(f"w[{i}]", w)
            coordinates[i] = circular_mean_2(torch.arange(l), w, l).item()

        return coordinates


class GridHippocampalScaffold:
    def __init__(
        self,
        shapes: torch.Tensor,
        N_h: int,
        sparse_matrix_initializer=None,
        relu_theta=0.5,
        ratshift=False,
        sanity_check=True,
        calculate_g_method="fast",
        smoothing=SoftmaxSmoothing(T=1e-3),
        device=None,
    ):
        assert calculate_g_method in ["hairpin", "fast", "spiral"]

        self.device = device
        self.shapes = torch.Tensor(shapes).int()
        """(M, d) where M is the number of grid modules and d is the dimensionality of the grid modules."""
        self.relu_theta = relu_theta
        self.ratshift = ratshift
        self.modules = [
            GridModule(shape, device=device, ratshift=ratshift, smoothing=smoothing)
            for shape in shapes
        ]
        """The list of grid modules in the scaffold."""
        # for module in self.modules:
        #     print(module.l)
        self.N_g = sum([module.l for module in self.modules])
        self.N_patts = np.prod([module.l for module in self.modules]).item()
        self.N_h = N_h

        print("module shapes: ", [module.shape for module in self.modules])
        print("N_g     : ", self.N_g)
        print("N_patts : ", self.N_patts)
        print("N_h     : ", self.N_h)

        self.sparse_matrix_initializer = sparse_matrix_initializer
        self.G = self._G(method=calculate_g_method)

        """The matrix of all possible grid states. Shape: `(N_patts, N_g)`"""

        if sparse_matrix_initializer is None:
            sparse_matrix_initializer = SparseMatrixBySparsityInitializer(
                sparsity=0.1, device=device
            )

        self.W_hg = sparse_matrix_initializer((self.N_h, self.N_g))

        """The matrix of weights to go from the grid layer to the hippocampal layer. Shape: `(N_h, N_g)`"""
        self.H = self.hippocampal_from_grid(self.G)  # (N_patts, N_h)

        """The matrix of all possible hippocampal states induced by `G` and `W_hg`. Shape: `(N_patts, N_h)`"""
        self.W_gh = self._W_gh()  # (N_g, N_h)
        if sanity_check:
            assert torch.all(
                self.G
                == self.denoise(
                    self.grid_from_hippocampal(self.hippocampal_from_grid(self.G))
                )
            ), "G -> H -> G should preserve G"

        self.g = self._g()
        """The current grid coding state tensor. Shape: `(N_g)`"""

    @torch.no_grad()
    def _G(self, method) -> torch.Tensor:
        """Calculates the matrix of all possible grid states. Shape: `(N_patts, N_g)`"""

        if method == "hairpin":
            # e.x. shapes: (3,3), (4,4), (5,5)
            dim_vecs = []
            for dim in range(len(self.shapes[0])):
                l = 1
                for shape in self.shapes:
                    l *= shape[dim]
                dim_vecs.append(torch.arange(l, device=self.device))

            grid_states = torch.cartesian_prod(*dim_vecs).int()
            gbook = torch.zeros(
                (self.N_g, *[len(d) for d in dim_vecs]), device=self.device
            )

            for state in tqdm(grid_states):
                i = 0
                for shape in self.shapes:
                    phis = torch.remainder(state, shape.to(self.device)).int()
                    gpattern = torch.zeros(tuple(shape.tolist()), device=self.device)
                    gpattern[tuple(phis)] = 1
                    gpattern = gpattern.flatten()
                    if len(state) == 2:
                        gbook[i : i + len(gpattern), state[0], state[1]] = gpattern
                    if len(state) == 3:
                        gbook[i : i + len(gpattern), state[0], state[1], state[2]] = (
                            gpattern
                        )
                    i += len(gpattern)

            return gbook.flatten(1).T
        if method == "fast":
            gbook = torch.zeros((self.N_patts, self.N_g), device=self.device)
            i = 0
            for module in self.modules:
                gbook[:, i : i + module.l] = torch.tile(
                    torch.eye(module.l, device=self.device),
                    (self.N_patts // module.l, 1),
                )
                i += module.l
            return gbook

    @torch.no_grad()
    def _g(self) -> torch.Tensor:
        """Calculates the current grid coding state tensor. Shape: `(N_g)`"""
        vecs = list()
        for module in self.modules:
            vecs.append(module.onehot())
        return torch.cat(vecs)

    @torch.no_grad()
    def grid_state_from_cartesian_coordinates(
        self, coordinates: torch.Tensor
    ) -> torch.Tensor:
        """
        Input shape: `(d)` where `d` is the dimensionality of the points.

        Output shape: `(N_g)` where `N_g` is the number of grid cells.
        """
        g = torch.zeros(self.N_g, device=self.device)
        i = 0
        for module in self.modules:
            pattern = module.grid_state_from_cartesian_coordinates(coordinates)
            g[i : i + len(pattern)] = pattern

        return g

    @torch.no_grad()
    def grid_state_from_cartesian_coordinates_extended(
        self, coordinates: torch.Tensor
    ) -> torch.Tensor:
        """
        Input shape: `(d)` where `d` is the dimensionality of the points.

        Output shape: `(N_g)` where `N_g` is the number of grid cells.
        """

        g = torch.zeros(self.N_g, device=self.device)
        i = 0
        for module in self.modules:
            pattern = module.grid_state_from_cartesian_coordinates_extended(coordinates)
            g[i : i + len(pattern)] = pattern
            i += len(pattern)
        return g

    @torch.no_grad()
    def cartesian_coordinates_from_grid_state(self, g: torch.Tensor) -> torch.Tensor:
        """
        Input shape: `(N_g)` where `N_g` is the number of grid cells.

        Output shape: `(d)` where `d` is the dimensionality of the points.
        """

        remainders = torch.zeros(
            (len(self.shapes), len(self.shapes[0])), device=self.device
        )

        i = 0
        for j, module in enumerate(self.modules):
            remainders[j] = module.cartesian_coordinates_from_grid_state(
                g[i + module.l]
            )
            i += module.l

        coordinates = torch.zeros(len(self.shapes[0]), device=self.device)
        for d in range(len(self.shapes[0])):
            coordinates[d] = chinese_remainder_theorem(
                self.shapes[:, d].int().cpu().numpy(),
                remainders[:, d].int().cpu().numpy(),
            )

        return coordinates

    @torch.no_grad()
    def cartesian_coordinates_from_grid_state_extended(
        self, g: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()

    @torch.no_grad()
    def _W_gh(self, noisy=False, noisy_std=1, Npatts=None) -> torch.Tensor:
        """Calculates the matrix of weights to go from the hippocampal layer to the grid layer heteroassociatively. Shape: `(N_g, N_h)`"""
        g_train = self.G[:Npatts] if Npatts is not None else self.G
        h_train = self.H[:Npatts] if Npatts is not None else self.H
        scale = Npatts if Npatts is not None else self.N_patts

        if noisy:
            h_train = h_train.clone().detach() + torch.normal(
                mean=0, std=noisy_std, size=h_train.shape, device=self.device
            )
        return torch.einsum("bi,bj->ij", g_train, h_train) / scale

    @torch.no_grad()
    def hippocampal_from_grid(self, G: torch.Tensor) -> torch.Tensor:
        """
        Input shape `(B, N_g)`

        Output shape `(B, N_h)`

        Args:
            G (torch.Tensor): Grid coding state tensor.
        """
        if G.ndim == 1:
            G = G.unsqueeze(0)
        return torch.relu(G @ self.W_hg.T - self.relu_theta)

    @torch.no_grad()
    def grid_from_hippocampal(self, H: torch.Tensor) -> torch.Tensor:
        """
        Input shape `(B, N_h)`

        Output shape `(B, N_g)`

        Args:
            H (torch.Tensor): Hippocampal state tensor.
        """
        if H.ndim == 1:
            H = H.unsqueeze(0)

        return H @ self.W_gh.T

    @torch.no_grad()
    def _W_gh(self, noisy=False, noisy_std=1, Npatts=None) -> torch.Tensor:
        """Calculates the matrix of weights to go from the hippocampal layer to the grid layer heteroassociatively. Shape: `(N_g, N_h)`"""
        g_train = self.G[:Npatts] if Npatts is not None else self.G
        h_train = self.H[:Npatts] if Npatts is not None else self.H
        scale = Npatts if Npatts is not None else self.N_patts

        if noisy:
            h_train = h_train.clone().detach() + torch.normal(
                mean=0, std=noisy_std, size=h_train.shape, device=self.device
            )
        return torch.einsum("bi,bj->ij", g_train, h_train) / scale

    @torch.no_grad()
    def shift(self, velocity):
        """Shifts the grid coding state by a given velocity.

        The length of `velocity` must be equal to the dimensionality of the grid modules.
        """
        for module in self.modules:
            module.shift(velocity)

        self.g = self._g()

    @torch.no_grad()
    def reset_g(self):
        """Reset the position of the scaffold to 0"""
        coordinates = self.cartesian_coordinates_from_grid_state(self.g)
        self.shift(-coordinates)

    @torch.no_grad()
    def denoise(self, G: torch.Tensor) -> torch.Tensor:
        """Denoise a batch of grid coding states.

        Input shape: `(B, N_g)`

        Output shape: `(B, N_g)`

        Args:
            G: Batch of grid coding states to denoise.
        """
        if G.ndim == 1:
            G = G.unsqueeze(0)

        pos = 0
        for module in self.modules:
            # for i in range(len(G)):
            #     print(G[i])
            x = G[:, pos : pos + module.l]
            # for i in range(len(x)):
            #     print(x[i])
            x_denoised = module.denoise_onehot(x)
            # print(x)
            # print(x_denoised)
            G[:, pos : pos + module.l] = x_denoised
            pos += module.l

        return G
