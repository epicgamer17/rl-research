import math
import torch
import numpy as np
from matrix_initializers import SparseMatrixBySparsityInitializer
from ratslam_velocity_shift import inject_activity
from vectorhash_functions import chinese_remainder_theorem, circular_mean_2
from tqdm import tqdm

class GridModule:
    def __init__(self, shape: tuple[int], device=None, T=1, ratshift=True) -> None:
        """Initializes a grid module with a given shape and a temperature.

        Args:
            shape (tuple[int]): The shape of the grid.
            T (int, optional): The temperature of the grid used with softmax when computing the onehot encoded state probabilities. Defaults to 1.
        """
        self.shape = shape
        # self.state = torch.rand(shape, device=device)
        self.state = torch.zeros(shape, device=device)
        self.ratshift = ratshift
        i = tuple(0 for _ in range(len(shape)))
        self.state[i] = 1
        self.l = torch.prod(torch.tensor(shape)).item()
        """The number of elements in the grid, i.e. the product of each element in the shape. Ex. a `3x3x4` grid has `l=36`."""
        self.T = T
        self.device = device

        einsum_indices = [
            chr(ord("a") + i) for i in range(len(self.shape))
        ]  # a, b, c, ...
        einsum_str = (
            ",".join(einsum_indices) + "->" + "".join(einsum_indices)
        )  # a,b,c, ...->abc...
        self.einsum_str = einsum_str
        """`a, b, c, ..., z -> abc...z`
        """

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
        dims = [i for i in range(1, len(self.shape) + 1)]  # 1, 2, ..., n
        maxes = torch.amax(state, dim=dims, keepdim=True)

        state = torch.where(
            state == maxes, torch.ones_like(state), torch.zeros_like(state)
        )
        scaled = state / torch.sum(state, dim=dims, keepdim=True)
        return scaled

        # Check if there are any non-zero elements in the module
        # if len(torch.nonzero(state)) > 0:
        #     max_value = np.max(state)  # Find the maximum value in the module
        #     print(max_value)
        #     # Create a binary array where positions with the max value are 1, others are 0
        #     max_positions = np.where(state == max_value)[0]
        #     random_position = np.random.choice(max_positions)
        #     denoised_module = np.zeros_like(state)
        #     denoised_module[random_position] = 1
        # else:
        #     # If the module is all zeros, create an array of the same shape with all zeros
        #     denoised_module = np.zeros_like(state)
        # return torch.tensor(denoised_module, device=self.device)

    def denoise_self(self):
        """Denoises this grid module's state"""
        self.state = self.denoise(self.state).squeeze(0)

    def onehot(self) -> torch.Tensor:
        """Returns the one-hot encoding of the state of this grid module.

        Output shape: `(l)` where `l` is the product of the shape of the grid (i.e. a 3x3x4 grid has l=36).
        """
        pdfs = list()
        dims = range(len(self.shape))
        for i in range(len(self.shape)):
            pdf = torch.sum(self.state, dim=[j for j in dims if j != i])
            pdfs.append(pdf)

        #
        r = torch.einsum(self.einsum_str, *pdfs).flatten()
        return (r / (self.T * r.sum(dim=0))).softmax(dim=0)

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

    def grid_state_from_cartesian_coordinates(self, coordinates: torch.Tensor):
        phis = torch.remainder(coordinates, torch.Tensor(self.shape)).int()
        gpattern = torch.zeros_like(self.state)
        gpattern[tuple(phis)] = 1
        gpattern = gpattern.flatten()
        return gpattern

    def cartesian_coordinates_from_grid_state(self, g: torch.Tensor) -> torch.Tensor:
        reshaped = g.view(*self.shape).nonzero()
        return reshaped

    def grid_state_from_cartesian_coordinates_extended(self, coordinates: torch.Tensor):
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

    def cartesian_coordinates_from_grid_state_extended(
        self, g: torch.Tensor
    ) -> torch.Tensor:
        sums = list()
        dims = range(len(self.shape))
        for i in range(len(self.shape)):
            dim = [j for j in dims if j != i]
            pdf = torch.sum(g.reshape(*self.shape), dim=dim)
            sums.append(pdf)

        coordinates = torch.zeros(len(self.shape))
        for i, l in enumerate(self.shape):
            w = sums[i]
            print(f"w[{i}]", w)
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
        T=1e-3,
        device=None,
    ):
        assert calculate_g_method in ["hairpin", "fast", "spiral"]

        self.device = device
        self.T = T
        self.shapes = torch.Tensor(shapes).int()
        """(M, d) where M is the number of grid modules and d is the dimensionality of the grid modules."""
        self.relu_theta = relu_theta
        self.ratshift = ratshift
        self.modules = [
            GridModule(shape, device=device, T=T, ratshift=ratshift) for shape in shapes
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

    def _g(self) -> torch.Tensor:
        """Calculates the current grid coding state tensor. Shape: `(N_g)`"""
        vecs = list()
        for module in self.modules:
            vecs.append(module.onehot())
        return torch.cat(vecs)

    def grid_state_from_cartesian_coordinates(
        self, coordinates: torch.Tensor
    ) -> torch.Tensor:
        """
        Input shape: `(d)` where `d` is the dimensionality of the points.

        Output shape: `(N_g)` where `N_g` is the number of grid cells.
        """
        g = torch.zeros(self.N_g, device=self.device)
        i = 0
        for shape in self.shapes:
            phis = torch.remainder(coordinates, shape.to(self.device)).int()
            gpattern = torch.zeros(tuple(shape.tolist()), device=self.device)
            gpattern[tuple(phis)] = 1
            gpattern = gpattern.flatten()
            g[i : i + len(gpattern)] = gpattern
            i += len(gpattern)

        return g

    def cartesian_coordinates_from_grid_state(self, g: torch.Tensor) -> torch.Tensor:
        """
        Input shape: `(N_g)` where `N_g` is the number of grid cells.

        Output shape: `(d)` where `d` is the dimensionality of the points.
        """

        remainders = torch.zeros(
            (len(self.shapes), len(self.shapes[0])), device=self.device
        )
        i = 0
        for j, shape in enumerate(self.shapes):
            l = torch.prod(torch.tensor(shape)).item()
            gpattern: torch.Tensor = g[i : i + l].clone().detach().view(tuple(shape))
            remainder = gpattern.nonzero()
            remainders[j] = remainder
            i += l

        coordinates = torch.zeros(len(self.shapes[0]), device=self.device)
        for d in range(len(self.shapes[0])):
            coordinates[d] = chinese_remainder_theorem(
                self.shapes[:, d].int().cpu().numpy(),
                remainders[:, d].int().cpu().numpy(),
            )

        return coordinates

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
        """Reset the position of the scaffold to 0
        """
        coordinates = self.cartesian_coordinates_from_grid_state(self.g)
        self.shift(-coordinates)

    @torch.no_grad()
    def onehot(self, g: torch.Tensor) -> torch.Tensor:
        """Returns the one-hot encoding of a given grid state.

        Input shape: `(N_g)`

        Args:
            G: The tensor of grid states.

        Output shape: `(N_g)`
        """
        pos = 0
        for module in self.modules:
            x = g[pos : pos + module.l]
            x_onehot = (x / (x.sum() * self.T)).softmax(dim=0)
            # print(x_onehot)
            g[pos : pos + module.l] = x_onehot
            pos += module.l

        return g

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