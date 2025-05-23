import torch
import numpy as np
from matrix_initializers import SparseMatrixBySparsityInitializer
from smoothing import *
import copy
from shifts import *
from vectorhash_functions import (
    chinese_remainder_theorem,
    circular_mean,
    expand_distribution,
)
from tqdm import tqdm
from grid_module import GridModule
from smoothing import ArgmaxSmoothing


class GridHippocampalScaffold:
    def __init__(
        self,
        shapes: torch.Tensor,
        N_h: int,
        sparse_matrix_initializer=None,
        relu_theta=0.5,
        sanity_check=True,
        calculate_g_method="fast",
        smoothing=SoftmaxSmoothing(T=1e-3),
        shift_method: Shift = None,
        device=None,
        relu=True,
        limits=None,  # assumes 1-1 mapping in units
    ):
        assert calculate_g_method in ["hairpin", "fast", "spiral"]
        self.relu = relu
        self.device = device
        self.shapes = torch.Tensor(shapes).int()
        """(M, d) where M is the number of grid modules and d is the dimensionality of the grid modules."""
        self.relu_theta = relu_theta

        if shift_method == None:
            shift_method = RollShift(device)
        self.shift_method = shift_method

        self.modules = [
            GridModule(shape, device=device, smoothing=smoothing) for shape in shapes
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
            ), f"G -> H -> G should preserve G: {self.G}, {self.denoise(self.grid_from_hippocampal(self.hippocampal_from_grid(self.G)))}"

        self.g = self._g()
        """The current grid coding state tensor. Shape: `(N_g)`"""

        self.scale_factor = torch.ones(len(self.shapes[0]), device=self.device)
        """ `scale_factor[d]` is the amount to multiply by to convert "world units" into "grid units" """

        self.grid_limits = torch.ones(len(self.shapes[0]), device=self.device)
        for d in range(len(self.shapes[0])):
            self.grid_limits[d] = torch.prod(self.shapes[:, d]).item()

        if limits != None:
            for d in range(len(self.shapes[0])):
                self.scale_factor[d] = self.grid_limits[d] / limits[d]

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
                g[i : i + module.l]
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
        if self.relu:
            return torch.relu(G @ self.W_hg.T - self.relu_theta)
        else:
            return G @ self.W_hg.T - self.relu_theta

    @torch.no_grad()
    def modules_from_g(self, g: torch.Tensor) -> list[GridModule]:
        """
        Input shape: `(N_g)`

        Output: list of length `M`, where each entry corresponds with a grid module

        Args:
            g (torch.Tensor): Grid coding state tensor.
        """
        pos = 0
        module_states = []
        for module in self.modules:
            x = g[pos : pos + module.l]
            module_states.append(x.reshape(*module.shape))
            pos += module.l

        new_modules = copy.deepcopy(self.modules)
        for i, module in enumerate(new_modules):
            module.state = module_states[i].to(self.device)
        return new_modules
    
    @torch.no_grad()
    def additive_shift(self, new_g):
        """
        Adds the two distributions together and normalizes the result.
        """
        modules = self.modules_from_g(new_g)
        for i, module in enumerate(modules):
            module.state = module.state + self.modules[i].state
        self.modules = modules
        self._g()
        self.g = self.denoise(G=self.g, onehot=True)[0]
        self.modules= self.modules_from_g(self.g)

    @torch.no_grad()
    def multiplicative_shift(self, new_g):
        """
        Multiplies the two distributions together and normalizes the result.
        """
        modules = self.modules_from_g(new_g)
        for i, module in enumerate(modules):
            module.state = module.state * self.modules[i].state
        self.modules = modules
        self._g()
        self.g = self.denoise(G=self.g, onehot=True)[0]
        self.modules= self.modules_from_g(self.g)

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
        """Shifts the grid coding state by a given displacement. The velocity is given in
        "world coordinates" and scaled to "grid coordinates" using self.scale_factor

        The length of `velocity` must be equal to the dimensionality of the grid modules.
        """

        self.shift_method(self.modules, velocity * self.scale_factor)

        self.g = self._g()

    @torch.no_grad()
    def reset_g(self):
        """Reset the position of the scaffold to 0"""
        for module in self.modules:
            module.zero()

    @torch.no_grad()
    def denoise(self, G: torch.Tensor, onehot=True) -> torch.Tensor:
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
            if onehot:
                x_denoised = module.denoise_onehot(x)
            else:
                x_denoised = module.denoise(x)
            # print(x)
            # print(x_denoised)
            G[:, pos : pos + module.l] = x_denoised
            pos += module.l

        return G

    def estimate_certainty(self, limits: torch.Tensor = None, g=None):
        """Estimate, over each dimension d, P( |X_d - mu_d| < k_d limits[d]) where
        limits[d] is in "world coordinates" and not "grid coordinates" which will be
        converted to "grid coordinates"

        Args:
            differences (torch.Tensor): _description_
            g (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if g == None:
            modules = self.modules_from_g(self.g)
        else:
            modules = self.modules_from_g(g)

        if limits == None:
            limits = torch.ones(len(self.shapes[0]), device=self.device)

        sums = torch.zeros(len(self.shapes[0]))
        for dim in range(len(self.shapes[0])):
            marginals = [module.get_marginal(dim) for module in modules]
            v = expand_distribution(marginals)
            mean = circular_mean(
                v * torch.arange(0, len(v), device=self.device), len(v)
            )
            if mean > len(v) // 2:
                mean -= len(v)
            low = torch.ceil(mean - self.scale_factor[dim] * limits[dim])
            high = torch.floor(mean + self.scale_factor[dim] * limits[dim])
            indices = torch.arange(low, high + 1, device=self.device).int()
            sums[dim] = torch.sum(v[indices])
        return sums

    def expand_distribution(self, dim: int):
        marginals = [module.get_marginal(dim) for module in self.modules]
        v = expand_distribution(marginals)
        return v

    def get_mean_positions(self):
        """Return mean positions in "world coordinates"

        Returns torch.Tensor (x,y,θ)
        """
        means = torch.zeros(len(self.shapes[0]))
        for d in range(len(self.shapes[0])):
            v = self.expand_distribution(d)
            mean = (
                circular_mean(v * torch.arange(0, len(v), device=self.device), len(v))
                / self.scale_factor[d]
            )
            means[d] = mean
        return means

    def get_onehot(self, g=None):
        smoothing = ArgmaxSmoothing()
        pos = 0
        if g == None:
            g = self.g

        onehotted = torch.zeros_like(self.g)
        for module in self.modules:
            x = g[pos : pos + module.l]
            x_onehot = smoothing(x.unsqueeze(0)).squeeze()
            # print(x)
            # print(x_denoised)
            onehotted[pos : pos + module.l] = x_onehot
            pos += module.l

        return onehotted


def get_dim_distribution_from_g(s: GridHippocampalScaffold, g: torch.Tensor, dim: int):
    marginals = []
    i = 0
    for module in s.modules:
        marginals.append(
            module.marginal_from_state(dim, g[i : i + module.l].reshape(module.shape))
        )
    print(marginals)
    v = expand_distribution(marginals)
    return v
