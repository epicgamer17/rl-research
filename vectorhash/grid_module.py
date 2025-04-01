import torch
from smoothing import SoftmaxSmoothing
from vectorhash_functions import outer

class GridModule:
    def __init__(
        self,
        shape: tuple[int],
        smoothing=SoftmaxSmoothing(T=1e-3),
        device=None,
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
        i = tuple(0 for _ in range(len(shape)))
        self.state[i] = 1
        self.l = torch.prod(torch.tensor(shape)).item()
        """The number of elements in the grid, i.e. the product of each element in the shape. Ex. a `3x3x4` grid has `l=36`."""
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

    @torch.no_grad()
    def zero(self):
        self.state = torch.zeros(self.shape, device=self.device)
        i = tuple(0 for _ in range(len(self.shape)))
        self.state[i] = 1

    # note that the two below methods assume that the distributions across each dimension of the module are independent
    @torch.no_grad()
    def get_marginal(self, dim):
        return self.marginal_from_state(dim, self.state)
    
    @torch.no_grad()
    def state_from_marginals(self, marginals):
        return outer(marginals)

    def marginal_from_state(self, dim, state):
        dims = range(len(self.shape))
        dim_to_sum = tuple(j for j in dims if not j == dim)
        return torch.sum(state, dim=dim_to_sum)

    @torch.no_grad()
    def onehot(self) -> torch.Tensor:
        """Get the current state as a flattened onehot vector (normalized)"""
        return self.state.flatten() / self.state.sum()
        # return self.state.flatten()

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
        raise NotImplemented
        # sums = list()
        # dims = range(len(self.shape))
        # reshaped = g.reshape(*self.shape)

        # for dim in dims:
        #     sums.append(self.sum_marginal(dim, reshaped))

        # coordinates = torch.zeros(len(self.shape))
        # for i, l in enumerate(self.shape):
        #     w = sums[i]
        #     # print(f"w[{i}]", w)
        #     coordinates[i] = circular_mean_2(torch.arange(l), w, l).item()

        # return coordinates
