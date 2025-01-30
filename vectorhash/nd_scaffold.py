import torch
import numpy as np
from matrix_initializers import SparseMatrixBySparsityInitializer


class GridModule:
    def __init__(self, shape: tuple[int], device=None, T=1) -> None:
        """Initializes a grid module with a given shape and a temperature.

        Args:
            shape (tuple[int]): The shape of the grid.
            T (int, optional): The temperature of the grid used with softmax when computing the onehot encoded state probabilities. Defaults to 1.
        """
        self.shape = shape
        # self.state = torch.rand(shape, device=device)
        self.state = torch.zeros(shape, device=device)

        i = tuple(0 for _ in range(len(shape)))
        self.state[i] = 1
        self.l = torch.prod(torch.tensor(shape)).item()
        """The number of elements in the grid, i.e. the product of each element in the shape. Ex. a `3x3x4` grid has `l=36`."""
        self.T = T
        self.device = device

    def denoise_onehot(self, onehot: torch.Tensor):
        """Denoise a batch of one-hot encoded states.

        Input shape: `(B, l)` where l is the product of the shape of the grid.

        Args:
            onehot: The tensor of one-hot encoded states.

        Output shape: `(B, l)`
        """
        if onehot.ndim == 1:
            onehot = onehot.unsqueeze(0)

        state = onehot.view((onehot.shape[0], *self.shape))
        return self.denoise(state).flatten(1)

    def denoise(self, state: torch.Tensor):
        """Denoise a batch of grid states. This finds the maximum value in the grid and sets it to 1, and all other values to 0.
        If there are multiple maximum values, they are all set to 1 / number of maximum values.

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

    def denoise_self(self):
        """Denoises this grid module's state"""
        self.state = self.denoise(self.state).squeeze(0)

    def onehot(self):
        """Returns the one-hot encoding of the state of this grid module.

        Output shape: `(l)` where `l` is the product of the shape of the grid (i.e. a 3x3x4 grid has l=36).
        """
        pdfs = list()
        dims = range(len(self.shape))
        for i in range(len(self.shape)):
            pdf = torch.sum(self.state, dim=[j for j in dims if j != i])
            pdfs.append(pdf)

        einsum_indices = [
            chr(ord("a") + i) for i in range(len(self.shape))
        ]  # a, b, c, ...
        einsum_str = (
            ",".join(einsum_indices) + "->" + "".join(einsum_indices)
        )  # a,b,c, ...->abc...
        return (torch.einsum(einsum_str, *pdfs).flatten() / self.T).softmax(dim=0)

    def shift(self, v: torch.Tensor):
        """Shifts the state of the grid module by a given velocity.

        Input shape: `(len(shape))`

        Args:
            v: The velocity by which to shift the grid state.
        """
        v_ = v.int()
        self.state = torch.roll(
            self.state,
            tuple([v_[i].item() for i in range(len(v_))]),
            dims=tuple(i for i in range(len(self.shape))),
        )


class GridScaffold:
    def __init__(
        self,
        shapes: torch.Tensor,
        N_h: int,
        input_size: int,
        device=None,
        sparse_matrix_initializer=None,
        relu_theta=0.5,
        from_checkpoint=False,
        T=1,
    ) -> None:
        self.shapes = torch.Tensor(shapes)
        self.input_size = input_size
        self.device = device
        self.relu_theta = relu_theta

        if from_checkpoint:
            return

        self.modules = [GridModule(shape, device=device, T=T) for shape in shapes]
        """The list of grid modules in the scaffold."""
        self.N_g = sum([module.l for module in self.modules])
        self.N_patts = np.prod([module.l for module in self.modules]).item()
        self.N_h = N_h

        print("module shapes: ", [module.shape for module in self.modules])
        print("N_g     : ", self.N_g)
        print("N_patts : ", self.N_patts)
        print("N_h     : ", self.N_h)

        self.sparse_matrix_initializer = sparse_matrix_initializer

        self.G = self._G()
        """The matrix of all possible grid states. Shape: `(N_patts, N_g)`"""

        if sparse_matrix_initializer is None:
            sparse_matrix_initializer = SparseMatrixBySparsityInitializer(
                sparsity=0.9, device=device
            )

        self.W_hg = sparse_matrix_initializer((self.N_h, self.N_g))
        """The matrix of weights to go from the grid layer to the hippocampal layer. Shape: `(N_h, N_g)`"""

        self.H = self.hippocampal_from_grid(self.G)  # (N_patts, N_h)
        """The matrix of all possible hippocampal states induced by `G` and `W_hg`. Shape: `(N_patts, N_h)`"""

        self.W_gh = self._W_gh()  # (N_g, N_h)
        self.W_sh = torch.zeros((self.input_size, self.N_h), device=device)
        self.W_hs = torch.zeros((self.N_h, self.input_size), device=device)

        """The current grid coding state tensor. Shape: `(N_g)`"""
        self.g = self._g()

    @torch.no_grad()
    def _G(self):
        """Calculates the matrix of all possible grid states. Shape: `(N_patts, N_g)`"""
        G = torch.zeros((self.N_patts, self.N_g), device=self.device)
        height = 0
        for module in self.modules:
            G[:, height : height + module.l] = torch.tile(
                torch.eye(module.l, device=self.device), (self.N_patts // module.l, 1)
            )
            height += module.l

        return G

    def _g(self):
        """Calculates the current grid coding state tensor. Shape: `(N_g)`"""
        vecs = list()
        for module in self.modules:
            vecs.append(module.onehot())
        return torch.cat(vecs)

    def checkpoint(self, path):
        checkpoint = {
            "shapes": self.shapes,
            "input_size": self.input_size,
            "device": self.device,
            "relu_theta": self.relu_theta,
            "N_g": self.N_g,
            "N_patts": self.N_patts,
            "N_h": self.N_h,
            "W_hg": self.W_hg,
            "W_gh": self.W_gh,
            "W_sh": self.W_sh,
            "W_hs": self.W_hs,
            "N_h": self.N_h,
            "G": self.G,
            "H": self.H,
            "grid_modules": self.modules,
            "g": self._g(),
        }
        torch.save(checkpoint, path)

    @staticmethod
    def load(path, device=None):
        checkpoint = torch.load(path)
        scaffold = GridScaffold(
            shapes=checkpoint["shapes"],
            input_size=checkpoint["input_size"],
            relu_theta=checkpoint["relu_theta"],
            from_checkpoint=True,
            device=device,
        )
        scaffold.modules = checkpoint["grid_modules"]
        scaffold.N_g = checkpoint["N_g"]
        scaffold.N_patts = checkpoint["N_patts"]
        scaffold.N_h = checkpoint["N_h"]
        scaffold.W_hg = checkpoint["W_hg"]
        scaffold.W_gh = checkpoint["W_gh"]
        scaffold.W_sh = checkpoint["W_sh"]
        scaffold.W_hs = checkpoint["W_hs"]
        scaffold.G = checkpoint["G"]
        scaffold.H = checkpoint["H"]
        scaffold.g = checkpoint["g"]
        return scaffold

    @torch.no_grad()
    def _W_gh(self):
        """Calculates the matrix of weights to go from the hippocampal layer to the grid layer heteroassociatively. Shape: `(N_g, N_h)`"""
        return (
            torch.einsum("bi,bj->bij", self.G, self.H).sum(dim=0, keepdim=False)
            / self.N_h
        )

    @torch.no_grad()
    def hippocampal_from_grid(self, G: torch.Tensor):
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
    def grid_from_hippocampal(self, H: torch.Tensor):
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
    def sensory_from_hippocampal(self, H: torch.Tensor):
        """
        Input shape `(B, N_h)`

        Output shape `(B, input_size)`

        Args:
            H (torch.Tensor): Hippocampal state tensor.
        """
        if H.ndim == 1:
            H = H.unsqueeze(0)

        return H @ self.W_sh.T

    @torch.no_grad()
    def hippocampal_from_sensory(self, S: torch.Tensor):
        """
        Input shape: `(B, input_size)`

        Output shape: `(B, N_h)`

        Args:
            S (torch.Tensor): Sensory input tensor.

        """
        if S.ndim == 1:
            S = S.unsqueeze(0)

        return torch.relu(S @ self.W_hs.T)

    @torch.no_grad()
    def calculate_update(self, input: torch.Tensor, output: torch.Tensor):
        # input: (N)
        # output: (M)
        # M: (M x N)
        ret = (torch.einsum("i,j->ji", input, output)) / (
            torch.linalg.norm(input) ** 2 + 1e-8
        )
        return ret

    @torch.no_grad()
    def store_memory(self, s: torch.Tensor):
        """Stores a memory in the scaffold.
        Input shape: `(input_size)`
        """
        # https://github.com/tmir00/TemporalNeuroAI/blob/c37e4d57d0d2d76e949a5f31735f902f4fd2c3c7/model/model.py#L55C1-L55C69
        h = torch.relu(self.W_hg @ self.g - self.relu_theta)

        self.W_gh += self.calculate_update(input=h, output=self.g)
        self.W_sh += self.calculate_update(input=h, output=s)
        self.W_hs += self.calculate_update(input=s, output=h)

    @torch.no_grad()
    def shift(self, velocity):
        """Shifts the grid coding state by a given velocity.

        The length of `velocity` must be equal to the dimensionality of the grid modules.
        """
        for module in self.modules:
            module.shift(velocity)

        self.g = self._g()

    @torch.no_grad()
    def denoise(self, G):
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
            G[:, pos : pos + module.l] = module.denoise_onehot(G[:, pos : pos + module.l])
            pos += module.l

        return G

    @torch.no_grad()
    def learn_path(self, observations, velocities):
        """Add a path of observations to the memory scaffold. It is assumed that the observations are taken at each time step and the velocities are the velocities directly after the observations."""

        from collections import defaultdict
        # i =0
        seen = defaultdict(lambda : defaultdict(int)) # debugging
        for obs, vel in zip(observations, velocities):
            self.learn(obs, vel)
            indexes = torch.isclose(self.g, self.g.max()).nonzero().flatten()
            if seen[indexes[0].item()][indexes[1].item()] > 0:
                print("Seen", indexes, "count:", seen[indexes[0].item()][indexes[1].item()])
            seen[indexes[0].item()][indexes[1].item()] += 1
            # if i % 100 == 0:
            #     print(indexes, "count:", seen[indexes[0].item()][indexes[1].item()])

    @torch.no_grad()
    def learn(self, observation, velocity):
        """Add a memory to the memory scaffold and shift the grid coding state by a given velocity.

        observation shape: `(input_size)`
        velocity shape: `(D)` where `D` is the dimensionality of the grid modules.
        """
        self.store_memory(observation)
        self.shift(velocity)

    @torch.no_grad()
    def recall(self, observations):
        """Recall a (batch of) sensory input(s) from the scaffold.

        Input shape: `(B, input_size)` where `B` is the batch size.
        Output shape: `(B, input_size)` where `B` is the batch size.

        Args:
            observations (torch.Tensor): The tensor of batched sensory inputs to recall

        Returns:
            The tensor of batched sensory inputs recalled from the scaffold.
        """
        # https://github.com/tmir00/TemporalNeuroAI/blob/c37e4d57d0d2d76e949a5f31735f902f4fd2c3c7/model/model.py#L96
        # noisy_observations: (N, input_size)
        H = self.hippocampal_from_sensory(observations)
        G = self.grid_from_hippocampal(H)
        G_ = self.denoise(G)
        H_ = self.hippocampal_from_grid(G_)
        S_ = self.sensory_from_hippocampal(H_)

        # print("H:", H)
        # print("G:", G)
        # print("G_:", G_)
        # print("G_[0]:", G_[0])
        # print("denoised_H:", H_)

        return S_

    def temporal_recall(self, noisy_observations: torch.Tensor):
        # https://github.com/tmir00/TemporalNeuroAI/blob/c37e4d57d0d2d76e949a5f31735f902f4fd2c3c7/model/model.py#L113
        H_ = self.hippocampal_from_sensory(noisy_observations)
        S_ = self.sensory_from_hippocampal(H_)

        return S_

    def graph_scaffold(self):
        pass

    def plot_cans(self):
        pass
