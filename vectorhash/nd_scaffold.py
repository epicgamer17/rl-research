import torch
import numpy as np
from matrix_initializers import SparseMatrixBySparsityInitializer


class GridModule:
    def __init__(self, shape, device=None) -> None:
        self.shape = shape
        self.state = torch.zeros(shape, device=device)
        self.l = torch.prod(torch.tensor(shape)).item()

    def denoise(self):
        indices = torch.amax(self.state)
        self.state = 0
        self.state[indices] = 1

    def onehot(self):
        pdfs = list()
        dims = range(len(self.shape))
        for i in range(len(self.shape)):
            pdf = torch.sum(self.state, dim=[j for j in dims if j != i])
            pdf = pdf / torch.sum(pdf)
            pdfs.append(pdf)

        einsum_indices = [chr(ord('a') + i) for i in range(len(self.shape))] # a, b, c, ...
        einsum_str = ",".join(einsum_indices) + "->" + "".join(einsum_indices) # a,b,c, ...->abc...
        return torch.einsum(einsum_str, *pdfs).flatten()

    def shift(self, v):
        self.state = torch.roll(self.state, v, dims=range(len(self.shape)))


class GridScaffold:
    def __init__(
        self,
        shapes: torch.Tensor,
        N_h: int,
        input_size: int,
        device=None,
        sparse_matrix_initializer=None,
        relu_theta=0.5,
    ) -> None:
        self.shapes = torch.Tensor(shapes) 
        self.device = device
        self.relu_theta = relu_theta

        self.modules = [GridModule(shape, device=device) for shape in shapes]

        self.input_size = input_size
        self.N_g = sum([module.l for module in self.modules])
        self.N_patts = np.prod([module.l for module in self.modules]).item()
        self.N_h = N_h

        print("N_g     : ", self.N_g)  
        print("N_patts : ", self.N_patts)
        print("N_h     : ", self.N_h)

        self.sparse_matrix_initializer = sparse_matrix_initializer

        self.G = self._G()

        if sparse_matrix_initializer is None:
            sparse_matrix_initializer = SparseMatrixBySparsityInitializer(sparsity=0.9, device=device)

        self.W_hg = sparse_matrix_initializer((self.N_h, self.N_g))
        self.H = self.hippocampal_from_grid(self.G)  # (N_patts, N_h)
        # self.H = torch.relu(self.W_hg @ self.G - self.relu_theta)  # (N_h, N_patts)
        self.W_gh = self._W_gh()  # (N_g, N_h)
        self.W_sh = torch.zeros((self.input_size, self.N_h), device=device)
        self.W_hs = torch.zeros((self.N_h, self.input_size), device=device)

        self.relu_theta = relu_theta

    @torch.no_grad()
    def _G(self):
        G = torch.zeros((self.N_patts, self.N_g), device=self.device)
        height = 0
        for module in self.modules:
            G[:, height : height + module.l] = torch.tile(
                torch.eye(module.l, device=self.device), (self.N_patts // module.l, 1)
            )
            height += module.l

        return G

    def checkpoint(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path, device=None):
        return torch.load(path)

    @torch.no_grad()
    def _W_gh(self):
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

    def learn_path(self, observations, velocities):
        for obs, vel in zip(observations, velocities):
            self.learn(obs, vel)

    def learn(self, observation, velocity):
        pass

    def recall(self, observation):
        pass

    def graph_scaffold(self):
        pass

    def plot_cans(self):
        pass
