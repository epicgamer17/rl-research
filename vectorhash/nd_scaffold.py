import torch
import numpy as np
from matrix_initializers import SparseMatrixBySparsityInitializer


class GridModule:
    def __init__(self, shape: tuple[int], device=None) -> None:
        self.shape = shape
        # self.state = torch.rand(shape, device=device)
        self.state = torch.zeros(shape, device=device)
        self.state[0, 0, 0] = 1
        self.l = torch.prod(torch.tensor(shape)).item()

    def denoise_onehot(self, onehot: torch.Tensor):
        if onehot.ndim == 1:
            onehot = onehot.unsqueeze(0)
        """Denoise a batch of one-hot encoded states.

        Input shape: `(B, l)` where l is the product of the shape of the grid.

        Args:
            onehot: The tensor of one-hot encoded states.

        Output shape: `(B, l)`
        """

        state = onehot.view((onehot.shape[0], *self.shape))
        return self.denoise(state).flatten(1)

    def denoise(self, state: torch.Tensor):
        if state.ndim == len(self.shape):
            state = state.unsqueeze(0)

        dims = [i for i in range(1, len(self.shape)+1)]  # 1, 2, ..., n
        maxes = torch.amax(state, dim=dims, keepdim=True)
        state = torch.where(
            state == maxes, torch.ones_like(state), torch.zeros_like(state)
        )
        scaled = state / torch.sum(state, dim=dims, keepdim=True)
        return scaled

    def denoise_self(self):
        self.state = self.denoise(self.state)

    def onehot(self):
        pdfs = list()
        dims = range(len(self.shape))
        for i in range(len(self.shape)):
            s = torch.sum(self.state, dim=[j for j in dims if j != i])
            pdf = torch.nn.functional.softmax(s)
            pdfs.append(pdf)

        einsum_indices = [
            chr(ord("a") + i) for i in range(len(self.shape))
        ]  # a, b, c, ...
        einsum_str = (
            ",".join(einsum_indices) + "->" + "".join(einsum_indices)
        )  # a,b,c, ...->abc...
        return torch.einsum(einsum_str, *pdfs).flatten()

    def shift(self, v):
        self.state = torch.roll(self.state, (1,1,0), dims=tuple(i for i in range(len(self.shape))))


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
    ) -> None:
        self.shapes = torch.Tensor(shapes)
        self.input_size = input_size
        self.device = device
        self.relu_theta = relu_theta

        if from_checkpoint:
            return

        self.modules = [GridModule(shape, device=device) for shape in shapes]
        self.N_g = sum([module.l for module in self.modules])
        self.N_patts = np.prod([module.l for module in self.modules]).item()
        self.N_h = N_h

        print("module shapes: ", [module.shape for module in self.modules])
        print("N_g     : ", self.N_g)
        print("N_patts : ", self.N_patts)
        print("N_h     : ", self.N_h)

        self.sparse_matrix_initializer = sparse_matrix_initializer

        self.G = self._G()

        if sparse_matrix_initializer is None:
            sparse_matrix_initializer = SparseMatrixBySparsityInitializer(
                sparsity=0.9, device=device
            )

        self.W_hg = sparse_matrix_initializer((self.N_h, self.N_g))
        self.H = self.hippocampal_from_grid(self.G)  # (N_patts, N_h)
        self.W_gh = self._W_gh()  # (N_g, N_h)
        self.W_sh = torch.zeros((self.input_size, self.N_h), device=device)
        self.W_hs = torch.zeros((self.N_h, self.input_size), device=device)
        self.g = self._g()

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

    def _g(self):
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
        ret = (torch.einsum("i,j->ji", input, output)) / (torch.linalg.norm(input) ** 2 + 1e-8)
        return ret

    @torch.no_grad()
    def store_memory(self, s: torch.Tensor, num_iterations=1):
        """
        Stores sensory input s into the memory model by learning weights.

        :param s: Sensory input vector. - (input_size)
        :param num_iterations: Number of iterations for updating the weights.
        :return: None
        """
        # https://github.com/tmir00/TemporalNeuroAI/blob/c37e4d57d0d2d76e949a5f31735f902f4fd2c3c7/model/model.py#L55C1-L55C69
        for _ in range(num_iterations):
            print("g:", self.g)
            print("h:", s)

            h = torch.relu(self.W_hg @ self.g - self.relu_theta)

            self.W_gh += self.calculate_update(input=h, output=self.g)
            self.W_sh += self.calculate_update(input=h, output=s)
            self.W_hs += self.calculate_update(input=s, output=h)

    @torch.no_grad()
    def shift(self, velocity):
        for module in self.modules:
            module.shift(velocity)

    @torch.no_grad()
    def denoise(self, G):
        """

        Input shape: `(B, N_g)`

        Output shape: `(B, N_g)`

        Args:
            G (_type_): Batch of grid coding states to denoise.
        """
        if G.ndim == 1:
            G = G.unsqueeze(0)

        for i, module in enumerate(self.modules):
            G[:, i : i + module.l] = module.denoise_onehot(G[:, i : i + module.l])
        
        return G

    @torch.no_grad()
    def learn_path(self, observations, velocities):
        for obs, vel in zip(observations, velocities):
            self.learn(obs, vel)

    @torch.no_grad()
    def learn(self, observation, velocity):
        self.store_memory(observation)
        self.shift(velocity)
        pass

    @torch.no_grad()
    def recall(self, observations):
        # https://github.com/tmir00/TemporalNeuroAI/blob/c37e4d57d0d2d76e949a5f31735f902f4fd2c3c7/model/model.py#L96
        # noisy_observations: (N, input_size)
        H = self.hippocampal_from_sensory(observations)
        G = self.grid_from_hippocampal(H)
        G_ = self.denoise(G)
        H_ = self.hippocampal_from_grid(G_)
        S_ = self.sensory_from_hippocampal(H_)

        print("H:", H)
        print("G:", G)
        print("denoised_H:", H_)

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
