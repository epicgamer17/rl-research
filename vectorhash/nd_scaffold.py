import math
import torch
import numpy as np
from matrix_initializers import SparseMatrixBySparsityInitializer
from ratslam_velocity_shift import inject_activity
from vectorhash_functions import expectation_of_relu_normal
import matplotlib.pyplot as plt


def plot_recall_info(info):
    fig, ax = plt.subplots(1, 2, dpi=200, figsize=(4, 5))

    ax[0].imshow(info["G"].cpu().numpy(), cmap="gray", aspect="auto")
    ax[0].set_xlabel("N_g")
    ax[0].set_ylabel("N_patts")
    ax[0].title.set_text("G")

    ax[1].imshow(info["G_denoised"].cpu().numpy(), cmap="gray", aspect="auto")
    ax[1].set_xlabel("N_g")
    ax[1].set_ylabel("N_patts")
    ax[1].title.set_text("G_denoised")

    fig, ax = plt.subplots(2, 1, dpi=400, figsize=(5, 3))

    ax[0].imshow(info["H"].cpu().numpy(), cmap="gray", aspect="auto")
    ax[0].set_xlabel("N_h")
    ax[0].set_ylabel("N_patts")
    ax[0].title.set_text("H")

    ax[1].imshow(info["H_denoised"].cpu().numpy(), cmap="gray", aspect="auto")
    ax[1].set_xlabel("N_h")
    ax[1].set_ylabel("N_patts")
    ax[1].title.set_text("H_denoised")

    fig, ax = plt.subplots(2, 2, dpi=400, figsize=(5, 8))

    ax[0][0].imshow(info["H"][:50, :50].cpu().numpy(), cmap="gray", aspect="auto")
    ax[0][0].set_xlabel("N_patts")
    ax[0][0].set_ylabel("N_h")
    ax[0][0].title.set_text("H, first 50")

    ax[1][0].imshow(
        info["H_denoised"][:50, :50].cpu().numpy(), cmap="gray", aspect="auto"
    )
    ax[1][0].set_xlabel("N_patts")
    ax[1][0].set_ylabel("N_h")
    ax[1][0].title.set_text("H_denoised, first 50")

    ax[0][1].imshow(info["H"][:50, :50].cpu().numpy() == 0, cmap="gray", aspect="auto")
    ax[0][1].set_xlabel("N_patts")
    ax[0][1].set_ylabel("N_h")
    ax[0][1].title.set_text("H, first 50, zero locations")

    ax[1][1].imshow(
        1 - (info["H_denoised"][:50, :50].cpu().numpy() == 0),
        cmap="gray",
        aspect="auto",
    )
    ax[1][1].set_xlabel("N_patts")
    ax[1][1].set_ylabel("N_h")
    ax[1][1].title.set_text("H_denoised, first 50, zero locations")


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


class GridScaffold:
    def __init__(
        self,
        shapes: torch.Tensor,
        N_h: int,
        input_size: int,
        normal_mean: float,
        normal_std: float,
        device=None,
        sparse_matrix_initializer=None,
        relu_theta=0.5,
        from_checkpoint=False,
        T=1,
        continualupdate=False,
        ratshift=False,
        initialize_W_gh_with_zeroes=True,
        pseudo_inverse=False,
        batch_update=False,
        use_h_fix = True,
        learned_pseudo=True,
    ) -> None:
        self.shapes = torch.Tensor(shapes)
        self.input_size = input_size
        self.device = device
        self.relu_theta = relu_theta

        self.normal_mean = normal_mean
        self.normal_std = normal_std

        if continualupdate == False:
            assert (
                initialize_W_gh_with_zeroes == False
            ), "initialize_W_gh_with_zeroes must be False if continualupdate is False"
        self.pseudo_inverse = pseudo_inverse
        # FOR TESTING, CONTINUAL UPDATE TURNS ON UPDATE EVERY STEP
        # FOR TESTING, RATSHIFT TURNS ON RATSHIFT INSTEAD OF ROLL
        self.continualupdate = continualupdate
        self.ratshift = ratshift

        if from_checkpoint:
            return

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
        self.G = self._G()

        """The matrix of all possible grid states. Shape: `(N_patts, N_g)`"""

        if sparse_matrix_initializer is None:
            sparse_matrix_initializer = SparseMatrixBySparsityInitializer(
                sparsity=0.1, device=device
            )

        self.W_hg = sparse_matrix_initializer((self.N_h, self.N_g))

        """The matrix of weights to go from the grid layer to the hippocampal layer. Shape: `(N_h, N_g)`"""
        self.H = self.hippocampal_from_grid(self.G)  # (N_patts, N_h)
        """The matrix of all possible hippocampal states induced by `G` and `W_hg`. Shape: `(N_patts, N_h)`"""

        if initialize_W_gh_with_zeroes:
            self.W_gh = torch.zeros((self.N_g, self.N_h), device=device)
        else:
            self.W_gh = self._W_gh()  # (N_g, N_h)
        # print(self.W_gh)
        # print(self.W_hg)
        # assert torch.all(
        #    self.G
        #    == self.denoise(
        #        self.grid_from_hippocampal(self.hippocampal_from_grid(self.G))
        #    )
        # ), "G -> H -> G should preserve G"
        self.learned_pseudo = learned_pseudo
        self.W_sh = torch.zeros((self.input_size, self.N_h), device=device)
        self.W_hs = torch.zeros((self.N_h, self.input_size), device=device)
        # self.inhibition_matrix_sh = torch.eye(self.N_h, device=device) / self.input_size
        self.inhibition_matrix_hs = torch.eye(self.input_size, device=device) / self.N_h

        """The current grid coding state tensor. Shape: `(N_g)`"""
        self.g = self._g()
        ### testing S such that Whs = H @ S^-1
        self.S = torch.zeros((self.N_patts, self.input_size), device=device)

        # self.batch_update = batch_update
        self.use_h_fix = use_h_fix

        # if self.batch_update:
        #     self.W_sh_update = torch.zeros((self.input_size, self.N_h), device=device)
        #     self.W_hs_update = torch.zeros((self.N_h, self.input_size), device=device)
        #     self.update_count = 0
        #     self.update_interval = 20

        self.mean_h = expectation_of_relu_normal(self.normal_mean, self.normal_std)

    @torch.no_grad()
    def _G(self) -> torch.Tensor:
        """Calculates the matrix of all possible grid states. Shape: `(N_patts, N_g)`"""
        G = torch.zeros((self.N_patts, self.N_g), device=self.device)
        height = 0
        for module in self.modules:
            G[:, height : height + module.l] = torch.tile(
                torch.eye(module.l, device=self.device), (self.N_patts // module.l, 1)
            )
            height += module.l

        return G

    def _g(self) -> torch.Tensor:
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
    def _W_gh(self) -> torch.Tensor:
        """Calculates the matrix of weights to go from the hippocampal layer to the grid layer heteroassociatively. Shape: `(N_g, N_h)`"""
        return (
            torch.einsum("bi,bj->bij", self.G, self.H).sum(dim=0, keepdim=False)
            / self.N_h
        )

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
    def sensory_from_hippocampal(self, H: torch.Tensor) -> torch.Tensor:
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
    def hippocampal_from_sensory(self, S: torch.Tensor) -> torch.Tensor:
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
    def calculate_update(
        self, input: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        # input: (N)
        # output: (M)
        # M: (M x N)
        # Eg : Wgh = 1/Nh * sum_i (G_i * H_iT) (outer product)
        ret = (torch.einsum("j,i->ji", output, input)) / (
            self.N_h
            # torch.linalg.norm(input + 1e-10)
            # ** 2
        )
        return ret

    @torch.no_grad()
    def calculate_update_Wsh_fix(
        self, input: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        if self.use_h_fix:
            return self.calculate_update(input=input - self.mean_h, output=output)
        
        else:
            return self.calculate_update(input=input, output=output)

    @torch.no_grad()
    def calculate_update_Whs(
        self, input: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        # input: (N)
        # output: (M)
        # M: (M x N)
        s_plus = torch.linalg.pinv(self.S)
        ret = s_plus @ self.H

        return ret.T

    def learned_pseudo_inverse(self, input, output):
        # print("s", input.shape)
        # print("h", output.shape)
        b_k_hs = (input.T @ self.inhibition_matrix_hs.T) / (
            1 + input.T @ self.inhibition_matrix_hs @ input
        )
        # print("b_k_hs", b_k_hs.shape)
        self.inhibition_matrix_hs -= (self.inhibition_matrix_hs @ input) @ b_k_hs
        print("inhibition matrix", self.inhibition_matrix_hs)
        # print(self.inhibition_matrix_hs.shape)
        # print((self.W_hs @ input).shape)
        # print((output - self.W_hs @ input).shape)
        # print(((h - self.W_hs @ s) @ b_k_hs.T).shape)
        # self.W_hs += (h - self.W_hs @ s) @ b_k_hs.T
        return torch.outer(output - self.W_hs @ input, b_k_hs)

        # b_k_sh = (h.T @ self.inhibition_matrix_sh.T) / (
        #     1 + h.T @ self.inhibition_matrix_sh @ h
        # )
        # self.inhibition_matrix_sh -= (self.inhibition_matrix_sh @ h) @ b_k_sh
        # self.W_sh += (s - self.W_sh @ h) @ b_k_sh.T
        # self.W_sh += torch.outer(s - self.W_sh @ h, b_k_sh)

    @torch.no_grad()
    def calculate_update_Wsh(
        self, input: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        # input: (N)
        # output: (M)
        # M: (M x N)
        h_plus = torch.linalg.pinv(self.H)
        ret = h_plus @ self.S

        return ret.T

    @torch.no_grad()
    def store_memory(self, s: torch.Tensor):
        """Stores a memory in the scaffold.
        Input shape: `(input_size)`
        """

        # if self.batch_update:
        #     h = torch.relu(self.W_hg @ self.g - self.relu_theta)
        #     self.W_sh_update += self.calculate_update(input=h, output=s)
        #     self.W_hs_update += self.calculate_update(input=s, output=h)
        #     self.update_count += 1

        #     if self.update_count == self.update_interval:
        #         self.W_sh += self.W_sh_update
        #         self.W_hs += self.W_hs_update

        #         self.W_sh_update = torch.zeros(
        #             (self.input_size, self.N_h), device=self.device
        #         )
        #         self.W_hs_update = torch.zeros(
        #             (self.N_h, self.input_size), device=self.device
        #         )
        #         self.update_count = 0
        #     return

        # https://github.com/tmir00/TemporalNeuroAI/blob/c37e4d57d0d2d76e949a5f31735f902f4fd2c3c7/model/model.py#L55C1-L55C69
        # replaces first empyty row in S with s
        if self.S[0].sum() == 0:
            self.S[0] = s
        else:
            self.S[self.S.nonzero()[-1][0] + 1] = s
        h = torch.relu(self.W_hg @ self.g - self.relu_theta)

        if self.continualupdate:
            self.W_gh += self.calculate_update(input=h, output=self.g)

        if self.pseudo_inverse:
            self.W_sh = self.calculate_update_Wsh_fix(input=h, output=s)
            self.W_hs = self.calculate_update_Whs(input=s, output=h)
        elif self.learned_pseudo:
            print("learned pseudo")
            self.W_hs += self.learned_pseudo_inverse(input=s, output=h)
            print(self.W_hs)
            self.W_sh += self.calculate_update_Wsh_fix(input=h, output=s)

        else:
            self.W_sh += self.calculate_update_Wsh_fix(input=h, output=s)
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
    def denoise(self, G) -> torch.Tensor:
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

    @torch.no_grad()
    def learn_path(self, observations, velocities):
        """Add a path of observations to the memory scaffold. It is assumed that the observations are taken at each time step and the velocities are the velocities directly after the observations."""

        from collections import defaultdict

        seen_gs = set()
        seen_hs = set()
        for obs, vel in zip(observations, velocities):
            seen_gs.add(tuple(self.g.tolist()))
            seen_hs.add(torch.relu(self.W_hg @ self.g - self.relu_theta))
            self.learn(obs, vel)
        print("Unique Gs seen while learning:", len(seen_gs))
        print("Unique Hs seen while learning:", len(seen_hs))

    @torch.no_grad()
    def learn(self, observation, velocity):
        """Add a memory to the memory scaffold and shift the grid coding state by a given velocity.

        observation shape: `(input_size)`
        velocity shape: `(D)` where `D` is the dimensionality of the grid modules.
        """
        self.store_memory(observation)
        self.shift(velocity)

    @torch.no_grad()
    def recall(self, observations) -> torch.Tensor:
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
        used_Hs = set()
        for h in H:
            used_Hs.add(tuple(h.tolist()))
        print("Unique Hs seen while recalling:", len(used_Hs))

        G = self.grid_from_hippocampal(H)
        used_gs = set()
        for g in G:
            # print(g)
            used_gs.add(tuple(g.tolist()))
        print("Unique Gs seen while recalling (before denoising):", len(used_gs))
        G_ = self.denoise(G)
        used_G_s = set()
        for g in G_:
            used_G_s.add(tuple(g.tolist()))
        print("Unique Gs seen while recalling (after denoising):", len(used_G_s))
        H_ = self.hippocampal_from_grid(G_)
        used_H_s = set()
        for h in H_:
            used_H_s.add(tuple(h.tolist()))
        print("Unique Hs seen while recalling (after denoising):", len(used_H_s))
        if self.use_h_fix:
            H_ -= self.mean_h
        S_ = self.sensory_from_hippocampal(H_)
        H_nonzero = torch.sum(H != 0, 1).float()
        print("avg nonzero H:", torch.mean(H_nonzero).item())
        H__nonzero = torch.sum(H_ != 0, 1).float()
        print("avg nonzero H_denoised:", torch.mean(H__nonzero).item())

        # G = list of multi hot vectors
        # g = a multi hot vector (M one hot vectors)
        # print(G)

        # print("H:", H)
        # print("H_indexes:", H.nonzero())
        # print("G:", G)
        # print("G_indexes", G.nonzero())
        # print("G_:", G_)
        # print("G__indexes:", G_.nonzero())
        # print("G_[0]:", G_[0])
        # print("H__indexes:", H_.nonzero())
        # print("denoised_H:", H_)

        # info = {
        #     "avg_nonzero_H": torch.mean(H_nonzero).item(),
        #     "std_nonzero_H": torch.std(H_nonzero).item(),
        #     "avg_nonzero_H_denoised": torch.mean(H__nonzero).item(),
        #     "std_nonzero_H_denoised": torch.std(H__nonzero).item(),
        #     "H_indexes": H.nonzero(),
        #     "G_indexes": G.nonzero(),
        #     "G_denoised_indexes": G_.nonzero(),
        #     "H_denoised_indexes": H_.nonzero(),
        #     "H": H,
        #     "G": G,
        #     "G_denoised": G_,
        #     "H_denoised": H_,
        # }
        # plot_recall_info(info)
        return S_

    def temporal_recall(self, noisy_observations: torch.Tensor) -> torch.Tensor:
        # https://github.com/tmir00/TemporalNeuroAI/blob/c37e4d57d0d2d76e949a5f31735f902f4fd2c3c7/model/model.py#L113
        H_ = self.hippocampal_from_sensory(noisy_observations)
        S_ = self.sensory_from_hippocampal(H_)

        return S_

    def graph_scaffold(self):
        pass

    def plot_cans(self):
        pass
