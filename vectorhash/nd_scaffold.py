from tqdm import tqdm as tqdm
import math
import torch
import numpy as np
from matrix_initializers import SparseMatrixBySparsityInitializer
from ratslam_velocity_shift import inject_activity
from scipy.stats import norm
from vectorhash_functions import expectation_of_relu_normal, Rk1MrUpdate
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
        r = torch.einsum(einsum_str, *pdfs).flatten()
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


class GridScaffold:
    def __init__(
        self,
        shapes: torch.Tensor,
        N_h: int,
        input_size: int,
        h_normal_mean: float,
        h_normal_std: float,
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
        use_h_fix=True,
        learned_pseudo=True,
        epsilon=0.01,
        calculate_update_scaling_method="norm",
        MagicMath=False,
        sanity_check=True,
        calculate_g_method="fast",
        scaling_updates=True,
        dream_fix=None,
        ZeroTol=1e-5,
    ) -> None:
        assert calculate_update_scaling_method in ["norm", "n_h"]
        assert calculate_g_method in ["hairpin", "fast", "spiral"]
        self.calculate_update_scaling_method = calculate_update_scaling_method
        self.scaling_updates = scaling_updates
        print("UPDATE SCALING BY SCHWARZ ERROR", scaling_updates)
        self.T = T
        if use_h_fix:
            assert (
                calculate_update_scaling_method == "norm"
            ), "use_h_fix only makes sense with norm scaling"

        self.shapes = torch.Tensor(shapes).int()
        self.input_size = input_size
        self.device = device
        self.relu_theta = relu_theta
        self.ZeroTol = ZeroTol
        self.MagicMath = MagicMath
        self.dream_fix = dream_fix
        self.h_normal_mean = h_normal_mean
        self.h_normal_std = h_normal_std

        if continualupdate == False:
            assert (
                initialize_W_gh_with_zeroes == False
            ), "initialize_W_gh_with_zeroes must be False if continualupdate is False"
        self.pseudo_inverse = pseudo_inverse
        # FOR TESTING, CONTINUAL UPDATE TURNS ON UPDATE EVERY STEP
        # FOR TESTING, RATSHIFT TURNS ON RATSHIFT INSTEAD OF ROLL
        self.continualupdate = continualupdate
        self.ratshift = ratshift
        self.epsilon = epsilon
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
        self.G = self._G(method=calculate_g_method)

        """The matrix of all possible grid states. Shape: `(N_patts, N_g)`"""

        if sparse_matrix_initializer is None:
            sparse_matrix_initializer = SparseMatrixBySparsityInitializer(
                sparsity=0.1, device=device
            )

        self.W_hg = sparse_matrix_initializer((self.N_h, self.N_g))

        """The matrix of weights to go from the grid layer to the hippocampal layer. Shape: `(N_h, N_g)`"""
        self.H = self.hippocampal_from_grid(self.G)  # (N_patts, N_h)

        # print("cosine similarity between all pairs of H")
        # for h1 in range(len(self.H)):
        #     for h2 in range(len(self.H)):
        #         # print(self.H[h1])
        #         # print(self.H[h2])
        #         print(
        #             "h{} and h{}: {}".format(
        #                 h1,
        #                 h2,
        #                 torch.nn.functional.cosine_similarity(
        #                     self.H[h1].reshape(1, -1), self.H[h2].reshape(1, -1)
        #                 ),
        #             )
        #         )

        """The matrix of all possible hippocampal states induced by `G` and `W_hg`. Shape: `(N_patts, N_h)`"""

        if initialize_W_gh_with_zeroes:
            self.W_gh = torch.zeros((self.N_g, self.N_h), device=device)
        else:
            self.W_gh = self._W_gh()  # (N_g, N_h)
        # print(self.W_gh)
        # print(self.W_hg)
        if sanity_check:
            assert torch.all(
                self.G
                == self.denoise(
                    self.grid_from_hippocampal(self.hippocampal_from_grid(self.G))
                )
            ), "G -> H -> G should preserve G"
        self.learned_pseudo = learned_pseudo
        self.W_sh = torch.zeros((self.input_size, self.N_h), device=device)
        # self.W_hs = sparse_matrix_initializer((self.N_h, self.input_size))
        # self.W_sh = torch.linalg.pinv(self.W_hs)
        self.W_hs = torch.zeros((self.N_h, self.input_size), device=device)
        # self.inhibition_matrix_sh = torch.eye(self.N_h, device=device) / (self.epsilon**2)
        # self.inhibition_matrix_hs = torch.eye(self.input_size, device=device) / (self.epsilon**2)
        self.inhibition_matrix_sh = torch.eye(self.N_h, device=device) / (
            self.input_size
        )
        self.inhibition_matrix_hs = torch.eye(self.input_size, device=device) / (
            self.N_h
        )

        self.updatesh = 0
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

        self.mean_h = expectation_of_relu_normal(self.h_normal_mean, self.h_normal_std)

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
            # e.x. (N_g, 60, 60)

            for state in grid_states:
                i = 0
                for shape in self.shapes:
                    phis = torch.remainder(state, shape.to(self.device)).int()
                    gpattern = torch.zeros(tuple(shape.tolist()), device=self.device)
                    gpattern[(phis)] = 1
                    gpattern = gpattern.flatten()
                    gbook[i : i + len(gpattern), tuple(state)] = gpattern
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
    def _W_hs(self) -> torch.Tensor:
        """Calculates the matrix of weights to go from the hippocampal layer to the grid layer heteroassociatively. Shape: `(N_g, N_h)`"""
        return torch.einsum("bi,bj->ij", self.S, self.H) / self.N_h

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
        if self.calculate_update_scaling_method == "norm":
            scale = torch.linalg.norm(input) ** 2
        elif self.calculate_update_scaling_method == "n_h":
            scale = self.N_h
        else:
            raise ValueError("Invalid calculate_update_scaling_method")
        if self.scaling_updates:
            if output.shape==torch.Size([self.N_h]):
                output = output-self.hippocampal_from_sensory(input)[0]
            if output.shape==torch.Size([self.input_size]):
                output = output-self.sensory_from_hippocampal(input)[0]
        ret = torch.einsum("j,i->ji", output, input) / (scale + 1e-10)

        # print("input", input)
        # print("output", output)
        # print("scale", scale)
        # print("ret", ret)
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
        self, sbook: torch.Tensor, hbook: torch.Tensor
    ) -> torch.Tensor:
        # input: (N)
        # output: (M)
        # M: (M x N)
        s_plus = torch.linalg.pinv(sbook)
        ret = torch.einsum("ki,kj->ij", s_plus, hbook)

        return ret.T

    def learned_pseudo_inversehs(self, input, output):
        # print("s", input.shape)
        # print("h", output.shape)
        b_k_hs = (self.inhibition_matrix_hs @ input) / (
            1 + input.T @ self.inhibition_matrix_hs @ input
        )
        # ERROR VECTOR EK
        # E_k = output - self.W_hs @ input

        # NORMALIZATION FACTOR

        # E = ((E_k.T @ E_k)/self.input_size) / (1 + input.T @ self.inhibition_matrix_hs @ input)
        # L2Enorm = torch.abs(E)

        # GAMMA CALCULATION

        # gamma = 1/(1+ ((1-torch.exp(-L2Enorm))/self.epsilon))

        # print("b_k_hs", b_k_hs.shape)
        # self.inhibition_matrix_hs = gamma * (self.inhibition_matrix_hs - self.inhibition_matrix_hs * input @ b_k_hs.T + ((1-torch.exp(-L2Enorm))/self.epsilon) * torch.eye(self.input_size, device=self.device))
        self.inhibition_matrix_hs = (
            self.inhibition_matrix_hs - self.inhibition_matrix_hs * input @ b_k_hs.T
        )
        # print(self.inhibition_matrix_hs.shape)
        # print((self.W_hs @ input).shape)
        # print((output - self.W_hs @ input).shape)
        # print(((h - self.W_hs @ s) @ b_k_hs.T).shape)
        # self.W_hs += (h - self.W_hs @ s) @ b_k_hs.T
        self.W_hs += torch.outer((output - self.W_hs @ input), b_k_hs.T)

        # b_k_sh = (h.T @ self.inhibition_matrix_sh.T) / (
        #     1 + h.T @ self.inhibition_matrix_sh @ h
        # )
        # self.inhibition_matrix_sh -= (self.inhibition_matrix_sh @ h) @ b_k_sh
        # self.W_sh += (s - self.W_sh @ h) @ b_k_sh.T
        # self.W_sh += torch.outer(s - self.W_sh @ h, b_k_sh)

    def learned_pseudo_inversesh(self, input, output):
        # print("s", input.shape)
        # print("h", output.shape)
        b_k_sh = (self.inhibition_matrix_sh @ input) / (
            1 + input.T @ self.inhibition_matrix_sh @ input
        )
        # ERROR VECTOR EK
        # E_k = output - self.W_sh @ input

        # NORMALIZATION FACTOR

        # E = ((E_k.T @ E_k)/self.input_size) / (1 + input.T @ self.inhibition_matrix_sh @ input)
        # L2Enorm = torch.abs(E)

        # GAMMA CALCULATION

        # gamma = 1/(1+ ((1-torch.exp(-L2Enorm))/self.epsilon))

        # print("b_k_hs", b_k_hs.shape)
        # self.inhibition_matrix_sh = gamma * (self.inhibition_matrix_sh - self.inhibition_matrix_sh * input @ b_k_sh.T + ((1-torch.exp(-L2Enorm))/self.epsilon) * torch.eye(self.N_h, device=self.device))
        self.inhibition_matrix_sh = (
            self.inhibition_matrix_sh - self.inhibition_matrix_sh * input @ b_k_sh.T
        )
        # print(self.inhibition_matrix_hs.shape)
        # print((self.W_hs @ input).shape)
        # print((output - self.W_hs @ input).shape)
        # print(((h - self.W_hs @ s) @ b_k_hs.T).shape)
        # self.W_hs += (h - self.W_hs @ s) @ b_k_hs.T

        self.W_sh += torch.outer((output - self.W_sh @ input), b_k_sh.T)

        # b_k_sh = (h.T @ self.inhibition_matrix_sh.T) / (
        #     1 + h.T @ self.inhibition_matrix_sh @ h
        # )
        # self.inhibition_matrix_sh -= (self.inhibition_matrix_sh @ h) @ b_k_sh
        # self.W_sh += (s - self.W_sh @ h) @ b_k_sh.T
        # self.W_sh += torch.outer(s - self.W_sh @ h, b_k_sh)

    @torch.no_grad()
    def calculate_update_Wsh(
        self, hbook: torch.Tensor, sbook: torch.Tensor
    ) -> torch.Tensor:
        # input: (N)
        # output: (M)
        # M: (M x N)
        h_plus = torch.linalg.pinv(hbook)
        ret = torch.einsum("ik,jk->ij", sbook, h_plus)

        return ret

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
            # print("learned pseudo")
            self.learned_pseudo_inversehs(input=s, output=h)
            # print(self.W_hs)
            self.W_sh += self.calculate_update_Wsh_fix(input=h, output=s)
        elif self.MagicMath:
            self.W_hs = Rk1MrUpdate(self.W_sh, self.W_hs, c=s, d=h, Zero_tol=self.ZeroTol, Case_Print_Flag=0)
            self.W_sh += self.calculate_update_Wsh_fix(input=h, output=s)
        else:
            self.W_hs += self.calculate_update(input=s, output=h)
            self.W_sh += self.calculate_update_Wsh_fix(input=h, output=s)

        # assert torch.allclose(
        #     self.W_sh @ h, s
        # ), f"Wsh should be the pseudo-inverse of Whs. Got {self.W_sh @ ((h - self.mean_h) if self.use_h_fix else h)} and expected {s}"
        # assert torch.allclose(
        #     self.W_hs @ s, h
        # ), f"Whs should be the pseudo-inverse of Wsh. Got {self.W_hs @ s} and expected {h}"

        # print("S", s)
        # print("H", h)
        # print("HIPPO FROM S", torch.relu(self.W_hs @ s))
        # print("S FROM HIPPO", self.W_sh @ h)

    @torch.no_grad()
    def shift(self, velocity):
        """Shifts the grid coding state by a given velocity.

        The length of `velocity` must be equal to the dimensionality of the grid modules.
        """
        for module in self.modules:
            module.shift(velocity)

        self.g = self._g()

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

    def estimate_position(self, obs: torch.Tensor, as_tuple_list=False) -> torch.Tensor:
        g = self.grid_from_hippocampal(self.hippocampal_from_sensory(obs))
        onehotted = self.onehot(g.squeeze())

        if not as_tuple_list:
            return onehotted
        
        pos = 0
        onehotted_list = []
        for module in self.modules:
            onehotted_list.append(onehotted[pos : pos + module.l].reshape(module.shape))
            pos += module.l

        return onehotted_list

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
    def dream(self, seen_gss):
        i=0
        print("seen_gs", seen_gss)
        #reverse the order of seen_gs
        seen_gs = list(seen_gss)
        seen_gs.reverse()

        for g in seen_gs:
            # if i<self.dream_fix:
            self.g = torch.tensor(list(g))
            h = torch.relu(self.W_hg @ self.g - self.relu_theta)
            s = self.W_sh @ h
            dh = self.W_hs @ s - h
            if dh.norm() > self.ZeroTol:
                self.W_hs += self.calculate_update(input=s, output=h)
            # i+=1


    @torch.no_grad()
    def learn_path(self, observations, velocities):
        """Add a path of observations to the memory scaffold. It is assumed that the observations are taken at each time step and the velocities are the velocities directly after the observations."""

        seen_gs = set()
        seen_gs_recall = set()
        seen_g_s_recall = set()
        seen_hs = set()
        seen_hs_recall = set()
        first_g = self.g
        first_obs = observations[0]
        second_obs = observations[1]

        first_image_grid_position_estimates = []
        second_image_grid_position_estimates = []
        first_image_grid_positions = []
        second_image_grid_positions = []

        i = 0
        for obs, vel in tqdm(zip(observations, velocities)):
            seen_gs.add(tuple(self.g.tolist()))
            seen_hs.add(torch.relu(self.W_hg @ self.g - self.relu_theta))
            self.learn(obs, vel)

            # testing code
            first_image_grid_position_estimates.append(self.estimate_position(first_obs).flatten().clone())
            first_image_grid_positions.append(self.denoise(self.grid_from_hippocampal(self.hippocampal_from_sensory(first_obs))).flatten().clone())
            

            if i > 0:
                second_image_grid_position_estimates.append(self.estimate_position(second_obs).flatten().clone())
                second_image_grid_positions.append(self.denoise(self.grid_from_hippocampal(self.hippocampal_from_sensory(second_obs))).flatten().clone())
            if self.dream_fix != None:
                if (i+1)%self.dream_fix == 0:
                    self.dream(seen_gs)


            i += 1
        print("Unique Gs seen while learning:", len(seen_gs))
        print("Unique Hs seen while learning:", len(seen_hs))
        print("Unique Hs seen while recalling:", len(seen_hs_recall))
        print(
            "Unique Gs seen while recalling (right after learning):",
            len(seen_gs_recall),
        )
        print(
            "Unique Gs seen while recalling (right after learning, after denoising):",
            len(seen_g_s_recall),
        )
        seen_g_s = set()
        for g in seen_gs:
            # print(self.denoise(torch.tensor(list(g))))
            seen_g_s.add(tuple(self.denoise(torch.tensor(list(g)))))
        print("Unique Gs seen while learning (after denoising):", len(seen_g_s))
        return first_image_grid_position_estimates, second_image_grid_position_estimates, first_image_grid_positions, second_image_grid_positions



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
            # print(g)
            used_G_s.add(tuple(g.tolist()))
        print("Unique Gs seen while recalling (after denoising):", len(used_G_s))
        H_ = self.hippocampal_from_grid(G_)
        used_H_s = set()
        for h in H_:
            used_H_s.add(tuple(h.tolist()))
        print("Unique Hs seen while recalling (after denoising):", len(used_H_s))
        H_nonzero = torch.sum(H != 0, 1).float()
        print("avg nonzero H:", torch.mean(H_nonzero).item())
        H__nonzero = torch.sum(H_ != 0, 1).float()
        print("avg nonzero H_denoised:", torch.mean(H__nonzero).item())

        if self.use_h_fix:
            H_ -= self.mean_h
        S_ = self.sensory_from_hippocampal(H_)
        # print("H_", H_)
        # print("H_[0]", H_[0])
        # print("H_ mean", torch.mean(H_).item())

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
