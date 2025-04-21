import torch


class HippocampalSensoryLayer:
    def __init__(
        self,
        input_size: int,
        N_h: int,
        device=None,
    ):
        self.input_size = input_size
        self.N_h = N_h
        self.device = device

    def hippocampal_from_sensory(self, S: torch.Tensor) -> torch.Tensor:
        """
        Input shape: `(B, input_size)`

        Output shape: `(B, N_h)`

        Args:
            S (torch.Tensor): Sensory input tensor.

        """
        pass

    def sensory_from_hippocampal(self, H: torch.Tensor) -> torch.Tensor:
        """
        Input shape `(B, N_h)`

        Output shape `(B, input_size)`

        Args:
            H (torch.Tensor): Hippocampal state tensor.
        """
        pass

    def learn(self, h: torch.Tensor, s: torch.Tensor):
        """
        Associate a sensory input to a hippocampal fixed point.

        `h`: The hippocampal fixed point
        `s`: The sensory input
        """
        pass


class IterativeBidirectionalPseudoInverseHippocampalSensoryLayer(
    HippocampalSensoryLayer
):
    def __init__(
        self,
        input_size: int,
        N_h: int,
        hidden_layer_factor: int,
        stationary=True,
        epsilon_sh=None,
        epsilon_hs=None,
        device=None,
        relu=False,
    ):
        super().__init__(input_size, N_h, device)

        self.hidden_layer_factor = hidden_layer_factor
        self.stationary = stationary
        self.epsilon_hs = epsilon_hs
        self.epsilon_sh = epsilon_sh
        self.relu = relu
        hidden_size_sh = self.N_h * self.hidden_layer_factor
        if hidden_size_sh == 0:
            hidden_size_sh = self.N_h
        else:
            self.hidden_sh = torch.rand((hidden_size_sh, self.N_h), device=device) - 0.5
        self.W_sh = torch.zeros((self.input_size, hidden_size_sh), device=device)

        if epsilon_sh == None:
            self.epsilon_sh = hidden_size_sh
        else:
            self.epsilon_sh = epsilon_sh
        self.inhibition_matrix_sh = torch.eye(hidden_size_sh, device=device) / (
            self.epsilon_sh**2
        )

        hidden_size_hs = self.input_size * self.hidden_layer_factor
        if hidden_size_hs == 0:
            hidden_size_hs = self.input_size
        else:
            self.hidden_hs = (
                torch.rand((hidden_size_hs, self.input_size), device=device) - 0.5
            )
        self.W_hs = torch.zeros((self.N_h, hidden_size_hs), device=device)
        if epsilon_hs == None:
            self.epsilon_hs = hidden_size_hs
        else:
            self.epsilon_hs = epsilon_hs
        self.inhibition_matrix_hs = torch.eye(hidden_size_hs, device=device) / (
            self.epsilon_hs**2
        )

    @torch.no_grad()
    def learned_pseudo_inverse_hs(self, input, output):
        if self.stationary:
            b_k_hs = (self.inhibition_matrix_hs @ input) / (
                1 + input.T @ self.inhibition_matrix_hs @ input
            )

            self.inhibition_matrix_hs = (
                self.inhibition_matrix_hs
                - self.inhibition_matrix_hs @ torch.outer(input, b_k_hs.T)
            )

            self.W_hs += torch.outer((output - self.W_hs @ input), b_k_hs.T)
        else:
            b_k_hs = (self.inhibition_matrix_hs @ input) / (
                1 + input.T @ self.inhibition_matrix_hs @ input
            )
            # ERROR VECTOR EK
            e_k = output - self.W_hs @ input

            # NORMALIZATION FACTOR
            E = ((e_k.T @ e_k) / self.inhibition_matrix_hs.shape[0]) / (
                1 + input.T @ self.inhibition_matrix_hs @ input
            )
            # E = torch.abs(E)

            # GAMMA CALCULATION
            gamma = 1 / (1 + ((1 - torch.exp(-E)) / self.epsilon_hs))

            self.inhibition_matrix_hs = gamma * (
                self.inhibition_matrix_hs
                - self.inhibition_matrix_hs @ torch.outer(input, b_k_hs.T)
                + ((1 - torch.exp(-E)) / self.epsilon_hs)
                * torch.eye(self.inhibition_matrix_hs.shape[0], device=self.device)
            )
            self.W_hs += torch.outer((output - self.W_hs @ input), b_k_hs.T)

    @torch.no_grad()
    def learned_pseudo_inverse_sh(self, input, output):
        if self.stationary:
            b_k_sh = (self.inhibition_matrix_sh @ input) / (
                1 + input.T @ self.inhibition_matrix_sh @ input
            )

            self.inhibition_matrix_sh = (
                self.inhibition_matrix_sh
                - self.inhibition_matrix_sh @ torch.outer(input, b_k_sh.T)
            )

            self.W_sh += torch.outer((output - self.W_sh @ input), b_k_sh.T)
        else:
            # (N_h, N_h) x (N_h, 1) / (1 + (1, N_h) x (N_h, N_h) x (N_h, 1)) = (N_h, 1)
            b_k_sh = (self.inhibition_matrix_sh @ input) / (
                1 + input.T @ self.inhibition_matrix_sh @ input
            )

            # (784, 1) - (784, N_h) x (N_h, 1) = (784, 1)
            e_k = output - self.W_sh @ input

            # ((1, 784) x (784, 1) / (1)) / ((1, N_h) x (N_h, N_h) x (N_h x 1))
            E = ((e_k.T @ e_k) / self.inhibition_matrix_sh.shape[0]) / (
                1 + input.T @ self.inhibition_matrix_sh @ input
            )
            # E = torch.abs(E)

            # scalar
            gamma = 1 / (1 + ((1 - torch.exp(-E)) / self.epsilon_sh))

            # (N_h, N_h) - (N_h, N_h) x (N_h, 1) x (1, N_h) + scalar * (N_h, N_h) = (N_h, N_h)
            self.inhibition_matrix_sh = gamma * (
                self.inhibition_matrix_sh
                - self.inhibition_matrix_sh @ torch.outer(input, b_k_sh.T)
                + ((1 - torch.exp(-E)) / self.epsilon_sh)
                * torch.eye(self.inhibition_matrix_sh.shape[0], device=self.device)
            )
            self.W_sh += torch.outer((output - self.W_sh @ input), b_k_sh.T)

    @torch.no_grad()
    def learn(self, h, s):
        self.learned_pseudo_inverse_hs(
            input=(
                torch.sigmoid(self.hidden_hs @ s)
                if self.hidden_layer_factor != 0
                else s
            ),
            output=h,
        )
        self.learned_pseudo_inverse_sh(
            input=(
                torch.sigmoid(self.hidden_sh @ h)
                if self.hidden_layer_factor != 0
                else h
            ),
            output=s,
        )

    @torch.no_grad()
    def sensory_from_hippocampal(self, H):
        if H.ndim == 1:
            H = H.unsqueeze(0)

        if self.hidden_layer_factor != 0:
            hidden = torch.sigmoid(H @ self.hidden_sh.T)
            return hidden @ self.W_sh.T
        else:
            return H @ self.W_sh.T

    @torch.no_grad()
    def hippocampal_from_sensory(self, S):
        if S.ndim == 1:
            S = S.unsqueeze(0)

        if self.hidden_layer_factor != 0:
            hidden = torch.sigmoid(S @ self.hidden_hs.T)
            if self.relu:
                return torch.relu(hidden @ self.W_hs.T)
            else:
                # to relu or not to relu, that is the question.
                return hidden @ self.W_hs.T
        else:
            if self.relu:
                return torch.relu(S @ self.W_hs.T)
            else:
                return S @ self.W_hs.T  # to relu or not to relu, that is the question.


class ExactPseudoInverseHippocampalSensoryLayer(HippocampalSensoryLayer):
    def __init__(self, input_size, N_h, N_patts, device=None):
        super().__init__(input_size, N_h, device)
        self.size = 0

        self.sbook = torch.zeros((N_patts, input_size), device=self.device)
        """Matrix of all previously seen sensory inputs. Shape: `(N_patts x input_size)`
        """
        self.hbook = torch.zeros((N_patts, N_h), device=self.device)
        """Matrix of all possible hippocampal states. Shape: `(N_patts x N_h)`
        """

        self.W_hs = torch.zeros((N_h, input_size), device=self.device)
        """Sensory to hippocampal weight matrix. Shape: `(N_h x input_size)`
        """

        self.W_sh = torch.zeros((input_size, N_h), device=self.device)
        """Hippocampal to sensory weight matrix. Shape: `(input_size x N_h)`
        """

    @torch.no_grad()
    def learn(self, h, s):
        self.sbook[self.size] = s
        self.hbook[self.size] = h
        # self.W_hs = torch.linalg.lstsq(self.hbook, self.sbook).solution
        # self.W_sh = torch.linalg.lstsq(self.sbook, self.hbook).solution
        self.W_hs = self.hbook.T @ self.sbook.pinverse().T
        self.W_sh = self.sbook.T @ self.hbook.pinverse().T
        self.size += 1

    @torch.no_grad()
    def hippocampal_from_sensory(self, S):
        if S.ndim == 1:
            S = S.unsqueeze(0)

        return torch.relu(S @ self.W_hs.T)

    @torch.no_grad()
    def sensory_from_hippocampal(self, H):
        if H.ndim == 1:
            H = H.unsqueeze(0)

        return H @ self.W_sh.T

    @torch.no_grad()
    def learn_batch(self, sbook: torch.Tensor, hbook: torch.Tensor):
        assert len(sbook) == len(hbook), f"length of sbook must be identical to hbook, sbook_length={len(sbook)}, hbook_length={len(hbook)}"
        self.size = len(sbook)
        self.sbook = sbook.clone()
        self.hbook = hbook.clone()
        
        # self.W_hs = torch.linalg.lstsq(hbook, sbook).solution
        # self.W_sh = torch.linalg.lstsq(sbook, hbook).solution
        self.W_hs = hbook.T @ sbook.pinverse().T
        self.W_sh = sbook.T @ hbook.pinverse().T


class HebbianHippocampalSensoryLayer(HippocampalSensoryLayer):
    def __init__(
        self,
        input_size,
        N_h,
        calculate_update_scaling_method="norm",
        use_h_fix=False,
        mean_h=None,
        scaling_updates=True,
        device=None,
    ):
        assert calculate_update_scaling_method in ["n_h", "norm"]
        if use_h_fix:
            assert (
                calculate_update_scaling_method == "norm"
            ), "use_h_fix only makes sense with norm scaling"

        super().__init__(input_size, N_h, device)

        self.W_hs = torch.zeros((N_h, input_size), device=self.device)
        """Sensory to hippocampal weight matrix. Shape: `(N_h x input_size)`
        """

        self.W_sh = torch.zeros((input_size, N_h), device=self.device)
        """Hippocampal to sensory weight matrix. Shape: `(input_size x N_h)`
        """

        self.calculate_update_scaling_method = calculate_update_scaling_method
        self.use_h_fix = use_h_fix
        if self.use_h_fix:
            assert mean_h != None, "mean_h must be defined to use use_h_fix"
            self.mean_h = mean_h
        else:
            print("mean_h is defined but will not be used because use_h_fix is false")

        self.scaling_updates = scaling_updates

    @torch.no_grad()
    def calculate_update_Wsh(
        self, input: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        if self.calculate_update_scaling_method == "norm":
            scale = torch.linalg.norm(input) ** 2
        elif self.calculate_update_scaling_method == "n_h":
            scale = self.N_h

        if self.scaling_updates:
            output = output - self.sensory_from_hippocampal(input)[0]

        ret = torch.einsum("j,i->ji", output, input) / (scale + 1e-10)
        return ret

    @torch.no_grad()
    def calculate_update_Whs(
        self, input: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        if self.calculate_update_scaling_method == "norm":
            scale = torch.linalg.norm(input) ** 2
        elif self.calculate_update_scaling_method == "n_h":
            scale = self.N_h

        if self.scaling_updates:
            output = output - self.hippocampal_from_sensory(input)[0]

        ret = torch.einsum("j,i->ji", output, input) / (scale + 1e-10)
        return ret

    @torch.no_grad()
    def learn(self, h, s):
        if self.use_h_fix:
            h_ = h - self.mean_h
        else:
            h_ = h
        self.W_hs += self.calculate_update_Whs(s, h_)
        self.W_sh += self.calculate_update_Wsh(h_, s)

    @torch.no_grad()
    def hippocampal_from_sensory(self, S):
        if S.ndim == 1:
            S = S.unsqueeze(0)

        ret = S @ self.W_hs.T
        if self.use_h_fix:
            ret += self.mean_h
        return torch.relu(ret)

    @torch.no_grad()
    def sensory_from_hippocampal(self, H):
        if H.ndim == 1:
            H = H.unsqueeze(0)

        if self.use_h_fix:
            H_ = H - self.mean_h
        else:
            H_ = H

        return H_ @ self.W_sh.T


class HSPseudoInverseSHHebbieanHippocampalSensoryLayer(HippocampalSensoryLayer):
    def __init__(
        self,
        input_size,
        N_h,
        hidden_layer_factor: int,
        stationary=False,
        epsilon_sh=None,
        epsilon_hs=None,
        calculate_update_scaling_method="norm",
        use_h_fix=False,
        mean_h=None,
        scaling_updates=True,
        device=None,
    ):
        assert calculate_update_scaling_method in ["n_h", "norm"]
        if use_h_fix:
            assert (
                calculate_update_scaling_method == "norm"
            ), "use_h_fix only makes sense with norm scaling"
        super().__init__(input_size, N_h, device)

        self.calculate_update_scaling_method = calculate_update_scaling_method
        self.use_h_fix = use_h_fix
        if self.use_h_fix:
            assert mean_h != None, "mean_h must be defined to use use_h_fix"
            self.mean_h = mean_h
        else:
            print("mean_h is defined but will not be used because use_h_fix is false")

        self.scaling_updates = scaling_updates

        self.hidden_layer_factor = hidden_layer_factor
        self.stationary = stationary
        self.epsilon_hs = epsilon_hs
        self.epsilon_sh = epsilon_sh

        self.W_sh = torch.zeros((self.input_size, self.N_h), device=device)

        hidden_size_hs = self.input_size * self.hidden_layer_factor
        if hidden_size_hs == 0:
            hidden_size_hs = self.input_size
        else:
            self.hidden_hs = (
                torch.rand((hidden_size_hs, self.input_size), device=device) - 0.5
            )
        self.W_hs = torch.zeros((self.N_h, hidden_size_hs), device=device)
        if epsilon_hs == None:
            self.epsilon_hs = hidden_size_hs
        else:
            self.epsilon_hs = epsilon_hs
        self.inhibition_matrix_hs = torch.eye(hidden_size_hs, device=device) / (
            self.epsilon_hs**2
        )

    @torch.no_grad()
    def sensory_from_hippocampal(self, H):
        if H.ndim == 1:
            H = H.unsqueeze(0)

        if self.use_h_fix:
            H_ = H - self.mean_h
        else:
            H_ = H

        return H_ @ self.W_sh.T

    @torch.no_grad()
    def hippocampal_from_sensory(self, S):
        if S.ndim == 1:
            S = S.unsqueeze(0)
        if self.hidden_layer_factor != 0:
            hidden = torch.sigmoid(S @ self.hidden_hs.T)
            return torch.relu(hidden @ self.W_hs.T)
        else:
            return torch.relu(
                S @ self.W_hs.T
            )  # to relu or not to relu, that is the question.

    @torch.no_grad()
    def learned_pseudo_inverse_hs(self, input, output):
        if self.stationary:
            b_k_hs = (self.inhibition_matrix_hs @ input) / (
                1 + input.T @ self.inhibition_matrix_hs @ input
            )

            self.inhibition_matrix_hs = (
                self.inhibition_matrix_hs
                - self.inhibition_matrix_hs @ torch.outer(input, b_k_hs.T)
            )

            self.W_hs += torch.outer((output - self.W_hs @ input), b_k_hs.T)
        else:
            b_k_hs = (self.inhibition_matrix_hs @ input) / (
                1 + input.T @ self.inhibition_matrix_hs @ input
            )
            # ERROR VECTOR EK
            e_k = output - self.W_hs @ input

            # NORMALIZATION FACTOR
            E = ((e_k.T @ e_k) / self.inhibition_matrix_hs.shape[0]) / (
                1 + input.T @ self.inhibition_matrix_hs @ input
            )
            # E = torch.abs(E)

            # GAMMA CALCULATION
            gamma = 1 / (1 + ((1 - torch.exp(-E)) / self.epsilon_hs))

            self.inhibition_matrix_hs = gamma * (
                self.inhibition_matrix_hs
                - self.inhibition_matrix_hs @ torch.outer(input, b_k_hs.T)
                + ((1 - torch.exp(-E)) / self.epsilon_hs)
                * torch.eye(self.inhibition_matrix_hs.shape[0], device=self.device)
            )
            self.W_hs += torch.outer((output - self.W_hs @ input), b_k_hs.T)

    @torch.no_grad()
    def calculate_update_Wsh(
        self, input: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        if self.use_h_fix:
            input_ = input - self.mean_h
        else:
            input_ = input

        if self.calculate_update_scaling_method == "norm":
            scale = torch.linalg.norm(input_) ** 2
        elif self.calculate_update_scaling_method == "n_h":
            scale = self.N_h

        if self.scaling_updates:
            output = output - self.sensory_from_hippocampal(input_)[0]

        ret = torch.einsum("j,i->ji", output, input) / (scale + 1e-10)
        return ret

    @torch.no_grad()
    def learn(self, h, s):
        hidden = (
            torch.sigmoid(self.hidden_hs @ s) if self.hidden_layer_factor != 0 else s
        )
        self.learned_pseudo_inverse_hs(input=hidden, output=h)
        # print(self.W_hs)
        self.W_sh += self.calculate_update_Wsh(input=h, output=s)
