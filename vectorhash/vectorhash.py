from clean_scaffold import GridHippocampalScaffold, SoftmaxSmoothing
from hippocampal_sensory_layers import *
from shifts import *
from tqdm import tqdm


class VectorHaSH:
    def __init__(
        self,
        scaffold: GridHippocampalScaffold,
        hippocampal_sensory_layer: HippocampalSensoryLayer,
        zero_tol=1e-2,
        dream_fix=False,
        self_certainty=None,
    ):
        self.scaffold = scaffold
        self.hippocampal_sensory_layer = hippocampal_sensory_layer
        self.zero_tol = zero_tol
        self.dream_fix = dream_fix
        self.certainty = self_certainty

    @torch.no_grad()
    def store_memory(self, s: torch.Tensor, debug=True, hard=False):
        """Stores a memory in the scaffold.
        Input shape: `(input_size)`
        """
        if hard:
            h= self.scaffold.hippocampal_from_grid(self.scaffold.denoise(self.scaffold.g))[0]
        else:
            h = self.scaffold.hippocampal_from_grid(self.scaffold.g)[0]
        # print("current h we are learning", h)
        # print("is h in the h_book", torch.allclose(h, self.H[self.H.nonzero()[-1][0]]))

        self.hippocampal_sensory_layer.learn(h, s)
        if debug:
            print("info for each h directly after learning it")
            h_from_s = self.hippocampal_sensory_layer.hippocampal_from_sensory(s)
            g_from_h_from_s = self.scaffold.grid_from_hippocampal(h_from_s)
            g_denoised = self.scaffold.denoise(g_from_h_from_s)
            h_from_s_denoised = self.scaffold.hippocampal_from_grid(g_denoised)

            print("h max, min, mean", torch.max(h), torch.min(h), torch.mean(h))
            print(
                "h_from_s max, min, mean",
                torch.max(h_from_s),
                torch.min(h_from_s),
                torch.mean(h_from_s),
            )
            print(
                "h_from_s_denoised max, min, mean",
                torch.max(h_from_s_denoised),
                torch.min(h_from_s_denoised),
                torch.mean(h_from_s_denoised),
            )

            print(
                "avg nonzero/greaterzero h from book:",
                torch.sum(h != 0),
                torch.sum(h > 0),
            )
            print(
                "avg nonzero/greaterzero h from s:",
                torch.sum(h_from_s != 0),
                torch.sum(h_from_s > 0),
            )
            print(
                "avg nonzero/greaterzero h from s denoised:",
                torch.sum(h_from_s_denoised != 0),
                torch.sum(h_from_s_denoised > 0),
            )
            # print(h.shape, h_from_s.shape)
            print(
                "mse/cosinesimilarity h from book and h from s",
                torch.nn.functional.mse_loss(h, h_from_s),
                torch.nn.functional.cosine_similarity(
                    h.reshape(1, -1), h_from_s.reshape(1, -1)
                ),
            )
            print(
                "mse/cosinesimilarity h from book and h from s denoised",
                torch.nn.functional.mse_loss(h, h_from_s_denoised),
                torch.nn.functional.cosine_similarity(
                    h.reshape(1, -1), h_from_s_denoised.reshape(1, -1)
                ),
            )
            s_from_h = self.hippocampal_sensory_layer.sensory_from_hippocampal(h)
            s_from_h_from_s = self.hippocampal_sensory_layer.sensory_from_hippocampal(
                h_from_s
            )
            s_from_h_from_s_denoised = (
                self.hippocampal_sensory_layer.sensory_from_hippocampal(
                    h_from_s_denoised
                )
            )
            print(
                "mse/cosinesimilarity s and s from h from s",
                torch.nn.functional.mse_loss(s, s_from_h_from_s),
                torch.nn.functional.cosine_similarity(
                    s.reshape(1, -1), s_from_h_from_s.reshape(1, -1)
                ),
            )
            print(
                "mse/cosinesimilarity s and s from h from s denoised",
                torch.nn.functional.mse_loss(s, s_from_h_from_s_denoised),
                torch.nn.functional.cosine_similarity(
                    s.reshape(1, -1), s_from_h_from_s_denoised.reshape(1, -1)
                ),
            )
            print(
                "mse/cosinesimilarity s and s from h",
                torch.nn.functional.mse_loss(s, s_from_h),
                torch.nn.functional.cosine_similarity(
                    s.reshape(1, -1), s_from_h.reshape(1, -1)
                ),
            )

            # hidden = torch.sigmoid(self.hidden_sh @ h)
            # print("S FROM HIPPO", self.W_sh @ hidden)

    def estimate_position(self, s: torch.Tensor, as_tuple_list=False):
        g = self.scaffold.grid_from_hippocampal(
            self.hippocampal_sensory_layer.hippocampal_from_sensory(s)
        )
        onehotted = self.scaffold.denoise(g.squeeze())

        if not as_tuple_list:
            return onehotted

        pos = 0
        onehotted_list = []
        for module in self.scaffold.modules:
            onehotted_list.append(onehotted[pos : pos + module.l].reshape(module.shape))
            pos += module.l

        return onehotted_list

    @torch.no_grad()
    def dream(self, seen_gss: set):
        i = 0
        # print("seen_gs", seen_gss)
        # reverse the order of seen_gs
        seen_gs = list(seen_gss)
        seen_gs.reverse()

        for g in seen_gs:
            # if i<self.dream_fix:
            self.scaffold.g = g
            h = self.scaffold.hippocampal_from_grid(g)[0]
            s = self.hippocampal_sensory_layer.sensory_from_hippocampal(h)[0]
            h_ = self.hippocampal_sensory_layer.hippocampal_from_sensory(s)[0]
            dh = h_ - h
            if dh.norm() > self.zero_tol:
                self.hippocampal_sensory_layer.W_hs += (
                    self.hippocampal_sensory_layer.calculate_update_Whs(s, h)
                )
            # i+=1

    @torch.no_grad()
    def learn_path(self, observations, velocities):
        """Add a path of observations to the memory scaffold. It is assumed that the observations are taken at each time step and the velocities are the velocities directly after the observations."""
        assert len(observations) == len(velocities)

        seen_gs = set()
        seen_gs_recall = set()
        seen_g_s_recall = set()
        seen_hs = set()
        seen_hs_recall = set()

        first_obs = observations[0]
        second_obs = observations[1]
        first_image_grid_position_estimates = []
        second_image_grid_position_estimates = []
        first_image_grid_positions = []
        second_image_grid_positions = []

        i = 0
        for i in tqdm(range(len(observations))):
            g = self.scaffold.g
            obs = observations[i]
            vel = velocities[i]

            seen_gs.add(tuple(g.tolist()))
            seen_hs.add(self.scaffold.hippocampal_from_grid(g)[0])

            self.learn(obs, vel)

            # testing code
            first_image_grid_position_estimates.append(
                self.estimate_position(first_obs).flatten().clone()
            )
            first_image_grid_positions.append(
                self.scaffold.denoise(
                    self.scaffold.grid_from_hippocampal(
                        self.hippocampal_sensory_layer.hippocampal_from_sensory(
                            first_obs
                        )
                    )
                )
                .flatten()
                .clone()
            )

            if i > 0:
                second_image_grid_position_estimates.append(
                    self.estimate_position(second_obs).flatten().clone()
                )
                second_image_grid_positions.append(
                    self.scaffold.denoise(
                        self.scaffold.grid_from_hippocampal(
                            self.hippocampal_sensory_layer.hippocampal_from_sensory(
                                second_obs
                            )
                        )
                    )
                    .flatten()
                    .clone()
                )

            if self.dream_fix != None:
                if (i + 1) % self.dream_fix == 0:
                    self.dream(seen_gs)

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
            seen_g_s.add(tuple(self.scaffold.denoise(torch.tensor(list(g)))))
        print("Unique Gs seen while learning (after denoising):", len(seen_g_s))
        return (
            first_image_grid_position_estimates,
            second_image_grid_position_estimates,
            first_image_grid_positions,
            second_image_grid_positions,
        )

    def learn_direct(self, observations, offset=0):
        for i in tqdm(range(len(observations))):
            self.g = self.G[i + offset]
            self.store_memory(observations[i], debug=False)

    @torch.no_grad()
    def learn(self, observation, velocity=None):
        """Add a memory to the memory scaffold and shift the grid coding state by a given velocity.

        observation shape: `(input_size)`
        velocity shape: `(D)` where `D` is the dimensionality of the grid modules.
        """

        self.store_memory(observation)
        if velocity:
            self.scaffold.shift(velocity)

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
        H = self.hippocampal_sensory_layer.hippocampal_from_sensory(observations)
        used_Hs = set()
        for h in H:
            used_Hs.add(tuple(h.tolist()))
        print("Unique Hs seen while recalling:", len(used_Hs))
        # print(used_Hs)
        G = self.scaffold.grid_from_hippocampal(H)
        used_gs = set()
        for g in G:
            # print(g)
            used_gs.add(tuple(g.tolist()))
        print("Unique Gs seen while recalling (before denoising):", len(used_gs))
        # print(used_gs)
        G_ = self.scaffold.denoise(G)
        used_G_s = set()
        for g in G_:
            # print(g)
            used_G_s.add(tuple(g.tolist()))
        print("Unique Gs seen while recalling (after denoising):", len(used_G_s))
        H_ = self.scaffold.hippocampal_from_grid(G_)
        used_H_s = set()
        for h in H_:
            used_H_s.add(tuple(h.tolist()))
        print("Unique Hs seen while recalling (after denoising):", len(used_H_s))
        H_nonzero = torch.sum(H != 0, 1).float()
        print("avg nonzero H:", torch.mean(H_nonzero).item())
        H__nonzero = torch.sum(H_ != 0, 1).float()
        print("avg nonzero H_denoised:", torch.mean(H__nonzero).item())

        S_ = self.hippocampal_sensory_layer.sensory_from_hippocampal(H_)
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
    
    @torch.no_grad()
    def main_loop(self, time_step_length, odometry, observation, k=1):
        dx, dy, dtheta = odometry
        self.scaffold.shift(self.scaffold.modules, (dx,dy,dtheta)) # integrate time_step_length in ratshift
        certainty = self.scaffold.estimate_certainty(k=k)
        if certainty >= self.certainty:
            self.store_memory(observation)
        else:
            print("Certainty not high enough, not storing memory.")


import math
from scipy.stats import norm
from matrix_initializers import *
from vectorhash_functions import expectation_of_relu_normal


def build_initializer(
    shapes,
    initalization_method="by_sparsity",
    W_gh_var=1,
    percent_nonzero_relu=0.9,
    sparse_initialization=0.1,
    device=None,
):
    if initalization_method == "by_scaling":
        W_hg_std = math.sqrt(W_gh_var)
        W_hg_mean = (
            -W_hg_std * norm.ppf(1 - percent_nonzero_relu) / math.sqrt(len(shapes))
        )
        h_normal_mean = len(shapes) * W_hg_mean
        h_normal_std = math.sqrt(len(shapes)) * W_hg_std
        relu_theta = 0
        sparse_initializer = SparseMatrixByScalingInitializer(
            mean=W_hg_mean, scale=W_hg_std, device=device
        )
    elif initalization_method == "by_sparsity":
        gamma = 1 - sparse_initialization
        relu_theta = math.sqrt(gamma * len(shapes)) * norm.ppf(1 - percent_nonzero_relu)
        W_hg_mean = 0
        W_hg_std = math.sqrt(gamma * len(shapes))
        h_normal_mean = -relu_theta
        h_normal_std = (1 - sparse_initialization) * len(shapes)
        sparse_initializer = SparseMatrixBySparsityInitializer(
            sparsity=sparse_initialization, device=device
        )

    return (
        sparse_initializer,
        relu_theta,
        expectation_of_relu_normal(h_normal_mean, h_normal_std),
    )

def build_shift(shift, device=None):
    assert shift in ["roll", "rat", "conv", "shift"]

    if shift == "roll":
        return RollShift(device)
    elif shift == "rat":
        return RatShift(device)
    elif shift == "conv":
        return ConvolutionalShift(device=device)
    else:
        return Shift(device)

def build_scaffold(
    shapes,
    N_h,
    initalization_method="by_sparsity",
    W_gh_var=1,
    percent_nonzero_relu=0.9,
    sparse_initialization=0.1,
    smoothing=SoftmaxSmoothing(T=1e-3),
    shift="roll",
    device=None,
    relu=False,
):
    initializer, relu_theta, mean_h = build_initializer(
        shapes,
        initalization_method=initalization_method,
        W_gh_var=W_gh_var,
        percent_nonzero_relu=percent_nonzero_relu,
        sparse_initialization=sparse_initialization,
        device=device,
    )
    smoothing = smoothing
    shift = build_shift(shift, device)
    scaffold = GridHippocampalScaffold(
        shapes=shapes,
        N_h=N_h,
        sparse_matrix_initializer=initializer,
        relu_theta=relu_theta,
        shift_method=shift,
        sanity_check=False, # breaks with soft smoothing
        calculate_g_method="fast",
        smoothing=smoothing,
        device=device,
        relu=relu,
    )

    return scaffold, mean_h


def build_vectorhash_architecture(
    shapes,
    N_h,
    input_size,
    initalization_method="by_scaling",
    W_gh_var=1,
    percent_nonzero_relu=0.9,
    sparse_initialization=0.1,
    device=None,
    hippocampal_sensory_layer_type="iterative_pseudoinverse",
    hidden_layer_factor=1,
    stationary=True,
    epsilon_hs=1,
    epsilon_sh=1,
    relu=False,
    shift="roll",
    smoothing=SoftmaxSmoothing(T=1e-6),
    shift="roll",
):
    assert initalization_method in ["by_scaling", "by_sparsity"]
    assert hippocampal_sensory_layer_type in [
        "exact_pseudoinverse",
        "iterative_pseudoinverse",
        "hebbian",
        "naive_hebbian",
        "mixed",
    ]
    assert shift in ["roll", "conv"]
    print(initalization_method)
    scaffold, mean_h = build_scaffold(
        shapes,
        N_h,
        initalization_method=initalization_method,
        W_gh_var=W_gh_var,
        percent_nonzero_relu=percent_nonzero_relu,
        sparse_initialization=sparse_initialization,
        device=device,
        relu=relu,
        smoothing=smoothing,
        shift=shift,
    )

    if hippocampal_sensory_layer_type == "exact_pseudoinverse":
        hippocampal_sensory_layer = ExactPseudoInverseHippocampalSensoryLayer(
            input_size=input_size,
            N_h=N_h,
            N_patts=scaffold.N_patts,
            hbook=scaffold.N_h,
            device=device,
        )
    elif hippocampal_sensory_layer_type == "hebbian":
        hippocampal_sensory_layer = HebbianHippocampalSensoryLayer(
            input_size=input_size,
            N_h=N_h,
            N_patts=scaffold.N_patts,
            use_h_fix=True,
            mean_h=mean_h,
            scaling_updates=True,
            device=device,
        )
    elif hippocampal_sensory_layer_type == "naive_hebbian":
        hippocampal_sensory_layer = HebbianHippocampalSensoryLayer(
            input_size=input_size,
            N_h=N_h,
            N_patts=scaffold.N_patts,
            use_h_fix=False,
            mean_h=mean_h,
            scaling_updates=False,
            device=device,
        )
    elif hippocampal_sensory_layer_type == "iterative_pseudoinverse":
        hippocampal_sensory_layer = (
            IterativeBidirectionalPseudoInverseHippocampalSensoryLayer(
                input_size=input_size,
                N_h=N_h,
                hidden_layer_factor=hidden_layer_factor,
                stationary=stationary,
                epsilon_hs=epsilon_hs,
                epsilon_sh=epsilon_sh,
                device=device,
                relu=relu,
            )
        )
    
    architecture = VectorHaSH(
        scaffold=scaffold,
        hippocampal_sensory_layer=hippocampal_sensory_layer,
        zero_tol=1e-2,
        dream_fix=None,

    )

    return architecture
