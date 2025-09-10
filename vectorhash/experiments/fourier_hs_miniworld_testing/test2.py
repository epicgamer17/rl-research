import sys

sys.path.append("../..")
import torch
from hippocampal_sensory_layers import (
    ComplexIterativeBidirectionalPseudoInverseHippocampalSensoryLayerComplexScalars,
    HippocampalSensoryLayer,
)
from fourier_scaffold import (
    FourierScaffold,
    HadamardShiftMatrixRat,
    GuassianFourierSmoothingMatrix,
)
from preprocessing_cnn import (
    Preprocessor,
    RescalePreprocessing,
    SequentialPreprocessing,
    GrayscaleAndFlattenPreprocessing,
)
from experiments.fourier_miniworld_gridsearch.room_env import RoomExperiment
import matplotlib.pyplot as plt
from graph_utils import plot_imgs_side_by_side
from agent import TrueData

forward_20 = [2] * 20
right_60_deg = [1] * 20
loop_path = (forward_20 + right_60_deg) * 6 + forward_20
device = "cuda"

shapes = [(3, 3, 3), (5, 5, 5)]
dim_sizes = torch.tensor(shapes).prod(dim=0).tolist()


D = 800
D_reshape_size_map = {
    4000: (50, 80),
    2000: (40, 50),
    800: (20, 40),
    600: (20, 30),
    300: (15, 20),
}
eps_v = 1


def make_env():
    return RoomExperiment([3, 0, 3], 0)


def combine(P1, P2):
    P = P1 @ P2
    return P / P.trace()


alpha = 0.3


def add_combine(P1, P2):
    P = (1 - alpha) * P1 + alpha * P2
    return P


def make_data(
    path: list[int],
    scaffold: FourierScaffold,
    layer: HippocampalSensoryLayer,
    preprocessing: Preprocessor,
    noise_dist=None,
):
    env = make_env()

    def get_true_pos():
        p_x, p_y, p_z = env.get_wrapper_attr("agent").pos
        angle = env.get_wrapper_attr("agent").dir
        p = torch.tensor([p_x, p_z, angle]).float().to(device)
        return p

    def _env_reset():
        obs, info = env.reset()
        img = obs
        processed_img = preprocessing.encode(img)
        p = get_true_pos()
        return processed_img, p

    def _obs_postpreprocess(step_tuple, action):
        obs, reward, terminated, truncated, info = step_tuple
        img = obs
        processed_img = preprocessing.encode(img)
        p = get_true_pos()
        return processed_img, p

    def P_from_h(h):
        P = torch.outer(h, h.conj())
        return P / P.trace()

    def g_avg(P):
        return torch.einsum("ijm,ij->m", scaffold.T_s, P)

    start_img, start_pos = _env_reset()
    v_cumulative = torch.zeros(3, device=device)

    true_data = TrueData(start_pos)

    true_positions = [true_data.true_position.clone()]
    true_Pbook = [scaffold.P.clone().unsqueeze(0)]
    sbook = [start_img]
    g = layer.hippocampal_from_sensory(start_img)[0]
    rec_h_book = [g.clone()]
    h_entropies = [scaffold.entropy(P_from_h(g))]
    scaffold_entropies = [scaffold.entropy(scaffold.P)]

    ### version 1
    # layer.learn(g_avg(scaffold.P), start_img)
    ###

    ### version 2
    # learn initially uniform dist
    uniform = scaffold.encode_probability(
        torch.ones(*dim_sizes, device=device) / torch.prod(torch.tensor(dim_sizes))
    )
    layer.learn(g_avg(uniform), start_img)
    ###
    combined_P_book = []
    sharpened_combined_P_book = []
    sharpened_traces = []

    for i, action in enumerate(path):
        ### env-specific observation processing
        step_tuple = env.step(action)

        ### this is the sensory input not flattened yet
        new_img, new_pos = _obs_postpreprocess(step_tuple, action)

        ### calculation of noisy input
        dp = new_pos - true_data.true_position
        true_data.true_position = new_pos

        print("new pos:", new_pos)
        noisy_dp = dp
        if noise_dist != None:
            noisy_dp += noise_dist.sample(3)

        dt = 1
        v = (noisy_dp / dt) * scaffold.scale_factor
        v_cumulative += v

        if v_cumulative.norm(p=float("inf")) < eps_v:
            continue
        true_positions += [true_data.true_position.clone()]
        sbook += [new_img]

        # shift + smooth
        scaffold.velocity_shift(v_cumulative)
        scaffold.smooth()
        scaffold_entropies += [scaffold.entropy(scaffold.P)]
        true_Pbook += [scaffold.P.clone().unsqueeze(0)]

        layer.learn(g_avg(scaffold.P), new_img)

        # sensory: s -> h-> P
        h_from_s = layer.hippocampal_from_sensory(new_img)[0]
        P_from_s = P_from_h(h_from_s.clone())
        rec_h_book.append(h_from_s.clone())
        h_entropies.append(scaffold.entropy(P_from_s))

        # combine
        ## version 1
        combined_P = combine(scaffold.P, torch.outer(h_from_s, h_from_s.conj()))
        combined_P_book += [combined_P.clone().unsqueeze(0)]
        # sharpen
        scaffold.P = combined_P
        scaffold.sharpen()
        sharpened_combined_P_book += [scaffold.P.clone().unsqueeze(0)]
        ## version 2
        # combined_P = add_combine(
        #     scaffold.P, combine(scaffold.P, torch.outer(h_from_s, h_from_s.conj()))
        # )
        ## version 3
        # combined_P = combine(
        #     scaffold.P,
        #     scaffold.smoothing(torch.outer(h_from_s, h_from_s.conj())),
        # )
        ## version 4
        # if scaffold.entropy(torch.outer(h_from_s, h_from_s.conj())) < 0.9:
        #     combined_P = scaffold.P
        # else:
        #     combined_P = combine(
        #         scaffold.P,
        #         scaffold.smoothing(torch.outer(h_from_s, h_from_s.conj())),
        #     )
        ## version 5
        # if i >= 5 and scaffold.entropy(P_from_s) > 0.5:
        #     combined_P = combine(scaffold.P, scaffold.smoothing(P_from_h(h_from_s)))
        #     combined_P_book += [combined_P.clone().unsqueeze(0)]
        #     # sharpen
        #     scaffold.sharpen()
        #     sharpened_combined_P_book += [scaffold.P.clone().unsqueeze(0)]
        # else:
        #     combined_P = scaffold.P
        #     combined_P_book += [combined_P.clone().unsqueeze(0)]
        #     # scaffold.sharpen()
        #     sharpened_combined_P_book += [scaffold.P.clone().unsqueeze(0)]
        #     pass

        # learn
        sharpened_traces.append(scaffold.P.trace())
        # layer.learn(g_avg(scaffold.P), new_img)

        v_cumulative = torch.zeros(3, device=device)

    return (
        torch.vstack(true_positions),
        torch.concat(true_Pbook, dim=0),
        torch.vstack(sbook),
        torch.vstack(rec_h_book),
        torch.concat(combined_P_book, dim=0),
        torch.concat(sharpened_combined_P_book, dim=0),
        torch.tensor(h_entropies, device=device),
        torch.tensor(sharpened_traces, device=device),
        torch.tensor(scaffold_entropies, device=device),
    )


def get_xy_distributions(scaffold: FourierScaffold, Pbook: torch.Tensor):
    c_x, c_y, c_th = 0, 0, 0
    r_x, r_y, r_th = 7, 7, 7
    l_x, l_y, l_th = 2 * r_x + 1, 2 * r_y + 1, 2 * r_th + 1
    omega = torch.cartesian_prod(
        torch.arange(c_x - r_x, c_x + r_x + 1, 1, device=device),
        torch.arange(c_y - r_y, c_y + r_y + 1, 1, device=device),
        torch.arange(c_th - r_th, c_th + r_th + 1, 1, device=device),
    )
    xy_distributions = torch.empty(len(Pbook), l_x, l_y)
    for i in range(len(Pbook)):
        print(i)
        dist = scaffold.get_probability_abs_batched(omega, P=Pbook[i])
        xy_distributions[i] = dist.reshape(l_x, l_y, l_th).sum(-1).T

    return xy_distributions


scaffold = FourierScaffold(
    shapes=torch.tensor(shapes),
    D=D,
    shift=HadamardShiftMatrixRat(shapes=torch.tensor(shapes)),
    smoothing=GuassianFourierSmoothingMatrix(
        kernel_radii=[10] * 3, kernel_sigmas=[1] * 3
    ),
    device=device,
    limits=[10, 10, 2 * torch.pi],
)
layer = ComplexIterativeBidirectionalPseudoInverseHippocampalSensoryLayerComplexScalars(
    input_size=300,
    N_h=D,
    hidden_layer_factor=0,
    epsilon_sh=0.1,
    epsilon_hs=0.1,
    device=device,
)
preprocessing = SequentialPreprocessing(
    transforms=[
        RescalePreprocessing(0.25),
        GrayscaleAndFlattenPreprocessing(device),
    ]
)

data = make_data(loop_path, scaffold, layer, preprocessing, noise_dist=None)


def lift(h):
    return torch.outer(h, h.conj())


(
    data_true_positions,
    data_true_Pbook,
    data_sbook,
    data_rec_h_book,
    data_combined_P_book,
    data_sharpened_combined_P_book,
    h_entropies,
    sharpened_traces,
    scaffold_entropies,
) = data

N = len(data_true_positions)
print("dists_Pbook ")
dists_Pbook = get_xy_distributions(scaffold, data_true_Pbook)
print("dists_rec_h_nook")
dists_rec_h_book = get_xy_distributions(scaffold, torch.vmap(lift)(data_rec_h_book))
print("dists_combined")
print(data_combined_P_book.shape)
dists_combined = get_xy_distributions(scaffold, data_combined_P_book)
print("dists_sharpeened")
dists_sharpeened = get_xy_distributions(scaffold, data_sharpened_combined_P_book)

fig, ax = plt.subplots(nrows=len(data_true_positions), ncols=4, figsize=(10, 48))

plot_imgs_side_by_side(
    imgs=dists_Pbook.cpu(),
    axs=ax[:, 0],
    titles=[f"{scaffold_entropies[i]:.2f}" for i in range(N)],
    fig=fig,
    use_first_img_scale=False,
)
plot_imgs_side_by_side(
    imgs=dists_rec_h_book.cpu(),
    axs=ax[:, 1],
    titles=[f"{h_entropies[i]:.2f}" for i in range(N)],
    fig=fig,
    use_first_img_scale=False,
)
plot_imgs_side_by_side(
    imgs=dists_combined.cpu(),
    axs=ax[1:, 2],
    titles=[""] * (N - 1),
    fig=fig,
    use_first_img_scale=False,
)
plot_imgs_side_by_side(
    imgs=dists_sharpeened.cpu(),
    axs=ax[1:, 3],
    titles=[f"{sharpened_traces[i]:.2f}" for i in range(N - 1)],
    fig=fig,
    use_first_img_scale=False,
)

for i in range(N):
    true_pos = data_true_positions[i]
    x = (true_pos[0] - 3) * scaffold.scale_factor[0] + 7
    y = (true_pos[1] - 3) * scaffold.scale_factor[1] + 7

    for j in range(4):
        ax[i, j].scatter(x.cpu(), y.cpu(), color="red", marker="x")

fig.savefig("test.png")
