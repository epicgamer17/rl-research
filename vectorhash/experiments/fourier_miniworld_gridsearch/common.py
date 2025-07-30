from room_env import RoomAgent
import sys

sys.path.append("../..")

import torch
import itertools
import numpy as np
from preprocessing_cnn import (
    PreprocessingCNN,
    RescalePreprocessing,
    SequentialPreprocessing,
    GrayscaleAndFlattenPreprocessing,
)
from fourier_vectorhash import (
    AdditiveCombine,
    MultiplicativeCombine,
    FourierVectorHaSH,
)
from fourier_scaffold import (
    GuassianFourierSmoothingMatrix,
    HadamardShiftMatrixRat,
    HadamardShiftMatrix,
    ContractionSharpening,
)
from experiments.fourier_miniworld_gridsearch.room_env import RoomExperiment
from hippocampal_sensory_layers import (
    ComplexIterativeBidirectionalPseudoInverseHippocampalSensoryLayerComplexScalars,
)
from fourier_scaffold import FourierScaffold
from agent_history import FourierVectorhashAgentHistory
from matplotlib import pyplot as plt


device = "cuda"


Ds = [400, 1200, 10000]
preprocessing_methods = ["no_cnn"]  # , "cnn"]
additive_shift_alphas = [0.1]  # , 0.3, 0.5, 0.7, 0.9]
combine_methods = [AdditiveCombine(alpha) for alpha in additive_shift_alphas] + [
    MultiplicativeCombine(),
]
shapes = [(3, 3, 3), (7, 7, 7)]
eps_vs = [1.0]
smoothings = [
    GuassianFourierSmoothingMatrix(kernel_radii=[10] * 3, kernel_sigmas=[sigma] * 3)
    for sigma in [0.5, 0.3, 0.1]
]
shifts = [HadamardShiftMatrixRat(torch.tensor(shapes)), HadamardShiftMatrix()]
sharpenings = [
    # ContractionSharpening(2),
    ContractionSharpening(1),
    # ContractionSharpening(3),
]

img_size_map = {"cnn": (16, 8), "no_cnn": (30, 40)}
N_s_map = {"cnn": 16 * 8, "no_cnn": 30 * 40}
preprocessor_map = {
    "cnn": PreprocessingCNN(device=device),
    "no_cnn": SequentialPreprocessing(
        [RescalePreprocessing(0.5), GrayscaleAndFlattenPreprocessing(device=device)]
    ),
}


def generate_env(with_red_box: bool, with_blue_box: bool):
    env = RoomExperiment(
        start_pos=[3, 0, 3],
        start_angle=0,
        place_red_box=with_red_box,
        place_blue_box=with_blue_box,
    )
    return env


def generate_combinations():
    combinations = list(
        itertools.product(
            Ds,
            preprocessing_methods,
            combine_methods,
            eps_vs,
            smoothings,
            shifts,
            sharpenings,
        )
    )
    return combinations


def generate_titles():
    titles = [
        f"D={D}, preprocessing_method={preprocessing_method}, combine_method={combine_method}, eps_v={eps_v}, smoothing={smoothing}, shift={shift}, sharpening={sharpening}"
        for (
            D,
            preprocessing_method,
            combine_method,
            eps_v,
            smoothing,
            shift,
            sharpening,
        ) in generate_combinations()
    ]
    return titles


#  (0,0)                 x
#    +-------------------------- . . .  ----------------->
#    |
#    |                                   __  red box
#    |                                   || (8, 1.5)
#    |        start*[1]                  --
#    |          (3,3)    (5,3)
#    |            x->----+
#    |           /        \
#    |          /          \ (6,4)* <- location to kidnap to after completing circuit
# y  |    (4,6) \          /
#    |           \        /
#    |            +------+
#    |          (3,5)    (5,5)
#    .
#    .
#    .                                  [1]: initial direcion (->) is 0 degrees
#    |
#    |
#    |  --   blue box
#    |  ||  (1.5,8)
#    |  --
#    ↓


def create_agent_for_test(
    env, D, preprocessing_method, combine_method, eps_v, smoothing, shift, sharpening
):
    scaffold = FourierScaffold(
        shapes=torch.tensor(shapes),
        D=D,
        smoothing=smoothing,
        device=device,
        shift=shift,
        sharpening=sharpening,
    )
    layer = (
        ComplexIterativeBidirectionalPseudoInverseHippocampalSensoryLayerComplexScalars(
            input_size=N_s_map[preprocessing_method],
            N_h=D,
            hidden_layer_factor=0,
            epsilon_hs=0.1,
            epsilon_sh=0.1,
            device=device,
        )
    )
    arch = FourierVectorHaSH(
        scaffold=scaffold,
        hippocampal_sensory_layer=layer,
        eps_H=100,
        eps_v=eps_v,
        combine=combine_method,
    )
    agent = RoomAgent(
        vectorhash=arch, env=env, preprocessor=preprocessor_map[preprocessing_method]
    )
    return agent


def write_animation(history: FourierVectorhashAgentHistory, target_dir, entry_name):
    anim = history.make_image_video()
    anim.save(
        f"{target_dir}/{entry_name.split('.')[0]}.gif",
        progress_callback=lambda step, total: print(f"frame {step+1}/{total}"),
    )


device = "cuda"


def analyze_history_errors(history: FourierVectorhashAgentHistory, kidnap_t=None):
    D, M, d = history._scaffold_features.shape
    scaffold = FourierScaffold(
        torch.tensor(shapes),
        D=D,
        features=history._scaffold_features.to(device),
        device=device,
        _skip_K_calc=True,
        _skip_gs_calc=True,
    )

    R = 5
    r = 2
    N = len(history._true_positions)
    masses_true = torch.zeros(N)
    masses_error = torch.zeros(N)

    current_dist = None
    for k in range(N):
        print(k)
        P = history._Ps[k]
        if P != None:
            P = P.to(device)
            x, y, theta = (
                torch.floor(history._true_positions[k][0]),
                torch.floor(history._true_positions[k][1]),
                torch.floor(history._true_positions[k][2]),
            )
            xs = torch.arange(start=x - R, end=x + R + 1, device=P.device)  # type: ignore
            ys = torch.arange(start=y - R, end=y + R + 1, device=P.device)  # type: ignore
            thetas = torch.arange(start=theta - R, end=theta + R + 1, device=P.device)  # type: ignore

            # (N,d)
            omega = torch.cartesian_prod(xs, ys, thetas)

            # (D,M,d)**(d,N)->(D,M,N)->(D,N)->(D,D,N)
            encodings = scaffold.encode_batch(omega.T)

            # (D,D) x (D,D,N) -> (N)
            probabilities = torch.einsum("ij,ijb->b", P, encodings.conj()).abs()

            # (N) -> (N_x, N_y, N_theta)
            current_dist = probabilities.reshape(len(xs), len(ys), len(thetas))

        true_mass = (
            current_dist[R - r : R + r + 1, R - r : R + r + 1, R - r : R + r + 1]
            .sum()
            .cpu()
        )
        masses_true[k] = true_mass
        masses_error[k] = current_dist.sum().cpu() - true_mass

    fig, ax = plt.subplots(figsize=(20, 9))
    if kidnap_t != None:
        ax.axvline(x=kidnap_t, ymin=0, ymax=100, label="kidnapped", linestyle="--")

    ax2 = ax.twinx()

    ax.plot(torch.arange(N), masses_true, label="true")
    ax.plot(torch.arange(N), masses_error, label="error")
    ax2.scatter(torch.arange(N), history._Hs_odometry, label="entropy odometry")
    ax2.scatter(torch.arange(N), history._Hs_sensory, label="entropy sensory")

    ax.set_xlabel("t")
    ax2.set_ylabel("H(P)")
    ax.set_ylabel("probability mass in true position")

    ax.set_xticks(torch.arange(0, N + 1, 20))
    ax.legend()
    ax2.legend()
    return fig


if __name__ == "__main__":
    combinations = generate_combinations()
    titles = generate_titles()
    for i in range(len(combinations)):
        print(f"test {i+1}/{len(combinations)}: {titles[i]}")
