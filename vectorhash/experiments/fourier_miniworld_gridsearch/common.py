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
from fourier_scaffold import GuassianFourierSmoothingMatrix
from experiments.fourier_miniworld_gridsearch.room_env import RoomExperiment
from hippocampal_sensory_layers import (
    ComplexIterativeBidirectionalPseudoInverseHippocampalSensoryLayerComplexScalars,
)
from fourier_scaffold import FourierScaffold


device = "cuda"


Ds = np.arange(100, 1001, 100)
preprocessing_methods = ["cnn", "no_cnn"]
additive_shift_alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
combine_methods = [
    MultiplicativeCombine(),
] + [AdditiveCombine(alpha) for alpha in additive_shift_alphas]
shapes = [(5, 5, 5), (7, 7, 7)]
eps_vs = [0.1, 0.3, 0.5, 0.7, 0.9]
smoothings = [
    GuassianFourierSmoothingMatrix(kernel_radii=[10] * 3, kernel_sigmas=[sigma] * 3)
    for sigma in [0.2, 0.4, 0.6, 0.8, 1]
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
            Ds, preprocessing_methods, combine_methods, eps_vs, smoothings
        )
    )
    return combinations


def generate_titles():
    titles = [
        f"D={D}, preprocessing_method={preprocessing_method}, combine_method={combine_method}, eps_v={eps_v}, smoothing={smoothing}"
        for (
            D,
            preprocessing_method,
            combine_method,
            eps_v,
            smoothing,
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
    env, D, preprocessing_method, combine_method, eps_v, smoothing
):
    scaffold = FourierScaffold(
        shapes=torch.tensor(shapes), D=D, smoothing=smoothing, device=device
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
        eps_H=1,
        eps_v=eps_v,
        combine=combine_method,
    )
    agent = RoomAgent(vectorhash=arch, env=env)
    return agent


if __name__ == "__main__":
    combinations = generate_combinations()
    titles = generate_titles()
    for i in range(len(combinations)):
        print(f"test {i+1}/{len(combinations)}: {titles[i]}")
