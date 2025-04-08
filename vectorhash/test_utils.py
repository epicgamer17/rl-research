import torch
from vectorhash import (
    build_initializer,
    GridHippocampalScaffold,
    ExactPseudoInverseHippocampalSensoryLayer,
    IterativeBidirectionalPseudoInverseHippocampalSensoryLayer,
    VectorHaSH,
    build_vectorhash_architecture,
)
from hippocampal_sensory_layers import *
from smoothing import ArgmaxSmoothing, SoftmaxSmoothing, PolynomialSmoothing
from tqdm import tqdm

UP = torch.tensor([0, 1])
DOWN = torch.tensor([0, -1])
LEFT = torch.tensor([-1, 0])
RIGHT = torch.tensor([1, 0])
# ROTATE_LEFT = torch.tensor([0, 0, 1])
# ROTATE_RIGHT = torch.tensor([0, 0, -1])


def get_action():
    a = None
    while a is None:
        action = input(
            "Enter 'w' to move up, 's' to move down, "
            "'a' to move left, 'd' to move right, 'q' to move forward, "
            "'q' to rotate left, 'e' to rotate right, 'quit' to quit: "
        )
        if action == "quit":
            return None
        elif action == "w":
            a = UP
        elif action == "s":
            a = DOWN
        elif action == "a":
            a = LEFT
        elif action == "d":
            a = RIGHT
        # elif action == 'q':
        #   a = ROTATE_LEFT
        # elif action == 'e':
        #   a = ROTATE_RIGHT
        else:
            print("Invalid action, type quit to exit")

    return a


def corrupt_p_1(codebook, p=0.1):
    if p == 0.0:
        return codebook
    rand_indices = torch.sign(
        torch.rand(size=codebook.shape, device=codebook.device) - p
    )
    return torch.multiply(codebook, rand_indices)


def dynamics_patts(
    scaffold: GridHippocampalScaffold,
    sensory_hippocampal_layer: HippocampalSensoryLayer,
    s_noisy,  # (Npatts, input_size)
    s_true,
    h_true,
    N_iter,
    N_patts,
    sign_output=False,
):
    s_noisy = s_noisy[:N_patts]
    h_in_original = sensory_hippocampal_layer.hippocampal_from_sensory(s_noisy)
    h = torch.clone(h_in_original)
    for i in range(N_iter):
        g = scaffold.denoise(scaffold.grid_from_hippocampal(h))
        print(g)
        h = scaffold.hippocampal_from_grid(g)

    s_out = sensory_hippocampal_layer.sensory_from_hippocampal(h)
    if sign_output:
        s_out = torch.sign(s_out)

    sbook = s_true[:N_patts]
    hbook = h_true[:N_patts]

    h_l2_err = torch.linalg.vector_norm(h - hbook) / scaffold.N_h
    s_l2_err = (
        torch.linalg.vector_norm(s_out - sbook) / sensory_hippocampal_layer.input_size
    )
    s_l1_err = torch.mean(torch.abs(s_out - sbook)) / 2

    return h_l2_err, s_l2_err, s_l1_err


def capacity_test(
    sbook: torch.Tensor,
    Npatts_list,
    nruns,
    device,
    sign_output=False,
    shapes=[(3,3), (5,5), (7,7)],
    N_h=1000,
    input_size=784,
    initalization_method="by_scaling",
    W_gh_var=1,
    percent_nonzero_relu=0.7,
    sparse_initialization=0.1,
    T=1e-6,
    hippocampal_sensory_layer_type="iterative_pseudoinverse",
    hidden_layer_factor=1,
    stationary=True,
    epsilon_hs=0.1,
    epsilon_sh=0.1,
    relu=False,
    smoothing_method=SoftmaxSmoothing(T=1e-6),
):
    
    model = build_vectorhash_architecture(
        shapes=shapes,
        N_h=N_h,
        input_size=input_size,
        initalization_method=initalization_method,
        W_gh_var=W_gh_var,
        percent_nonzero_relu=percent_nonzero_relu,
        sparse_initialization=sparse_initialization,
        T=T,
        device=device,
        hippocampal_sensory_layer_type=hippocampal_sensory_layer_type,
        hidden_layer_factor=hidden_layer_factor,
        stationary=stationary,
        epsilon_hs=epsilon_hs,
        epsilon_sh=epsilon_sh,
        relu=relu,
        smoothing=smoothing_method,
    )

    err_h_l2 = -1 * torch.ones((len(Npatts_list), nruns), device=device)
    err_s_l1 = -1 * torch.ones((len(Npatts_list), nruns), device=device)
    err_s_l2 = -1 * torch.ones((len(Npatts_list), nruns), device=device)



    for k in tqdm(range(len(Npatts_list))):
        Npatts = Npatts_list[k]
        for j in tqdm(range(Npatts)):
            model.hippocampal_sensory_layer.learn(model.scaffold.H[j], sbook[j])

        for r in range(nruns):
            sbook_noisy = sbook  # corrupt_p_1(sbook)[:Npatts]
            err_h_l2[k, r], err_s_l2[k, r], err_s_l1[k, r] = dynamics_patts(
                model.scaffold,
                model.hippocampal_sensory_layer,
                sbook,
                sbook_noisy,
                model.scaffold.H,
                nruns,
                Npatts,
                sign_output=sign_output,
                relu=relu,
            )

    return err_h_l2, err_s_l2, err_s_l1


def capacity1(
    shapes,
    Np_lst,
    Npatts_lst,
    nruns,
    sbook,
    device,
    init_method="by_scaling",
    W_gh_var=1,
    percent_nonzero_relu=0.7,
    sparse_initialization=0.1,
    T=1e-6,
    hippocampal_sensory_layer_type="iterative_pseudoinverse",
    hidden_layer_factor=1,
    stationary=True,
    epsilon_hs=0.1,
    epsilon_sh=0.1,
    relu=False,
    sign_output=False,
    smoothing_method=SoftmaxSmoothing(T=1e-6),
):
    
    err_h_l2 = -1 * torch.ones((len(Np_lst), len(Npatts_lst), nruns), device=device)
    err_s_l2 = -1 * torch.ones((len(Np_lst), len(Npatts_lst), nruns), device=device)
    err_s_l1 = -1 * torch.ones((len(Np_lst), len(Npatts_lst), nruns), device=device)

    sbook_torch = torch.from_numpy(sbook).to(device).float().T

    for l, Np in enumerate(Np_lst):
        err_h_l2[l], err_s_l2[l], err_s_l1[l] = capacity_test(
            N_h=Np,
            sbook=sbook_torch,
            Npatts_list=Npatts_lst,
            nruns=nruns,
            device=device,
            sign_output=sign_output,
            shapes=shapes,
            input_size=sbook_torch.shape[0],
            initalization_method=init_method,
            W_gh_var=W_gh_var,
            percent_nonzero_relu=percent_nonzero_relu,
            T=T,
            hippocampal_sensory_layer_type=hippocampal_sensory_layer_type,
            hidden_layer_factor=hidden_layer_factor,
            stationary=stationary,
            epsilon_hs=epsilon_hs,
            epsilon_sh=epsilon_sh,
            relu=relu,
            smoothing_method=smoothing_method,
        )

    return err_h_l2, err_s_l2, err_s_l1

def generate_animalai_path(path_length=100):
    return torch.randint(0, 9, (path_length,)).tolist()