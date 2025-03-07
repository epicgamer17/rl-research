from nd_scaffold import *

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


from vectorhash import (
    build_initializer,
    GridHippocampalScaffold,
    ExactPseudoInverseHippocampalSensoryLayer,
    IterativeBidirectionalPseudoInverseHippocampalSensoryLayer,
)
from hippocampal_sensory_layers import *


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
        h = scaffold.hippocampal_from_grid(g)
    h_out = torch.clone(h)

    if sign_output:
        s_out = torch.sign(sensory_hippocampal_layer.sensory_from_hippocampal(h))
    else:
        s_out = sensory_hippocampal_layer.sensory_from_hippocampal(h)

    sbook = s_true[:N_patts]
    hbook = h_true[:N_patts]

    h_l2_err = torch.linalg.vector_norm(h_out - hbook) / scaffold.N_h
    s_l2_err = (
        torch.linalg.vector_norm(s_out - sbook) / sensory_hippocampal_layer.input_size
    )
    s_l1_err = torch.mean(torch.abs(s_out - sbook)) / 2

    return h_l2_err, s_l2_err, s_l1_err


def capacity_test(
    shapes,
    N_h,
    sbook: torch.Tensor,
    Npatts_list,
    nruns,
    device,
    method,
    sign_output=False,
):
    assert method in ["exact", "iterative"]

    err_h_l2 = -1 * torch.ones((len(Npatts_list), nruns), device=device)
    err_s_l1 = -1 * torch.ones((len(Npatts_list), nruns), device=device)
    err_s_l2 = -1 * torch.ones((len(Npatts_list), nruns), device=device)

    initializer, _, _ = build_initializer(
        shapes, sparse_initialization=0.4, device=device
    )

    for k in tqdm(range(len(Npatts_list))):
        Npatts = Npatts_list[k]
        scaffold = GridHippocampalScaffold(
            shapes=shapes,
            N_h=N_h,
            sanity_check=False,
            device=device,
            sparse_matrix_initializer=initializer,
            relu_theta=0.5,
        )
        if method == "exact":
            sensory_hippocampal_layer = ExactPseudoInverseHippocampalSensoryLayer(
                input_size=sbook.shape[1],
                N_h=scaffold.N_h,
                N_patts=scaffold.N_patts,
                hbook=scaffold.H[:Npatts],
                device=device,
            )
            sensory_hippocampal_layer.learn_batch(sbook[:Npatts])
        elif method == "iterative":
            sensory_hippocampal_layer = (
                IterativeBidirectionalPseudoInverseHippocampalSensoryLayer(
                    input_size=sbook.shape[1],
                    N_h=scaffold.N_h,
                    hidden_layer_factor=1,
                    epsilon_hs=0.1,
                    epsilon_sh=0.1,
                    device=device,
                )
            )
            for j in tqdm(range(Npatts)):
                sensory_hippocampal_layer.learn(scaffold.H[j], sbook[j])

        for r in range(nruns):
            sbook_noisy = sbook  # corrupt_p_1(sbook)[:Npatts]
            err_h_l2[k, r], err_s_l2[k, r], err_s_l1[k, r] = dynamics_patts(
                scaffold,
                sensory_hippocampal_layer,
                sbook,
                sbook_noisy,
                scaffold.H,
                1,
                Npatts,
                sign_output=sign_output,
            )

    return err_h_l2, err_s_l2, err_s_l1


def capacity1(shapes, Np_lst, Npatts_lst, nruns, sbook, device, method, sign_output):
    err_h_l2 = -1 * torch.ones((len(Np_lst), len(Npatts_lst), nruns), device=device)
    err_s_l2 = -1 * torch.ones((len(Np_lst), len(Npatts_lst), nruns), device=device)
    err_s_l1 = -1 * torch.ones((len(Np_lst), len(Npatts_lst), nruns), device=device)

    sbook_torch = torch.from_numpy(sbook).to(device).float().T

    for l, Np in enumerate(Np_lst):
        err_h_l2[l], err_s_l2[l], err_s_l1[l] = capacity_test(
            shapes,
            Np,
            sbook_torch,
            Npatts_lst,
            nruns,
            device,
            method=method,
            sign_output=sign_output,
        )

    return err_h_l2, err_s_l2, err_s_l1
