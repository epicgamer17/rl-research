# MIGHT BE DOUBLE COUNTING P[i][j][k], since i do the += and do a copy (not fresh)
import torch
import math


def generate_epsilon(N_x, N_y, sigma):
    """
    Generate a 2D Gaussian kernel (epsilon) for the given dimensions and standard deviation.
    """
    kernel = torch.zeros((N_x, N_y))
    for i in range(N_x):
        for j in range(N_y):
            x = i - N_x // 2
            y = j - N_y // 2
            kernel[i, j] = math.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel


def update_internal_P_jk(P, epsilon):
    assert len(P.shape) == 3, "P should be a 3D matrix. (x, y, theta)"
    updated_P = torch.clone(P)
    # for every layer
    p_x = (
        ((len(P[0])) // 2, (len(P[0]) - 1) // 2)
        if len(P[0]) % 2 == 0
        else ((len(P[0])) // 2, (len(P[0]) // 2))
    )
    p_y = (
        ((len(P[0][0])) // 2, (len(P[0][0]) - 1) // 2)
        if len(P[0][0]) % 2 == 0
        else ((len(P[0][0])) // 2, (len(P[0][0]) // 2))
    )
    padded = torch.nn.functional.pad(
        P,
        p_x + p_y,
        # mode="circular",
        mode="constant",
        value=0,
    )  # pad the tensor with zeros
    # print(padded)
    print(f"padded shape: {padded.shape}")
    updated_P = torch.nn.functional.conv2d(
        padded.unsqueeze(1),
        epsilon.unsqueeze(0).unsqueeze(0),
        padding="valid",
        stride=1,
    )
    print(f"updated shape: {updated_P.shape}")
    updated_P = updated_P.reshape(P.shape)

    return P + updated_P


def generate_delta(N_theta, sigma, gamma=2):
    delta = torch.zeros(N_theta)
    for i in range(N_theta):
        x = i - N_theta // 2
        delta[i] = math.exp(-(x**2) / (2 * sigma**2))

    return delta[
        (N_theta // 2) - gamma : (N_theta // 2) + gamma + 1
    ]  # cut off the first and last gamma values


def update_inter_layer_P_ijk(P, delta):
    assert len(P.shape) == 3, f"P should be a 3D matrix. (x, y, theta), got {P.shape}"
    updated_P = torch.clone(P)
    print(f"P shape: {P.unsqueeze(0).shape}")
    # pad theta layers circularly
    padded = torch.nn.functional.pad(
        P.permute(1, 2, 0).unsqueeze(0),
        (len(delta) // 2, len(delta) // 2) + (0, 0) + (0, 0),
        mode="circular",
    )
    print(f"padded shape: {padded.shape}")
    padded = padded.reshape(P.shape[0] * P.shape[1], 1, -1)
    # print(padded[0])
    updated_P = torch.nn.functional.conv1d(
        padded,
        delta.unsqueeze(0).unsqueeze(0),
        padding="valid",
        stride=1,
    )
    print(f"updated shape: {updated_P.shape}")
    updated_P = updated_P.permute(2, 0, 1).reshape(P.shape)
    return P + updated_P


def global_inhibition(P, inhibition_constant=0.004):
    assert len(P.shape) == 3, "P should be a 3D matrix. (x, y, theta)"
    updated_P = torch.clip(
        P + inhibition_constant * (P - torch.max(P)), min=0
    )  # clip to avoid negative values
    return updated_P


def normalize(P):
    return P / torch.sum(P)  # normalize the probabilities
