import torch
from typing import Tuple


def generate_epsilon(N_x, N_y, sigma, device=None):
    """
    Generate a 2D Gaussian kernel (epsilon) for the given dimensions and standard deviation.
    """
    x = torch.arange(N_x, device=device) - N_x // 2
    y = torch.arange(N_y, device=device) - N_y // 2
    x, y = torch.meshgrid(x, y)
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel


def generate_delta(N_theta, sigma, gamma=2, device=None):
    theta = torch.arange(N_theta, device=device) - N_theta // 2
    delta = torch.exp(-(theta**2) / (2 * sigma**2))
    delta = delta[
        (N_theta // 2) - gamma : (N_theta // 2) + gamma + 1
    ]  # cut off the first and last gamma values
    return delta


def update_internal_P_jk(P, epsilon):
    """P shape (theta, x, y)"""
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
    # print(f"padded shape: {padded.shape}")
    updated_P = torch.nn.functional.conv2d(
        padded.unsqueeze(1),
        epsilon.unsqueeze(0).unsqueeze(0),
        padding="valid",
        stride=1,
    )
    # print(f"updated shape: {updated_P.shape}")
    updated_P = updated_P.reshape(P.shape)

    return P + updated_P


def update_inter_layer_P_ijk(P, delta):
    assert len(P.shape) == 3, f"P should be a 3D matrix. (theta, x, y), got {P.shape}"
    updated_P = torch.clone(P)
    print(f"P shape: {P.unsqueeze(0).shape}")
    # pad theta layers circularly
    padded = torch.nn.functional.pad(
        P.permute(1, 2, 0).unsqueeze(0),
        (len(delta) // 2, len(delta) // 2) + (0, 0) + (0, 0),
        mode="circular",
    )
    print(f"padded shape: {padded.shape}")
    padded = padded.reshape(P.shape[1] * P.shape[2], 1, -1)
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


def calculate_padding(i: int, k: int, s: int) -> Tuple[int, int]:
    """Calculate both padding sizes along 1 dimension for a given input length, kernel length, and stride

    Args:
        i (int): input length
        k (int): kernel length
        s (int): stride

    Returns:
        (p_1, p_2): where p_1 = p_2 - 1 for uneven padding and p_1 == p_2 for even padding
    """

    p = (i - 1) * s - i + k
    p_1 = p // 2
    p_2 = (p + 1) // 2
    return (p_1, p_2)


def update_internal_P_jk_batched(P, epsilon):
    """P shape: (B, x, y, theta)"""
    assert len(P.shape) == 4, "P should be a 4D matrix. (B, x, y, theta)"

    B, N_x, N_y, N_theta = P.shape
    x_padding = calculate_padding(N_x, epsilon.shape[0], 1)
    y_padding = calculate_padding(N_y, epsilon.shape[1], 1)

    permuted = P.permute(0, 3, 1, 2)  # to (B, theta, x, y)
    reshaped = permuted.reshape(B * N_theta, N_x, N_y)  # to (B*theta, x, y)
    padded = torch.nn.functional.pad(
        reshaped,
        y_padding + x_padding,
        mode="constant",
        value=0,
    )
    updated_P = torch.nn.functional.conv2d(
        padded.unsqueeze(1),
        epsilon.unsqueeze(0).unsqueeze(0),
    )
    updated_P = updated_P.reshape(B, N_theta, N_x, N_y)  # to (B, theta, x, y)
    updated_P = updated_P.permute(0, 2, 3, 1)  # to (B, x, y, theta)
    return P + updated_P


def update_inter_layer_P_ijk_batched(P, delta):
    """P shape: (B, x, y, theta)"""
    assert len(P.shape) == 4, "P should be a 4D matrix. (B, x, y, theta)"

    B, N_x, N_y, N_theta = P.shape
    theta_padding = calculate_padding(N_theta, delta.shape[0], 1)

    reshaped = P.reshape(B * N_x * N_y, N_theta)  # to (B*x*y, theta)
    padded = torch.nn.functional.pad(
        reshaped,
        theta_padding,
        mode="circular",
    )
    updated_P = torch.nn.functional.conv1d(
        padded.unsqueeze(1),
        delta.unsqueeze(0).unsqueeze(0),
    )
    updated_P = updated_P.reshape(B, N_x, N_y, N_theta)  # to (B, x, y, theta)
    return P + updated_P


def global_inhibition(P, inhibition_constant=0.004):
    assert len(P.shape) == 3, "P should be a 3D matrix. (x, y, theta)"
    updated_P = torch.clip(
        P + inhibition_constant * (P - torch.max(P)), min=0
    )  # clip to avoid negative values
    return updated_P


def global_inhibition_batched(P, inhibition_constant=0.004):
    """P shape: (B, x, y, theta)"""
    assert len(P.shape) == 4, "P should be a 4D matrix. (B, x, y, theta)"
    B, N_x, N_y, N_theta = P.shape

    y = P.flatten(1).T  # to (B, x*y*theta)
    updated_P = torch.clip(
        y + inhibition_constant * (y - y.max(dim=0).values), min=0
    ).T.reshape(B, N_x, N_y, N_theta)

    return updated_P


def batch_rescale(x):
    """Input shape: (B, ...)"""
    reshaped = x.flatten(1).T
    return (reshaped / reshaped.sum(dim=0, keepdim=True)).T.reshape(x.shape)
