import math
import torch


# for every pose cell, calculate the velocity shift
def inject_activity(P, v, k_x=1, k_y=1, k_theta=1):
    """P shape: (x, y, theta)"""
    v_x, v_y, v_theta = k_x * v[0], k_y * v[1], k_theta * v[2]
    delta_x, delta_y, delta_theta = (
        math.floor(v_x),
        math.floor(v_y),
        math.floor(v_theta),
    )
    delta_f_x, delta_f_y, delta_f_theta = (
        v_x - delta_x,
        v_y - delta_y,
        v_theta - delta_theta,
    )
    alpha_x_length, alpha_y_length, alpha_theta_length = (
        min(2, P.shape[0]),
        min(2, P.shape[1]),
        min(2, P.shape[2]),
    )
    x_padding, y_padding, theta_padding = (
        (alpha_x_length // 2, 0),
        (alpha_y_length // 2, 0),
        (alpha_theta_length // 2, 0),
    )
    alpha = calculate_alpha(
        delta_f_x,
        delta_f_y,
        delta_f_theta,
        shape=(alpha_x_length, alpha_y_length, alpha_theta_length),
        device=P.device,
    )
    alpha_flipped = torch.flip(alpha, (0, 1, 2))
    shifted = torch.roll(P, shifts=(delta_x, delta_y, delta_theta), dims=(0, 1, 2))
    padded = torch.nn.functional.pad(
        shifted.unsqueeze(0).unsqueeze(0),
        theta_padding + y_padding + x_padding,
        mode="circular",
    )
    updated_P = torch.nn.functional.conv3d(
        padded,
        alpha_flipped.unsqueeze(0).unsqueeze(0),
        padding=0,
    )
    return updated_P.squeeze(0).squeeze(0)


def calculate_velocity_shift(P, l, m, n, delta_x, delta_y, delta_theta, alpha):
    change = 0
    # print(alpha.shape)
    for t in range(delta_theta, delta_theta + len(alpha)):
        for x in range(delta_x, delta_x + len(alpha[0])):
            for y in range(delta_y, delta_y + len(alpha[0][0])):
                change += (
                    alpha[t - delta_theta][x - delta_x][
                        y - delta_y
                    ]  # paper does x y theta (but there alpha indices are weird)
                    * P[(n + t) % len(P)][(l + x) % len(P[0])][(m + y) % len(P[0][0])]
                )

    print("change", change)
    return change


def calculate_alpha(delta_f_x, delta_f_y, delta_f_theta, shape=(2, 2, 2), device=None):
    alpha = torch.zeros(shape, device=device)
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            for k in range(0, shape[2]):
                alpha[i][j][k] = g(delta_f_x, i) * g(delta_f_y, j) * g(delta_f_theta, k)
    return alpha


# this seemingly has some errors in the paper as b can be values other than 0 or 1, but we can have any (integer) value for b (maybe we need to tune k? or is it mod?)
def g(a, b):
    if b == 0:
        return 1 - a
    else:
        return a
