import math
import torch


# for every pose cell, calculate the velocity shift
def inject_activity(P, v, theta, omega, k_x=1, k_y=1, k_theta=1):
    updated_P = torch.clone(P)
    v_x = v * math.cos(theta) * k_x
    v_y = v * math.sin(theta) * k_y
    v_theta = k_theta * omega
    print(f"v_x: {v_x}, v_y: {v_y}, v_theta: {v_theta}")
    delta_x = math.floor(v_x)
    delta_f_x = v_x - delta_x
    delta_y = math.floor(v_y)
    delta_f_y = v_y - delta_y
    delta_theta = math.floor(v_theta)
    delta_f_theta = v_theta - delta_theta
    print(
        f"delta_f_x: {delta_f_x}, delta_x: {delta_x}, delta_f_y: {delta_f_y}, delta_y: {delta_y}, delta_f_theta: {delta_f_theta}, delta_theta: {delta_theta}"
    )
    kD, kH, kW = min(2, P.shape[0]), min(2, P.shape[1]), min(2, P.shape[2])

    alpha = calculate_alpha(
        delta_f_x,
        delta_f_y,
        delta_f_theta,
        shape=(kD, kH, kW),
        device=P.device
    )

    shifted = torch.roll(P, shifts=(delta_theta, delta_x, delta_y), dims=(0, 1, 2))
    print("shifted", shifted)
    padded = torch.nn.functional.pad(
        shifted.unsqueeze(0).unsqueeze(0),
        (kW // 2, 0, kH // 2, 0, kD // 2, 0),
        mode="circular",
    )  # pad with circular padding
    # apply the convulution in the flipped directions
    print("padded", padded)
    flipped = torch.flip(padded, dims=(0, 1, 2, 3, 4))
    print("flipped", flipped)
    updated_P = (
        torch.nn.functional.conv3d(
            flipped,
            alpha.unsqueeze(0).unsqueeze(0),
            stride=1,
            padding=0,
        )
        .squeeze(0)
        .squeeze(0)
    )
    print("updated_P", updated_P)
    return updated_P.flip(0, 1, 2)  # flip back to original orientation


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


def calculate_alpha(
    delta_f_x, delta_f_y, delta_f_theta, shape=(2, 2, 2), device=None
):  # delta_x, delta_y, delta_theta
    alpha = torch.zeros(shape, device=device)
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            for k in range(0, shape[2]):
                alpha[i][j][k] = g(delta_f_theta, i) * g(delta_f_x, j) * g(delta_f_y, k)
    print("alpha", alpha)
    return alpha


# this seemingly has some errors in the paper as b can be values other than 0 or 1, but we can have any (integer) value for b (maybe we need to tune k? or is it mod?)
def g(a, b):
    if b == 0:
        return 1 - a
    else:
        return a
