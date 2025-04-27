from calendar import c
from hmac import new
import math
import torch


# for every pose cell, calculate the velocity shift
def inject_activity(P, v, theta, omega, k_x=1, k_y=1, k_theta=1):
    updated_P = torch.clone(P)

    v_x = -v * math.cos(theta) * k_x
    v_y = -v * math.sin(theta) * k_y
    v_theta = k_theta * -omega
    delta_f_x, delta_x = math.modf(v_x)
    delta_f_y, delta_y = math.modf(v_y)
    delta_f_theta, delta_theta = math.modf(v_theta)

    alpha = calculate_alpha(
        delta_f_x,
        delta_f_y,
        delta_f_theta,
        shape=(min(2, P.shape[0]), min(2, P.shape[1]), min(2, P.shape[2])),
    )

    for g_theta in range(P.shape[0]):
        for g_x in range(P.shape[1]):
            for g_y in range(P.shape[2]):
                print("updating pose cell", g_theta, g_x, g_y)
                updated_P[g_theta][g_x][g_y] = calculate_velocity_shift(
                    P,
                    g_x,
                    g_y,
                    g_theta,
                    int(delta_x),
                    int(delta_y),
                    int(delta_theta),
                    alpha,
                )  # -v for same functionality as ezra had implimented, and - omega same reason

    return updated_P


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
    delta_f_x, delta_f_y, delta_f_theta, shape=(2, 2, 2)
):  # delta_x, delta_y, delta_theta
    alpha = torch.zeros(shape)
    for i in range(0, shape[0]):
        for j in range(0, shape[1]):
            for k in range(0, shape[2]):
                alpha[i][j][k] = (
                    g(
                        delta_f_x, i
                    )  # paper has i - delta_x, but its really just saying floored value gets 1 - a, and the other gets a
                    * g(delta_f_y, j)
                    * g(delta_f_theta, k)
                )
    return alpha


# this seemingly has some errors in the paper as b can be values other than 0 or 1, but we can have any (integer) value for b (maybe we need to tune k? or is it mod?)
def g(a, b):
    if b == 0:
        return 1 - a
    else:
        return a
