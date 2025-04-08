from calendar import c
from hmac import new
import math
import torch


# grid cell to real world size scaling


# for every pose cell, calculate the velocity shift
def inject_activity(pose_cells, v, theta, omega, k_x=1, k_y=1, k_theta=1):
    new_pose_cells = torch.clone(pose_cells)
    for g_x in range(pose_cells.shape[0]):
        for g_y in range(pose_cells.shape[1]):
            for g_theta in range(pose_cells.shape[2]):
                # new_pose_cells[g_x][g_y][g_theta] = pose_cells[g_x][g_y][
                #     g_theta
                # ] + calculate_velocity_shift(
                #     pose_cells, g_x, g_y, g_theta, v, theta, omega
                # )
                # not keeping old beliefs
                new_pose_cells[g_x][g_y][g_theta] = calculate_velocity_shift(
                    pose_cells, g_x, g_y, g_theta, -v, theta, -omega, k_x, k_y, k_theta
                )  # -v for same functionality as ezra had implimented, and - omega same reason

    return new_pose_cells


def calculate_velocity_shift(pose_cells, l, m, n, v, theta, omega, k_x, k_y, k_theta):
    change = 0
    delta_x, delta_y, delta_theta = calculate_deltas(
        v, theta, omega, k_x, k_y, k_theta
    )  # speed and angular velocity of the robot
    delta_f_x, delta_f_y, delta_f_theta = calculate_delta_fs(
        delta_x, delta_y, delta_theta, v, theta, omega, k_x, k_y, k_theta
    )
    alpha = calculate_alpha(delta_f_x, delta_f_y, delta_f_theta)
    # print("deltas", delta_x, delta_y, delta_theta)
    for x in range(delta_x, delta_x + 2):
        for y in range(delta_y, delta_y + 2):
            for theta in range(delta_theta, delta_theta + 2):
                change += (
                    alpha[x - delta_x][y - delta_y][
                        theta - delta_theta
                    ]  # paper does x y theta (but there alpha indices are weird)
                    * pose_cells[(l + x) % len(pose_cells)][
                        (m + y) % len(pose_cells[0])
                    ][(n + theta) % len(pose_cells[0][0])]
                )
                # print(
                #     "alpha cell value",
                #     alpha[x - delta_x][y - delta_y][theta - delta_theta],
                # )
                # print(
                #     "observed cell",
                #     pose_cells[(l + x) % len(pose_cells)][(m + y) % len(pose_cells[0])][
                #         (n + theta) % len(pose_cells[0][0])
                #     ],
                # )
                # print("intermediate change", change)
    # print("change", change)
    return change


def calculate_deltas(
    velocity, theta, omega, k_x, k_y, k_theta
):  # velocity is really a speed, theta a global direction
    delta_x = math.floor(velocity * math.cos(theta) * k_x)
    delta_y = math.floor(velocity * math.sin(theta) * k_y)
    delta_theta = math.floor(k_theta * omega)

    return delta_x, delta_y, delta_theta


def calculate_delta_fs(
    delta_x, delta_y, delta_theta, v, theta, omega, k_x, k_y, k_theta
):
    delta_f_x = (
        k_x * v * math.cos(theta) - delta_x
    )  # i think they forgot this in the paper equations
    delta_f_y = k_y * v * math.sin(theta) - delta_y
    delta_f_theta = k_theta * omega - delta_theta

    return delta_f_x, delta_f_y, delta_f_theta


def calculate_alpha(
    delta_f_x, delta_f_y, delta_f_theta
):  # delta_x, delta_y, delta_theta
    alpha = torch.zeros((2, 2, 2))
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                alpha[i][j][k] = (
                    g(
                        delta_f_x, i
                    )  # paper has i - delta_x, but its really just saying floored value gets 1 - a, and the other gets a
                    * g(delta_f_y, j)
                    * g(delta_f_theta, k)
                )
    # print("alpha", alpha)
    return alpha


# this seemingly has some errors in the paper as b can be values other than 0 or 1, but we can have any (integer) value for b (maybe we need to tune k? or is it mod?)
def g(a, b):
    if b == 0:
        return 1 - a
    else:
        return a
