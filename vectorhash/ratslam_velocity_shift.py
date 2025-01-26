from calendar import c
import math
import numpy as np


# grid cell to real world size scaling
k_x = 1 / 2
k_y = 1 / 2
k_theta = 1


# for every pose cell, calculate the velocity shift
def inject_activity(pose_cells, v, theta, omega):
    for g_x in range(pose_cells.shape[0]):
        for g_y in range(pose_cells.shape[1]):
            for g_theta in range(pose_cells.shape[2]):
                pose_cells[g_x][g_y][g_theta] = pose_cells[g_x][g_y][
                    g_theta
                ] + calculate_velocity_shift(
                    pose_cells, g_x, g_y, g_theta, v, theta, omega
                )
    return pose_cells


def calculate_velocity_shift(pose_cells, l, m, n, v, theta, omega):
    change = 0
    delta_x, delta_y, delta_theta = calculate_deltas(
        v, theta, omega
    )  # speed and angular velocity of the robot
    print(delta_x, delta_y, delta_theta)
    delta_f_x, delta_f_y, delta_f_theta = calculate_delta_fs(
        delta_x, delta_y, delta_theta, theta, omega
    )
    alpha = calculate_alpha(
        delta_x, delta_y, delta_theta, delta_f_x, delta_f_y, delta_f_theta
    )
    for x in range(delta_x, delta_x + 1):
        for y in range(delta_y, delta_y + 1):
            for theta in range(delta_theta, delta_theta + 1):
                change += alpha[x][y][theta] * pose_cells[l + x][m + y][n + theta]
    print("change", change)
    return change


def calculate_deltas(
    velocity, theta, omega
):  # velocity is really a speed, theta a global direction
    delta_x = math.floor(velocity * np.cos(theta) * k_x)
    delta_y = math.floor(velocity * np.sin(theta) * k_y)
    delta_theta = k_theta * omega

    return delta_x, delta_y, delta_theta


def calculate_delta_fs(delta_x, delta_y, delta_theta, theta, omega):
    delta_f_x = k_x * np.cos(theta) - delta_x
    delta_f_y = k_y * np.sin(theta) - delta_y
    delta_f_theta = k_theta * omega - delta_theta

    return delta_f_x, delta_f_y, delta_f_theta


def calculate_alpha(delta_x, delta_y, delta_theta, delta_f_x, delta_f_y, delta_f_theta):
    alpha = np.zeros((2, 2, 2))
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                alpha[i][j][k] = (
                    g(delta_f_x, i - delta_x)
                    * g(delta_f_y, j - delta_y)
                    * g(delta_f_theta, k - delta_theta)
                )
    print("alpha", alpha)
    return alpha


# this seemingly has some errors in the paper as b can be values other than 0 or 1
def g(a, b):
    if b == 0:
        return 1 - a
    elif b == 1:
        return a
    else:
        raise ValueError("b must be 0 or 1", b)
