import numpy as np


def generate_epsilon(N_x, N_y, sigma):
    """
    Generate a 2D Gaussian kernel (epsilon) for the given dimensions and standard deviation.
    """
    x = np.linspace(-N_x // 2, N_x // 2, N_x)
    y = np.linspace(-N_y // 2, N_y // 2, N_y)
    X, Y = np.meshgrid(x, y)
    epsilon = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return epsilon / np.sum(epsilon)  # Normalize the kernel


def update_P_jk(P, N_x, N_y, epsilon):
    updated_P = P.copy()
    for j in range(len(P)):
        for k in range(len(P[0])):
            for a in range(N_x):
                for b in range(N_y):
                    updated_P[j][k] += epsilon[a - j % N_y][b - k % N_y] * P[a][b]
    return updated_P
