import torch
from functools import reduce
from scipy.stats import norm
import math
import numpy as np
import scipy
from scipy.constants import pi


def solve_mean(p, var=1):
    """
    Solve for the mean (mu) of a normal distribution such that P(N(mu, var) > x) = p.

    Parameters:
    x (float): The threshold value.
    p (float): The probability P(N(mu, var) > x).
    var (float, optional): The variance of the normal distribution (default is 1).

    Returns:
    float: The mean (mu) that satisfies the equation.
    """
    sigma = math.sqrt(var)  # Convert variance to standard deviation
    z = norm.ppf(1 - p)  # Inverse CDF
    mu = -sigma * z
    return mu


def calculate_big_theta(num_modules, targetp, sparsity):
    var = num_modules * sparsity
    return math.sqrt(var) * scipy.stats.norm.ppf(targetp)


def difference_of_guassians(
    v: torch.Tensor, alpha: float, sigma1: float, sigma2: float
):
    """
    Computes the difference of two guassians with different standard deviations

    :param v: torch.Tensor, input tensor (N x N) grid
    :param sigma1: float, standard deviation of the first guassian
    :param sigma2: float, standard deviation of the second guassian

    :return: torch.Tensor, difference of the two guassians
    """
    p = torch.pow(v, 2)
    return alpha * torch.exp(-sigma1 * p) - torch.exp(-sigma2 * p)


def circular_mean(points: torch.Tensor, grid_size: int):
    """
    Computes the mean of a set of N points that "wrap around" on a d-dimensional toroidal grid

    :param points: torch.Tensor, N x d tensor of points
    :param grid_size: int, length of the grid

    :return: torch.Tensor, mean of the points
    """

    # rescale points to [-pi, pi) to be viewed as angles
    rescaled = (points * 2 * torch.pi / grid_size) - torch.pi  # (N x d)

    # transform to complex numbers
    Im = torch.sin(rescaled)  # (N x d)
    Re = torch.cos(rescaled)  # (N x d)

    # compute the mean
    mean_Im = torch.mean(Im, dim=0)
    mean_Re = torch.mean(Re, dim=0)

    # compute the angle of the mean and rescale to [0, 2 * pi)
    Arg = torch.atan2(mean_Im, mean_Re) + torch.pi

    # rescale to [0, grid_size)
    return Arg * grid_size / (2 * torch.pi)


def softmax_2d(x: torch.Tensor):
    """
    Computes the softmax of a 2d tensor

    :param x: torch.Tensor, input tensor (N x N)

    :return: torch.Tensor, softmax of the input tensor
    """
    x = x - torch.max(x)
    exp = torch.exp(x)
    return exp / torch.sum(exp)


def sort_polygon_vertices(vertices: torch.Tensor):
    """
    Sorts the vertices of a polygon in counter-clockwise order

    :param vertices: torch.Tensor, N x 2 tensor of vertices

    :return: torch.Tensor, N x 2 tensor of sorted vertices
    """
    center = torch.mean(vertices, dim=0)
    angles = torch.atan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    sorted_indices = torch.argsort(angles)
    return vertices[sorted_indices]


def chinese_remainder_theorem(modules, remainders):
    """
    Finds the solution to the system of congruence equations with forms x = r1 (modulo m1) for all modules and remainders

    :param modules: Array of modules m1 -- mn
    :param remainders: Array of remainders r1--rn

    :return: integer gives solution x
    """
    c = []
    b = []
    p = reduce((lambda x, y: x * y), modules)
    for i in range(len(modules)):
        c.append(p / modules[i])
    for i in range(len(c)):
        b.append(modulo_inverse(modules[i], c[i]))

    sum = 0
    for i in range(len(modules)):
        sum = sum + (remainders[i] * b[i] * c[i])
    return sum % p


def modulo_inverse(modulo, number):
    """
    Finds the solution to the equation ax = 1 (mod m) where a, m are known

    :param modulo: integer
    :param number: integer

    :return: integer
    """

    target = number % modulo
    i = 1
    if target == 1:
        return 1
    else:
        while ((target * i) % modulo) != 1:
            i = i + 1
    return i


def spacefillingcurve(modules):
    number_of_module_dims = len(modules[0])
    assert all(len(module) == number_of_module_dims for module in modules)

    return addcurves(number_of_module_dims - 1, modules, [])


def addcurves(dim, modules, velocities):
    dims = len(modules[0])

    if dim == 0:
        a = 1
        for module in modules:
            a = a * module[dim]
        for i in range(a - 1):
            b = torch.zeros(dims)
            b[1] = 1
            velocities.append(b)
        b = torch.zeros(dims)
        b[dim] = 1
        b[dim + 1] = 1
        velocities.append(b)
        return velocities

    a = 1
    for module in modules:
        a = a * module[dim]
    for i in range(a - 1):
        ## in spot so add curve(n-1, mods)
        addcurves(dim - 1, modules, velocities)
        ## add a vector of dimesnion n, all 0 but a 1 in the nth dimension
    b = torch.zeros(dims)
    b[dim] = 1
    if dim != (dims - 1):
        b[dim + 1] = 1
    velocities.append(b)

    return velocities


def expectation_of_relu_normal(mu, std):
    """
    Calculate the expectation of a ReLU applied to a normal distribution with mean mu and standard deviation std.
    """

    mu_over_std = mu / std
    return mu * norm.cdf(mu_over_std) + std / math.sqrt(2 * pi) * np.exp(-mu_over_std ** 2 / 2)

if __name__ == "__main__":
    # print(chinese_remainder_theorem([3, 5, 7], [1, 2, 3]))
    N = 10000
    std=3
    means = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Ns = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    results = np.zeros((len(means), len(Ns)))
    for i, mean in enumerate(means):
        expected = expectation_of_relu_normal(mean, std)
        for j, N in enumerate(Ns):
            x1 = torch.relu(torch.normal(mean, std, (N,))) - expected
            x2 = torch.relu(torch.normal(mean, std, (N,))) - expected
            results[i, j] = torch.mean(x1 * x2)
            print(f"mean={mean}, N={N}, similarity={results}")
    
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for i, mean in enumerate(means):
        ax.plot(Ns, results[i, :], label=f"mean={mean}")
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("similarity")
    ax.legend()


    plt.savefig("sim.png")
