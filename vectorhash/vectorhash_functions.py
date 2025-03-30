import scipy.special
import torch
from functools import reduce
from scipy.stats import norm
import math
import numpy as np
import scipy
from scipy.constants import pi
from torch import abs
from matplotlib import pyplot as plt


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


def circular_mean_2(samples: torch.Tensor, weights: torch.Tensor, high=2 * pi):
    modified = torch.where(samples > high / 2, samples - high, samples)
    weighted = modified * weights / weights.sum()
    indices = weighted.nonzero()
    if len(indices) == 0:
        return torch.zeros(1)
    mean = torch.mean(weighted[indices])
    ret = torch.where(mean > 0, mean, mean + high)

    # print(
    #     f"""
    #      samples: {samples} 
    #      weights: {weights} 
    #      modified: {modified}
    #      weighted: {modified * weights / weights.sum()}
    #      indices: {weighted.nonzero()}
    #      weighted_nonzero: {weighted[weighted.nonzero()]}
    #      mean: {mean}
    #      ret: {ret}
          
    #       """
    # )
    return ret


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


def space_filling_curve(modules):
    number_of_module_dims = len(modules[0])
    assert all(len(module) == number_of_module_dims for module in modules)

    return add_curves(number_of_module_dims - 1, modules, [])


def add_curves(dim, modules, velocities):
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
        add_curves(dim - 1, modules, velocities)
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
    return mu * norm.cdf(mu_over_std) + std / math.sqrt(2 * pi) * np.exp(
        -(mu_over_std**2) / 2
    )


if __name__ == "__main__":
    # print(chinese_remainder_theorem([3, 5, 7], [1, 2, 3]))
    N = 10000
    std = 3
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


def Rk1MrUpdate(A, A_pinv, c, d, zero_tol, Case_Print_Flag):
    # size c = [n,1]
    # size d = [m,1]
    c = c.reshape(-1, 1)
    d = d.reshape(-1, 1)
    n = c.shape[0]
    m = d.shape[0]
    V = A_pinv @ c

    b = 1 + d.T @ V
    N = A_pinv.T @ d
    W = (torch.eye(n) - A @ A_pinv) @ c
    M = (torch.eye(m) - A_pinv @ A) @ d
    ## squared norm of the two abovesaid vectors
    w_snorm = torch.norm(W, p=2) ** 2
    m_snorm = torch.norm(M, p=2) ** 2

    if w_snorm >= zero_tol and m_snorm >= zero_tol:
        if Case_Print_Flag == 1:
            print("case 1")
        G = (
            ((-1 / w_snorm) * V @ W.T)
            - ((1 / m_snorm) * (M @ N.T))
            + ((b / m_snorm / w_snorm) * (M @ W.T))
        )

    elif w_snorm < zero_tol and m_snorm >= zero_tol and abs(b) < zero_tol:
        if Case_Print_Flag == 1:
            print("case 2")
        v_snorm = torch.norm(V, 2) ** 2
        G = (-1 / v_snorm) * (V @ V.T) @ A_pinv + (-(1 / m_snorm) * M @ N.T)

    elif w_snorm < zero_tol and abs(b) > zero_tol:
        if Case_Print_Flag == 1:
            print("case 3")
        v_snorm = torch.norm(V, 2) ** 2
        G = (1 / b) * M @ V.T @ A_pinv - ((b / (v_snorm * m_snorm + b**2))) * (
            (v_snorm / b) * M + V
        ) @ ((m_snorm / b) * A_pinv.T @ V + N).T

    elif m_snorm < zero_tol and w_snorm >= zero_tol and abs(b) < zero_tol:
        if Case_Print_Flag == 1:
            print("case 4")
        n_snorm = torch.norm(N, 2) ** 2
        G = (-1 / n_snorm) * A_pinv @ (N @ N.T) + (-(1 / w_snorm) * V @ W.T)

    elif m_snorm < zero_tol and abs(b) > zero_tol:
        if Case_Print_Flag == 1:
            print("case 5")
        n_snorm = torch.norm(N, 2) ** 2
        G = ((1 / b) * A_pinv @ N @ W.T) - (
            ((b / (w_snorm * n_snorm + b**2)) * ((w_snorm / b) * A_pinv @ N + V))
            @ (((n_snorm / b) * W + N).T)
        )

    elif m_snorm < zero_tol and w_snorm < zero_tol and abs(b) < zero_tol:
        if Case_Print_Flag == 1:
            print("case 6")
        v_snorm = torch.norm(V, 2) ** 2
        n_snorm = torch.norm(N, 2) ** 2
        G = (
            ((-1 / v_snorm) * (V @ V.T) @ A_pinv)
            - ((1 / n_snorm) * A_pinv @ (N @ N.T))
            + (((V.T @ A_pinv @ N) / (v_snorm * n_snorm)) @ (V @ N.T))
        )

    A_pinv_New = A_pinv + G
    return A_pinv_New


def ConvertToXYNew(gin, shape):
    """
    Converts a 1D vector to a grid mod shape for x and y

    :param gin: torch.Tensor, 1D vector, shape (n, m)
    :param shape: tuple, shape of the grid

    :return: torch.Tensor, grid
    """

    half_prob = torch.sum(gin).item() / 2

    gin = gin.reshape(shape)

    # y sum
    x = []
    for i in range(shape[1]):
        x.append(sum(gin[:, i]))
    # x sum
    y = []
    for i in range(shape[0]):
        y.append(sum(gin[i, :]))
    y_med = continuous_median_1d(y, half=half_prob)
    x_med = continuous_median_1d(x, half=half_prob)
    return x_med, y_med


def continuous_median_1d(prob, half):
    """
    Given a 1D probability vector 'prob' (with prob.sum() == 1)
    that represents a density which is uniform over each interval [i, i+1),
    compute the continuous median coordinate.

    For example, if the median falls at x = 2.3, then 0.2 of the probability
    comes from [0,1), 0.3 from [1,2), and 0.3 (i.e. 0.3 out of the cell's mass)
    from the interval [2,3), so that the median is 2 + 0.3.
    """
    # Compute the cumulative sum of probabilities.
    cumsum = np.cumsum(prob)
    # Find the first index i where the cumulative sum >= 0.5.
    i = np.searchsorted(cumsum, half)
    # Cumulative probability before cell i.
    L = cumsum[i - 1] if i > 0 else 0.0
    # The probability mass in the cell [i, i+1)
    p_cell = prob[i]
    # Linear interpolation: fraction of the cell needed to reach 0.5.
    # (Assumes uniform density in the cell.)
    fraction = (half - L) / p_cell
    median_coordinate = i + fraction
    return median_coordinate


def GraphGrid(ax, shape, points, first_point=None, title=None):
    """
    Given a grid shape and a set of points, return the grid figure with the points

    :param shape: tuple, shape of the grid
    :param points: list, list of points

    :return: image
    """
    t = np.arange(len(points))

    if len(points) == 2:
        c = ["red", "blue"]
        for i in range(2):
            xs = [point[1] for point in points[i]]
            ys = [point[0] for point in points[i]]
            sc = ax.scatter(xs, ys, c=t, label=f"after learning {c}", cmap="viridis")
            # print number of points on the grid
            # have plot start at 0 and finish at shape + 1
            # show grid lines but only at integer values
    else:
        xs = [point[1] for point in points]
        ys = [point[0] for point in points]
        sc = ax.scatter(xs, ys, c=t, label="after learning", cmap="viridis")
        # print number of points on the grid
        print(len(points))
        # have plot start at 0 and finish at shape + 1
        # show grid lines but only at integer values

    if title is not None:
        ax.set_title(title)
    if first_point is not None:
        ax.plot(
            first_point[1],
            first_point[0],
            c="green",
            label="start",
            marker="x",
            markersize=10,
        )
    # ax.colorbar()
    ax.set_xticks(np.arange(0, shape[0], 1))
    ax.set_yticks(np.arange(0, shape[1], 1))
    ax.grid()
    ax.set_xlim(0, shape[0])
    ax.set_ylim(0, shape[1])
    ax.legend()
    return sc


def GraphGrids(points_lists, shapes_lists, first_points, titles, main_title):
    assert len(points_lists) == len(shapes_lists) == len(first_points) == len(titles)
    fig, axs = plt.subplots(1, len(points_lists), figsize=(20, 7))
    for i in range(len(points_lists)):
        sc = GraphGrid(
            axs[i], shapes_lists[i], points_lists[i], first_points[i], titles[i]
        )

    fig.suptitle(main_title)

    # add colorbar on bottom going from 0 to len(points)

    # [left, bottom, width, height]
    cax = fig.add_axes([0.1, 0.05, 0.8, 0.02])

    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
    plt.show()


def ConvertXtoYOld(gin, shape):
    """
    Converts a 1D vector to a grid

    :param gin: torch.Tensor, 1D vector

    :return: torch.Tensor, grid
    """
    # return coords of center of cell with max value
    index = torch.argmax(gin)
    x = index % shape[1]
    y = index // shape[1]
    return x + 0.5, y + 0.5


def ConvertToXYZ(gin: torch.Tensor, shape):
    half_prob = torch.sum(gin).item() / 2
    gin = gin.reshape(shape)
    [x, y, z] = [torch.sum(gin, dim=i).flatten().cpu().numpy() for i in range(3)]

    print(x.shape, y.shape, z.shape)

    x_med = continuous_median_1d(x, half=half_prob)
    y_med = continuous_median_1d(y, half=half_prob)
    z_med = continuous_median_1d(z, half=half_prob)
    return x_med, y_med, z_med


def calculate_theoretical_capacity(shapes, N_h, input_size):
    N_g = 0
    for shape in shapes:
        l = torch.prod(torch.tensor(shape)).item()
        N_g += l

    return N_g * N_h / input_size


def outer(tensors):
    einsum_indices = [chr(ord("a") + i) for i in range(len(tensors))]  # a, b, c, ...

    einsum_str = (
        ",".join(einsum_indices) + "->" + "".join(einsum_indices)
    )  # a,b,c, ...->abc...

    return torch.einsum(einsum_str, *tensors)


def generate_1d_gaussian_kernel(radius, mu=0, sigma=1, device=None):
    """
    Genereate a 1-D Gaussian convolution kernel.
    """
    x = torch.arange(-radius, radius + 1, device=device)

    low = (x - mu - 0.5) / (sigma * 2**0.5)
    high = (x - mu + 0.5) / (sigma * 2**0.5)
    w = 0.5 * (scipy.special.erf(high) - scipy.special.erf(low))
    return w
