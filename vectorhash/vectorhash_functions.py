import torch
from functools import reduce


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
    c =[]
    b= []
    p = reduce((lambda x, y: x * y), modules)
    for i in range(len(modules)):
        c.append(p/modules[i])
    for i in range(len(c)):
        b.append(modulo_inverse(modules[i], c[i]))
    
    sum = 0
    for i in range(len(modules)):
        sum = sum + (remainders[i] * b[i] * c[i])
    return (sum % p)

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
        while ((target*i) % modulo) != 1:
            i = i+1
    return i


if __name__ == "__main__":
    print(chinese_remainder_theorem([3,5,7], [1,2,3]))

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
    for i in range(a-1):
        ## in spot so add curve(n-1, mods)
        addcurves(dim - 1, modules, velocities)
        ## add a vector of dimesnion n, all 0 but a 1 in the nth dimension
    ## now add one last vector like a[n] + a[n+1]
    b = torch.zeros(dims)
    # open the torch array and set the nth and n+1th dimension to 1
    b[dim] = 1
    if dim != (dims - 1):
        b[dim + 1] = 1
    velocities.append(b)
    
    return velocities

modules = [(2,3), (3,4)]
v = spacefillingcurve(modules)
