import torch


def difference_of_guassians(v: torch.Tensor, alpha: float, sigma1: float, sigma2: float):
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
    rescaled = (points * 2 * torch.pi / grid_size) - torch.pi # (N x d)

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