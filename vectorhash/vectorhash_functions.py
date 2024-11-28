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
