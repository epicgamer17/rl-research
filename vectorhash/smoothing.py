import torch

class Smoothing:
    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Smooth a batch of tensors.

        Input shape: (B, ...)

        """
        pass

    def __str__(self):
        return self.__class__.__name__


class SoftmaxSmoothing(Smoothing):
    def __init__(self, T=1e-3):
        super().__init__()
        assert T > 0
        self.T = T

    def __call__(self, x):
        y = x.flatten(1).T
        maxes = torch.max(y, dim=0).values
        y = y - maxes
        exp = torch.exp(y / self.T)
        out = (exp / torch.sum(exp, dim=0)).T
        return out.reshape(*x.shape)
    
    def __str__(self):
        return super().__str__() + f" (T={self.T})"


class PolynomialSmoothing(Smoothing):
    def __init__(self, k):
        super().__init__()
        assert k > 0
        self.k = k
        pass

    def __call__(self, x):
        y = x.flatten(1).T
        y = y**self.k
        out = (y / torch.sum(y, dim=0)).T
        return out.reshape(*x.shape)
    
    def __str__(self):
        return super().__str__() + f" (k={self.k})"


class ArgmaxSmoothing(Smoothing):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        y = x.flatten(1).T
        maxes = torch.max(y, dim=0).values
        y = torch.where(y == maxes, torch.ones_like(y), torch.zeros_like(y))
        scaled = (y / torch.sum(y, dim=0, keepdim=True)).T
        return scaled
    
    def __str__(self):
        return super().__str__()

class IdentitySmoothing(Smoothing):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x.detach().clone()
    
    def __str__(self):
        return super().__str__()