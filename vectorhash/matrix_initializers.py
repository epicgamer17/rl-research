import torch

class SparseMatrixInitializer:
    def __init__(self, device=None):
        self.device = device
        pass

    def __call__(self, *args, **kwds):
        pass


class SparseMatrixBySparsityInitializer(SparseMatrixInitializer):
    def __init__(self, sparsity, device=None):
        """

        :param sparsity: float, sparsity of the matrix. sparsity = 1 means all zeros, sparsity = 0 means no zeros

        """
        super().__init__(device=device)
        self.sparsity = sparsity

    def __call__(self, shape):
        mask = (torch.rand(shape, device=self.device) < 1-self.sparsity).float()
        return torch.normal(0, 1, shape, device=self.device) * mask


class SparseMatrixByScalingInitializer(SparseMatrixInitializer):
    def __init__(self, scale, mean=0, device=None):
        super().__init__(device=device)
        self.device = device
        self.mean = mean
        self.std = scale

    def __call__(self, shape):
        return torch.normal(self.mean, self.std, shape, device=self.device)
