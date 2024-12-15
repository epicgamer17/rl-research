import torch


class Initializer:
    def __init__(self, grid_size, device):
        self.grid_size = grid_size
        self.device = device

    def __call__(self, shape) -> torch.Tensor:
        return self._initialize(shape)

    def _initialize(self, shape):
        pass


class BlobInitializer(Initializer):
    """
    Initializes the grid with a guassian blob in the center
    """

    def _initialize(self, shape):
        x, y = torch.arange(shape[0]), torch.arange(shape[1])
        X, Y = torch.meshgrid(x, y, indexing="ij")
        x0, y0 = shape[0] // 2, shape[1] // 2
        grid = torch.exp(-((X - x0) ** 2 + (Y - y0) ** 2))
        return grid


class BlobInitializer100(Initializer):
    """
    Generates 100 randomly positioned blobs in the grid
    """

    def _initialize(self, shape):
        grid = torch.zeros(shape)
        x, y = torch.arange(grid.shape[0]), torch.arange(grid.shape[1])
        X, Y = torch.meshgrid(x, y)

        for i in range(100):
            x0 = torch.randint(0, grid.shape[0], (1,))
            y0 = torch.randint(0, grid.shape[1], (1,))

            grid += torch.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / 100)

        return grid


class RandomInitializer(Initializer):
    """
    Initializes the grid with random values
    """

    def __init__(self, grid_size, device, mean=1, std=0.5):
        super().__init__(grid_size, device)

        self.mean = mean
        self.std = std

    def _initialize(self, shape):
        return torch.normal(self.mean, self.std, shape)
