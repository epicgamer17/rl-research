from torch import nn


class NoisyDense(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma: float = None,
    ):
        super(NoisyDense, self).__init__()

    def forward(self, inputs):
        pass

    def reset_noise(self):
        pass
