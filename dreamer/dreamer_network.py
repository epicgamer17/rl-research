from typing import Callable, Tuple

from torch import nn, Tensor

from modules.conv import Conv2dStack
from modules.dense import DenseStack, build_dense
from utils import to_lists
from modules.residual import ResidualStack


class SequenceModelNetwork(nn.Module):
    def __init__():
        pass


class Encoder(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        is_image: bool,
        hidden_size: int,
        norm: Callable = nn.RMSNorm,
        activation: Callable = nn.SELU,
    ):
        self.activation = activation
        self.is_image = is_image
        self.hidden_size = hidden_size
        if is_image:
            assert len(input_shape) == 3
            assert input_shape[1] == input_shape[2]
            num_layers = 0
            width = input_shape[0]  # should be square image
            while width > 6:  # go to images of size 6x6 or 4x4
                width = width // 2
                num_layers += 1

            filters = [hidden_size // 4] * num_layers
            filters[0] = hidden_size // 16
            filters[1] = hidden_size // 8
            # code uses hidden_size // 16 * [2, 3, 4, 4]
            # paper says first layer has hidden_size // 16 filters

            self.conv_layers = nn.ModuleList()
            self.norm_layers = nn.ModuleList()
            for i in range(filters):
                self.conv_layers.append(
                    nn.Conv2d(
                        in_channels=3,
                        out_channels=filters[i],
                        kernel_size=5,  # param used in DreamerV3 code
                        stride=2,
                        padding="same",
                    )
                )

                self.norm_layers.append(norm(filters[i]))

        else:
            raise NotImplementedError("Only image inputs are supported")

        self.output_layer = nn.Linear(filters[-1], hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        if self.is_image:
            for conv, norm in zip(self.conv_layers, self.norm_layers):
                x = self.activation(norm(conv(x)))
            x = x.flatten(start_dim=1)
        else:
            raise NotImplementedError("Only image inputs are supported")
        return self.activation(self.output_layer(x))


class Decoder(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        is_image: bool,
        hidden_size: int,
        norm: Callable = nn.RMSNorm,
        activation: Callable = nn.SELU,
        output_activation: Callable = nn.Sigmoid,
        output_size: int = 3,
    ):
        pass


class DynamicsPredictor(nn.Module):
    def __init__():
        pass


class RewardPredictor(nn.Module):
    def __init__():
        pass


class ContinuePredictor(nn.Module):
    def __init__():
        pass
