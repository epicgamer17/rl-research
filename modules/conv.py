from typing import Callable, Tuple

from torch import nn, Tensor
from utils import calculate_padding


def unpack(x: int | Tuple):
    if isinstance(x, Tuple):
        assert len(x) == 2
        return x
    else:
        try:
            x = int(x)
            return x, x
        except Exception as e:
            print(f"error converting {x} to int: ", e)


class Conv2dStack(nn.Module):
    @staticmethod
    def calculate_same_padding(i, k, s) -> Tuple[None | Tuple[int], None | str | Tuple]:
        """Calculate pytorch inputs for same padding

        Args:
            i (int, int) or int: (h, w) or (w, w)
            k (int, int) or int: (k_h, k_w) or (k, k)
            s (int, int) or int: (s_h, s_w) or (s, s)

        Returns:
            Tuple[manual_pad_padding, torch_conv2d_padding_input]: Either the manual padding that must be applied (first element of tuple) or the input to the torch padding argument of the Conv2d layer
        """

        if s == 1:
            return None, "same"
        h, w = unpack(i)
        k_h, k_w = unpack(k)
        s_h, s_w = unpack(s)
        p_h = calculate_padding(h, k_h, s_h)
        p_w = calculate_padding(w, k_w, s_w)
        if p_h[0] == p_h[1] and p_w[0] == p_w[1]:
            return None, (p_h[0], p_w[0])
        else:
            # not torch compatiable, manually pad with torch.nn.functional.pad
            return (*p_w, *p_h), None

    def __init__(
        self,
        input_shape: tuple[int],
        filters: list[int],
        kernel_sizes: list[int | Tuple[int, int]],
        strides: list[int | Tuple[int, int]],
        activation: nn.Module = nn.ReLU(),
        noisy_sigma: float = 0,
    ):
        """A sequence of convolution layers with the activation function applied after each layer.
        Always applies the minimum zero-padding that ensures the output shape is equal to the input shape.
        Input shape in "BCHW" form, i.e. (batch_size, input_channels, height, width)
        """
        super(Conv2dStack, self).__init__()
        self.conv_layers = nn.ModuleList()

        self.activation = activation

        # [B, C_in, H, W]
        assert len(input_shape) == 4
        assert len(filters) == len(kernel_sizes) == len(strides)
        assert len(filters) > 0

        self.noisy = noisy_sigma != 0
        if self.noisy:
            print("warning: Noisy convolutions not implemented yet")
            # raise NotImplementedError("")

        current_input_channels = input_shape[1]

        for i in range(len(filters)):
            h, w = input_shape[2], input_shape[3]
            manual_padding, torch_padding = self.calculate_same_padding(
                (h, w), kernel_sizes[i], strides[i]
            )

            if not torch_padding is None:
                layer = nn.Conv2d(
                    in_channels=current_input_channels,
                    out_channels=filters[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=torch_padding,
                )
            else:
                layer = nn.Sequential(
                    nn.ZeroPad2d(manual_padding),
                    nn.Conv2d(
                        in_channels=current_input_channels,
                        out_channels=filters[i],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                    ),
                )

            self.conv_layers.append(layer)
            current_input_channels = filters[i]

        self._output_len = current_input_channels

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        def initialize_if_conv(m: nn.Module):
            if isinstance(m, nn.Conv2d):
                initializer(m.weight)

        self.apply(initialize_if_conv)

    def forward(self, inputs):
        x = inputs
        for layer in self.conv_layers:
            x = self.activation(layer(x))
        return x

    def reset_noise(self):
        assert self.noisy

        # noisy not implemented

        # for layer in self.conv_layers:
        #     # layer.reset_noise()
        # return

    @property
    def output_channels(self):
        return self._output_len
