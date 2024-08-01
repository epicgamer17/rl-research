from typing import Callable, Tuple

from torch import nn, Tensor

from agent_configs import RainbowConfig
from modules.conv import Conv2dStack
from modules.dense import DenseStack, build_dense
from utils import to_lists
from modules.residual import ResidualStack


class RainbowNetwork(nn.Module):
    def __init__(
        self,
        config: RainbowConfig,
        output_size: int,
        input_shape: Tuple[int],
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.has_residual_layers = len(config.residual_layers) > 0
        self.has_conv_layers = len(config.conv_layers) > 0
        self.has_dense_layers = len(config.dense_layer_widths) > 0
        assert (
            self.has_conv_layers or self.has_dense_layers or self.has_residual_layers
        ), "At least one of the layers should be present."

        self.has_value_hidden_layers = len(config.value_hidden_layer_widths) > 0
        self.has_advantage_hidden_layers = len(config.advantage_hidden_layer_widths) > 0
        if not self.config.dueling:
            assert not (
                self.has_value_hidden_layers or self.has_advantage_hidden_layers
            ), "Value or Advantage hidden layers are only used in dueling networks"

        self.output_size = output_size

        current_shape = input_shape
        B = current_shape[0]

        if self.has_residual_layers:
            assert (
                len(input_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(input_shape)
            filters, kernel_sizes, strides = to_lists(config.residual_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.residual_layers = ResidualStack(
                input_shape=input_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.residual_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_conv_layers:
            assert (
                len(input_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(input_shape)
            filters, kernel_sizes, strides = to_lists(config.conv_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.conv_layers = Conv2dStack(
                input_shape=input_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert len(current_shape) == 2
                initial_width = current_shape[1]

            # (B, width_in) -> (B, width_out)
            self.dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )
            current_shape = (
                B,
                self.dense_layers.output_width,
            )

        if len(current_shape) == 4:
            initial_width = current_shape[1] * current_shape[2] * current_shape[3]
        else:
            assert (
                len(current_shape) == 2
            ), "Input shape should be (B, width), got {}".format(current_shape)
            initial_width = current_shape[1]

        if self.config.dueling:
            if self.has_value_hidden_layers:
                # (B, width_in) -> (B, value_in_features) -> (B, atom_size)
                self.value_hidden_layers = DenseStack(
                    initial_width=initial_width,
                    widths=self.config.value_hidden_layer_widths,
                    activation=self.config.activation,
                    noisy_sigma=self.config.noisy_sigma,
                )
                value_in_features = self.value_hidden_layers.output_width
            else:
                value_in_features = initial_width
            # (B, value_in_features) -> (B, atom_size)
            self.value_layer = build_dense(
                in_features=value_in_features,
                out_features=config.atom_size,
                sigma=config.noisy_sigma,
            )

            if self.has_advantage_hidden_layers:
                # (B, width_in) -> (B, advantage_in_features)
                self.advantage_hidden_layers = DenseStack(
                    initial_width=initial_width,
                    widths=self.config.advantage_hidden_layer_widths,
                    activation=self.config.activation,
                    noisy_sigma=self.config.noisy_sigma,
                )
                advantage_in_features = self.advantage_hidden_layers.output_width
            else:
                advantage_in_features = initial_width
            # (B, advantage_in_features) -> (B, output_size * atom_size)
            self.advantage_layer = build_dense(
                in_features=advantage_in_features,
                out_features=output_size * config.atom_size,
                sigma=self.config.noisy_sigma,
            )
        else:
            self.distribution_layer = build_dense(
                in_features=initial_width,
                out_features=self.output_size * self.config.atom_size,
                sigma=self.config.noisy_sigma,
            )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if self.has_residual_layers:
            self.residual_layers.initialize(initializer)
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)
        if self.has_value_hidden_layers:
            self.value_hidden_layers.initialize(initializer)
        if self.has_advantage_hidden_layers:
            self.advantage_hidden_layers.initialize(initializer)
        if self.config.dueling:
            self.value_layer.initialize(initializer)
            self.advantage_layer.initialize(initializer)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.has_conv_layers:
            assert inputs.dim() == 4

        # (B, *)
        S = inputs
        # (B, C_in, H, W) -> (B, C_out, H, W)
        if self.has_residual_layers:
            S = self.residual_layers(S)

        # (B, C_in, H, W) -> (B, C_out, H, W)
        if self.has_conv_layers:
            S = self.conv_layers(S)

        # (B, *) -> (B, dense_features_in)
        S = S.flatten(1, -1)

        # (B, dense_features_in) -> (B, dense_features_out)
        if self.has_dense_layers:
            S = self.dense_layers(S)

        if self.config.dueling:
            # (B, value_hidden_in) -> (B, value_hidden_out)
            if self.has_value_hidden_layers:
                v = self.value_hidden_layers(S)
            else:
                v = S

            # (B, value_hidden_in || dense_features_out) -> (B, atom_size) -> (B, 1, atom_size)
            v: Tensor = self.value_layer(v).view(-1, 1, self.config.atom_size)

            # (B, adv_hidden_in) -> (B, adv_hidden_out)
            if self.has_advantage_hidden_layers:
                A = self.advantage_hidden_layers(S)
            else:
                A = S

            # (B, adv_hidden_out || dense_features_out) -> (B, output_size * atom_size) -> (B, output_size, atom_size)
            A: Tensor = self.advantage_layer(A).view(
                -1, self.output_size, self.config.atom_size
            )

            # (B, output_size, atom_size) -[mean(1)]-> (B, 1, atom_size)
            a_mean = A.mean(1, keepdim=True)

            # (B, 1, atom_size) +
            # (B, output_size, atom_size) +
            # (B, 1, atom_size)
            # is valid broadcasting operation
            Q = v + A - a_mean

            # -[softmax(2)]-> turns the atom dimension into a valid p.d.f.
            # ONLY CLIP FOR CATEGORICAL CROSS ENTROPY LOSS TO PREVENT NAN
            # MIGHT BE ABLE TO REMOVE CLIPPING ENTIRELY SINCE I DONT THINK THE TENSORFLOW LOSSES CAN RETURN NaN
            # q.clip(1e-3, 1)
        else:
            # (B, dense_features_out) -> (B, output_size, atom_size)
            Q = self.distribution_layer(S).view(
                -1, self.output_size, self.config.atom_size
            )

        if self.config.atom_size == 1:
            return Q.squeeze(-1)
        else:
            return Q.softmax(dim=-1)

    def reset_noise(self):
        if self.config.noisy_sigma != 0:
            if self.has_residual_layers:
                self.residual_layers.reset_noise()
            if self.has_conv_layers:
                self.conv_layers.reset_noise()
            if self.has_dense_layers:
                self.dense_layers.reset_noise()
            if self.has_value_hidden_layers:
                self.value_hidden_layers.reset_noise()
            if self.has_advantage_hidden_layers:
                self.advantage_hidden_layers.reset_noise()
            if self.config.dueling:
                self.value_layer.reset_noise()
                self.advantage_layer.reset_noise()

    def remove_noise(self):
        if self.config.noisy_sigma != 0:
            if self.has_residual_layers:
                self.residual_layers.remove_noise()
            if self.has_conv_layers:
                self.conv_layers.remove_noise()
            if self.has_dense_layers:
                self.dense_layers.remove_noise()
            if self.has_value_hidden_layers:
                self.value_hidden_layers.remove_noise()
            if self.has_advantage_hidden_layers:
                self.advantage_hidden_layers.remove_noise()
            if self.config.dueling:
                self.value_layer.remove_noise()
                self.advantage_layer.remove_noise()
