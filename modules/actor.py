from modules.action_heads import ContinuousActionHead, DiscreteActionHead
from modules.conv import Conv2dStack
from packages.agent_configs.agent_configs.base_config import Config
from packages.utils.utils.utils import to_lists
from torch import nn, Tensor
from typing import Callable, Tuple
from modules.dense import DenseStack, build_dense
from modules.utils import zero_weights_initializer


class ActorNetwork(nn.Module):
    def __init__(
        self,
        config: Config,
        input_shape: Tuple[int],
        output_size: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.has_conv_layers = len(config.actor_conv_layers) > 0
        self.has_dense_layers = len(config.actor_dense_layer_widths) > 0

        current_shape = input_shape
        B = current_shape[0]
        if self.has_conv_layers:
            # WITH BATCHNORM FOR EVERY CONV LAYER
            assert len(current_shape) == 4
            filters, kernel_sizes, strides = to_lists(config.actor_conv_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.conv_layers = Conv2dStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
                norm_type=config.norm_type,
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
                assert (
                    len(current_shape) == 2
                ), "Input shape should be (B, width), got {}".format(current_shape)
                initial_width = current_shape[1]

            # (B, width_in) -> (B, width_out)
            self.dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.actor_dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
                norm_type=config.norm_type,
            )

            current_shape = (
                B,
                self.dense_layers.output_width,
            )

        if len(current_shape) == 4:
            initial_width = current_shape[1] * current_shape[2] * current_shape[3]
        else:
            assert len(current_shape) == 2
            initial_width = current_shape[1]

        if self.config.game.is_discrete:
            self.actions = DiscreteActionHead(
                config,
                input_width=initial_width,
                output_size=output_size,
            )
        else:
            self.actions = ContinuousActionHead(
                config,
                input_width=initial_width,
                output_size=output_size,
            )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)
        # Policy (actions) is the main categorical output, initialize it to zero weights.
        if self.config.game.is_discrete:
            self.actions.apply(zero_weights_initializer)
        else:
            self.actions.initialize(initializer)

    def forward(self, inputs: Tensor):
        if self.has_conv_layers:
            assert (
                inputs.dim() == 4
            ), "Input shape should be (B, C, H, W), got {}".format(inputs.shape)

        x = inputs
        if self.has_conv_layers:
            x = self.conv_layers(x)

        x = x.flatten(1, -1)
        if self.has_dense_layers:
            x = self.dense_layers(x)
        actions = self.actions(x)
        return actions

    def reset_noise(self):
        if self.has_conv_layers:
            self.conv_layers.reset_noise()
        if self.has_dense_layers:
            self.dense_layers.reset_noise()
        self.actions.reset_noise()
