from typing import Callable, Tuple, Optional
import torch
from torch import nn, Tensor
from modules.dense import build_dense
from modules.network_block import NetworkBlock
from modules.utils import zero_weights_initializer, initialize_module
from agent_configs.base_config import Config


class BaseHead(nn.Module):
    """
    Base class for all heads. Handles optional backbone (NetworkBlock) and initialization.
    """

    def __init__(
        self,
        config: Config,
        input_shape: Tuple[int],
        layer_prefix: Optional[str] = None,
    ):
        super().__init__()
        self.config = config

        # Optional: A head can have its own "neck" of layers (e.g., Reward head has reward_conv_layers)
        if layer_prefix:
            self.backbone = NetworkBlock(config, input_shape, layer_prefix)
            self.input_flat_dim = self._get_flat_dim(self.backbone.output_shape)
        else:
            self.backbone = nn.Identity()
            self.input_flat_dim = self._get_flat_dim(input_shape)

        self.dequant = torch.ao.quantization.DeQuantStub()

    def _get_flat_dim(self, shape: Tuple[int]) -> int:
        if len(shape) == 4:  # (B, C, H, W)
            return shape[1] * shape[2] * shape[3]
        return shape[1]

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if isinstance(self.backbone, NetworkBlock):
            self.backbone.initialize(initializer)

        # Initialize the final output layer
        # If probabilistic, we often zero-init to start with uniform probability
        if self.is_probabilistic:
            if self.config.prob_layer_initializer is not None:
                self.output_layer.apply(self.config.prob_layer_initializer)
            else:
                self.output_layer.apply(zero_weights_initializer)
        elif hasattr(self.output_layer, "initialize"):
            self.output_layer.initialize(initializer)
        else:
            initialize_module(self.output_layer, initializer)

    def reset_noise(self):
        if self.backbone.reset_noise:
            self.backbone.reset_noise()
        if self.output_layer.reset_noise:
            self.output_layer.reset_noise()

    def _process_backbone(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        if x.dim() > 2:
            x = x.flatten(1, -1)
        return x


class ScalarHead(BaseHead):
    """
    Predicts a scalar quantity (Reward, Value).
    Handles switching between simple regression and MuZero support (categorical buckets).
    """

    def __init__(
        self,
        config: Config,
        input_shape: Tuple[int],
        layer_prefix: Optional[str] = None,
    ):
        super().__init__(config, input_shape, layer_prefix)

        # Determine output size based on support range
        if config.support_range is not None:
            self.output_size = 2 * config.support_range + 1
            self.is_probabilistic = True  # It outputs a distribution over buckets
        else:
            self.output_size = 1
            self.is_probabilistic = False

        self.output_layer = build_dense(
            in_features=self.input_flat_dim,
            out_features=self.output_size,
            sigma=config.noisy_sigma,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._process_backbone(x)
        x = self.output_layer(x)
        x = self.dequant(x)

        if self.is_probabilistic:
            return x.softmax(dim=-1)
        return x


class CategoricalHead(BaseHead):
    """
    Predicts a categorical distribution.
    Used for Discrete Actions (Policy) and To-Play prediction.
    """

    def __init__(
        self,
        config: Config,
        input_shape: Tuple[int],
        output_size: int,
        layer_prefix: Optional[str] = None,
    ):
        super().__init__(config, input_shape, layer_prefix)
        self.is_probabilistic = True

        self.output_layer = build_dense(
            in_features=self.input_flat_dim,
            out_features=output_size,
            sigma=config.noisy_sigma,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self._process_backbone(x)
        x = self.output_layer(x)
        x = self.dequant(x)
        return x.softmax(dim=-1)


class ContinuousHead(BaseHead):
    """
    Predicts Mean and Std for continuous actions.
    """

    def __init__(
        self,
        config: Config,
        input_shape: Tuple[int],
        output_size: int,
        layer_prefix: Optional[str] = None,
    ):
        super().__init__(config, input_shape, layer_prefix)
        self.is_probabilistic = False  # Not strictly a categorical distribution

        self.mean = build_dense(
            self.input_flat_dim, output_size, sigma=config.noisy_sigma
        )
        self.std = build_dense(
            self.input_flat_dim, output_size, sigma=config.noisy_sigma
        )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if isinstance(self.backbone, NetworkBlock):
            self.backbone.initialize(initializer)
        self.mean.initialize(initializer)
        self.std.initialize(initializer)

    def reset_noise(self):
        super().reset_noise()
        self.mean.reset_noise()
        self.std.reset_noise()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self._process_backbone(x)
        mean = self.mean(x)
        mean = self.dequant(mean).tanh()

        std = self.std(x)
        std = self.dequant(std).softplus()
        return mean, std
