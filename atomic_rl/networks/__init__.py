"""
Neural network architectures, initialization routines, and target network updating for atomic_rl.
"""

from . import architectures
from . import initialization
from . import update_target_net

from .architectures import (
    NoisyLinear,
    ResNetBlock2d,
    ResNetBlock1d,
    ResNetBackbone,
    ResNetBlock,
    Transpose,
    AtariCNN,
    Conv2dBackbone,
    PositionalEncoding,
    MultiHeadSelfAttention,
    TransformerBlock,
    TransformerEncoder,
    AutoregressivePolicyHead,
)
from .initialization import (
    make_gnt_init,
    layer_init_,
    lecun_uniform_,
    sparse_init_weight_,
    make_sparse_init,
)
from .update_target_net import (
    hard_update_target_network_,
    soft_update_target_network_,
)

__all__ = [
    "architectures",
    "initialization",
    "update_target_net",
    "NoisyLinear",
    "ResNetBlock2d",
    "ResNetBlock1d",
    "ResNetBackbone",
    "ResNetBlock",
    "Transpose",
    "AtariCNN",
    "Conv2dBackbone",
    "PositionalEncoding",
    "MultiHeadSelfAttention",
    "TransformerBlock",
    "TransformerEncoder",
    "AutoregressivePolicyHead",
    "make_gnt_init",
    "layer_init_",
    "lecun_uniform_",
    "sparse_init_weight_",
    "make_sparse_init",
    "hard_update_target_network_",
    "soft_update_target_network_",
]
