"""
Neural network components layer for reinforcement learning.

Exports common network building blocks (ResNets, CNN backbones, Noisy layers, Transformer skeletons)
while keeping algorithm orchestration in example scripts.
"""

from .noisy_linear import NoisyLinear
from .resnet import (
    ResNetBlock2d,
    ResNetBlock1d,
    ResNetBackbone,
    ResNetBlock,
)
from .cnn import Transpose, AtariCNN, Conv2dBackbone
from .transformer import (
    PositionalEncoding,
    MultiHeadSelfAttention,
    TransformerBlock,
    TransformerEncoder,
)
from .autoregressive import AutoregressivePolicyHead

__all__ = [
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
]
