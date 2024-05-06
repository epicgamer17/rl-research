from .base_config import ConfigBase
import tensorflow as tf
from tensorflow import keras
from keras.initializers import (
    VarianceScaling,
    Orthogonal,
    GlorotUniform,
    GlorotNormal,
    HeNormal,
    HeUniform,
    LecunNormal,
    LecunUniform,
    Initializer,
)

from keras.activations import (
    relu,
    sigmoid,
    softplus,
    softsign,
    hard_sigmoid,
    elu,
    selu,
)

from keras.optimizers import Optimizer, Adam

from keras.losses import Loss

from tensorflow.nn import (
    silu,
    swish,
    gelu,
)

import numpy as np
from game_configs import GameConfig

import yaml


class PPOCriticConfig(ConfigBase):
    def __init__(self, config_dict):
        super().__init__(config_dict)

        self.adam_epsilon = self.parse_field("adam_epsilon", 1e-7)
        self.learning_rate = self.parse_field("learning_rate", 0.005)
        self.clipnorm = self.parse_field("clipnorm", None)
        self.optimizer: Optimizer = self.parse_field(
            "optimizer",
            Adam,
            wrapper=lambda optimizer: optimizer(
                self.learning_rate, epsilon=self.adam_epsilon, clipnorm=self.clipnorm
            ),
        )
