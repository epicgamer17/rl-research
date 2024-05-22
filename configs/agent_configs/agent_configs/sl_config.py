import tensorflow as tf
from .base_config import Config, ConfigBase
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


class SupervisedConfig(ConfigBase):
    def __init__(self, config_dict):
        super().__init__(config_dict)
        self.adam_epsilon = self.parse_field("sl_adam_epsilon", 1e-7)
        self.learning_rate = self.parse_field("sl_learning_rate", 0.005)
        self.clipnorm = self.parse_field("sl_clipnorm", None, required=False)
        self.optimizer: Optimizer = self.parse_field(
            "sl_optimizer",
            Adam,
            wrapper=lambda optimizer: optimizer(
                self.learning_rate, epsilon=self.adam_epsilon, clipnorm=self.clipnorm
            ),
        )
        self.training_steps = self.parse_field("training_steps", required=True)
        self.training_iterations = self.parse_field("sl_training_iterations", 1)
        self.num_minibatches = self.parse_field("sl_num_minibatches", 1)
        self.minibatch_size = self.parse_field("sl_minibatch_size", 32)
        self.min_replay_buffer_size = self.parse_field(
            "sl_min_replay_buffer_size", self.minibatch_size
        )
        self.replay_buffer_size = self.parse_field(
            "sl_replay_buffer_size", self.training_steps
        )
        self.activation = self.parse_field(
            "sl_activation", "relu", wrapper=prepare_activations
        )
        self.kernel_initializer = self.parse_field(
            "sl_kernel_initializer", GlorotUniform
        )

        self.conv_layers = self.parse_field("sl_conv_layers", [])
        self.conv_layers_noisy = self.parse_field("sl_conv_layers_noisy", False)
        self.dense_layers = self.parse_field("sl_dense_layers", 1)
        self.dense_layers_noisy = self.parse_field("sl_dense_layers_noisy", False)
        self.width = self.parse_field("sl_width", 128)

        self.game = None
