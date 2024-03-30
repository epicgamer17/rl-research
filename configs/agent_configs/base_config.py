import tensorflow as tf
from tensorflow.keras.initializers import (
    VarianceScaling,
    Orthogonal,
    GlorotUniform,
    GlorotNormal,
    HeNormal,
    HeUniform,
    LecunNormal,
    LecunUniform,
)

from tensorflow.keras.activations import (
    relu,
    sigmoid,
    softplus,
    softsign,
    hard_sigmoid,
    elu,
    selu,
)

from tensorflow.nn import (
    silu,
    swish,
    gelu,
)

import numpy as np
from configs.game_configs.game_config import GameConfig


class Config:
    def __init__(self, config_dict, game_config: GameConfig) -> None:
        # could take in a game config and set an action space and observation shape here
        # OR DO THAT IN BASE AGENT?
        self.game = game_config

        self._verify_game()

        # ADD LEARNING RATE SCHEDULES

        if "optimizer" in config_dict:
            self.optimizer = config_dict["optimizer"]
        else:
            self.optimizer = tf.keras.optimizers.legacy.Adam
            print("Using default optimizer: Adam")

        if "adam_epsilon" in config_dict:
            self.adam_epsilon = config_dict["adam_epsilon"]
        else:
            self.adam_epsilon = 1e-6
            print("Using default Adam epsilon: 1e-6")

        if "learning_rate" in config_dict:
            self.learning_rate = config_dict["learning_rate"]
        else:
            self.learning_rate = 0.01
            print("Using default learning rate: 0.01")

        if "clipnorm" in config_dict:
            self.clipnorm = config_dict["clipnorm"]
        else:
            self.clipnorm = None
            print("No clipping norm set")

        if "loss_function" in config_dict:
            self.loss_function = config_dict["loss_function"]
        else:
            self.loss_function = None
            print("No loss function set")
            # assert self.loss_function is not None, "Loss function must be defined"

        if "training_iterations" in config_dict:
            self.training_iterations = config_dict["training_iterations"]
        else:
            self.training_iterations = 1
            print("Using default training iterations: 1")

        if "num_minibatches" in config_dict:
            self.num_minibatches = config_dict["num_minibatches"]
        else:
            self.num_minibatches = 1
            print("Using default number of minibatches: 1")

        if "minibatch_size" in config_dict:
            self.minibatch_size = config_dict["minibatch_size"]
        else:
            self.minibatch_size = 32
            print("Using default minibatch size: 32")

        if "replay_buffer_size" in config_dict:
            self.replay_buffer_size = config_dict["replay_buffer_size"]
        else:
            self.replay_buffer_size = 1024
            print("Using default replay buffer size: 1024")

        if "min_replay_buffer_size" in config_dict:
            self.min_replay_buffer_size = config_dict["min_replay_buffer_size"]
        else:
            self.min_replay_buffer_size = 0
            print("Using default min replay buffer size: 0")

        if "training_steps" in config_dict:
            self.training_steps = config_dict["training_steps"]
        else:
            self.training_steps = 10000
            print("Using default training steps: 10000")

        self.activation = self._prepare_activations(config_dict["activation"])
        self.kernel_initializer = config_dict["kernel_initializer"]

    def _verify_game(self):
        raise NotImplementedError

    def prepare_kernel_initializers(self):
        if self.kernel_initializer == "glorot_uniform":
            return GlorotUniform(seed=np.random.seed())
        elif self.kernel_initializer == "glorot_normal":
            return GlorotNormal(seed=np.random.seed())
        elif self.kernel_initializer == "he_uniform":
            return HeUniform(seed=np.random.seed())
        elif self.kernel_initializer == "he_normal":
            return HeNormal(seed=np.random.seed())
        elif self.kernel_initializer == "variance_baseline":
            return VarianceScaling(seed=np.random.seed())
        elif self.kernel_initializer == "variance_0.1":
            return VarianceScaling(scale=0.1, seed=np.random.seed())
        elif self.kernel_initializer == "variance_0.3":
            return VarianceScaling(scale=0.3, seed=np.random.seed())
        elif self.kernel_initializer == "variance_0.8":
            return VarianceScaling(scale=0.8, seed=np.random.seed())
        elif self.kernel_initializer == "variance_3":
            return VarianceScaling(scale=3, seed=np.random.seed())
        elif self.kernel_initializer == "variance_5":
            return VarianceScaling(scale=5, seed=np.random.seed())
        elif self.kernel_initializer == "variance_10":
            return VarianceScaling(scale=10, seed=np.random.seed())
        elif self.kernel_initializer == "lecun_uniform":
            return LecunUniform(seed=np.random.seed())
        elif self.kernel_initializer == "lecun_normal":
            return LecunNormal(seed=np.random.seed())
        elif self.kernel_initializer == "orthogonal":
            return Orthogonal(seed=np.random.seed())

        raise ValueError(f"Invalid kernel initializer: {self.kernel_initializer}")

    def _prepare_activations(self, activation=None):
        if activation == "linear":
            return None
        elif activation == "relu":
            return relu
        elif activation == "relu6":
            return relu(max_value=6)
        elif activation == "sigmoid":
            return sigmoid
        elif activation == "softplus":
            return softplus
        elif activation == "soft_sign":
            return softsign
        elif activation == "silu":
            return silu
        elif activation == "swish":
            return swish
        # elif activation == "log_sigmoid":
        #     return log_sigmoid
        elif activation == "hard_sigmoid":
            return hard_sigmoid
        # elif activation == "hard_silu":
        #     return hard_silu
        # elif activation == "hard_swish":
        #     return hard_swish
        # elif activation == "hard_tanh":
        #     return hard_tanh
        elif activation == "elu":
            return elu
        # elif activation == "celu":
        #     return celu
        elif activation == "selu":
            return selu
        elif activation == "gelu":
            return gelu
        # elif activation == "glu":
        #     return glu

        raise ValueError(f"Activation {activation} not recognized")
