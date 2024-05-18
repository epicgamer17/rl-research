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


def prepare_kernel_initializers(kernel_initializer=None):
    if kernel_initializer == "glorot_uniform":
        return GlorotUniform(seed=np.random.seed())
    elif kernel_initializer == "glorot_normal":
        return GlorotNormal(seed=np.random.seed())
    elif kernel_initializer == "he_uniform":
        return HeUniform(seed=np.random.seed())
    elif kernel_initializer == "he_normal":
        return HeNormal(seed=np.random.seed())
    elif kernel_initializer == "variance_baseline":
        return VarianceScaling(seed=np.random.seed())
    elif kernel_initializer == "variance_0.1":
        return VarianceScaling(scale=0.1, seed=np.random.seed())
    elif kernel_initializer == "variance_0.3":
        return VarianceScaling(scale=0.3, seed=np.random.seed())
    elif kernel_initializer == "variance_0.8":
        return VarianceScaling(scale=0.8, seed=np.random.seed())
    elif kernel_initializer == "variance_3":
        return VarianceScaling(scale=3, seed=np.random.seed())
    elif kernel_initializer == "variance_5":
        return VarianceScaling(scale=5, seed=np.random.seed())
    elif kernel_initializer == "variance_10":
        return VarianceScaling(scale=10, seed=np.random.seed())
    elif kernel_initializer == "lecun_uniform":
        return LecunUniform(seed=np.random.seed())
    elif kernel_initializer == "lecun_normal":
        return LecunNormal(seed=np.random.seed())
    elif kernel_initializer == "orthogonal":
        return Orthogonal(seed=np.random.seed())

    raise ValueError(f"Invalid kernel initializer: {kernel_initializer}")


def prepare_activations(activation=None):
    # print("Activation to prase: ", activation)
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


class ConfigBase:
    def parse_field(self, field_name, default=None, wrapper=None, required=True):
        if field_name in self.config_dict:
            val = self.config_dict[field_name]
            # print("value: ", val)
            print(f"Using {field_name}: {val}")
            if wrapper is not None:
                return wrapper(val)
            return self.config_dict[field_name]

        if default is not None:
            print(f"Using default {field_name}: {default}")
            if wrapper is not None:
                return wrapper(default)
            return default

        if required:
            raise ValueError(
                f"Missing required field without default value: {field_name}"
            )

    def __init__(self, config_dict: dict):
        self.config_dict = config_dict

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r") as f:
            o = yaml.load(f, yaml.Loader)
            print(o)
            a = cls(config_dict=o["config_dict"])

        return a

    def dump(self, filepath: str):
        to_dump = dict(config_dict=self.config_dict)

        with open(filepath, "w") as f:
            yaml.dump(to_dump, f, yaml.Dumper)


class Config(ConfigBase):
    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r") as f:
            o = yaml.load(f, yaml.Loader)
            print(o)
            a = cls(config_dict=o["config_dict"], game_config=o["game"])

        return a

    def dump(self, filepath: str):
        to_dump = dict(config_dict=self.config_dict, game=self.game)

        with open(filepath, "w") as f:
            yaml.dump(to_dump, f, yaml.Dumper)

    def __init__(self, config_dict: dict, game_config: GameConfig) -> None:
        super().__init__(config_dict)
        # could take in a game config and set an action space and observation shape here
        # OR DO THAT IN BASE AGENT?
        self.game = game_config

        self._verify_game()

        # not hyperparameters but utility things
        self.save_intermediate_weights: bool = self.parse_field(
            "save_intermediate_weights", True
        )

        # ADD LEARNING RATE SCHEDULES

        self.adam_epsilon: float = self.parse_field("adam_epsilon", 1e-6)
        self.learning_rate: float = self.parse_field("learning_rate", 0.01)
        self.clipnorm: int | None = self.parse_field("clipnorm", None, required=False)
        self.optimizer: Optimizer = self.parse_field(
            "optimizer",
            Adam,
            wrapper=lambda optimizer: optimizer(
                self.learning_rate, epsilon=self.adam_epsilon, clipnorm=self.clipnorm
            ),
        )
        self.loss_function: Loss = self.parse_field(
            "loss_function", None, required=False
        )
        self.training_iterations: int = self.parse_field("training_iterations", 1)
        self.num_minibatches: int = self.parse_field("num_minibatches", 1)
        self.minibatch_size: int = self.parse_field("minibatch_size", 32)
        self.replay_buffer_size: int = self.parse_field("replay_buffer_size", 1024)
        self.min_replay_buffer_size: int = self.parse_field(
            "min_replay_buffer_size", self.minibatch_size
        )
        self.training_steps: int = self.parse_field("training_steps", 10000)

        self.activation = self.parse_field(
            "activation", "relu", wrapper=prepare_activations
        )
        self.kernel_initializer = self.parse_field("kernel_initializer")

    def _verify_game(self):
        raise NotImplementedError
