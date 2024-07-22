import torch
from utils import Loss
import yaml

from game_configs import GameConfig
from utils import (
    prepare_kernel_initializers,
    prepare_activations,
)


class ConfigBase:
    def parse_field(
        self, field_name, default=None, wrapper=None, required=True, dtype=None
    ):
        if field_name in self.config_dict:
            val = self.config_dict[field_name]
            # print("value: ", val)
            print(f"Using         {field_name:30}: {val}")
            if wrapper is not None:
                return wrapper(val)
            return self.config_dict[field_name]

        if default is not None:
            print(f"Using default {field_name:30}: {default}")
            if wrapper is not None:
                return wrapper(default)
            return default

        if required:
            raise ValueError(
                f"Missing required field without default value: {field_name}"
            )
        else:
            print(f"Using         {field_name:30}: {default}")

        if field_name in self._parsed_fields:
            print("warning: duplicate field: ", field_name)
        self._parsed_fields.add(field_name)

    def __init__(self, config_dict: dict):
        self.config_dict = config_dict
        self._parsed_fields = set()

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
            "save_intermediate_weights", False
        )

        # ADD LEARNING RATE SCHEDULES
        self.training_steps: int = self.parse_field(
            "training_steps", 10000, wrapper=int
        )

        self.adam_epsilon: float = self.parse_field("adam_epsilon", 1e-6)
        self.learning_rate: float = self.parse_field("learning_rate", 0.001)
        self.clipnorm: int = self.parse_field("clipnorm", 0)
        self.optimizer: torch.optim.Optimizer = self.parse_field(
            "optimizer", torch.optim.Adam
        )
        self.weight_decay: float = self.parse_field("weight_decay", 0.0)
        self.loss_function: Loss = self.parse_field("loss_function", required=True)
        self.activation = self.parse_field(
            "activation", "relu", wrapper=prepare_activations
        )
        self.kernel_initializer = self.parse_field(
            "kernel_initializer",
            None,
            required=False,
            wrapper=kernel_initializer_wrapper,
        )

        self.minibatch_size: int = self.parse_field("minibatch_size", 64, wrapper=int)
        self.replay_buffer_size: int = self.parse_field(
            "replay_buffer_size", 5000, wrapper=int
        )
        self.min_replay_buffer_size: int = self.parse_field(
            "min_replay_buffer_size", self.minibatch_size, wrapper=int
        )
        self.num_minibatches: int = self.parse_field("num_minibatches", 1, wrapper=int)
        self.training_iterations: int = self.parse_field(
            "training_iterations", 1, wrapper=int
        )

    def _verify_game(self):
        raise NotImplementedError


def kernel_initializer_wrapper(x):
    if x is None:
        return x
    elif isinstance(x, str):
        return prepare_kernel_initializers(x)
    else:
        assert callable(x)
        return x
