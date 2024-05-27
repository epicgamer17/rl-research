import torch
import yaml

from game_configs import GameConfig
from utils import prepare_kernel_initializers, prepare_activations


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
            raise ValueError(f"Missing required field without default value: {field_name}")

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
        self.save_intermediate_weights: bool = self.parse_field("save_intermediate_weights", True)

        # ADD LEARNING RATE SCHEDULES

        self.adam_epsilon: float = self.parse_field("adam_epsilon", 1e-6)
        self.learning_rate: float = self.parse_field("learning_rate", 0.001)
        self.clipnorm: int | None = self.parse_field("clipnorm", None, required=False)
        self.optimizer: torch.optim.Optimizer = self.parse_field("optimizer", torch.optim.Adam)
        self.loss_function: torch.nn.CrossEntropyLoss = self.parse_field(
            "loss_function", torch.nn.CrossEntropyLoss(reduction="none"), required=False
        )
        self.training_iterations: int = self.parse_field("training_iterations", 1)
        self.num_minibatches: int = self.parse_field("num_minibatches", 1)
        self.minibatch_size: int = self.parse_field("minibatch_size", 64)
        self.replay_buffer_size: int = self.parse_field("replay_buffer_size", 5000)
        self.min_replay_buffer_size: int = self.parse_field("min_replay_buffer_size", self.minibatch_size)
        self.training_steps: int = self.parse_field("training_steps", 10000)

        self.activation = self.parse_field("activation", "relu", wrapper=prepare_activations)

        self.kernel_initializer = self.parse_field(
            "kernel_initializer",
            None,
            required=False,
            wrapper=kernel_initializer_wrapper,
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
