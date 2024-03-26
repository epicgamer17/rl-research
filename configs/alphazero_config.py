from base_config import Config
import tensorflow as tf


class AlphaZeroConfig(Config):
    def __init__(self, config_dict, game_config):
        super(AlphaZeroConfig, self).__init__(config_dict)
        self.games_per_generation = (
            config_dict["games_per_generation"]
            if "games_per_generation" in config_dict
            else 100
        )  # kind of like a replay period or steps per epoch sort of thing

        self.value_loss_factor = (
            config_dict["value_loss_factor"]
            if "value_loss_factor" in config_dict
            else 1.0
        )
        self.weight_decay = (
            config_dict["weight_decay"] if "weight_decay" in config_dict else 1e-4
        )

        self.replay_batch_size = (
            config_dict["replay_batch_size"]
            if "replay_batch_size" in config_dict
            else 32
        )

        self.root_dirichlet_alpha = (
            config_dict["root_dirichlet_alpha"]
            if "root_dirichlet_alpha" in config_dict
            else None
        )
        assert (
            self.root_dirichlet_alpha is not None
        ), "Root dirichlet alpha should be defined to a game specific value"

        self.root_exploration_fraction = (
            config_dict["root_exploration_fraction"]
            if "root_exploration_fraction" in config_dict
            else 0.25
        )
        self.num_simulations = (
            config_dict["num_simulations"] if "num_simulations" in config_dict else 800
        )
        self.num_sampling_moves = (
            config_dict["num_sampling_moves"]
            if "num_sampling_moves" in config_dict
            else 30
        )
        self.initial_temperature = (
            config_dict["initial_temperature"]
            if "initial_temperature" in config_dict
            else 1.0
        )
        self.exploitation_temperature = (
            config_dict["exploitation_temperature"]
            if "exploitation_temperature" in config_dict
            else 0.1
        )

        self.pb_c_base = (
            config_dict["pb_c_base"] if "pb_c_base" in config_dict else 19652
        )
        self.pb_c_init = (
            config_dict["pb_c_init"] if "pb_c_init" in config_dict else 1.25
        )
