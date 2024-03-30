from configs.agent_configs.base_config import Config
import tensorflow as tf


class RainbowConfig(Config):
    def __init__(self, config_dict, game_config):
        super(RainbowConfig, self).__init__(config_dict, game_config)

        # Network Arcitecture
        if "width" in config_dict:
            self.width = config_dict["width"]
        else:
            self.width = 128
            print("Using default width: 128")

        if "noisy_sigma" in config_dict:
            self.noisy_sigma = config_dict["noisy_sigma"]
        else:
            self.noisy_sigma = 0.5
            print("Using default noisy sigma: 0.5")

        if "conv_layers" in config_dict:
            self.conv_layers = config_dict["conv_layers"]
        else:
            self.conv_layers = None
            print("No convolutional layers set")
            assert not (
                self.game.is_image and self.conv_layers is not None
            ), "Convolutional layers must be defined for image based games"

        if "conv_layers_noisy" in config_dict:
            self.conv_layers_noisy = config_dict["conv_layers_noisy"]
        else:
            self.conv_layers_noisy = False
            print("No convolutional layers noisy set")

        if "dense_layers" in config_dict:
            self.dense_layers = config_dict["dense_layers"]
        else:
            self.dense_layers = 1
            print("Using default number of dense layers: 1")

        if "dense_layers_noisy" in config_dict:
            self.dense_layers_noisy = config_dict["dense_layers_noisy"]
        else:
            self.dense_layers_noisy = False
            print("No dense layers noisy set")

        if "value_hidden_layers" in config_dict:
            self.value_hidden_layers = config_dict["value_hidden_layers"]
        else:
            self.value_hidden_layers = 0
            print("Using default number of value hidden layers: 0")

        if "advantage_hidden_layers" in config_dict:
            self.advantage_hidden_layers = config_dict["advantage_hidden_layers"]
        else:
            self.advantage_hidden_layers = 0
            print("Using default number of advantage hidden layers: 0")

        if "discount_factor" in config_dict:
            self.discount_factor = config_dict["discount_factor"]
        else:
            self.discount_factor = 0.99
            print("Using default discount factor: 0.99")

        if "soft_update" in config_dict:
            self.soft_update = config_dict["soft_update"]
        else:
            self.soft_update = False
            print("No soft update set")

        if "transfer_interval" in config_dict:
            self.transfer_interval = int(config_dict["transfer_interval"])
        else:
            self.transfer_interval = 512
            print("Using default transfer interval: 512")

        if "ema_beta" in config_dict:
            self.ema_beta = config_dict["ema_beta"]
        else:
            self.ema_beta = 0.99
            print("Using default ema beta: 0.99")

        if "replay_interval" in config_dict:
            self.replay_interval = int(config_dict["replay_interval"])
        else:
            self.replay_interval = 4
            print("Using default replay interval: 4")

        if "per_alpha" in config_dict:
            self.per_alpha = config_dict["per_alpha"]
        else:
            self.per_alpha = 0.6
            print("Using default per alpha: 0.6")

        if "per_beta" in config_dict:
            self.per_beta = config_dict["per_beta"]
        else:
            self.per_beta = 0.5
            print("Using default per beta: 0.5")

        if "per_epsilon" in config_dict:
            self.per_epsilon = config_dict["per_epsilon"]
        else:
            self.per_epsilon = 1e-6
            print("Using default per epsilon: 1e-6")

        if "n_step" in config_dict:
            self.n_step = config_dict["n_step"]
        else:
            self.n_step = 3
            print("Using default n step: 3")

        if "atom_size" in config_dict:
            self.atom_size = config_dict["atom_size"]
        else:
            self.atom_size = 51
            print("Using default atom size: 51")

        # maybe don't use a game config, since if tuning for multiple games this should be the same regardless of the game <- (it is really a hyper parameter if you are tuning for multiple games or a game with unknown bounds)

        # could use a MuZero min-max config and just constantly update the suport size (would this break the model?) <- might mean this is not in the config but just a part of the model

        self.v_min = game_config.min_score
        self.v_max = game_config.max_score

    def _verify_game(self):
        assert self.game.is_discrete, "Rainbow only supports discrete action spaces"
