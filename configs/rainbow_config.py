from base_config import Config
import tensorflow as tf


class RainbowConfig(Config):
    def __init__(self, config_dict, game_config):
        super(RainbowConfig, self).__init__(config_dict)
        self.discount_factor = (
            config_dict["discount_factor"] if "discount_factor" in config_dict else 0.99
        )

        self.soft_update = (
            config_dict["soft_update"] if "soft_update" in config_dict else False
        )
        self.transfer_frequency = (
            int(config_dict["transfer_frequency"])
            if "transfer_frequency" in config_dict
            else 512
        )
        self.ema_beta = config_dict["ema_beta"] if "ema_beta" in config_dict else 0.99

        self.per_beta = config_dict["per_beta"] if "per_beta" in config_dict else 0.5
        self.per_epsilon = (
            config_dict["per_epsilon"] if "per_epsilon" in config_dict else 1e-6
        )

        self.n_step = config_dict["n_step"] if "n_step" in config_dict else 3

        self.atom_size = config_dict["atom_size"] if "atom_size" in config_dict else 51

        # maybe don't use a game config, since if tuning for multiple games this should be the same regardless of the game <- (it is really a hyper parameter if you are tuning for multiple games or a game with unknown bounds)
        self.v_min = game_config.min_score
        self.v_max = game_config.max_score
