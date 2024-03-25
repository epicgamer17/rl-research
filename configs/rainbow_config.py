from base_config import Config
import tensorflow as tf


class RainbowConfig(Config):
    def __init__(self, config_dict, game_config):
        super(RainbowConfig, self).__init__(config_dict)
        if config_dict != {}:
            self.loss_function = config_dict["loss_function"]

            self.discount_factor = config_dict["discount_factor"]

            self.soft_update = config_dict["soft_update"]
            self.transfer_frequency = int(config_dict["transfer_frequency"])
            self.ema_beta = config_dict["ema_beta"]

            self.per_beta = config_dict["per_beta"]
            self.per_epsilon = config_dict["per_epsilon"]

            self.n_step = config_dict["n_step"]

            self.atom_size = config_dict["atom_size"]
        else:
            self._set_default_config_dict()

        self.v_min = game_config.min_score
        self.v_max = game_config.max_score

    def _set_default_config_dict(self):
        self.loss_function = tf.keras.losses.KLDivergence()

        self.discount_factor = 0.99

        self.soft_update = False
        self.transfer_frequency = 512
        self.ema_beta = 0.99

        self.per_beta = 0.5
        self.per_alpha = 0.5
        self.per_epsilon = 1e-6

        self.n_step = 3

        self.atom_size = 51
