from base_config import Config
import tensorflow as tf


class PPOCriticConfig(Config):
    def __init__(self, config_dict, game_config):
        super(PPOCriticConfig, self).__init__(config_dict)

    def _set_default_config_dict(self):
        pass
