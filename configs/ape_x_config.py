from rainbow_config import RainbowConfig
import tensorflow as tf


class ApeXConfig(RainbowConfig):
    def __init__(self, config_dict, game_config):
        super(ApeXConfig, self).__init__(config_dict)
