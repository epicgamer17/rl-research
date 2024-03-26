from base_config import Config
import tensorflow as tf


class PPOActorConfig(Config):
    def __init__(self, config_dict, game_config):
        super(PPOActorConfig, self).__init__(config_dict)

    def _set_default_config_dict(self):
        pass
