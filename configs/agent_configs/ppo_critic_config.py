import tensorflow as tf
from base_config import ConfigBase


class PPOCriticConfig(ConfigBase):
    def __init__(self, config_dict):
        super().__init__(config_dict)

        self.optimizer = self.parse_field("optimizer", tf.keras.optimizers.legacy.Adam)
        self.adam_epsilon = self.parse_field("adam_epsilon", 1e-7)
        self.learning_rate = self.parse_field("learning_rate", 0.005)
        self.clipnorm = self.parse_field("clipnorm", None)
