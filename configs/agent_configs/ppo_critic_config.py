import tensorflow as tf


class PPOCriticConfig:
    def __init__(self, config_dict):
        if "optimizer" in config_dict:
            self.optimizer = config_dict["optimizer"]
        else:
            self.optimizer = tf.keras.optimizers.legacy.Adam
            print("Using default critic optimizer: Adam")

        if "adam_epsilon" in config_dict:
            self.adam_epsilon = config_dict["adam_epsilon"]
        else:
            self.adam_epsilon = 1e-7
            print("Using default critic Adam epsilon: 1e-7")

        if "learning_rate" in config_dict:
            self.learning_rate = config_dict["learning_rate"]
        else:
            self.learning_rate = 0.005
            print("Using default critic learning rate: 0.005")

        if "clipnorm" in config_dict:
            self.clipnorm = config_dict["clipnorm"]
        else:
            self.clipnorm = None
            print("No critic clipping norm set")