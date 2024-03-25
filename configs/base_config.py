import tensorflow as tf


class Config:
    def __init__(self, config) -> None:

        # could take in a game config and set an action space and observation shape here
        # OR DO THAT IN BASE AGENT?

        if config != {}:
            self.optimizer = config["optimizer"]
            self.adam_epsilon = config["adam_epsilon"]
            self.learning_rate = config["learning_rate"]
            self.clipnorm = config["clipnorm"]

            self.replay_batch_size = int(config["replay_batch_size"])
            self.replay_period = int(config["replay_period"])
            self.replay_buffer_size = max(
                int(config["replay_buffer_size"]), self.replay_batch_size
            )
            self.min_replay_buffer_size = int(config["min_replay_buffer_size"])

            self.num_training_steps = int(config["num_training_steps"])
        else:
            self._set_default_config()

    def _set_default_config(self):
        self.optimizer = tf.keras.optimizers.legacy.Adam
        self.adam_epsilon = 1e-6
        self.learning_rate = 0.01
        self.clipnorm = None

        self.replay_batch_size = 32
        self.replay_period = 1
        self.replay_buffer_size = 1024
        self.min_replay_buffer_size = 0

        self.num_training_steps = 100
