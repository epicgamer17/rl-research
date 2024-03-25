import tensorflow as tf


class Config:
    def __init__(self, config) -> None:
        # could take in a game config and set an action space and observation shape here
        # OR DO THAT IN BASE AGENT?

        # ADD LEARNING RATE SCHEDULES

        self.optimizer = (
            config["optimizer"]
            if "optimizer" in config
            else tf.keras.optimizers.legacy.Adam
        )
        self.adam_epsilon = config["adam_epsilon"] if "adam_epsilon" in config else 1e-6
        self.learning_rate = (
            config["learning_rate"] if "learning_rate" in config else 0.01
        )
        self.clipnorm = config["clipnorm"] if "clipnorm" in config else None

        self.loss_function = (
            config["loss_function"] if "loss_function" in config else None
        )
        assert self.loss_function is not None, "Loss function must be defined"

        self.replay_batch_size = (
            int(config["replay_batch_size"]) if "replay_batch_size" in config else 32
        )
        self.replay_period = (
            int(config["replay_period"]) if "replay_period" in config else 1
        )
        self.replay_buffer_size = (
            max(int(config["replay_buffer_size"]), self.replay_batch_size)
            if "replay_buffer_size" in config
            else 1024
        )
        self.min_replay_buffer_size = (
            int(config["min_replay_buffer_size"])
            if "min_replay_buffer_size" in config
            else 0
        )

        self.training_steps = (
            int(config["training_steps"]) if "training_steps" in config else 100
        )
