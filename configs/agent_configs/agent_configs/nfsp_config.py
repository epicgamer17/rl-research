from .base_config import Config


class NFSPConfig(Config):
    def __init__(self, config_dict, game_config, rl_config_type):
        # rl_config_dict = {
        #     key: config_dict[key] for key in keys
        # }  # normal rainbow config
        self.rl_config = rl_config_type(config_dict, game_config)
        keys = [
            "sl_adam_epsilon",
            "sl_learning_rate",
            "sl_clipnorm",
            "sl_optimizer",
            "sl_loss_function",
            "sl_training_iterations",
            "sl_num_minibatches",
            "sl_minibatch_size",
            "sl_replay_buffer_size",
            "sl_min_replay_buffer_size",
            "sl_training_steps",
            "sl_activation",
            "sl_kernel_initializer",
        ]

        sl_config_dict = {
            key[3:]: config_dict[key] for key in keys
        }  # remove sl_ from keys
        self.sl_config = Config(sl_config_dict, game_config)

    def _verify_game(self):
        assert self.game.is_discrete, "NFSP only supports discrete action spaces"
