from .base_config import Config


class DQNConfig(Config):
    def __init__(self, config_dict, game_config):
        super(DQNConfig, self).__init__(config_dict, game_config)

        # Network Arcitecture

        self.width: int = self.parse_field("width", 128)
        self.noisy_sigma: float = 0
        self.deuling: bool = False

        self.conv_layers: list = self.parse_field("conv_layers", [])
        self.dense_layers: int = self.parse_field("dense_layers", 1)
        self.discount_factor: float = self.parse_field("discount_factor", 0.99)
        self.soft_update: bool = False
        self.transfer_interval: int = 1
        self.replay_interval: int = self.parse_field("replay_interval", 4)
        self.per_alpha: float = 0
        self.per_beta: float = 0
        self.per_epsilon: float = 0
        self.n_step: int = 1
        self.atom_size: int = 1

        assert not (
            self.game.is_image and len(self.conv_layers) == 0
        ), "Convolutional layers must be defined for image based games"

        # maybe don't use a game config, since if tuning for multiple games this should be the same regardless of the game <- (it is really a hyper parameter if you are tuning for multiple games or a game with unknown bounds)

        # could use a MuZero min-max config and just constantly update the suport size (would this break the model?) <- might mean this is not in the config but just a part of the model

        self.v_min = game_config.min_score
        self.v_max = game_config.max_score

    def _verify_game(self):
        assert self.game.is_discrete, "Rainbow only supports discrete action spaces"
