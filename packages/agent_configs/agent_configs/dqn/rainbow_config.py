from ..base_config import Config
from utils import Loss, CategoricalCrossentropy


class RainbowConfig(Config):
    def __init__(self, config_dict: dict, game_config):
        super(RainbowConfig, self).__init__(config_dict, game_config)

        self.conv_layers: list = self.parse_field("conv_layers", [])
        self.dense_layers_widths: int = self.parse_field("dense_layers_widths", [128])
        self.value_hidden_layers_widths = self.parse_field(
            "value_hidden_layers_widths", []
        )
        self.advantage_hidden_layers_widths: int = self.parse_field(
            "advantage_hidden_layers_widths", []
        )
        self.noisy_sigma: float = self.parse_field("noisy_sigma", 0.5)

        self.deuling: bool = self.parse_field("deuling", True)
        self.discount_factor: float = self.parse_field("discount_factor", 0.99)
        self.soft_update: bool = self.parse_field("soft_update", False)
        self.transfer_interval: int = self.parse_field("transfer_interval", 512)
        self.ema_beta: float = self.parse_field("ema_beta", 0.99)
        self.replay_interval: int = self.parse_field("replay_interval", 4)
        self.per_alpha: float = self.parse_field("per_alpha", 0.6)
        self.per_beta: float = self.parse_field("per_beta", 0.5)
        self.per_epsilon: float = self.parse_field("per_epsilon", 1e-6)
        self.n_step: int = self.parse_field("n_step", 5)
        self.atom_size: int = self.parse_field("atom_size", 51)

        assert not (
            self.game.is_image and len(self.conv_layers) == 0
        ), "Convolutional layers must be defined for image based games"

        if len(self.conv_layers) > 0:
            assert len(self.conv_layers[0]) == 3

        # maybe don't use a game config, since if tuning for multiple games this should be the same regardless of the game <- (it is really a hyper parameter if you are tuning for multiple games or a game with unknown bounds)

        # could use a MuZero min-max config and just constantly update the suport size (would this break the model?) <- might mean this is not in the config but just a part of the model

        self.v_min = game_config.min_score
        self.v_max = game_config.max_score

        if self.atom_size != 1:
            assert self.v_min != None and self.v_max != None

        if self.loss_function == None:
            print("Manually setting rainbow loss function to categorical crossentropy")
            self.loss_function: Loss = CategoricalCrossentropy()

    def _verify_game(self):
        assert self.game.is_discrete, "Rainbow only supports discrete action spaces"
