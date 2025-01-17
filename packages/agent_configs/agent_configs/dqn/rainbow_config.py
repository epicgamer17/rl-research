from ..base_config import Config
from utils import CategoricalCrossentropyLoss, tointlists


class RainbowConfig(Config):
    def __init__(self, config_dict: dict, game_config):
        super(RainbowConfig, self).__init__(config_dict, game_config)
        print("RainbowConfig")
        self.residual_layers: list = self.parse_field("residual_layers", [])
        self.conv_layers: list = self.parse_field("conv_layers", [])
        self.dense_layer_widths: int = self.parse_field(
            "dense_layer_widths", [128], tointlists
        )
        self.value_hidden_layer_widths = self.parse_field(
            "value_hidden_layer_widths", [], tointlists
        )
        self.advantage_hidden_layer_widths: int = self.parse_field(
            "advantage_hidden_layer_widths", [], tointlists
        )

        self.noisy_sigma: float = self.parse_field("noisy_sigma", 0.5)
        self.eg_epsilon: float = self.parse_field("eg_epsilon", 0.00)
        self.eg_epsilon_final: float = self.parse_field("eg_epsilon_final", 0.00)
        self.eg_epsilon_decay_type: str = self.parse_field(
            "eg_epsilon_decay_type", "linear"
        )
        self.eg_epsilon_final_step: int = self.parse_field(
            "eg_epsilon_final_step", self.training_steps
        )

        self.dueling: bool = self.parse_field("dueling", True)
        self.discount_factor: float = self.parse_field("discount_factor", 0.99)
        self.soft_update: bool = self.parse_field("soft_update", False)
        self.transfer_interval: int = self.parse_field(
            "transfer_interval", 512, wrapper=int
        )
        self.ema_beta: float = self.parse_field("ema_beta", 0.99)
        self.replay_interval: int = self.parse_field("replay_interval", 1, wrapper=int)
        self.per_alpha: float = self.parse_field("per_alpha", 0.6)
        self.per_beta: float = self.parse_field("per_beta", 0.5)
        self.per_beta_final: float = self.parse_field("per_beta_final", 1.0)
        self.per_epsilon: float = self.parse_field("per_epsilon", 1e-6)
        self.n_step: int = self.parse_field("n_step", 3)
        self.atom_size: int = self.parse_field("atom_size", 51, wrapper=int)
        # assert (
        #     self.atom_size > 1
        # ), "Atom size must be greater than 1, as softmax and Q distribution to Q value calculation requires more than 1 atom"

        # assert not (
        #     self.game.is_image
        #     and len(self.conv_layers) == 0
        #     and len(self.residual_layers) == 0
        # ), "Convolutional layers must be defined for image based games"

        if len(self.conv_layers) > 0:
            assert len(self.conv_layers[0]) == 3

        # maybe don't use a game config, since if tuning for multiple games this should be the same regardless of the game <- (it is really a hyper parameter if you are tuning for multiple games or a game with unknown bounds)

        # could use a MuZero min-max config and just constantly update the suport size (would this break the model?) <- might mean this is not in the config but just a part of the model

        self.v_min = game_config.min_score
        self.v_max = game_config.max_score

        if self.atom_size != 1:
            assert self.v_min != None and self.v_max != None
        
    def _verify_game(self):
        assert self.game.is_discrete, "Rainbow only supports discrete action spaces"
