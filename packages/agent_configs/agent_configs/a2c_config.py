from .base_config import Config
from utils import CategoricalCrossentropyLoss, tointlists


class A2CConfig(Config):
    def __init__(self, config_dict: dict, game_config):
        super(A2CConfig, self).__init__(config_dict, game_config)
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
        self.discount_factor: float = self.parse_field("discount_factor", 0.99)
        self.replay_interval: int = self.parse_field("replay_interval", 1, wrapper=int)

        self.shared_networks: bool = self.parse_field("shared_networks", False)

        if len(self.conv_layers) > 0:
            assert len(self.conv_layers[0]) == 3

    def _verify_game(self):
        pass
