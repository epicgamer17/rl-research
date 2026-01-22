from ..base_config import Config, DistributionalConfig, NoisyConfig, EpsilonGreedyConfig
from losses.basic_losses import CategoricalCrossentropyLoss
from utils.utils import tointlists


class RainbowConfig(Config, DistributionalConfig, NoisyConfig, EpsilonGreedyConfig):
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

        # Mixin: Noisy
        self.parse_noisy_params()  # Default 0.5 in code below, careful with mixin default

        # Mixin: Epsilon Greedy
        self.parse_epsilon_greedy_params()

        self.dueling: bool = self.parse_field("dueling", True)

        # self.discount_factor parsed in Config
        self.soft_update: bool = self.parse_field("soft_update", False)
        self.transfer_interval: int = self.parse_field(
            "transfer_interval", 512, wrapper=int
        )
        self.ema_beta: float = self.parse_field("ema_beta", 0.99)
        self.replay_interval: int = self.parse_field("replay_interval", 1, wrapper=int)

        # self.per_alpha etc parsed in Config

        # Mixin: Distributional (Atom Size)
        self.parse_distributional_params()

        # Overwriting default parse behavior for Noisy to match Rainbow default 0.5 if not present
        if "noisy_sigma" not in self.config_dict:
            self.noisy_sigma = 0.5  # Restore Rainbow Default

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

        # Logic moved to DistributionalConfig, but verifying assignment
        self.v_min = game_config.min_score
        self.v_max = game_config.max_score

        if self.atom_size != 1:
            assert self.v_min != None and self.v_max != None

    def _verify_game(self):
        assert self.game.is_discrete, "Rainbow only supports discrete action spaces"
