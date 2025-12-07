from .base_config import Config
from utils import CategoricalCrossentropyLoss, tointlists


class AlphaZeroConfig(Config):
    def __init__(self, config_dict, game_config):
        super().__init__(config_dict, game_config)

        # Network Arcitecture
        self.residual_layers: list = self.parse_field(
            "residual_layers", [(256, 3, 1)] * 20
        )
        self.conv_layers: list = self.parse_field("conv_layers", [])
        self.dense_layer_widths: int = self.parse_field(
            "dense_layer_widths", [], tointlists
        )

        self.critic_conv_layers: list = self.parse_field("conv_layers", [(32, 3, 1)])
        self.critic_dense_layer_widths: int = self.parse_field(
            "dense_layer_widths", [256], tointlists
        )
        self.actor_conv_layers: list = self.parse_field("conv_layers", [(32, 3, 1)])
        self.actor_dense_layer_widths: int = self.parse_field(
            "dense_layer_widths", [256], tointlists
        )

        self.noisy_sigma: float = self.parse_field("noisy_sigma", 0.0)

        # Training
        self.games_per_generation: int = self.parse_field("games_per_generation", 100)
        self.value_loss_factor: float = self.parse_field("value_loss_factor", 1.0)
        self.weight_decay: float = self.parse_field("weight_decay", 1e-4)

        # MCTS
        self.root_dirichlet_alpha: float = self.parse_field(
            "root_dirichlet_alpha", required=True
        )
        self.root_exploration_fraction: float = self.parse_field(
            "root_exploration_fraction", 0.25
        )
        self.num_simulations: int = self.parse_field("num_simulations", 800)
        self.num_sampling_moves: int = self.parse_field("num_sampling_moves", 30)
        self.exploration_temperature: float = self.parse_field(
            "exploration_temperature", 1.0
        )
        self.exploitation_temperature: float = self.parse_field(
            "exploitation_temperature", 0.1
        )
        self.clip_low_prob: float = self.parse_field("clip_low_prob", 0.0)
        self.pb_c_base: int = self.parse_field("pb_c_base", 19652)
        self.pb_c_init: float = self.parse_field("pb_c_init", 1.25)

    def _verify_game(self):
        assert (
            self.game.is_deterministic
        ), "AlphaZero only works for deterministic games (board games)"

        assert (
            self.game.is_discrete
        ), "AlphaZero only works for discrete action space games (board games)"

        assert (
            self.game.is_image
        ), "AlphaZero only works for image based games (board games)"

        assert (
            self.game.has_legal_moves
        ), "AlphaZero only works for games where legal moves are provided so it can do action masking (board games)"
