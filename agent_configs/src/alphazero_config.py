from base_config import Config


class AlphaZeroConfig(Config):
    def __init__(self, config_dict, game_config):
        super().__init__(config_dict, game_config)

        # Network Arcitecture
        self.kernel_size: int = self.parse_field("kernel_size", 3)
        self.num_filters: int = self.parse_field("num_filters", 256)
        self.residual_blocks: int = self.parse_field("residual_blocks", 20)
        self.critic_conv_layers: int = self.parse_field("critic_conv_layers", 1)
        self.critic_conv_filters: int = self.parse_field("critic_conv_filters", 32)
        self.critic_dense_layers: int = self.parse_field("critic_dense_layers", 1)
        self.critic_dense_size: int = self.parse_field("critic_dense_size", 256)
        self.actor_conv_layers: int = self.parse_field("actor_conv_layers", 1)
        self.actor_conv_filters: int = self.parse_field("actor_conv_filters", 32)
        self.actor_dense_layers: int = self.parse_field("actor_dense_layers", 0)
        self.actor_dense_size: int = self.parse_field("actor_dense_size", 256)

        # Training
        self.games_per_generation: int = self.parse_field("games_per_generation", 100)
        self.value_loss_factor: float = self.parse_field("value_loss_factor", 1.0)
        self.weight_decay: float = self.parse_field("weight_decay", 1e-4)

        # MCTS
        self.root_dirichlet_alpha: float = self.parse_field(
            "root_dirichlet_alpha", required=False
        )
        if self.root_dirichlet_alpha is None:
            print("Root dirichlet alpha should be defined to a game specific value")

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
