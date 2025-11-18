from typing import Callable

from torch import Tensor
from .base_config import Config
from utils import CategoricalCrossentropyLoss, tointlists, Loss, MSELoss
import copy


class MuZeroConfig(Config):
    def __init__(self, config_dict, game_config):
        super(MuZeroConfig, self).__init__(config_dict, game_config)
        # SAME AS VMIN AND VMAX
        self.known_bounds = self.parse_field("known_bounds", None)

        # Network Arcitecture
        self.residual_layers: list = self.parse_field(
            "residual_layers", [(256, 3, 1)] * 20
        )
        self.conv_layers: list = self.parse_field("conv_layers", [])
        self.dense_layer_widths: int = self.parse_field(
            "dense_layer_widths", [], tointlists
        )

        self.representation_residual_layers: list = self.parse_field(
            "representation_residual_layers", copy.deepcopy(self.residual_layers)
        )
        self.representation_conv_layers: list = self.parse_field(
            "representation_conv_layers", copy.deepcopy(self.conv_layers)
        )
        self.representation_dense_layer_widths: int = self.parse_field(
            "representation_dense_layer_widths", copy.deepcopy(self.dense_layer_widths)
        )

        self.dynamics_residual_layers: list = self.parse_field(
            "dynamics_residual_layers", copy.deepcopy(self.residual_layers)
        )
        self.dynamics_conv_layers: list = self.parse_field(
            "dynamics_conv_layers", copy.deepcopy(self.conv_layers)
        )
        self.dynamics_dense_layer_widths: int = self.parse_field(
            "dynamics_dense_layer_widths", copy.deepcopy(self.dense_layer_widths)
        )

        self.reward_conv_layers: list = self.parse_field(
            "reward_conv_layers", [(32, 3, 1)]
        )
        self.reward_dense_layer_widths: int = self.parse_field(
            "reward_dense_layer_widths", [256], tointlists
        )

        self.to_play_conv_layers: list = self.parse_field(
            "to_play_conv_layers", [(32, 3, 1)]
        )
        self.to_play_dense_layer_widths: int = self.parse_field(
            "to_play_dense_layer_widths", [256], tointlists
        )

        self.critic_conv_layers: list = self.parse_field(
            "critic_conv_layers", [(32, 3, 1)]
        )
        self.critic_dense_layer_widths: int = self.parse_field(
            "critic_dense_layer_widths", [256], tointlists
        )
        self.actor_conv_layers: list = self.parse_field(
            "actor_conv_layers", [(32, 3, 1)]
        )
        self.actor_dense_layer_widths: int = self.parse_field(
            "actor_dense_layer_widths", [256], tointlists
        )

        self.noisy_sigma: float = self.parse_field("noisy_sigma", 0.0)

        # Training
        self.games_per_generation: int = self.parse_field("games_per_generation", 100)
        self.value_loss_factor: float = self.parse_field("value_loss_factor", 1.0)
        self.to_play_loss_factor: float = self.parse_field("to_play_loss_factor", 1.0)
        self.weight_decay: float = self.parse_field("weight_decay", 1e-4)

        # MCTS
        self.root_dirichlet_alpha: float = self.parse_field(
            "root_dirichlet_alpha", required=True
        )
        self.root_exploration_fraction: float = self.parse_field(
            "root_exploration_fraction", 0.25
        )
        self.num_simulations: int = self.parse_field("num_simulations", 800)

        self.temperatures = self.parse_field("temperatures", [1.0, 0.0])
        self.temperature_updates = self.parse_field("temperature_updates", [5])
        self.temperature_with_training_steps = self.parse_field(
            "temperature_with_training_steps", False
        )
        assert len(self.temperatures) == len(self.temperature_updates) + 1

        self.clip_low_prob: float = self.parse_field("clip_low_prob", 0.0)
        self.pb_c_base: int = self.parse_field("pb_c_base", 19652)
        self.pb_c_init: float = self.parse_field("pb_c_init", 1.25)

        self.value_loss_function: Loss = self.parse_field(
            "value_loss_function", MSELoss()
        )

        self.reward_loss_function: Loss = self.parse_field(
            "reward_loss_function", MSELoss()
        )

        self.policy_loss_function: Loss = self.parse_field(
            "policy_loss_function", CategoricalCrossentropyLoss()
        )

        self.to_play_loss_function: Loss = self.parse_field(
            "to_play_loss_function", CategoricalCrossentropyLoss()
        )

        self.action_function: Callable = self.parse_field(
            "action_function", required=True
        )

        self.n_step: int = self.parse_field("n_step", 5)
        self.discount_factor: float = self.parse_field("discount_factor", 1.0)
        self.unroll_steps: int = self.parse_field("unroll_steps", 5)

        self.per_alpha: float = self.parse_field("per_alpha", 0.5)
        self.per_beta: float = self.parse_field("per_beta", 0.5)
        self.per_beta_final: float = self.parse_field("per_beta_final", 1.0)
        self.per_epsilon: float = self.parse_field("per_epsilon", 1e-6)
        self.per_use_batch_weights: bool = self.parse_field(
            "per_use_batch_weights", False
        )
        self.per_initial_priority_max: bool = self.parse_field(
            "per_initial_priority_max", False
        )

        self.support_range: int = self.parse_field("support_range", None)

        self.multi_process: bool = self.parse_field("multi_process", True)
        self.num_workers: int = self.parse_field("num_workers", 4)
        self.lr_ratio: float = self.parse_field("lr_ratio", float("inf"))
        self.transfer_interval: int = self.parse_field("transfer_interval", 1000)

        self.gumbel: bool = self.parse_field("gumbel", False)
        self.gumbel_m = self.parse_field("gumbel_m", 16)
        self.gumbel_cvisit = self.parse_field("gumbel_cvisit", 50)
        self.gumbel_cscale = self.parse_field("gumbel_cscale", 1.0)

    def _verify_game(self):
        # override alphazero game verification since muzero can play those games
        # assert self.game.is_image, "MuZero only supports image-based games right now"
        pass
