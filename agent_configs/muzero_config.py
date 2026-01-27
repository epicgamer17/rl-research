from typing import Callable

from torch import Tensor

# from muzero.muzero_world_model import MuzeroWorldModel
from .base_config import (
    Config,
    SearchConfig,
    ValuePrefixConfig,
    ConsistencyConfig,
    DistributionalConfig,
    NoisyConfig,
)
from losses.basic_losses import CategoricalCrossentropyLoss, MSELoss
from utils.utils import tointlists
import copy


class MuZeroConfig(
    Config,
    SearchConfig,
    ValuePrefixConfig,
    ConsistencyConfig,
    DistributionalConfig,
    NoisyConfig,
):
    def __init__(self, config_dict, game_config):
        super(MuZeroConfig, self).__init__(config_dict, game_config)

        self.world_model_cls = self.parse_field("world_model_cls", None, required=True)
        # self.norm_type parsed in Config
        # SAME AS VMIN AND VMAX?
        self.known_bounds = self.parse_field(
            "known_bounds", default=None, required=False
        )

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

        # Mixin: Noisy
        self.parse_noisy_params()

        # Training
        self.games_per_generation: int = self.parse_field("games_per_generation", 100)
        self.value_loss_factor: float = self.parse_field("value_loss_factor", 1.0)
        self.to_play_loss_factor: float = self.parse_field("to_play_loss_factor", 1.0)
        # self.weight_decay parsed in Config

        # Mixin: Search (MCTS)
        self.parse_search_params()

        self.temperatures = self.parse_field("temperatures", [1.0, 0.0])
        self.temperature_updates = self.parse_field("temperature_updates", [5])
        self.temperature_with_training_steps = self.parse_field(
            "temperature_with_training_steps", False
        )
        assert len(self.temperatures) == len(self.temperature_updates) + 1

        self.clip_low_prob: float = self.parse_field("clip_low_prob", 0.0)

        self.value_loss_function = self.parse_field("value_loss_function", MSELoss())

        self.reward_loss_function = self.parse_field("reward_loss_function", MSELoss())

        self.policy_loss_function = self.parse_field(
            "policy_loss_function", CategoricalCrossentropyLoss()
        )

        self.to_play_loss_function = self.parse_field(
            "to_play_loss_function", CategoricalCrossentropyLoss()
        )

        # self.n_step parsed in Config
        # self.discount_factor parsed in Config
        self.unroll_steps: int = self.parse_field("unroll_steps", 5)

        # self.per_alpha, beta, epsilon etc parsed in Config

        # Mixin: Distributional (Support Range)
        self.parse_distributional_params()

        self.multi_process: bool = self.parse_field("multi_process", True)
        self.num_workers: int = self.parse_field("num_workers", 4)
        self.lr_ratio: float = self.parse_field("lr_ratio", float("inf"))
        self.transfer_interval: int = self.parse_field("transfer_interval", 1000)

        self.reanalyze_ratio: float = self.parse_field("reanalyze_ratio", 0.0)
        self.use_quantization: bool = self.parse_field("use_quantization", False)

        self.reanalyze_method: bool = self.parse_field("reanalyze_method", "mcts")
        self.reanalyze_tau: float = self.parse_field("reanalyze_tau", 0.3)
        self.injection_frac: float = self.parse_field(
            "injection_frac", 0.0
        )  # 0.25 for unplugged
        self.reanalyze_noise: bool = self.parse_field(
            "reanalyze_noise", False
        )  # true for gumbel
        self.reanalyze_update_priorities: bool = self.parse_field(
            "reanalyze_update_priorities", False
        )  # default false for most implementations

        # Mixin: Consistency
        self.parse_consistency_params()

        self.mask_absorbing = self.parse_field("mask_absorbing", True)

        # Mixin: Value Prefix
        self.parse_value_prefix_params()

        self.q_estimation_method: str = self.parse_field("q_estimation_method", "v_mix")

        self.stochastic: bool = self.parse_field("stochastic", False)
        self.use_true_chance_codes: bool = self.parse_field(
            "use_true_chance_codes", False
        )
        self.num_chance: int = self.parse_field("num_chance", 32)
        self.sigma_loss = self.parse_field("sigma_loss", CategoricalCrossentropyLoss())
        self.afterstate_residual_layers: list = self.parse_field(
            "afterstate_residual_layers", copy.deepcopy(self.dynamics_residual_layers)
        )
        self.afterstate_conv_layers: list = self.parse_field(
            "afterstate_conv_layers", copy.deepcopy(self.dynamics_conv_layers)
        )
        self.afterstate_dense_layer_widths: int = self.parse_field(
            "afterstate_dense_layer_widths",
            copy.deepcopy(self.dynamics_dense_layer_widths),
        )
        self.chance_conv_layers: list = self.parse_field(
            "chance_conv_layers", [(32, 3, 1)]
        )
        self.chance_dense_layer_widths: int = self.parse_field(
            "chance_dense_layer_widths", [256], tointlists
        )
        self.vqvae_commitment_cost_factor: float = self.parse_field(
            "vqvae_commitment_cost_factor", 1.0
        )

        self.action_embedding_dim = self.parse_field("action_embedding_dim", 32)
        self.single_action_plane = self.parse_field("single_action_plane", False)

        self.latent_viz_method = self.parse_field("latent_viz_method", "umap")
        self.latent_viz_interval = self.parse_field(
            "latent_viz_interval", 10
        )  # how often within learn() to update buffer

    def _verify_game(self):
        # override alphazero game verification since muzero can play those games
        # assert self.game.is_image, "MuZero only supports image-based games right now"
        pass
