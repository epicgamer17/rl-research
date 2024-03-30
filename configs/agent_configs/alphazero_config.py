from configs.agent_configs.base_config import Config
import tensorflow as tf


class AlphaZeroConfig(Config):
    def __init__(self, config_dict, game_config):
        super(AlphaZeroConfig, self).__init__(config_dict, game_config)

        # Network Arcitecture
        if "kernel_size" in config_dict:
            self.kernel_size = config_dict["kernel_size"]
        else:
            self.kernel_size = 3
            print("Using default kernel size: 3")

        if "num_filters" in config_dict:
            self.num_filters = config_dict["num_filters"]
        else:
            self.num_filters = 256
            print("Using default number of filters: 256")

        if "residual_blocks" in config_dict:
            self.residual_blocks = config_dict["residual_blocks"]
        else:
            self.residual_blocks = 20
            print("Using default number of residual blocks: 20")

        if "critic_conv_layers" in config_dict:
            self.critic_conv_layers = config_dict["critic_conv_layers"]
        else:
            self.critic_conv_layers = 1
            print("Using default number of critic conv layers: 1")

        if "critic_conv_filters" in config_dict:
            self.critic_conv_filters = config_dict["critic_conv_filters"]
        else:
            self.critic_conv_filters = 32
            print("Using default number of critic conv filters: 32")

        if "critic_dense_layers" in config_dict:
            self.critic_dense_layers = config_dict["critic_dense_layers"]
        else:
            self.critic_dense_layers = 1
            print("Using default number of critic dense layers: 1")

        if "critic_dense_size" in config_dict:
            self.critic_dense_size = config_dict["critic_dense_size"]
        else:
            self.critic_dense_size = 256
            print("Using default number of critic dense size: 256")

        if "actor_conv_layers" in config_dict:
            self.actor_conv_layers = config_dict["actor_conv_layers"]
        else:
            self.actor_conv_layers = 1
            print("Using default number of actor conv layers: 1")

        if "actor_conv_filters" in config_dict:
            self.actor_conv_filters = config_dict["actor_conv_filters"]
        else:
            self.actor_conv_filters = 32
            print("Using default number of actor conv filters: 32")

        if "actor_dense_layers" in config_dict:
            self.actor_dense_layers = config_dict["actor_dense_layers"]
        else:
            self.actor_dense_layers = 0
            print("Using default number of actor dense layers: 0")

        if "actor_dense_size" in config_dict:
            self.actor_dense_size = config_dict["actor_dense_size"]
        else:
            self.actor_dense_size = 256
            print("Using default number of actor dense size: 256")

        # Training
        if "games_per_generation" in config_dict:
            self.games_per_generation = config_dict["games_per_generation"]
        else:
            self.games_per_generation = 100
            print("Using default games per generation: 100")

        if "value_loss_factor" in config_dict:
            self.value_loss_factor = config_dict["value_loss_factor"]
        else:
            self.value_loss_factor = 1.0
            print("Using default value loss factor: 1.0")

        if "weight_decay" in config_dict:
            self.weight_decay = config_dict["weight_decay"]
        else:
            self.weight_decay = 1e-4
            print("Using default weight decay: 1e-4")

        # MCTS
        if "root_dirichlet_alpha" in config_dict:
            self.root_dirichlet_alpha = config_dict["root_dirichlet_alpha"]
        else:
            self.root_dirichlet_alpha = None
            print("Root dirichlet alpha should be defined to a game specific value")

        if "root_exploration_fraction" in config_dict:
            self.root_exploration_fraction = config_dict["root_exploration_fraction"]
        else:
            self.root_exploration_fraction = 0.25
            print("Using default root exploration fraction: 0.25")

        if "num_simulations" in config_dict:
            self.num_simulations = config_dict["num_simulations"]
        else:
            self.num_simulations = 800
            print("Using default number of simulations: 800")

        if "num_sampling_moves" in config_dict:
            self.num_sampling_moves = config_dict["num_sampling_moves"]
        else:
            self.num_sampling_moves = 30
            print("Using default number of sampling moves: 30")

        if "exploration_temperature" in config_dict:
            self.exploration_temperature = config_dict["exploration_temperature"]
        else:
            self.exploration_temperature = 1.0
            print("Using default exploration temperature: 1.0")

        if "exploitation_temperature" in config_dict:
            self.exploitation_temperature = config_dict["exploitation_temperature"]
        else:
            self.exploitation_temperature = 0.1
            print("Using default exploitation temperature: 0.1")

        if "pb_c_base" in config_dict:
            self.pb_c_base = config_dict["pb_c_base"]
        else:
            self.pb_c_base = 19652
            print("Using default pb_c_base: 19652")

        if "pb_c_init" in config_dict:
            self.pb_c_init = config_dict["pb_c_init"]
        else:
            self.pb_c_init = 1.25
            print("Using default pb_c_init: 1.25")

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
