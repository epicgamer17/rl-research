from agent_configs.actor_config import ActorConfig
from agent_configs.critic_config import CriticConfig
from .base_config import Config
from utils import CategoricalCrossentropyLoss, tointlists


class A2CConfig(Config):
    def __init__(self, config_dict: dict, game_config, actor_config: ActorConfig,
        critic_config: CriticConfig,
):
        super(A2CConfig, self).__init__(config_dict, game_config)
        self.shared_networks: bool = self.parse_field("shared_networks", False)
        # config_dict["optimizer"] = -1
        # config_dict["adam_epsilon"] = -1
        # config_dict["learning_rate"] = -1
        # config_dict["clipnorm"] = -1
        # config_dict["loss_function"] = -1
        # config_dict["training_iterations"] = -1
        # config_dict["min_replay_buffer_size"] = -1

        assert (
            not "replay_buffer_size" in config_dict
        ), "Replay buffer size must not be set for PPO as it is the same as steps per epoch"
        config_dict["replay_buffer_size"] = config_dict[
            "max_steps_per_episode"
        ]  # times number of agents
        # could change to just storing a config and accessing it as ppo_config.actor_config.learning_rate etc
        # self.actor_optimizer = actor_config.optimizer
        # self.actor_epsilon = actor_config.adam_epsilon
        # self.actor_learning_rate = actor_config.learning_rate
        # self.actor_clipnorm = actor_config.clipnorm
        self.actor = (
            actor_config  # maybe go back since it is not inherriting anything anymore
        )

        # could change to just storing a config and accessing it as ppo_config.critic_config.learning_rate etc
        # self.critic_optimizer = critic_config.optimizer
        # self.critic_epsilon = critic_config.adam_epsilon
        # self.critic_learning_rate = critic_config.learning_rate
        # self.critic_clipnorm = critic_config.clipnorm
        self.critic = (
            critic_config  # maybe go back since it is not inherriting anything anymore
        )

        # Network Arcitecture
        # COULD SET ALL ACTOR STUFF IN ACTOR CONFIG AND CRITIC STUFF IN CRITIC CONFIG FOR NETWORK ARCHITECTURE
        self.steps_per_epoch = self.parse_field("max_steps_per_episode", 500)
        self.noisy_sigma = self.parse_field("noisy_sigma", 0.0)
        self.critic_conv_layers = self.parse_field("conv_layers", [])
        self.actor_conv_layers = self.parse_field("conv_layers", [])
        self.critic_dense_layer_widths = self.parse_field("critic_dense_layers", [])
        self.actor_dense_layer_widths = self.parse_field("actor_dense_layers", [])
        self.max_steps_per_episode = self.parse_field("max_steps_per_episode", 1000)
        self.clip_param = self.parse_field("clip_param", 0.2)
        self.replay_buffer_size = self.parse_field(
            "replay_buffer_size", self.max_steps_per_episode
        )
        self.discount_factor: float = self.parse_field("discount_factor", 0.99)
        self.entropy_coefficient = self.parse_field("entropy_coefficient", 0.01)
        self.critic_coefficient = self.parse_field("critic_coefficient", 0.5)

        self.clip_low_prob = self.parse_field("clip_low_prob", 0.00)


        assert not (
            self.game.is_image
            and self.actor_conv_layers is not None
            and self.critic_conv_layers is not None
        ), "Convolutional layers must be defined for image based games"

    def _verify_game(self):
        pass

