import numpy as np
from .base_config import Config
from .ppo_actor_config import PPOActorConfig
from .ppo_critic_config import PPOCriticConfig
import tensorflow as tf


class PPOConfig(Config):
    def __init__(
        self,
        config_dict,
        game_config,
        actor_config: PPOActorConfig,
        critic_config: PPOCriticConfig,
    ):

        config_dict["optimizer"] = -1
        config_dict["adam_epsilon"] = -1
        config_dict["learning_rate"] = -1
        config_dict["clipnorm"] = -1
        config_dict["loss_function"] = -1
        config_dict["training_iterations"] = -1
        config_dict["min_replay_buffer_size"] = -1
        assert (
            not "replay_buffer_size" in config_dict
        ), "Replay buffer size must not be set for PPO as it is the same as steps per epoch"
        config_dict["replay_buffer_size"] = config_dict[
            "steps_per_epoch"
        ]  # times number of agents
        assert (
            not "minibatch_size" in config_dict
        ), "Minibatch size must not be set for PPO as it is the same as steps per epoch"
        config_dict["minibatch_size"] = config_dict["steps_per_epoch"]

        super(PPOConfig, self).__init__(config_dict, game_config)

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
        if "conv_layers" in config_dict:
            self.conv_layers = config_dict["conv_layers"]
        else:
            self.conv_layers = None
            print("No convolutional layers set")
            assert not (
                self.game.is_image and self.conv_layers is not None
            ), "Convolutional layers must be defined for image based games"

        if "conv_layers_noisy" in config_dict:
            self.conv_layers_noisy = config_dict["conv_layers_noisy"]
        else:
            self.conv_layers_noisy = False
            print("No convolutional layers noisy set")

        if "critic_width" in config_dict:
            self.critic_width = config_dict["critic_width"]
        else:
            self.critic_width = 128
            print("Using default critic width: 128")

        if "critic_dense_layers" in config_dict:
            self.critic_dense_layers = config_dict["critic_dense_layers"]
        else:
            self.critic_dense_layers = 1
            print("Using default number of critic dense layers: 1")

        if "critic_dense_layers_noisy" in config_dict:
            self.critic_dense_layers_noisy = config_dict["critic_dense_layers_noisy"]
        else:
            self.critic_dense_layers_noisy = False
            print("No critic dense layers noisy set")

        if "actor_width" in config_dict:
            self.actor_width = config_dict["actor_width"]
        else:
            self.actor_width = 128
            print("Using default actor width: 128")

        if "actor_dense_layers" in config_dict:
            self.actor_dense_layers = config_dict["actor_dense_layers"]
        else:
            self.actor_dense_layers = 1
            print("Using default number of actor dense layers: 1")

        if "actor_dense_layers_noisy" in config_dict:
            self.actor_dense_layers_noisy = config_dict["actor_dense_layers_noisy"]
        else:
            self.actor_dense_layers_noisy = False
            print("No actor dense layers noisy set")

        if "clip_param" in config_dict:
            self.clip_param = config_dict["clip_param"]
        else:
            self.clip_param = 0.2
            print("Using default clip param: 0.2")

        if "steps_per_epoch" in config_dict:
            self.steps_per_epoch = int(config_dict["steps_per_epoch"])
        else:
            self.steps_per_epoch = 4800
            print("Using default steps per epoch: 4800")

        # could move to actor_config
        if "train_policy_iterations" in config_dict:
            self.train_policy_iterations = int(config_dict["train_policy_iterations"])
        else:
            self.train_policy_iterations = 5
            print("Using default train policy iterations: 5")
        # could move to critic config
        if "train_value_iterations" in config_dict:
            self.train_value_iterations = int(config_dict["train_value_iterations"])
        else:
            self.train_value_iterations = 5
            print("Using default train value iterations: 5")

        if "target_kl" in config_dict:
            self.target_kl = config_dict["target_kl"]
        else:
            self.target_kl = 0.02
            print("Using default target kl: 0.02")

        if "discount_factor" in config_dict:
            self.discount_factor = config_dict["discount_factor"]
        else:
            self.discount_factor = 0.99
            print("Using default discount factor: 0.99")

        if "gae_lambda" in config_dict:
            self.gae_lambda = config_dict["gae_lambda"]
        else:
            self.gae_lambda = 0.98
            print("Using default gae lambda: 0.98")

        if "entropy_coefficient" in config_dict:
            self.entropy_coefficient = config_dict["entropy_coefficient"]
        else:
            self.entropy_coefficient = 0.001
            print("Using default entropy coefficient: 0.001")

    def _verify_game(self):
        pass
