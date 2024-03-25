from base_config import Config
from ppo_actor_config import PPOActorConfig
from ppo_critic_config import PPOCriticConfig
import tensorflow as tf


class PPOConfig(Config):
    def __init__(
        self,
        config_dict,
        game_config,
        actor_config: PPOActorConfig,
        critic_config: PPOCriticConfig,
    ):
        super(PPOConfig, self).__init__(config_dict)

        # could check for discrete and not discrete here or in the game config

        # could change to just storing a config and accessing it as ppo_config.actor_config.learning_rate etc
        # self.actor_optimizer = actor_config.optimizer
        # self.actor_epsilon = actor_config.adam_epsilon
        # self.actor_learning_rate = actor_config.learning_rate
        # self.actor_clipnorm = actor_config.clipnorm
        self.actor = actor_config

        # could change to just storing a config and accessing it as ppo_config.critic_config.learning_rate etc
        # self.critic_optimizer = critic_config.optimizer
        # self.critic_epsilon = critic_config.adam_epsilon
        # self.critic_learning_rate = critic_config.learning_rate
        # self.critic_clipnorm = critic_config.clipnorm
        self.critic = critic_config

        self.clip_param = (
            config_dict["clip_param"] if "clip_param" in config_dict else 0.2
        )

        self.num_minibatches = (
            config_dict["num_minibatches"] if "num_minibatches" in config_dict else 4
        )  # leads to replay_batch_size because it is memory size / num minibatches = replay batch size
        self.num_epochs = self.training_steps // self.num_minibatches
        self.steps_per_epoch = (
            int(config_dict["steps_per_epoch"])
            if "steps_per_epoch" in config_dict
            else 4800
        )  # replay period

        self.replay_buffer_size = self.steps_per_epoch  # times number of agents

        # could move to actor_config
        self.train_policy_iterations = (
            int(config_dict["train_policy_iterations"])
            if "train_policy_iterations" in config_dict
            else 5
        )
        # could move to critic config
        self.train_value_iterations = (
            int(config_dict["train_value_iterations"])
            if "train_value_iterations" in config_dict
            else 5
        )
        self.target_kl = (
            config_dict["target_kl"] if "target_kl" in config_dict else 0.02
        )

        self.discount_factor = (
            config_dict["discount_factor"] if "discount_factor" in config_dict else 0.99
        )
        self.gae_labmda = (
            config_dict["gae_lambda"] if "gae_lambda" in config_dict else 0.98
        )
        self.entropy_coefficient = (
            config_dict["entropy_coefficient"]
            if "entropy_coefficient" in config_dict
            else 0.001
        )
