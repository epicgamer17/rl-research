from agent_configs.distributed_configs import (
    DistributedActorConfig,
    DistributedLearnerConfig,
)
from .rainbow_config import RainbowConfig


class ApeXConfig(RainbowConfig):
    def __init__(self, config_dict, game_config):
        # initializes the rainbow config
        super(ApeXConfig, self).__init__(config_dict, game_config)
        # CHANGE CONFIGS TO WORK FOR APE-X AGENT
        # SHOULD BE ONE CONFIG BUT CAN HAVE CHILD CONFIGS (LOOK AT PPO)
        # so could just be like num actors etc <- since i think the others can just use the rainbow parameters from this config


class ApeXLearnerConfig(ApeXConfig, DistributedLearnerConfig):
    def __init__(self, config_dict, game_config):
        super(ApeXLearnerConfig, self).__init__(config_dict, game_config)
        super(DistributedLearnerConfig, self).__init__(config_dict)

        self.num_actors: int = self.parse_field("num_actors", required=True, wrapper=int)
        self.noisy_sigma = 0
        self.config_dict['noisy_sigma'] = 0

        print("Loading actor config:")
        self.actor_config: ApeXActorConfig = ApeXActorConfig.load(
            self.parse_field("distributed_actor_config_file", required=True)
        )

        self.actor_config.noisy_sigma = 0
        self.actor_config.config_dict['noisy_sigma '] = 0


class ApeXActorConfig(ApeXConfig, DistributedActorConfig):
    def __init__(self, config_dict, game_config):
        super(ApeXActorConfig, self).__init__(config_dict, game_config)
        super(DistributedActorConfig, self).__init__(config_dict)
        # DIFFERENT EPSILONS PER ACTOR
        self.replay_buffer_size: int = self.parse_field("actor_buffer_size", 128, wrapper=int)
        self.eg_epsilon: float = self.parse_field("eg_epsilon", 0.95)
        self.minibatch_size = self.replay_buffer_size
        self.training_steps = 10000000000  # just a big number
