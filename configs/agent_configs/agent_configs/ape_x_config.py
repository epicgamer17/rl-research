from agent_configs.distributed_configs import (
    DistributedActorConfig,
    DistributedLearnerConfig,
)
from .base_config import ConfigBase
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


class ApeXActorConfig(ApeXConfig, DistributedActorConfig):
    def __init__(self, config_dict, game_config):
        super(ApeXActorConfig, self).__init__(config_dict, game_config)
        super(DistributedActorConfig, self).__init__(config_dict)
        # DIFFERENT EPSILONS PER ACTOR
        self.actor_buffer_size: int = self.parse_field("actor_buffer_size", 128)
        self.replay_buffer_size = self.actor_buffer_size
        self.training_steps = 10000000000  # just a big number
