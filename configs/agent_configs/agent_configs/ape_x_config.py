from .base_config import ConfigBase
from .rainbow_config import RainbowConfig


class DistributedConfig(ConfigBase):
    def __init__(self, config_dict):
        super().__init__(config_dict)

        self.learner_addr: str = self.parse_field("learner_addr", required=True)
        self.learner_port: int = self.parse_field("learner_port", required=True)
        self.replay_addr: str = self.parse_field("replay_addr", required=True)
        self.replay_port: int = self.parse_field("replay_port", required=True)


class LearnerApeXMixin(ConfigBase):
    def __init__(self, config_dict):
        super().__init__(config_dict)

        self.samples_queue_size: int = self.parse_field("samples_queue_size", 16)
        self.updates_queue_size: int = self.parse_field("updates_queue_size", 16)
        self.remove_old_experiences_interval: int = self.parse_field(
            "remove_old_experiences_interval", 1000
        )

        # not used
        self.push_params_interval: int = self.parse_field("push_params_interval", 100)


class ActorApeXMixin(ConfigBase):
    def __init__(self, config_dict):
        super().__init__(config_dict)

        self.poll_params_interval: int = self.parse_field("poll_params_interval", 100)


class ApeXConfig(RainbowConfig):
    def __init__(self, config_dict, game_config):
        # initializes the rainbow config
        super(ApeXConfig, self).__init__(config_dict, game_config)
        # CHANGE CONFIGS TO WORK FOR APE-X AGENT
        # SHOULD BE ONE CONFIG BUT CAN HAVE CHILD CONFIGS (LOOK AT PPO)
        # so could just be like num actors etc <- since i think the others can just use the rainbow parameters from this config
