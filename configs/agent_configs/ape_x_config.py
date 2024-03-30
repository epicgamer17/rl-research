from configs.agent_configs.rainbow_config import RainbowConfig
import tensorflow as tf


class ApeXConfig(RainbowConfig):
    def __init__(self, config_dict, game_config):
        # initializes the rainbow config
        super(ApeXConfig, self).__init__(config_dict, game_config)
        # CHANGE CONFIGS TO WORK FOR APE-X AGENT
        # SHOULD BE ONE CONFIG BUT CAN HAVE CHILD CONFIGS (LOOK AT PPO)
        # so could just be like num actors etc <- since i think the others can just use the rainbow parameters from this config
        if "poll_params_interval" in config_dict:
            self.poll_params_interval = config_dict["poll_params_interval"]
        else:
            self.poll_params_interval = 100
            print("Using default poll params interval: 100")

        if "buffer_size" in config_dict:
            self.buffer_size = config_dict["buffer_size"]
        else:
            self.buffer_size = 100
            print("Using default buffer size: 100")

        if "learner_addr" in config_dict:
            self.learner_addr = config_dict["learner_addr"]
        else:
            self.learner_addr = None
            assert self.learner_addr is not None, "Learner address must be set"

        if "learner_port" in config_dict:
            self.learner_port = config_dict["learner_port"]
        else:
            self.learner_port = None
            assert self.learner_port is not None, "Learner port must be set"

        if "replay_addr" in config_dict:
            self.replay_addr = config_dict["replay_addr"]
        else:
            self.replay_addr = None
            assert self.replay_addr is not None, "Replay address must be set"

        if "replay_port" in config_dict:
            self.replay_port = config_dict["replay_port"]
        else:
            self.replay_port = None
            assert self.replay_port is not None, "Replay port must be set"

        if "remove_old_experiences_interval" in config_dict:
            self.remove_old_experiences_interval = config_dict[
                "remove_old_experiences_interval"
            ]
        else:
            self.remove_old_experiences_interval = 1000
            print("Using default remove old experiences interval: 1000")

        if "push_weights_interval" in config_dict:
            self.push_weights_interval = config_dict["push_weights_interval"]
        else:
            self.push_weights_interval = 20
            print("Using default push weights interval: 20")

        if "samples_queue_size" in config_dict:
            self.samples_queue_size = config_dict["samples_queue_size"]
        else:
            self.samples_queue_size = 16
            print("Using default samples queue size: 16")

        if "updates_queue_size" in config_dict:
            self.updates_queue_size = config_dict["updates_queue_size"]
        else:
            self.updates_queue_size = 16
            print("Using default updates queue size: 16")

        if "port" in config_dict:
            self.port = config_dict["port"]
        else:
            self.port = None
            assert self.port is not None, "Port must be set"
