from agent_configs.base_config import ConfigBase


class DistributedConfig(ConfigBase):
    def __init__(self, config_dict):
        super().__init__(config_dict)

        self.replay_addr: str = self.parse_field("replay_addr", required=True)
        self.replay_port: int = self.parse_field("replay_port", required=True)

        self.storage_hostname: str = self.parse_field("storage_hostname", required=True)
        self.storage_port: int = self.parse_field("storage_port", required=True)
        self.storage_username: str = self.parse_field("storage_username", required=True)
        self.storage_password: str = self.parse_field("storage_password", required=True)


class DistributedLearnerConfig(DistributedConfig):
    def __init__(self, config_dict):
        super().__init__(config_dict)

        self.samples_queue_size: int = self.parse_field("samples_queue_size", 16)
        self.updates_queue_size: int = self.parse_field("updates_queue_size", 16)
        self.remove_old_experiences_interval: int = self.parse_field(
            "remove_old_experiences_interval", 1000
        )

        self.push_params_interval: int = self.parse_field("push_params_interval", 100)


class DistributedActorConfig(DistributedConfig):
    def __init__(self, config_dict):
        super().__init__(config_dict)

        self.poll_params_interval: int = self.parse_field("poll_params_interval", 100)
