from agent_configs.base_config import ConfigBase


class DistributedConfig(ConfigBase):
    def __init__(self, config_dict):
        super().__init__(config_dict)

        self.rank: int = self.parse_field("rank", required=True)
        self.worker_name: int = self.parse_field("worker_name", required=True)
        self.world_size: int = self.parse_field("world_size", required=True)
        self.rpc_port: int = self.parse_field("rpc_port", required=True)

        self.replay_addr: str = self.parse_field("replay_addr", required=True)
        self.storage_addr: str = self.parse_field("storage_addr", required=True)


class DistributedLearnerConfig(DistributedConfig):
    def __init__(self, config_dict):
        super().__init__(config_dict)

        self.samples_queue_size: int = self.parse_field("samples_queue_size", 16)
        self.updates_queue_size: int = self.parse_field("updates_queue_size", 16)
        self.push_params_interval: int = self.parse_field("push_params_interval", 100)


class DistributedActorConfig(DistributedConfig):
    def __init__(self, config_dict):
        super().__init__(config_dict)

        self.poll_params_interval: int = self.parse_field("poll_params_interval", 100)
