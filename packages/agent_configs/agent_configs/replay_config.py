from agent_configs.base_config import ConfigBase


class ReplayBufferConfig(ConfigBase):
    def __init__(self, config_dict):
        super().__init__(config_dict)

        self.observation_dimensions = self.parse_field("observation_dimensions")
        self.max_size: int = self.parse_field("max_size", 100000)
        self.min_size: int = self.parse_field("min_size", 512)
        self.batch_size: int = self.parse_field("batch_size")
        self.max_priority: float = self.parse_field("max_priority", 1.0)
        self.per_alpha: float = self.parse_field("per_alpha", 0.5)
        self.n_step: int = self.parse_field("n_step", 1)
        self.discount_factor: float = self.parse_field("discount_factor", 0.99)
