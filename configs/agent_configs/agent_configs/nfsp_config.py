from agent_configs.sl_config import SupervisedConfig
from .base_config import Config, ConfigBase
from keras.optimizers import Optimizer, Adam


class NFSPConfig(Config):
    def __init__(self, config_dict, game_config, rl_config_type):
        super(NFSPConfig, self).__init__(config_dict, game_config)
        self.num_players = self.parse_field("num_players", required=True)
        self.rl_configs = [
            rl_config_type(config_dict, game_config) for _ in range(self.num_players)
        ]
        self.sl_configs = [
            SupervisedConfig(config_dict) for _ in range(self.num_players)
        ]
        for c in self.sl_configs:
            c.game = game_config

        self.replay_interval = self.parse_field("replay_interval", 1)

        self.anticipatory_param = self.parse_field("anticipatory_param", 0.5)

    def _verify_game(self):
        assert self.game.is_discrete, "NFSP only supports discrete action spaces"
