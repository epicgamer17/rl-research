from agent_configs.dqn.rainbow_config import RainbowConfig
from agent_configs.sl_config import SupervisedConfig
from ..base_config import Config
from torch.optim import Optimizer, Adam


class NFSPDQNConfig(Config):
    def __init__(self, config_dict, game_config):
        # Config type should be a DQN Type
        super(NFSPDQNConfig, self).__init__(config_dict, game_config)
        print("NFSPDQNConfig")
        self.rl_configs = [
            RainbowConfig(config_dict, game_config) for _ in range(self.num_players)
        ]
        self.sl_configs = [
            SupervisedConfig(config_dict) for _ in range(self.num_players)
        ]
        for c in self.sl_configs:
            c.game = game_config

        self.replay_interval = self.parse_field("replay_interval", 16)

        self.anticipatory_param = self.parse_field("anticipatory_param", 0.1)

        # if self.anticipatory_param == 1.0 and self.game.is_deterministic:

        self.shared_networks_and_buffers = self.parse_field(
            "shared_networks_and_buffers", False
        )

    def _verify_game(self):
        assert self.game.is_discrete, "NFSP only supports discrete action spaces"
