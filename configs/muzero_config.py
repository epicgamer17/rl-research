from alphazero_config import AlphaZeroConfig
import tensorflow as tf


class MuZeroConfig(AlphaZeroConfig):
    def __init__(self, config_dict, game_config):
        super(MuZeroConfig, self).__init__(config_dict)
        self.known_bounds = (
            config_dict["known_bounds"] if "known_bounds" in config_dict else None
        )  # this is the min and max value of the reward (in one step and not score over the whole game (so not the same as vmin and vmax in rainbow i think))
        self.num_players = (
            game_config["num_players"] if "num_players" in game_config else None
        )
        assert (
            self.num_players is not None
        ), "Number of players should be defined in the game config"
