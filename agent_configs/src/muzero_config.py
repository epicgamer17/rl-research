from alphazero_config import AlphaZeroConfig
import tensorflow as tf


class MuZeroConfig(AlphaZeroConfig):
    def __init__(self, config_dict, game_config):
        super(MuZeroConfig, self).__init__(config_dict, game_config)
        if "known_bounds" in config_dict:
            self.known_bounds = config_dict[
                "known_bounds"
            ]  # this is the min and max value of the reward (in one step and not score over the whole game (so not the same as vmin and vmax in rainbow i think))
        else:
            self.known_bounds = None
            print("No known bounds set")

        self.num_players = (
            game_config["num_players"] if "num_players" in game_config else None
        )
        assert (
            self.num_players is not None
        ), "Number of players should be defined in the game config"
        self.known_bounds = (
            config_dict["known_bounds"] if "known_bounds" in config_dict else None
        )

    def _verify_game(self):
        # override alphazero game verification since muzero can play those games
        assert self.game.is_image, "MuZero only supports image-based games right now"
