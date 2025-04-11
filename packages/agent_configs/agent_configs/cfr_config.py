from .base_config import Config


class CFRConfig():
    def __init__ (self,
        config_dict,
        game_config):
        print("CFRConfig")

        self.num_players = game_config["num_players"]
        self.network = config_dict["network"]
        self.replay_buffer_size = config_dict["replay_buffer_size"]
        self.minibatch_size = config_dict["minibatch_size"]
        self.steps_per_epoch = config_dict["steps_per_epoch"]
        self.traversals = config_dict["traversals"]
        self.training_steps = config_dict["training_steps"]
        self.observation_space = game_config["observation_space"]
        self.action_space = game_config["action_space"]