from .base_config import ConfigBase
from utils import prepare_activations, prepare_kernel_initializers
from torch.optim import Optimizer, Adam


class SupervisedConfig(ConfigBase):
    def __init__(self, config_dict):
        super().__init__(config_dict)
        self.adam_epsilon = self.parse_field("sl_adam_epsilon", 1e-7)
        self.learning_rate = self.parse_field("sl_learning_rate", 0.005)
        self.clipnorm = self.parse_field("sl_clipnorm", None, required=False)
        self.optimizer: Optimizer = self.parse_field("sl_optimizer", Adam)
        self.training_steps = self.parse_field("training_steps", required=True)
        self.training_iterations = self.parse_field("sl_training_iterations", 1)
        self.num_minibatches = self.parse_field("sl_num_minibatches", 1)
        self.minibatch_size = self.parse_field("sl_minibatch_size", 32)
        self.min_replay_buffer_size = self.parse_field("sl_min_replay_buffer_size", self.minibatch_size)
        self.replay_buffer_size = self.parse_field("sl_replay_buffer_size", self.training_steps)
        self.activation = self.parse_field("sl_activation", "relu", wrapper=prepare_activations)
        self.kernel_initializer = self.parse_field(
            "sl_kernel_initializer", "glorot_uniform", wrapper=prepare_kernel_initializers
        )

        self.noisy_sigma = self.parse_field("sl_noisy_sigma", False)
        self.conv_layers = self.parse_field("sl_conv_layers", [])
        self.dense_layers_widths = self.parse_field("sl_dense_layers", [128])

        self.game = None
