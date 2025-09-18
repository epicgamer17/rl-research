from utils import Loss
from .base_config import ConfigBase, kernel_initializer_wrapper
from utils import prepare_activations, prepare_kernel_initializers
from torch.optim import Optimizer, Adam


class SupervisedConfig(ConfigBase):
    def __init__(self, config_dict):
        super().__init__(config_dict)
        print("SupervisedConfig")
        self.adam_epsilon = self.parse_field("sl_adam_epsilon", 1e-7)
        self.learning_rate = self.parse_field("sl_learning_rate", 0.005)
        self.momentum = self.parse_field("sl_momentum", 0.9)
        self.loss_function: Loss = self.parse_field("sl_loss_function", required=True)
        self.clipnorm = self.parse_field("sl_clipnorm", 0)
        self.optimizer: Optimizer = self.parse_field("sl_optimizer", Adam)
        self.weight_decay = self.parse_field("sl_weight_decay", 0.0)
        self.training_steps = self.parse_field("training_steps", required=True)
        self.training_iterations = self.parse_field("sl_training_iterations", 1)
        self.num_minibatches = self.parse_field("sl_num_minibatches", 1)
        self.minibatch_size = self.parse_field("sl_minibatch_size", 32)
        self.min_replay_buffer_size = self.parse_field(
            "sl_min_replay_buffer_size", self.minibatch_size
        )
        self.replay_buffer_size = self.parse_field(
            "sl_replay_buffer_size", self.training_steps
        )
        self.activation = self.parse_field(
            "sl_activation", "relu", wrapper=prepare_activations
        )
        self.kernel_initializer = self.parse_field(
            "sl_kernel_initializer",
            None,
            required=False,
            wrapper=kernel_initializer_wrapper,
        )

        self.clip_low_prob = self.parse_field("sl_clip_low_prob", 0.00)

        self.noisy_sigma = self.parse_field("sl_noisy_sigma", 0)
        self.residual_layers = self.parse_field("sl_residual_layers", [])
        self.conv_layers = self.parse_field("sl_conv_layers", [])
        self.dense_layers_widths = self.parse_field("sl_dense_layer_widths", [128])

        self.game = None
