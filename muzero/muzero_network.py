from typing import Callable, Tuple

from agent_configs.muzero_config import MuZeroConfig
from torch import nn, Tensor
from modules.utils import zero_weights_initializer
from utils.utils import to_lists

from modules.conv import Conv2dStack
from modules.dense import DenseStack, build_dense
from modules.residual import ResidualStack
import torch
import torch.nn.functional as F


class Representation(nn.Module):
    def __init__(
        self,
        config: MuZeroConfig,
        input_shape: Tuple[int],
    ):
        assert (
            config.game.is_discrete
        ), "AlphaZero only works for discrete action space games (board games)"

        self.config = config

        super(Representation, self).__init__()

        self.has_residual_layers = len(config.representation_residual_layers) > 0
        self.has_conv_layers = len(config.representation_conv_layers) > 0
        self.has_dense_layers = len(config.representation_dense_layer_widths) > 0
        assert (
            self.has_conv_layers or self.has_dense_layers or self.has_residual_layers
        ), "At least one of the layers should be present."

        current_shape = input_shape
        B = current_shape[0]

        # INPUTS = CONV + BATCHNORM + maybe RELU? into residual etc

        if self.has_residual_layers:
            assert (
                len(current_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
            filters, kernel_sizes, strides = to_lists(
                config.representation_residual_layers
            )

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.residual_layers = ResidualStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.residual_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_conv_layers:
            assert (
                len(current_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
            filters, kernel_sizes, strides = to_lists(config.representation_conv_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.conv_layers = Conv2dStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert len(current_shape) == 2
                initial_width = current_shape[1]
            # print(initial_width)
            # (B, width_in) -> (B, width_out)
            self.dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.representation_dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )
            current_shape = (
                B,
                self.dense_layers.output_width,
            )
        self.output_shape = current_shape

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if self.has_residual_layers:
            self.residual_layers.initialize(initializer)
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)

    def forward(self, inputs: Tensor):
        if self.has_conv_layers:
            assert inputs.dim() == 4

        # (B, *)
        S = inputs
        # INPUT CONV LAYERS???
        # input batch norm
        # relu?

        # (B, C_in, H, W) -> (B, C_out, H, W)
        if self.has_residual_layers:
            S = self.residual_layers(S)

        # (B, C_in, H, W) -> (B, C_out, H, W)
        if self.has_conv_layers:
            S = self.conv_layers(S)

        # (B, *) -> (B, dense_features_in)

        # (B, dense_features_in) -> (B, dense_features_out)
        if self.has_dense_layers:
            # print("S", S.shape)
            S = S.flatten(1, -1)
            # print("Flattened S", S.shape)
            S = self.dense_layers(S)

            # normalize inputs as per paper
            min_hidden_state = S.min(1, keepdim=True)[0]
            max_hidden_state = S.max(1, keepdim=True)[0]
            scale_hidden_state = max_hidden_state - min_hidden_state
            scale_hidden_state[scale_hidden_state < 1e-5] += 1e-5
            hidden_state = (S - min_hidden_state) / scale_hidden_state
        else:
            # normalize inputs as per paper
            min_hidden_state = (
                S.view(
                    -1,
                    S.shape[1],
                    S.shape[2] * S.shape[3],
                )
                .min(2, keepdim=True)[0]
                .unsqueeze(-1)
            )
            max_hidden_state = (
                S.view(
                    -1,
                    S.shape[1],
                    S.shape[2] * S.shape[3],
                )
                .max(2, keepdim=True)[0]
                .unsqueeze(-1)
            )
            scale_hidden_state = max_hidden_state - min_hidden_state
            scale_hidden_state[scale_hidden_state < 1e-5] += 1e-5
            hidden_state = (S - min_hidden_state) / scale_hidden_state

        return hidden_state


class Dynamics(nn.Module):
    def __init__(
        self,
        config: MuZeroConfig,
        input_shape: Tuple[int],
    ):
        assert (
            config.game.is_discrete
        ), "MuZero only works for discrete action space games (board games)"

        print("dynamics input shape", input_shape)
        self.config = config

        super(Dynamics, self).__init__()

        self.has_residual_layers = len(config.dynamics_residual_layers) > 0
        self.has_conv_layers = len(config.dynamics_conv_layers) > 0
        self.has_dense_layers = len(config.dynamics_dense_layer_widths) > 0

        assert (
            self.has_conv_layers or self.has_dense_layers or self.has_residual_layers
        ), "At least one of the layers should be present."

        current_shape = input_shape
        B = current_shape[0]

        # INPUTS = CONV + BATCHNORM + maybe RELU? into residual etc

        if self.has_residual_layers:
            assert (
                len(current_shape) == 4
            ), "dynamics residual layers expected an input shape should be (B, C, H, W), got {}".format(
                current_shape
            )
            filters, kernel_sizes, strides = to_lists(config.dynamics_residual_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.residual_layers = ResidualStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.residual_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_conv_layers:
            assert (
                len(current_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
            filters, kernel_sizes, strides = to_lists(config.dynamics_conv_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.conv_layers = Conv2dStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert len(current_shape) == 2
                initial_width = current_shape[1]

            # (B, width_in) -> (B, width_out)
            # print("initial_width", initial_width)
            self.dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.dynamics_dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )
            current_shape = (
                B,
                self.dense_layers.output_width,
            )

        self.output_shape = current_shape

        self.has_reward_conv_layers = len(config.reward_conv_layers) > 0
        self.has_reward_dense_layers = len(config.reward_dense_layer_widths) > 0

        if self.has_reward_conv_layers:
            assert (
                len(current_shape) == 4
            ), "reward layers expected an input shape should be (B, C, H, W), got {}".format(
                current_shape
            )
            filters, kernel_sizes, strides = to_lists(config.reward_conv_layers)

            self.reward_conv_layers = Conv2dStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )

            current_shape = (
                B,
                self.reward_conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_reward_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert len(current_shape) == 2
                initial_width = current_shape[1]

            self.reward_dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.reward_dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )

            current_shape = (
                B,
                self.reward_dense_layers.output_width,
            )

        if len(current_shape) == 4:
            initial_width = current_shape[1] * current_shape[2] * current_shape[3]
        else:
            assert len(current_shape) == 2
            initial_width = current_shape[1]

        if self.config.value_prefix:
            self.lstm = nn.LSTM(
                input_size=initial_width, hidden_size=self.config.lstm_hidden_size
            )
            initial_width = self.config.lstm_hidden_size

        if config.support_range is not None:
            self.full_support_size = 2 * config.support_range + 1
            self.reward = build_dense(
                in_features=initial_width,
                out_features=self.full_support_size,
                sigma=0,
            )
        else:
            self.reward = build_dense(
                in_features=initial_width,
                out_features=1,
                sigma=0,
            )

        current_shape = self.output_shape

        self.has_to_play_conv_layers = len(config.to_play_conv_layers) > 0
        self.has_to_play_dense_layers = len(config.to_play_dense_layer_widths) > 0

        if self.has_to_play_conv_layers:
            assert (
                len(current_shape) == 4
            ), "to_play layers expected an input shape should be (B, C, H, W), got {}".format(
                current_shape
            )
            filters, kernel_sizes, strides = to_lists(config.to_play_conv_layers)

            self.to_play_conv_layers = Conv2dStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )

            current_shape = (
                B,
                self.to_play_conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_to_play_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert len(current_shape) == 2
                initial_width = current_shape[1]

            self.to_play_dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.to_play_dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )

            current_shape = (
                B,
                self.to_play_dense_layers.output_width,
            )

        if len(current_shape) == 4:
            initial_width = current_shape[1] * current_shape[2] * current_shape[3]
        else:
            assert len(current_shape) == 2
            initial_width = current_shape[1]

        self.to_play = build_dense(
            in_features=initial_width,
            out_features=self.config.game.num_players,
            sigma=0,
        )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if self.has_residual_layers:
            self.residual_layers.initialize(initializer)
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)
        if self.has_reward_conv_layers:
            self.reward_conv_layers.initialize(initializer)
        if self.has_reward_dense_layers:
            self.reward_dense_layers.initialize(initializer)

        if self.config.support_range is not None:
            self.reward.apply(zero_weights_initializer)
        else:
            self.reward.initialize(
                initializer
            )  # Standard initialization for scalar output

        # To Play (always a probability distribution)
        self.to_play.apply(zero_weights_initializer)

    def forward(self, hidden_state: Tensor, reward_hidden: Tensor):
        if self.has_conv_layers:
            assert hidden_state.dim() == 4

        # (B, *)
        S = hidden_state
        # INPUT CONV LAYERS???
        # input batch norm
        # relu?

        # SHOULD I HAVE AN INPUT HERE THAT REDUCES THE CHANNELS BY 1 SO THAT THE NETWORK IS "the same" as the representation?

        # (B, C_in, H, W) -> (B, C_out, H, W)
        if self.has_residual_layers:
            S = self.residual_layers(S)

        # (B, C_in, H, W) -> (B, C_out, H, W)
        if self.has_conv_layers:
            S = self.conv_layers(S)

        # (B, *) -> (B, dense_features_in)
        if self.has_dense_layers:
            # print("S Shape", S.shape)
            flattened_hidden_state = S.flatten(1, -1)
            # print("flattened shape", flattened_hidden_state.shape)
            S = self.dense_layers(flattened_hidden_state)

            # normalize inputs as per paper
            min_hidden_state = S.min(1, keepdim=True)[0]
            max_hidden_state = S.max(1, keepdim=True)[0]
            scale_encoded_state = max_hidden_state - min_hidden_state
            scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
            hidden_state = (S - min_hidden_state) / scale_encoded_state
        else:
            # normalize inputs as per paper
            min_hidden_state = (
                S.view(
                    -1,
                    S.shape[1],
                    S.shape[2] * S.shape[3],
                )
                .min(2, keepdim=True)[0]
                .unsqueeze(-1)
            )
            max_hidden_state = (
                S.view(
                    -1,
                    S.shape[1],
                    S.shape[2] * S.shape[3],
                )
                .max(2, keepdim=True)[0]
                .unsqueeze(-1)
            )
            scale_hidden_state = max_hidden_state - min_hidden_state
            scale_hidden_state[scale_hidden_state < 1e-5] += 1e-5
            hidden_state = (S - min_hidden_state) / scale_hidden_state

        # (B, dense_features_in) -> (B, dense_features_out)
        if self.has_reward_conv_layers:
            reward_vector = self.reward_conv_layers(S)
            flattened_reward_vector = reward_vector.flatten(1, -1)
        else:
            flattened_reward_vector = hidden_state.flatten(1, -1)

        if self.has_reward_dense_layers:
            flattened_reward_vector = self.reward_dense_layers(flattened_reward_vector)

        if self.config.value_prefix:
            flattened_reward_vector = flattened_reward_vector.unsqueeze(0)
            flattened_reward_vector, reward_hidden = self.lstm(
                flattened_reward_vector, reward_hidden
            )
            flattened_reward_vector = flattened_reward_vector.squeeze(0)
            # flattened_reward_vector = self.bn_value_prefix(flattened_reward_vector)
        if self.config.support_range is None:
            reward = self.reward(flattened_reward_vector)
        else:
            # TODO: should this be turned into an expected value and then in the loss function into a two hot?
            reward = self.reward(flattened_reward_vector).softmax(dim=-1)

        # (B, dense_features_in) -> (B, dense_features_out)
        if self.has_to_play_conv_layers:
            to_play_vector = self.to_play_conv_layers(S)
            flattened_to_play_vector = to_play_vector.flatten(1, -1)
        else:
            flattened_to_play_vector = hidden_state.flatten(1, -1)

        if self.has_to_play_dense_layers:
            flattened_to_play_vector = self.to_play_dense_layers(
                flattened_to_play_vector
            )

        to_play = self.to_play(flattened_to_play_vector).softmax(dim=-1)

        return reward, hidden_state, to_play, reward_hidden


class Prediction(nn.Module):
    def __init__(
        self,
        config: MuZeroConfig,
        output_size: int,
        input_shape: Tuple[int],
    ):
        assert (
            config.game.is_discrete
        ), "AlphaZero only works for discrete action space games (board games)"

        self.config = config

        super(Prediction, self).__init__()

        self.has_residual_layers = len(config.residual_layers) > 0
        self.has_conv_layers = len(config.conv_layers) > 0
        self.has_dense_layers = len(config.dense_layer_widths) > 0
        if not (
            self.has_conv_layers or self.has_dense_layers or self.has_residual_layers
        ):
            print("Warning no layers set for prediction network.")

        current_shape = input_shape
        B = current_shape[0]

        # INPUTS = CONV + BATCHNORM + maybe RELU?

        if self.has_residual_layers:
            assert (
                len(current_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
            filters, kernel_sizes, strides = to_lists(config.residual_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.residual_layers = ResidualStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.residual_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_conv_layers:
            assert (
                len(current_shape) == 4
            ), "Input shape should be (B, C, H, W), got {}".format(current_shape)
            filters, kernel_sizes, strides = to_lists(config.conv_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.conv_layers = Conv2dStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert len(current_shape) == 2
                initial_width = current_shape[1]

            # (B, width_in) -> (B, width_out)
            self.dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )
            current_shape = (
                B,
                self.dense_layers.output_width,
            )

        self.critic = CriticNetwork(config, current_shape)
        self.actor = ActorNetwork(config, current_shape, output_size)

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if self.has_residual_layers:
            self.residual_layers.initialize(initializer)
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)

        self.actor.initialize(initializer)
        self.critic.initialize(initializer)

    def forward(self, inputs: Tensor):
        if self.has_conv_layers:
            assert inputs.dim() == 4

        # (B, *)
        S = inputs
        # INPUT CONV LAYERS???

        # (B, C_in, H, W) -> (B, C_out, H, W)
        if self.has_residual_layers:
            S = self.residual_layers(S)

        # (B, C_in, H, W) -> (B, C_out, H, W)
        if self.has_conv_layers:
            S = self.conv_layers(S)

        # (B, *) -> (B, dense_features_in)

        # (B, dense_features_in) -> (B, dense_features_out)
        if self.has_dense_layers:
            S = S.flatten(1, -1)
            S = self.dense_layers(S)

        return self.critic(S), self.actor(S)


class AfterstateDynamics(nn.Module):
    """
    Afterstate dynamics for stochastic MuZero:
      - models deterministic afterstate (hidden_state after action)
      - outputs chance_logits for stochastic branching (categorical over num_chance_outcomes)
      - outputs reward and to_play similarly to your Dynamics
    Inputs:
      - hidden_state: Tensor (B, C, H, W) or (B, D)
      - action: LongTensor (B,) of action indices OR one-hot FloatTensor (B, A)
      - reward_hidden: LSTM hidden (if value_prefix used) else None
    Returns:
      chance_logits (or None), afterstate_hidden, reward, to_play, reward_hidden
    """

    def __init__(self, config, input_shape: Tuple[int]):
        super().__init__()
        self.config = config
        self.input_shape = input_shape

        # reuse config naming where sensible; add defaults where config may not define afterstate-specific fields
        self.has_residual_layers = len(config.afterstate_residual_layers) > 0
        self.has_conv_layers = len(config.afterstate_conv_layers) > 0
        self.has_dense_layers = len(config.afterstate_dense_layer_widths) > 0

        # number of chance outcomes (categorical). If 1 or None -> deterministic
        self.num_chance = config.num_chance

        current_shape = input_shape
        B = current_shape[0]

        # Residual stack (if configured)
        if self.has_residual_layers:
            filters, kernel_sizes, strides = to_lists(config.afterstate_residual_layers)
            assert len(current_shape) == 4, "afterstate residual layers expect 4D input"
            self.residual_layers = ResidualStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.residual_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        # Conv stack
        if self.has_conv_layers:
            filters, kernel_sizes, strides = to_lists(config.afterstate_conv_layers)
            assert len(current_shape) == 4, "afterstate conv layers expect 4D input"
            self.conv_layers = Conv2dStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        # Dense stack (after flattening conv output)
        if self.has_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                initial_width = current_shape[1]
            # we will concatenate the action embedding to this flattened vector -> so increase input by action_embed_size
            self.dense_layers = DenseStack(
                initial_width=initial_width,
                widths=config.afterstate_dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (B, self.dense_layers.output_width)

        # remember output shape (afterstate_hidden will be same spatial/dim as original template's output)
        self.output_shape = current_shape

        # Chance head: maps afterstate features -> logits over possible chance outcomes (categorical)
        # chance head on flattened features

        if len(self.output_shape) == 4:
            initial_width = (
                self.output_shape[1] * self.output_shape[2] * self.output_shape[3]
            )
        else:
            initial_width = self.output_shape[1]
        self.has_chance_conv_layers = len(config.chance_conv_layers) > 0
        self.has_chance_dense_layers = len(config.chance_dense_layer_widths) > 0

        if self.has_chance_conv_layers:
            assert (
                len(current_shape) == 4
            ), "chance layers expected an input shape should be (B, C, H, W), got {}".format(
                current_shape
            )
            filters, kernel_sizes, strides = to_lists(config.chance_conv_layers)

            self.chance_conv_layers = Conv2dStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )

            current_shape = (
                B,
                self.chance_conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_chance_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert len(current_shape) == 2
                initial_width = current_shape[1]

            self.chance_dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.chance_dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )

            current_shape = (
                B,
                self.chance_dense_layers.output_width,
            )

        if len(current_shape) == 4:
            initial_width = current_shape[1] * current_shape[2] * current_shape[3]
        else:
            assert len(current_shape) == 2
            initial_width = current_shape[1]

        self.chance_head = build_dense(
            in_features=initial_width, out_features=self.num_chance, sigma=0
        )

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        # initialize internal stacks
        if self.has_residual_layers:
            self.residual_layers.initialize(initializer)
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)

        if self.has_chance_conv_layers:
            self.chance_conv_layers.initialize(initializer)
        if self.has_chance_dense_layers:
            self.chance_dense_layers.initialize(initializer)

        # initialize chance head to near-zero so initially the model is near-deterministic
        self.chance_head.apply(zero_weights_initializer)

    def forward(
        self,
        hidden_state: torch.Tensor,
    ):
        """
        action: LongTensor (B,) OR one-hot FloatTensor (B, A)
        reward_hidden: tuple(h, c) for LSTM if used, else None
        """
        assert hidden_state is not None
        S = hidden_state  # keep local name

        # run residual / conv stacks
        if self.has_residual_layers:
            S = self.residual_layers(S)
        if self.has_conv_layers:
            S = self.conv_layers(S)

        # (B, *) -> (B, dense_features_in)
        if self.has_dense_layers:
            # print("S Shape", S.shape)
            flattened_afterstate = S.flatten(1, -1)
            # print("flattened shape", flattened_hidden_state.shape)
            S = self.dense_layers(flattened_afterstate)

            # normalize inputs as per paper
            min_afterstate = S.min(1, keepdim=True)[0]
            max_afterstate = S.max(1, keepdim=True)[0]
            scale_encoded_state = max_afterstate - min_afterstate
            scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
            afterstate = (S - min_afterstate) / scale_encoded_state
        else:
            # normalize inputs as per paper
            min_afterstate = (
                S.view(
                    -1,
                    S.shape[1],
                    S.shape[2] * S.shape[3],
                )
                .min(2, keepdim=True)[0]
                .unsqueeze(-1)
            )
            max_afterstate = (
                S.view(
                    -1,
                    S.shape[1],
                    S.shape[2] * S.shape[3],
                )
                .max(2, keepdim=True)[0]
                .unsqueeze(-1)
            )
            scale_afterstate = max_afterstate - min_afterstate
            scale_afterstate[scale_afterstate < 1e-5] += 1e-5
            afterstate = (S - min_afterstate) / scale_afterstate

        if self.has_chance_conv_layers:
            chance_vector = self.chance_conv_layers(S)
            flattened_chance_vector = chance_vector.flatten(1, -1)
        else:
            flattened_chance_vector = hidden_state.flatten(1, -1)

        if self.has_chance_dense_layers:
            flattened_chance_vector = self.chance_dense_layers(flattened_chance_vector)

        chance_logits = self.chance_head(flattened_chance_vector)  # (B, num_chance)

        # return chance_logits.softmax(dim=-1), afterstate
        return afterstate


class AfterstatePrediction(nn.Module):
    """
    Prediction network for afterstate nodes. Mirrors your Prediction template,
    but intended to be run on afterstate_hidden representations.
    """

    def __init__(self, config, output_size: int, input_shape: Tuple[int]):
        super().__init__()
        self.config = config

        self.has_residual_layers = len(config.afterstate_residual_layers) > 0
        self.has_conv_layers = len(config.afterstate_conv_layers) > 0
        self.has_dense_layers = len(config.afterstate_dense_layer_widths) > 0

        current_shape = input_shape
        B = current_shape[0]

        # Residual stack (if configured)
        if self.has_residual_layers:
            filters, kernel_sizes, strides = to_lists(config.afterstate_residual_layers)
            assert len(current_shape) == 4, "afterstate residual layers expect 4D input"
            self.residual_layers = ResidualStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.residual_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        # Conv stack
        if self.has_conv_layers:
            filters, kernel_sizes, strides = to_lists(config.afterstate_conv_layers)
            assert len(current_shape) == 4, "afterstate conv layers expect 4D input"
            self.conv_layers = Conv2dStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        # Dense stack (after flattening conv output)
        if self.has_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                initial_width = current_shape[1]
            # we will concatenate the action embedding to this flattened vector -> so increase input by action_embed_size
            self.dense_layers = DenseStack(
                initial_width=initial_width,
                widths=config.afterstate_dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (B, self.dense_layers.output_width)

        # remember output shape (afterstate_hidden will be same spatial/dim as original template's output)
        self.output_shape = current_shape

        # Chance head: maps afterstate features -> logits over possible chance outcomes (categorical)
        # chance head on flattened features

        if len(self.output_shape) == 4:
            initial_width = (
                self.output_shape[1] * self.output_shape[2] * self.output_shape[3]
            )
        else:
            initial_width = self.output_shape[1]

        # TODO: Move the chance prediction here and use an "actor network"
        # final actor & critic using same modules as your Prediction class
        self.critic = CriticNetwork(config, current_shape)
        self.sigma = ActorNetwork(config, current_shape, config.num_chance)

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        if self.has_residual_layers:
            self.residual_layers.initialize(initializer)
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)

        self.critic.initialize(initializer)
        self.sigma.initialize(zero_weights_initializer)

    def forward(self, inputs: torch.Tensor):
        # inputs: afterstate_hidden (B, C, H, W) or (B, D)
        x = inputs
        if self.has_residual_layers:
            x = self.residual_layers(x)
        if self.has_conv_layers:
            x = self.conv_layers(x)
        if self.has_dense_layers:
            x = x.flatten(1, -1)
            x = self.dense_layers(x)

        # actor and critic expect flattened / dense-ready shapes via their constructors
        return self.critic(x), self.sigma(x)

    def reset_noise(self):
        self.critic.reset_noise()


class CriticNetwork(nn.Module):
    def __init__(self, config: MuZeroConfig, input_shape: Tuple[int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.has_conv_layers = len(config.critic_conv_layers) > 0
        self.has_dense_layers = len(config.critic_dense_layer_widths) > 0

        current_shape = input_shape
        B = current_shape[0]
        if self.has_conv_layers:
            # WITH BATCHNORM FOR EVERY CONV LAYER
            assert len(current_shape) == 4
            filters, kernel_sizes, strides = to_lists(config.critic_conv_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.conv_layers = Conv2dStack(
                input_shape=input_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert len(current_shape) == 2
                initial_width = current_shape[1]

            # (B, width_in) -> (B, width_out)
            self.dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.critic_dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )
            current_shape = (
                B,
                self.dense_layers.output_width,
            )

        if len(current_shape) == 4:
            initial_width = current_shape[1] * current_shape[2] * current_shape[3]
        else:
            assert len(current_shape) == 2
            initial_width = current_shape[1]

        if config.support_range is not None:
            self.full_support_size = 2 * config.support_range + 1
            self.value = build_dense(
                in_features=initial_width,
                out_features=self.full_support_size,
                sigma=0,
            )
        else:
            self.value = build_dense(
                in_features=initial_width,
                out_features=1,
                sigma=0,
            )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)
        # Initialize categorical layers with zero weights
        # Value (if support_range is not None, it's a probability distribution)
        if self.config.support_range is not None:
            self.value.apply(zero_weights_initializer)
        else:
            self.value.initialize(
                initializer
            )  # Standard initialization for scalar output

    def forward(self, inputs: Tensor):
        if self.has_conv_layers:
            assert (
                inputs.dim() == 4
            ), "Input shape should be (B, C, H, W), got {}".format(inputs.shape)

        x = inputs
        if self.has_conv_layers:
            x = self.conv_layers(x)
        x = x.flatten(1, -1)
        if self.has_dense_layers:
            x = self.dense_layers(x)
        value = self.value(x)
        if self.config.support_range is None:
            return value
        else:
            # TODO: should this be turned into an expected value and then in the loss function into a two hot?
            return value.softmax(dim=-1)

    def reset_noise(self):
        if self.has_conv_layers:
            self.conv_layers.reset_noise()
        if self.has_dense_layers:
            self.dense_layers.reset_noise()
        self.value.reset_noise()


class ActorNetwork(nn.Module):
    def __init__(
        self,
        config: MuZeroConfig,
        input_shape: Tuple[int],
        output_size: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.has_conv_layers = len(config.actor_conv_layers) > 0
        self.has_dense_layers = len(config.actor_dense_layer_widths) > 0

        current_shape = input_shape
        B = current_shape[0]
        if self.has_conv_layers:
            # WITH BATCHNORM FOR EVERY CONV LAYER
            assert len(current_shape) == 4
            filters, kernel_sizes, strides = to_lists(config.actor_conv_layers)

            # (B, C_in, H, W) -> (B, C_out H, W)
            self.conv_layers = Conv2dStack(
                input_shape=current_shape,
                filters=filters,
                kernel_sizes=kernel_sizes,
                strides=strides,
                activation=self.config.activation,
                noisy_sigma=config.noisy_sigma,
            )
            current_shape = (
                B,
                self.conv_layers.output_channels,
                current_shape[2],
                current_shape[3],
            )

        if self.has_dense_layers:
            if len(current_shape) == 4:
                initial_width = current_shape[1] * current_shape[2] * current_shape[3]
            else:
                assert (
                    len(current_shape) == 2
                ), "Input shape should be (B, width), got {}".format(current_shape)
                initial_width = current_shape[1]

            # (B, width_in) -> (B, width_out)
            self.dense_layers = DenseStack(
                initial_width=initial_width,
                widths=self.config.actor_dense_layer_widths,
                activation=self.config.activation,
                noisy_sigma=self.config.noisy_sigma,
            )

            current_shape = (
                B,
                self.dense_layers.output_width,
            )

        if len(current_shape) == 4:
            initial_width = current_shape[1] * current_shape[2] * current_shape[3]
        else:
            assert len(current_shape) == 2
            initial_width = current_shape[1]

        self.actions = build_dense(
            in_features=initial_width,
            out_features=output_size,
            sigma=self.config.noisy_sigma,
        )

    def initialize(self, initializer: Callable[[Tensor], None]) -> None:
        if self.has_conv_layers:
            self.conv_layers.initialize(initializer)
        if self.has_dense_layers:
            self.dense_layers.initialize(initializer)
        # Policy (actions) is the main categorical output, initialize it to zero weights.
        self.actions.apply(zero_weights_initializer)

    def forward(self, inputs: Tensor):
        if self.has_conv_layers:
            assert (
                inputs.dim() == 4
            ), "Input shape should be (B, C, H, W), got {}".format(inputs.shape)

        x = inputs
        if self.has_conv_layers:
            x = self.conv_layers(x)

        x = x.flatten(1, -1)
        if self.has_dense_layers:
            x = self.dense_layers(x)
        actions = self.actions(x)
        return actions.softmax(dim=-1)

    def reset_noise(self):
        if self.has_conv_layers:
            self.conv_layers.reset_noise()
        if self.has_dense_layers:
            self.dense_layers.reset_noise()
        self.actions.reset_noise()


class Projector(nn.Module):
    def __init__(self, input_dim: int, config: MuZeroConfig):
        super().__init__()

        # Paper defaults (SimSiam/EfficientZero)
        # Hidden dim is typically 2048 for ResNet50, but for MuZero
        # it usually matches the representation size or slightly larger.
        proj_hidden_dim = config.projector_hidden_dim
        proj_output_dim = config.projector_output_dim
        pred_hidden_dim = config.predictor_hidden_dim
        pred_output_dim = config.predictor_output_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.BatchNorm1d(proj_output_dim),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, pred_output_dim),
        )

    def forward(self, x):
        x = self.projection(x)
        return self.projection_head(x)


import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        input_shape,
        num_codes: int = 128,
        channel_first: bool = True,
    ):
        """
        Args:
            input_shape: tuple, e.g. (C, H, W) or (B, C, H, W).
            num_codes: embedding size output by encoder.
            channel_first: if True, treats input as (..., C, H, W).
        """
        super().__init__()
        self.channel_first = channel_first
        self.num_codes = num_codes

        # --- Image / 4D Path ---
        if len(input_shape) == 4 or (len(input_shape) == 3 and input_shape[0] > 1):
            # Heuristic: treating len=4 or len=3 (C,H,W) as image data
            self.use_conv = True

            # Determine input channels based on flag
            # Assuming input_shape excludes Batch dim if len==3, or includes it if len==4
            if len(input_shape) == 4:
                c = input_shape[1] if self.channel_first else input_shape[3]
            else:
                c = input_shape[0] if self.channel_first else input_shape[2]

            self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1)
            self.act = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

            # --- REMOVED POOLING HERE ---
            # previously: self.pool = nn.AdaptiveAvgPool2d(1)

            # We must calculate the flattened size dynamically because
            # we no longer force the output to 1x1.
            with torch.no_grad():
                # Create a dummy input to trace shape
                # We assume input_shape provided matches the user's tensor setup
                if len(input_shape) == 3:
                    dummy_in = torch.zeros(1, *input_shape)
                else:
                    dummy_in = torch.zeros(*input_shape)

                # Check for channel last input and permute for PyTorch convs if necessary
                if not self.channel_first and dummy_in.shape[-1] == c:
                    dummy_in = dummy_in.permute(0, 3, 1, 2)

                d = self.conv1(dummy_in)
                d = self.conv2(d)
                d = self.conv3(d)
                flat_dim = d.flatten(1).shape[1]

            # Linear head: maps (64 * H' * W') -> num_codes
            self.fc = nn.Linear(flat_dim, num_codes)

        # --- Vector / 2D Path ---
        else:
            self.use_conv = False
            # Assuming input_shape is (Batch, Features) or just (Features,)
            f = input_shape[-1]
            self.vec_net = nn.Sequential(
                nn.Linear(f, max(f // 2, num_codes)),
                nn.ReLU(inplace=True),
                nn.Linear(max(f // 2, num_codes), num_codes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            probs: (B, num_codes) - Softmax probabilities
            one_hot_st: (B, num_codes) - Straight-Through gradient flow
        """
        # 1. Processing to Logits
        if self.use_conv:
            # Ensure proper channel ordering for PyTorch Conv2d (B, C, H, W)
            if not self.channel_first:
                x = x.permute(0, 3, 1, 2)

            x = self.act(self.conv1(x))
            x = self.act(self.conv2(x))
            x = self.act(self.conv3(x))

            # Flatten directly (preserving spatial info in the flat vector)
            x = x.flatten(1, -1)
            x = self.fc(x)  # (B, num_codes)
        else:
            # Vector path returns LOGITS (removed .softmax inside here to avoid double softmax)
            x = self.vec_net(x)

        # 2. Softmax
        probs = x.softmax(dim=-1)

        # 3. Hard Quantization (One-Hot)
        # (B, 1) tensor containing the index of the max value
        quantized_indices = probs.argmax(dim=-1, keepdim=True)

        # Convert to one-hot (B, num_codes)
        one_hot = torch.zeros_like(probs).scatter_(-1, quantized_indices, 1.0)

        # 4. Straight-Through Estimator
        # Forward: use one_hot
        # Backward: use gradients of probs
        one_hot_st = (one_hot - probs).detach() + probs

        return probs, one_hot_st


class Network(nn.Module):
    def __init__(
        self,
        config: MuZeroConfig,
        output_size: int,
        input_shape: Tuple[int],
        channel_first: bool = True,
    ):
        super(Network, self).__init__()
        self.config = config
        self.channel_first = channel_first
        self.num_chance = config.num_chance

        # --- 1. Representation Network ---
        self.representation = Representation(config, input_shape)
        self.num_actions = output_size

        hidden_state_shape = self.representation.output_shape
        print("Hidden state shape:", hidden_state_shape)

        # # --- 2. Determine Action Encoding and Dynamics Input ---
        # if len(hidden_state_shape) == 4:
        #     print("image hidden state")
        #     # Case A: Image/Feature Map State (e.g., ConvNet)

        #     # Get the spatial/grid size (H, W) from the hidden state shape
        #     self.spatial_shape = (
        #         hidden_state_shape[2:]
        #         if self.channel_first
        #         else hidden_state_shape[1:-1]
        #     )
        #     print("spatial shape", self.spatial_shape)
        #     # Use a dummy action to get the shape (B=1, C_a, H, W) or (B=1, H, W, C_a)
        #     action_plane_shape_with_batch = action_function(
        #         self.num_actions, self.spatial_shape, torch.tensor(0).unsqueeze(0)
        #     )

        #     if self.channel_first:
        #         action_plane_shape_with_batch = action_plane_shape_with_batch.unsqueeze(
        #             1
        #         ).shape
        #         hidden_channels = hidden_state_shape[1]
        #         action_channels = action_plane_shape_with_batch[1]
        #         new_channels = hidden_channels + action_channels
        #         # Dynamics input shape: (C_h + C_a, H, W)
        #         dynamics_input_shape = torch.Size(
        #             [self.config.minibatch_size]
        #             + [new_channels]
        #             + list(hidden_state_shape[2:])
        #         )
        #     else:
        #         action_plane_shape_with_batch = action_plane_shape_with_batch.unsqueeze(
        #             -1
        #         ).shape
        #         hidden_channels = hidden_state_shape[-1]
        #         action_channels = action_plane_shape_with_batch[-1]
        #         new_channels = hidden_channels + action_channels
        #         # Dynamics input shape: (H, W, C_h + C_a)
        #         dynamics_input_shape = torch.Size(
        #             list(hidden_state_shape[:-1]) + [new_channels]
        #         )

        # elif len(hidden_state_shape[1:]) == 1:
        #     print("vector hidden state")
        #     # Case B: Vector State (e.g., MLP)

        #     # The spatial_shape argument is irrelevant for vector action encoding
        #     self.spatial_shape = None

        #     # Use a dummy action to get the shape (B=1, D_a)
        #     action_vector_shape_with_batch = action_function(
        #         self.num_actions, self.spatial_shape, torch.tensor(0).unsqueeze(0)
        #     ).shape

        #     # The hidden state dimension D_h
        #     hidden_dim = hidden_state_shape[1]
        #     # The action vector dimension D_a (typically num_actions)
        #     action_dim = action_vector_shape_with_batch[1]

        #     # Dynamics input shape: (D_h + D_a,)
        #     dynamics_input_shape = torch.Size(
        #         [self.config.minibatch_size] + [hidden_dim + action_dim]
        #     )

        # else:
        #     raise ValueError(f"Unsupported hidden state shape: {hidden_state_shape}")

        # print("Dynamics input shape (excluding batch):", dynamics_input_shape)

        # --- 3. Dynamics and Prediction Networks ---
        if config.stochastic:
            self.afterstate_dynamics = AfterstateDynamics(
                config, self.representation.output_shape
            )
            self.afterstate_prediction = AfterstatePrediction(
                config, config.num_chance, self.representation.output_shape
            )
        self.dynamics = Dynamics(config, self.representation.output_shape)
        # Dynamics output must match Representation output shape
        assert (
            self.dynamics.output_shape == self.representation.output_shape
        ), f"{self.dynamics.output_shape} = {self.representation.output_shape}"
        self.prediction = Prediction(config, output_size, hidden_state_shape)

        # --- 4. EFFICIENT ZERO Projector ---
        # The flat hidden dimension is simply the total size of the hidden state
        self.flat_hidden_dim = torch.Size(hidden_state_shape[1:]).numel()
        self.projector = Projector(self.flat_hidden_dim, config)

        hs_flat = torch.zeros(self.representation.output_shape).flatten(1, -1)
        flat_size = hs_flat.shape[1]
        # lazily create a linear mapping that maps (flat_size + num_actions) -> flat_size
        # this makes the layer persistent (trainable) after first forward.
        self._afterstate_action_linear = nn.Linear(
            flat_size + self.num_actions, flat_size
        )

        hs_flat = torch.zeros(self.representation.output_shape).flatten(1, -1)
        flat_size = hs_flat.shape[1]
        # lazily create a linear mapping that maps (flat_size + num_actions) -> flat_size
        # this makes the layer persistent (trainable) after first forward.
        self._afterstate_action_linear = nn.Linear(
            flat_size + self.num_actions, flat_size
        )

        if self.config.stochastic:
            self._action_linear = nn.Linear(flat_size + self.num_chance, flat_size)
        else:
            self._action_linear = nn.Linear(flat_size + self.num_actions, flat_size)

        self.encoder = Encoder(
            input_shape,
            num_codes=self.config.num_chance,
            channel_first=channel_first,
        )

    def initial_inference(self, x):
        # print("Initial inference")
        hidden_state = self.representation(x)
        value, policy = self.prediction(hidden_state)
        # print("Hidden state:", hidden_state.shape)
        return value, policy, hidden_state

    def recurrent_inference(
        self,
        hidden_state,
        action,
        reward_h_states,
        reward_c_states,
    ):
        # Hidden_state is now expected to be (B, ...)

        # Call the batch-enabled action function
        # action_plane shape is (B, C_action, H, W) or (B, H, W, C_action)
        # print(action.shape)
        # action = self.action_function(
        #     self.num_actions,
        #     self.spatial_shape,
        #     action,  # action is (B,)
        # ).to(hidden_state.device)

        # # 1. Prepare action input
        # if len(action.shape) == 3:  # image with no channel
        #     # it is an image
        #     # print("image")
        #     if self.channel_first:
        #         # The action plane will be concatenated along the C dimension (dim=1)
        #         concat_dim = 1
        #     else:
        #         # The action plane will be concatenated along the C dimension (dim=-1)
        #         concat_dim = -1
        #     action = action.unsqueeze(concat_dim)
        # else:
        #     concat_dim = 1

        # # action_plane needs to be broadcast to the correct shape if it's 1D (like a one-hot vector)
        # # Assuming action_plane is now correctly shaped for concatenation (e.g., (B, 1, H, W) for plane or (B, N) for onehot)
        # # Note: If self.action_function is action_as_onehot_batch, it returns (B, num_actions),
        # # which can only be concatenated if the dynamics network expects a flattened input.
        # # We assume action_as_plane_batch is used here, which returns (B, C_action, H, W)

        # # 2. Concatenate Hidden State and Action Plane
        # # nn_input shape will be (B, C + C_action, H, W)
        # # print(hidden_state.shape)
        # # print(action.shape)
        # nn_input = torch.cat(
        #     (hidden_state, action),
        #     dim=concat_dim,
        # )

        # Hidden_state is now expected to be (B, ...)
        # action is (B,)

        # --- simple action -> hidden-shaped plane transform (replaces self.action_function) ---
        B = hidden_state.shape[0]
        # print(hidden_state.shape)
        hs_shape = tuple(hidden_state.shape[1:])  # e.g. (C, H, W) or (D,)
        # compute flattened size of one sample (product of spatial/channel dims)
        # print(tuple(hidden_state.shape[1:]))
        # flat_size = 1
        # for s in hs_shape:
        #     flat_size *= int(s)

        # flatten hidden state (B, flat_size)
        # hs_flat = hidden_state.reshape(B, flat_size)
        # print(flat_size)
        hs_flat = hidden_state.flatten(1, -1)
        flat_size = hs_flat.shape[1]

        if self.config.stochastic:
            # TODO: add an assert that this should already be a onehot of a code, instead of remaking the onehot
            # THIS IS A CODE, MAKE SURE
            # code_one_hot = F.gumbel_softmax(
            #     action.float(), tau=1e-20, hard=True, dim=-1
            # )
            # concat -> (B, flat_size + num_actions)
            # print(action_onehot.shape)
            # print(hs_flat.shape)
            concat = torch.cat([hs_flat, action], dim=1)
            # print("concat shape", concat.shape)
        else:
            action = action.view(-1).to(hidden_state.device)
            # one-hot the action -> (B, num_actions)
            action_one_hot = (
                F.one_hot(action.long(), num_classes=self.num_actions)
                .float()
                .to(hidden_state.device)
            )

            # concat -> (B, flat_size + num_actions)
            concat = torch.cat([hs_flat, action_one_hot], dim=1)

        # If you prefer a non-trainable, temporary linear (not recommended for training), replace above with:
        # linear = nn.Linear(flat_size + self.num_actions, flat_size).to(hidden_state.device)
        # action_plane_flat = linear(concat)

        action_plane_flat = self._action_linear(concat)  # (B, flat_size)

        # reshape back to hidden-state spatial shape: (B, *hs_shape)
        nn_input = action_plane_flat.view(hidden_state.shape)
        # -------------------------------------------------------------------------------

        # Now `action_plane` has the same per-sample shape as `hidden_state` (except batch dim).
        # Continue with your existing logic that expects an action tensor to concat with hidden_state.

        # 1. Prepare action input
        # if (
        #     len(action_plane.shape) == 3
        # ):  # image with no channel (unlikely if hidden_state had channels)
        #     if self.channel_first:
        #         concat_dim = 1
        #     else:
        #         concat_dim = -1
        #     action_plane = action_plane.unsqueeze(concat_dim)
        # else:
        #     concat_dim = 1

        # # 2. Concatenate Hidden State and Action Plane
        # nn_input = torch.cat((hidden_state, action_plane), dim=concat_dim)

        # 3. Dynamics and Prediction
        # nn_input is already batched, no unsqueeze(0) needed
        # print("reward h", reward_h_states.shape)
        # print("reward c", reward_c_states.shape)
        reward, hidden_state, to_play, reward_hidden = self.dynamics(
            nn_input, (reward_h_states, reward_c_states)
        )  # hidden_state is now s_{t+1}
        value, policy = self.prediction(
            hidden_state
        )  # value, policy are predictions from s_{t+1}

        # The outputs are already batched (B, ...)
        return reward, hidden_state, value, policy, to_play, reward_hidden

    def afterstate_recurrent_inference(
        self,
        hidden_state,
        action,
    ):
        # --- simple action -> hidden-shaped plane transform (replaces self.action_function) ---
        B = hidden_state.shape[0]
        hs_shape = tuple(hidden_state.shape[1:])  # e.g. (C, H, W) or (D,)
        # compute flattened size of one sample (product of spatial/channel dims)
        # flat_size = 1
        # for s in hs_shape:
        #     flat_size *= int(s)

        # flatten hidden state (B, flat_size)

        # print(action.shape)
        # one-hot the action -> (B, num_actions)
        action_one_hot = (
            F.one_hot(action.long(), num_classes=self.num_actions)
            .float()
            .to(hidden_state.device)
        )
        hs_flat = hidden_state.flatten(1, -1)
        # concat -> (B, flat_size + num_actions)
        concat = torch.cat([hs_flat, action_one_hot], dim=1)

        # If you prefer a non-trainable, temporary linear (not recommended for training), replace above with:
        # linear = nn.Linear(flat_size + self.num_actions, flat_size).to(hidden_state.device)
        # action_plane_flat = linear(concat)

        action_plane_flat = self._afterstate_action_linear(concat)  # (B, flat_size)

        # reshape back to hidden-state spatial shape: (B, *hs_shape)
        nn_input = action_plane_flat.view(hidden_state.shape)
        # -------------------------------------------------------------------------------

        # Now `action_plane` has the same per-sample shape as `hidden_state` (except batch dim).
        # Continue with your existing logic that expects an action tensor to concat with hidden_state.

        # 1. Prepare action input
        # if (
        #     len(action_plane.shape) == 3
        # ):  # image with no channel (unlikely if hidden_state had channels)
        #     if self.channel_first:
        #         concat_dim = 1
        #     else:
        #         concat_dim = -1
        #     action_plane = action_plane.unsqueeze(concat_dim)
        # else:
        #     concat_dim = 1

        # # 2. Concatenate Hidden State and Action Plane
        # nn_input = torch.cat((hidden_state, action_plane), dim=concat_dim)

        # 3. Dynamics and Prediction
        # nn_input is already batched, no unsqueeze(0) needed
        # print("reward h", reward_h_states.shape)
        # print("reward c", reward_c_states.shape)
        afterstate = self.afterstate_dynamics(
            nn_input,
        )  # hidden_state is now s_{t+1}
        value, sigma = self.afterstate_prediction(
            hidden_state
        )  # value, policy are predictions from s_{t+1}

        # The outputs are already batched (B, ...)
        return afterstate, value, sigma

    def project(self, hidden_state: Tensor, grad=True) -> Tensor:
        """
        Projects the hidden state (s_t) into the embedding space.
        Used for both the 'real' target observation and the 'predicted' latent.
        """
        # Flatten the spatial dimensions (B, C, H, W) -> (B, C*H*W)
        flat_hidden = hidden_state.flatten(1, -1)
        proj = self.projector.projection(flat_hidden)

        # with grad, use proj_head
        if grad:
            proj = self.projector.projection_head(proj)
            return proj
        else:
            return proj.detach()
