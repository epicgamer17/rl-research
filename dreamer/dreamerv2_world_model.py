from typing import Callable, Optional, Tuple, Dict

from torch import Tensor
import torch
from modules.action_encoder import ActionEncoder
from modules.conv import Conv2dStack
from modules.network_block import NetworkBlock
from modules.residual import ResidualStack
from modules.utils import _normalize_hidden_state
from modules.world_model import WorldModelInterface
from packages.agent_configs.agent_configs.muzero_config import MuZeroConfig

from torch import nn
import torch.nn.functional as F
from modules.dense import DenseStack, build_dense
from modules.utils import zero_weights_initializer
from packages.utils.utils.utils import to_lists


class RewardToPlayHead(nn.Module):
    """
    The head of the Dynamics network for predicting reward and 'to play'.
    """

    # TODO: change configs so that this can be a MuZero config or like a MuZero Network config or something (we have a circular imports right now)
    def __init__(self, config: MuZeroConfig, input_shape: Tuple[int]):
        super().__init__()
        self.config = config
        current_shape = input_shape
        B = current_shape[0]

        # --- Reward Prediction Layers ---
        self.reward_block = NetworkBlock(config, current_shape, "reward")

        # Get final flattened width after reward block (or input width if block is Identity)
        reward_block_output_shape = self.reward_block.output_shape
        if len(reward_block_output_shape) == 4:
            reward_width = (
                reward_block_output_shape[1]
                * reward_block_output_shape[2]
                * reward_block_output_shape[3]
            )
        else:
            reward_width = reward_block_output_shape[1]

        # LSTM for value prefix
        if self.config.value_prefix:
            self.lstm = nn.LSTM(
                input_size=reward_width, hidden_size=self.config.lstm_hidden_size
            )
            reward_width = self.config.lstm_hidden_size

        # Reward output head
        if config.support_range is not None:
            self.full_support_size = 2 * config.support_range + 1
            reward_out_features = self.full_support_size
        else:
            reward_out_features = 1

        self.reward = build_dense(
            in_features=reward_width,
            out_features=reward_out_features,
            sigma=0,
        )

        # --- To Play Prediction Layers ---
        self.to_play_block = NetworkBlock(config, current_shape, "to_play")

        # Get final flattened width after to_play block (or input width if block is Identity)
        to_play_block_output_shape = self.to_play_block.output_shape
        if len(to_play_block_output_shape) == 4:
            to_play_width = (
                to_play_block_output_shape[1]
                * to_play_block_output_shape[2]
                * to_play_block_output_shape[3]
            )
        else:
            to_play_width = to_play_block_output_shape[1]

        self.to_play = build_dense(
            in_features=to_play_width,
            out_features=self.config.game.num_players,
            sigma=0,
        )

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        self.reward_block.initialize(initializer)
        self.to_play_block.initialize(initializer)

        # Reward head initialization
        if self.config.support_range is not None:
            self.reward.apply(zero_weights_initializer)
        else:
            # Standard initialization for scalar output
            if hasattr(self.reward, "initialize"):
                self.reward.initialize(initializer)
            else:
                self.reward.apply(initializer)

        # To Play head initialization (always a probability distribution)
        self.to_play.apply(zero_weights_initializer)

    def forward(
        self, S: torch.Tensor, reward_hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        # --- Reward Path ---
        reward_vector = self.reward_block(S)
        flattened_reward_vector = reward_vector.flatten(1, -1)

        if self.config.value_prefix:
            flattened_reward_vector = flattened_reward_vector.unsqueeze(0)
            flattened_reward_vector, new_reward_hidden = self.lstm(
                flattened_reward_vector, reward_hidden
            )
            flattened_reward_vector = flattened_reward_vector.squeeze(0)
        else:
            new_reward_hidden = reward_hidden

        if self.config.support_range is None:
            reward = self.reward(flattened_reward_vector)
        else:
            reward = self.reward(flattened_reward_vector).softmax(dim=-1)

        # --- To Play Path ---
        to_play_vector = self.to_play_block(S)
        flattened_to_play_vector = to_play_vector.flatten(1, -1)
        to_play = self.to_play(flattened_to_play_vector).softmax(dim=-1)

        return reward, to_play, new_reward_hidden


# --- Refactored Primary Modules ---
class Representation(nn.Module):
    def __init__(self, config: MuZeroConfig, input_shape: Tuple[int]):
        super().__init__()
        assert (
            config.game.is_discrete
        ), "AlphaZero only works for discrete action space games (board games)"
        self.config = config

        self.net = NetworkBlock(config, input_shape, "representation")
        self.output_shape = self.net.output_shape

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        self.net.initialize(initializer)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        S = self.net(inputs)
        # Apply normalization to the final output of the representation network
        return _normalize_hidden_state(S)


class BaseDynamics(nn.Module):
    """Base class for Dynamics and AfterstateDynamics, handling action fusion and core block."""

    def __init__(
        self,
        config: MuZeroConfig,
        input_shape: Tuple[int],
        num_actions: int,
        action_embedding_dim: int,
        layer_prefix: str,
    ):
        super().__init__()
        self.config = config
        self.action_embedding_dim = action_embedding_dim
        is_continuous = not self.config.game.is_discrete

        # 1. Action Encoder
        self.action_encoder = ActionEncoder(
            action_space_size=num_actions,
            embedding_dim=self.action_embedding_dim,
            is_continuous=is_continuous,
            single_action_plane=(
                layer_prefix == "dynamics"
            ),  # Assuming standard dynamics uses single_action_plane=True
        )

        # 2. Fusion Layer (Move from ActionEncoder to Dynamics)
        if len(input_shape) == 4:
            self.num_channels = input_shape[1]
            in_channels = self.num_channels + self.action_embedding_dim
            out_channels = self.num_channels
            self.fusion = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=1, bias=False
            )
            self.fusion_bn = nn.BatchNorm2d(out_channels)
        else:
            self.input_size = input_shape[1]
            in_features = self.input_size + self.action_embedding_dim
            out_features = self.input_size
            self.fusion = nn.Linear(in_features, out_features, bias=False)
            self.fusion_bn = nn.BatchNorm1d(out_features)

        # 3. Core Network Block
        self.net = NetworkBlock(config, input_shape, layer_prefix)
        self.output_shape = self.net.output_shape

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        self.net.initialize(initializer)
        # Additional initializations for fusion layers if needed

    def _fuse_and_process(
        self, hidden_state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        # Embed action
        action_embedded = self.action_encoder(action, hidden_state.shape)

        # Concatenate and fuse
        x = torch.cat((hidden_state, action_embedded), dim=1)
        x = self.fusion(x)
        # x = self.fusion_bn(x) # BN is often omitted or placed after ReLU in some MuZero implementations

        # Residual Connection
        x = x + hidden_state
        S = F.relu(x)

        # Process through the main network block
        S = self.net(S)

        # Apply normalization to the final output of the dynamics network
        next_hidden_state = _normalize_hidden_state(S)

        return next_hidden_state


class Dynamics(BaseDynamics):
    def __init__(
        self,
        config: MuZeroConfig,
        input_shape: Tuple[int],
        num_actions: int,
        action_embedding_dim: int,
    ):
        # dynamics layers uses the "dynamics" prefix
        super().__init__(
            config, input_shape, num_actions, action_embedding_dim, "dynamics"
        )
        self.reward_to_play_head = RewardToPlayHead(config, self.net.output_shape)

    def initialize(self, initializer: Callable[[torch.Tensor], None]) -> None:
        super().initialize(initializer)
        self.reward_to_play_head.initialize(initializer)

    def forward(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
        reward_hidden: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
    ]:

        next_hidden_state_unnormalized = self._fuse_and_process(hidden_state, action)

        # NOTE: _fuse_and_process already normalizes the hidden state, so next_hidden_state is the final output.
        S = next_hidden_state_unnormalized

        # Predict reward and to_play
        reward, to_play, new_reward_hidden = self.reward_to_play_head(S, reward_hidden)

        return reward, S, to_play, new_reward_hidden


class AfterstateDynamics(BaseDynamics):
    def __init__(
        self,
        config: MuZeroConfig,
        input_shape: Tuple[int],
        num_actions: int,
        action_embedding_dim: int,
    ):
        # afterstate dynamics uses the "dynamics" prefix for its network block, which seems to be the intent
        super().__init__(
            config, input_shape, num_actions, action_embedding_dim, "dynamics"
        )

    def forward(self, hidden_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # The base class handles fusion and processing, returning the normalized hidden state (afterstate)
        afterstate = self._fuse_and_process(hidden_state, action)
        return afterstate


class MuzeroWorldModel(WorldModelInterface, nn.Module):
    def __init__(self, config: MuZeroConfig, input_shape: Tuple[int], num_actions: int):
        nn.Module.__init__(self)
        self.config = config
        # --- 1. Representation Network ---
        self.representation = Representation(config, input_shape)
        self.num_actions = num_actions
        self.num_chance = config.num_chance

        hidden_state_shape = self.representation.output_shape
        print("Hidden state shape:", hidden_state_shape)

        # --- 3. Dynamics and Prediction Networks ---
        if self.config.stochastic:
            self.afterstate_dynamics = AfterstateDynamics(
                self.config,
                self.representation.output_shape,
                num_actions=self.num_actions,
                action_embedding_dim=self.config.action_embedding_dim,
            )

        if self.config.stochastic:
            self.dynamics = Dynamics(
                self.config,
                self.representation.output_shape,
                num_actions=self.num_chance,
                action_embedding_dim=self.config.action_embedding_dim,
            )
        else:
            self.dynamics = Dynamics(
                self.config,
                self.representation.output_shape,
                num_actions=self.num_actions,
                action_embedding_dim=self.config.action_embedding_dim,
            )

        # Dynamics output must match Representation output shape
        assert (
            self.dynamics.output_shape == self.representation.output_shape
        ), f"{self.dynamics.output_shape} = {self.representation.output_shape}"

    def initial_inference(self, observation: Tensor) -> Tensor:
        hidden_state = self.representation(observation)
        return hidden_state

    def recurrent_inference(
        self,
        hidden_state: Tensor,
        action: Tensor,
        reward_h_states: Optional[Tensor],
        reward_c_states: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if not self.config.stochastic:
            action = action.view(-1).to(hidden_state.device)
            # one-hot the action -> (B, num_actions)
            action = (
                F.one_hot(action.long(), num_classes=self.num_actions)
                .float()
                .to(hidden_state.device)
            )
        # print("hidden_state", hidden_state.shape)
        # print("action", action.shape)

        reward, next_hidden_state, to_play, reward_hidden = self.dynamics(
            hidden_state, action, (reward_h_states, reward_c_states)
        )

        return reward, next_hidden_state, to_play, reward_hidden

    def afterstate_recurrent_inference(
        self,
        hidden_state,
        action,
    ):
        # The AfterstateDynamics class handles (State, Action) -> Fusion -> Afterstate internally.
        action = action.view(-1).to(hidden_state.device)
        # one-hot the action -> (B, num_actions)
        action = (
            F.one_hot(action.long(), num_classes=self.num_actions)
            .float()
            .to(hidden_state.device)
        )

        afterstate = self.afterstate_dynamics(hidden_state, action)
        return afterstate

    def get_networks(self) -> Dict[str, nn.Module]:
        return {
            "representation_network": self.representation,
            "dynamics_network": self.dynamics,
            "afterstate_dynamics_network": self.afterstate_dynamics,
        }
