from typing import Callable, Optional, Tuple, Dict

from torch import Tensor
import torch
from modules.action_encoder import ActionEncoder
from modules.heads import CategoricalHead, ScalarHead
from modules.network_block import NetworkBlock
from modules.utils import _normalize_hidden_state
from modules.world_model import WorldModelInterface
from agent_configs.muzero_config import MuZeroConfig

from torch import nn
import torch.nn.functional as F


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
        # 3. Heads
        # Reward Head: Has its own layers (reward_conv...) defined in config
        self.reward_head = ScalarHead(config, self.output_shape, layer_prefix="reward")

        # To-Play Head: Has its own layers (to_play_conv...) defined in config
        self.to_play_head = CategoricalHead(
            config,
            self.output_shape,
            output_size=config.game.num_players,
            layer_prefix="to_play",
        )

        # LSTM Support for Value Prefix
        if self.config.value_prefix:
            self.lstm = nn.LSTM(
                input_size=self.reward_head.input_flat_dim,
                hidden_size=self.config.lstm_hidden_size,
            )
            # Re-build reward output layer to match LSTM size
            self.reward_head.output_layer = self.reward_head.output_layer.__class__(
                in_features=self.config.lstm_hidden_size,
                out_features=self.reward_head.output_size,
                sigma=config.noisy_sigma,
            )

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

        next_hidden_state = self._fuse_and_process(hidden_state, action)

        # Predict reward and to_play
        r_x = self.reward_head._process_backbone(next_hidden_state)

        if self.config.value_prefix:
            r_x = r_x.unsqueeze(0)
            r_x, new_reward_hidden = self.lstm(r_x, reward_hidden)
            r_x = r_x.squeeze(0)
        else:
            new_reward_hidden = reward_hidden

        reward = self.reward_head.output_layer(r_x)
        if self.reward_head.is_probabilistic:
            reward = reward.softmax(dim=-1)

        # To Play
        to_play = self.to_play_head(next_hidden_state)
        return reward, next_hidden_state, to_play, new_reward_hidden


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
