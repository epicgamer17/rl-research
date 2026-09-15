"""Action encoding transformations for model-based RL (e.g., MuZero, AlphaZero).

In model-based reinforcement learning, discrete actions must be converted into
feature representations (such as spatial planes, column strips, or feature maps)
to be concatenated with latent states in dynamics networks.

This module provides functional utilities to pre-materialize static action
encodings, enabling fast $O(1)$ tensor indexing during batched inference and training.
"""

from typing import Callable, Tuple
import torch


def build_action_encodings(
    num_actions: int,
    encoding_shape: Tuple[int, ...],
    mapping_fn: Callable[[int, torch.Tensor], None],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pre-materializes a lookup tensor of action encodings of shape `[num_actions, *encoding_shape]`.

    Depending on the domain, actions are spatially represented in distinct ways:
      - Grid games (TicTacToe, Go, Gomoku): action $a$ activates coordinate $(r, c)$ in a board plane.
      - Column/Gravity games (Connect-4): action $c$ activates an entire column across all rows.
      - Piece games (Chess, Shogi): action $a$ activates source/destination or move-plane channels.
      - Visual/Atari: action $a$ is tiled as a constant plane ($a / |A|$) or one-hot channel.

    By pre-materializing the encoding bank into a single tensor, encoding a batch of actions
    `actions: torch.Tensor` during dynamics rollout reduces to a single tensor indexing operation:
    `action_planes = action_encodings[actions]`.

    Args:
        num_actions: Total number of discrete actions in the environment action space.
        encoding_shape: Spatial shape of each individual action encoding (e.g., `(1, 3, 3)`
            for a 3x3 board or `(1, 6, 7)` for Connect-4).
        mapping_fn: A callable with signature `fn(action: int, target: torch.Tensor) -> None`
            that modifies `target` (a zero-initialized tensor of shape `encoding_shape`) in-place
            to represent `action`.
        dtype: Data type of the pre-materialized tensor (default: `torch.float32`).

    Returns:
        torch.Tensor of shape `(num_actions, *encoding_shape)`.

    Example:
        >>> # Tic-Tac-Toe (3x3 grid)
        >>> def tictactoe_map(action: int, target: torch.Tensor) -> None:
        ...     target[0, action // 3, action % 3] = 1.0
        >>> encodings = build_action_encodings(9, (1, 3, 3), tictactoe_map)
        >>> encodings.shape
        torch.Size([9, 1, 3, 3])
        >>> # Batched O(1) indexing during forward dynamics
        >>> batch_actions = torch.tensor([0, 4, 8])
        >>> batch_planes = encodings[batch_actions]
        >>> batch_planes.shape
        torch.Size([3, 1, 3, 3])
    """
    encodings = torch.zeros((num_actions, *encoding_shape), dtype=dtype)
    for a in range(num_actions):
        mapping_fn(a, encodings[a])
    return encodings
