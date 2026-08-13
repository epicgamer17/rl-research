"""
Pure functional environment dynamics and observations for 3x3 TicTacToe.
Provides fast tensor simulation operations for MCTS and MuZero algorithms.
"""

from typing import Optional, Tuple
import torch

# TODO: this folder is kind of a utils folder, maybe rename it to that.


def check_tictactoe_winner(board: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Checks 3x3 board tensor (+1 for P0, -1 for P1, 0 for empty) for win or terminal draw.

    Args:
        board: [B, 3, 3] board tensor (+1 for P0, -1 for P1, 0 for empty).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (reward_vec [B, 2], is_terminal [B])
            reward_vec: [r_p0, r_p1] where r_p0 is P0's reward (+1, -1, 0) and r_p1 is P1's reward (-1, +1, 0).
    """
    batch_size = board.shape[0]

    rows = board.sum(dim=2)  # [B, 3]
    cols = board.sum(dim=1)  # [B, 3]
    diag1 = torch.stack([board[:, 0, 0], board[:, 1, 1], board[:, 2, 2]], dim=1).sum(
        dim=1, keepdim=True
    )
    diag2 = torch.stack([board[:, 0, 2], board[:, 1, 1], board[:, 2, 0]], dim=1).sum(
        dim=1, keepdim=True
    )

    lines = torch.cat([rows, cols, diag1, diag2], dim=1)  # [B, 8]

    p0_wins = (lines == 3).any(dim=1)
    p1_wins = (lines == -3).any(dim=1)
    board_full = (board != 0).all(dim=1).all(dim=1)

    winner = board.new_zeros(batch_size)
    winner = torch.where(p0_wins, board.new_tensor(1.0), winner)
    winner = torch.where(p1_wins, board.new_tensor(-1.0), winner)

    reward_vec = torch.stack([winner, -winner], dim=-1)
    is_terminal = p0_wins | p1_wins | board_full
    return reward_vec, is_terminal


def tictactoe_dynamics_fn(
    embeddings: torch.Tensor,
    actions_taken: torch.Tensor,
    to_play: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transition dynamics for MCTS simulation operating on 2-channel PettingZoo observations.
    Supports both CHW [B, 2, 3, 3] and HWC [B, 3, 3, 2] representations.

    Args:
        embeddings: PettingZoo observation tensor [B, 2, 3, 3] (CHW) or [B, 3, 3, 2] (HWC)
                    (Channel 0: Active Player pieces, Channel 1: Opponent pieces).
        actions_taken: [B] Flat action index 0..8.
        to_play: [B] Active player index (0 for player_1, 1 for player_2).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            (next_embeddings [B, 2, 3, 3] or [B, 3, 3, 2], reward_vec [B, 2], next_to_play [B], is_terminal [B], next_legal_mask [B, 9])
    """
    is_chw = embeddings.ndim == 4 and (
        embeddings.shape[2] == 3 and embeddings.shape[3] == 3
    )
    batch_size = embeddings.shape[0]
    device = embeddings.device
    batch_range = torch.arange(batch_size, device=device)

    total_channels = embeddings.shape[1] if is_chw else embeddings.shape[-1]
    has_player_plane = total_channels % 2 == 1
    num_frames = (total_channels - 1) // 2 if has_player_plane else total_channels // 2

    if is_chw:
        my_pieces = embeddings[:, 0].clone()  # [B, 3, 3]
        opp_pieces = embeddings[:, 1]  # [B, 3, 3]
    else:
        my_pieces = embeddings[..., 0].clone()  # [B, 3, 3]
        opp_pieces = embeddings[..., 1]  # [B, 3, 3]

    row = actions_taken // 3
    col = actions_taken % 3

    if (actions_taken < 0).any() or (actions_taken > 8).any():
        invalid_actions = actions_taken[(actions_taken < 0) | (actions_taken > 8)]
        raise ValueError(
            f"Action out of bounds [0..8]. Got actions: {invalid_actions.tolist()}"
        )

    # Check if target cell is occupied in either channel
    target_my = my_pieces[batch_range, row, col]
    target_opp = opp_pieces[batch_range, row, col]
    illegal_mask = (target_my != 0.0) | (target_opp != 0.0)

    if illegal_mask.any():
        illegal_batch_indices = batch_range[illegal_mask].tolist()
        illegal_actions = actions_taken[illegal_mask].tolist()

        raise ValueError(
            f"\n[FATAL ERROR] Illegal action(s) detected in dynamics transition!\n"
            f" - Affected Batch Indices: {illegal_batch_indices}\n"
            f" - Illegal Actions Attempted: {illegal_actions}\n"
        )

    new_my_pieces = my_pieces.clone()
    new_my_pieces[batch_range, row, col] = 1.0

    # Signed board for winner check (+1 for P0, -1 for P1)
    to_play_expanded = to_play.view(-1, 1, 1)
    signed_board = torch.where(
        to_play_expanded == 0,
        new_my_pieces - opp_pieces,
        opp_pieces - new_my_pieces,
    )

    reward_vec, is_terminal = check_tictactoe_winner(signed_board)

    next_to_play = 1 - to_play

    # Next state's active player is opponent -> swap piece channels across historical frames!
    next_channels = []

    # 1. Next Frame 0 (Current step T+1)
    next_channels.append(opp_pieces)
    next_channels.append(new_my_pieces)

    # 2. Shift historical frames (Frame 1..K-1) per agent-turn (2 steps) matching PettingZoo FrameStackWrapper
    for i in range(1, num_frames):
        if is_chw:
            hist_next_active = embeddings[:, 2 * i + 1] if 2 * i + 1 < total_channels - (1 if has_player_plane else 0) else embeddings.new_zeros(embeddings.shape[0], 3, 3)
            hist_prev_active = embeddings[:, 2 * (i - 1)]
        else:
            hist_next_active = embeddings[..., 2 * i + 1] if 2 * i + 1 < total_channels - (1 if has_player_plane else 0) else embeddings.new_zeros(embeddings.shape[0], 3, 3)
            hist_prev_active = embeddings[..., 2 * (i - 1)]

        next_channels.append(hist_next_active)
        next_channels.append(hist_prev_active)

    # 3. Next Player Plane (if included)
    if has_player_plane:
        p_val = (
            (next_to_play == 0)
            .to(dtype=embeddings.dtype)
            .view(-1, 1, 1)
            .expand(-1, 3, 3)
        )
        next_channels.append(p_val)

    if is_chw:
        next_embeddings = torch.stack(next_channels, dim=1)
    else:
        next_embeddings = torch.stack(next_channels, dim=-1)

    occupied = (new_my_pieces == 1.0) | (opp_pieces == 1.0)
    next_legal_mask = (~occupied).view(batch_size, -1)

    return next_embeddings, reward_vec, next_to_play, is_terminal, next_legal_mask
