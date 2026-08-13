import functools
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from typing import Union, Optional, Tuple
from .fpu import (
    FPUFn,
    fpu_zero,
    fpu_parent_value,
    fpu_mixed_value,
    fpu_min_value,
)


def get_qvalues(
    tree: TensorDict, parent_nodes: torch.Tensor, active_players: torch.Tensor
) -> torch.Tensor:
    """Computes Q-values for parent_nodes matching mctx.Tree.qvalues().

    Q(s, a) = rewards(s, a) + discounts(s, a) * V(s')
    """
    batch_size = tree.batch_size[0]
    batch_range = torch.arange(batch_size, device=tree.device)

    assert parent_nodes.shape == (
        batch_size,
    ), f"parent_nodes shape mismatch: expected [{batch_size}], got {tuple(parent_nodes.shape)}"

    # Fetch N-player tensors
    rewards = tree["children_rewards"][batch_range, parent_nodes]  # [B, num_actions, P]
    discounts = tree["children_discounts"][
        batch_range, parent_nodes
    ]  # [B, num_actions, P]
    values = tree["children_values"][batch_range, parent_nodes]  # [B, num_actions, P]

    # Compute vector Q-values for all players: [B, num_actions, P]
    q_vec = rewards + discounts * values

    # Slice for the acting player p at each batch element -> [B, num_actions]
    p_idx = active_players.view(-1, 1, 1).expand(-1, q_vec.shape[1], 1)
    q_active = q_vec.gather(2, p_idx).squeeze(-1)

    return q_active


def get_completed_qvalues(
    tree: TensorDict,
    parent_nodes: torch.Tensor,  # [B]
    *,
    fpu_fn: FPUFn = fpu_zero,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes Q-values for parent_nodes and completes unvisited actions using `fpu_fn`.

    Returns:
        completed_qvalues: [B, num_actions]
        visit_counts: [B, num_actions]
    """
    batch_size = tree.batch_size[0]
    batch_range = torch.arange(batch_size, device=tree.device)
    active_players = tree["to_play"][batch_range, parent_nodes]  # [B]

    assert parent_nodes.shape == (
        batch_size,
    ), f"parent_nodes shape mismatch: expected [{batch_size}], got {tuple(parent_nodes.shape)}"

    qvalues = get_qvalues(tree, parent_nodes, active_players)  # [B, A]
    visit_counts = tree["children_visits"][batch_range, parent_nodes]  # [B, A]

    # Evaluate FPU strategy callable
    unvisited_value = fpu_fn(tree, parent_nodes)  # [B]

    # Expand unvisited value for broadcasting over actions: [B, 1]
    unvisited_exp = (
        unvisited_value.unsqueeze(-1) if unvisited_value.ndim == 1 else unvisited_value
    )

    completed_qvalues = torch.where(visit_counts > 0, qvalues, unvisited_exp)
    return completed_qvalues, visit_counts


def _format_bound(
    val: Union[torch.Tensor, float], target_tensor: torch.Tensor
) -> torch.Tensor:
    """Helper to reshape min/max bounds into [B, 1] tensors for safe broadcasting."""
    if not isinstance(val, torch.Tensor):
        val = torch.tensor(val, device=target_tensor.device, dtype=target_tensor.dtype)
    if val.ndim == 0:
        return val.unsqueeze(0).unsqueeze(-1)
    if val.ndim == 1:
        return val.unsqueeze(-1)
    return val


def qtransform_identity(
    tree: TensorDict,
    parent_nodes: torch.Tensor,  # [B]
    *,
    fpu_fn: FPUFn = fpu_zero,
) -> torch.Tensor:
    """Returns completed Q-values for parent_nodes without normalization.

    Shape: [B, num_actions].
    """
    completed_qvalues, _ = get_completed_qvalues(tree, parent_nodes, fpu_fn=fpu_fn)
    return completed_qvalues


def qtransform_by_min_max(
    tree: TensorDict,
    parent_nodes: torch.Tensor,  # [B]
    *,
    min_value: torch.Tensor,  # [B] or scalar
    max_value: torch.Tensor,  # [B] or scalar
    fpu_fn: Optional[FPUFn] = None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Returns Q-values normalized by the given `min_value` and `max_value`.

    If `fpu_fn` is None, defaults to `functools.partial(fpu_min_value, min_value=min_value)`
    for full backward compatibility with mctx semantics (unvisited actions mapped to min_value).
    Pass `fpu_fn=fpu_zero` to assign 0.0 before normalization (mapping to 0.5 in [0, 1] for min=-1, max=1).
    Shape: [B, num_actions].
    """
    if fpu_fn is None:
        fpu_fn = functools.partial(fpu_min_value, min_value=min_value)

    completed_qvalues, _ = get_completed_qvalues(tree, parent_nodes, fpu_fn=fpu_fn)

    min_val = _format_bound(min_value, completed_qvalues)
    max_val = _format_bound(max_value, completed_qvalues)

    denom = torch.clamp(max_val - min_val, min=epsilon)
    normalized = (completed_qvalues - min_val) / denom

    return normalized


def qtransform_by_parent_and_siblings(
    tree: TensorDict,
    parent_nodes: torch.Tensor,  # [B]
    *,
    fpu_fn: Optional[FPUFn] = None,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Returns Q-values normalized by min/max over V(parent) and visited Q-values.

    If `fpu_fn` is None, defaults to `functools.partial(fpu_min_value, min_value=min_value)`
    where min_value is computed dynamically over parent value and safe Q-values (matching mctx).
    Shape: [B, num_actions].
    """
    batch_size = tree.batch_size[0]
    batch_range = torch.arange(batch_size, device=tree.device)
    active_players = tree["to_play"][batch_range, parent_nodes]  # [B]

    qvalues = get_qvalues(tree, parent_nodes, active_players)  # [B, A]
    visit_counts = tree["children_visits"][batch_range, parent_nodes]  # [B, A]
    node_values_vec = tree["node_values"][batch_range, parent_nodes]  # [B, P]
    node_value = node_values_vec.gather(1, active_players.unsqueeze(-1))  # [B, 1]

    # Calculate safe_qvalues (unvisited -> parent node_value) to derive dynamic min/max bounds
    safe_qvalues = torch.where(visit_counts > 0, qvalues, node_value)
    min_value = torch.minimum(node_value, safe_qvalues.amin(dim=-1, keepdim=True))
    max_value = torch.maximum(node_value, safe_qvalues.amax(dim=-1, keepdim=True))

    if fpu_fn is None:
        fpu_fn = functools.partial(fpu_min_value, min_value=min_value.squeeze(-1))

    completed_qvalues, _ = get_completed_qvalues(tree, parent_nodes, fpu_fn=fpu_fn)

    denom = torch.clamp(max_value - min_value, min=epsilon)
    normalized = (completed_qvalues - min_value) / denom

    return normalized


def qtransform_completed_by_mix_value(
    tree: TensorDict,
    parent_nodes: torch.Tensor,  # [B]
    *,
    fpu_fn: Optional[FPUFn] = None,
    use_mixed_value: bool = True,
    value_scale: float = 0.1,
    maxvisit_init: float = 50.0,
    rescale_values: bool = True,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Computes completed Q-values for Gumbel MuZero.

    Applies linear scaling: (maxvisit_init + max(visit_counts)) * value_scale * completed_qvalues.
    Shape: [B, num_actions].
    """
    if fpu_fn is None:
        fpu_fn = fpu_mixed_value if use_mixed_value else fpu_parent_value

    completed_qvalues, visit_counts = get_completed_qvalues(
        tree, parent_nodes, fpu_fn=fpu_fn
    )

    if rescale_values:
        completed_qvalues = _rescale_qvalues(completed_qvalues, epsilon)

    maxvisit = visit_counts.amax(dim=-1, keepdim=True).to(completed_qvalues.dtype)
    visit_scale = maxvisit_init + maxvisit
    return visit_scale * value_scale * completed_qvalues


def _rescale_qvalues(qvalues: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Rescales completed Q-values to [0, 1] per batch item."""
    min_value = qvalues.amin(dim=-1, keepdim=True)
    max_value = qvalues.amax(dim=-1, keepdim=True)
    denom = torch.clamp(max_value - min_value, min=epsilon)
    return (qvalues - min_value) / denom


def _compute_mixed_value(
    raw_value: torch.Tensor,  # [B]
    qvalues: torch.Tensor,  # [B, A]
    visit_counts: torch.Tensor,  # [B, A]
    prior_probs: torch.Tensor,  # [B, A]
) -> torch.Tensor:
    """Interpolates raw_value and prior-weighted Q-values as defined in Gumbel MuZero."""
    sum_visit_counts = visit_counts.sum(dim=-1).to(qvalues.dtype)  # [B]

    # Ensure non-zero prior_probs to avoid Division-by-Zero / NaNs
    tiny = torch.finfo(prior_probs.dtype).tiny
    safe_priors = torch.clamp(prior_probs, min=tiny)

    # Sum probabilities of visited actions
    visited_mask = visit_counts > 0
    sum_probs = torch.where(visited_mask, safe_priors, 0.0).sum(dim=-1, keepdim=True)
    sum_probs_safe = torch.where(sum_probs > 0, sum_probs, 1.0)

    # Re-normalize priors over visited actions only
    normalized_priors = safe_priors / sum_probs_safe
    weighted_q = torch.where(visited_mask, normalized_priors * qvalues, 0.0).sum(
        dim=-1
    )  # [B]

    return (raw_value + sum_visit_counts * weighted_q) / (sum_visit_counts + 1.0)
