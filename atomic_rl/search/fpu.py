import torch
import torch.nn.functional as F
from tensordict import TensorDict
from typing import Callable, Union

FPUFn = Callable[[TensorDict, torch.Tensor], torch.Tensor]


def fpu_zero(
    tree: TensorDict,
    parent_nodes: torch.Tensor,  # [B]
) -> torch.Tensor:
    """Returns 0.0 (neutral expectation for [-1, 1] zero-sum games) for unvisited actions.

    Shape: [B].
    """
    batch_size = tree.batch_size[0]
    assert parent_nodes.shape == (
        batch_size,
    ), f"parent_nodes shape mismatch: expected [{batch_size}], got {tuple(parent_nodes.shape)}"
    return tree["node_values"].new_zeros(batch_size)


def fpu_constant(
    tree: TensorDict,
    parent_nodes: torch.Tensor,  # [B]
    *,
    value: float = 0.0,
) -> torch.Tensor:
    """Returns a constant scalar FPU value for unvisited actions.

    Use with `functools.partial(fpu_constant, value=val)`. Shape: [B].
    """
    batch_size = tree.batch_size[0]
    assert parent_nodes.shape == (
        batch_size,
    ), f"parent_nodes shape mismatch: expected [{batch_size}], got {tuple(parent_nodes.shape)}"
    return tree["node_values"].new_full((batch_size,), float(value))


def fpu_parent_value(
    tree: TensorDict,
    parent_nodes: torch.Tensor,  # [B]
) -> torch.Tensor:
    """Returns parent node value V(s) for unvisited actions.

    Shape: [B].
    """
    batch_size = tree.batch_size[0]
    assert parent_nodes.shape == (
        batch_size,
    ), f"parent_nodes shape mismatch: expected [{batch_size}], got {tuple(parent_nodes.shape)}"
    batch_range = torch.arange(batch_size, device=tree.device)
    active_players = tree["to_play"][batch_range, parent_nodes]  # [B]
    node_values = tree["node_values"][batch_range, parent_nodes]  # [B, P]
    return node_values.gather(1, active_players.unsqueeze(-1)).squeeze(-1)  # [B]


def fpu_mixed_value(
    tree: TensorDict,
    parent_nodes: torch.Tensor,  # [B]
) -> torch.Tensor:
    """Returns prior-weighted mixed value V_mix for unvisited actions (Gumbel MuZero).

    Shape: [B].
    """
    batch_size = tree.batch_size[0]
    assert parent_nodes.shape == (
        batch_size,
    ), f"parent_nodes shape mismatch: expected [{batch_size}], got {tuple(parent_nodes.shape)}"
    batch_range = torch.arange(batch_size, device=tree.device)
    active_players = tree["to_play"][batch_range, parent_nodes]  # [B]

    # Import get_qvalues lazily to avoid circular module dependency
    from .qtransforms import get_qvalues

    qvalues = get_qvalues(tree, parent_nodes, active_players)  # [B, A]
    visit_counts = tree["children_visits"][batch_range, parent_nodes]  # [B, A]
    raw_values = tree["raw_values"][batch_range, parent_nodes]  # [B, P]
    raw_value = raw_values.gather(1, active_players.unsqueeze(-1)).squeeze(-1)  # [B]
    prior_logits = tree["children_prior_logits"][batch_range, parent_nodes]  # [B, A]
    prior_probs = F.softmax(prior_logits, dim=-1)  # [B, A]

    sum_visit_counts = visit_counts.sum(dim=-1).to(qvalues.dtype)  # [B]
    tiny = torch.finfo(prior_probs.dtype).tiny
    safe_priors = torch.clamp(prior_probs, min=tiny)

    visited_mask = visit_counts > 0
    sum_probs = torch.where(visited_mask, safe_priors, 0.0).sum(dim=-1, keepdim=True)
    sum_probs_safe = torch.where(sum_probs > 0, sum_probs, 1.0)

    normalized_priors = safe_priors / sum_probs_safe
    weighted_q = torch.where(visited_mask, normalized_priors * qvalues, 0.0).sum(
        dim=-1
    )  # [B]

    return (raw_value + sum_visit_counts * weighted_q) / (sum_visit_counts + 1.0)


def fpu_min_value(
    tree: TensorDict,
    parent_nodes: torch.Tensor,  # [B]
    *,
    min_value: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Returns min_value (pessimistic value) for unvisited actions.

    Use with `functools.partial(fpu_min_value, min_value=min_val)`. Shape: [B].
    """
    batch_size = tree.batch_size[0]
    assert parent_nodes.shape == (
        batch_size,
    ), f"parent_nodes shape mismatch: expected [{batch_size}], got {tuple(parent_nodes.shape)}"
    if not isinstance(min_value, torch.Tensor):
        return tree["node_values"].new_full((batch_size,), float(min_value))
    if min_value.ndim == 0:
        return min_value.expand(batch_size).to(
            device=tree.device, dtype=tree["node_values"].dtype
        )
    assert min_value.shape == (
        batch_size,
    ), f"min_value shape mismatch: expected [{batch_size}], got {tuple(min_value.shape)}"
    return min_value.to(device=tree.device, dtype=tree["node_values"].dtype)
