import torch


# TODO: should I make this accept a tree?
def get_mcts_visit_policy(
    visit_counts: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Computes a target policy distribution from MCTS visit counts.

    Formula:
        for tau > 0: pi(a|s) = N(s, a)^(1/tau) / sum_b N(s, b)^(1/tau)
        for tau = 0: pi(a|s) = one_hot(argmax_a N(s, a))

    Args:
        visit_counts: Tensor of visit counts [B, A] or [A].
        temperature: Temperature parameter tau >= 0.

    Returns:
        torch.Tensor: Target policy probability distribution with same shape as visit_counts.
    """
    assert temperature >= 0.0, f"Temperature must be non-negative, got {temperature}"
    assert visit_counts.ndim in [
        1,
        2,
    ], f"visit_counts must be 1D [A] or 2D [B, A], got shape {tuple(visit_counts.shape)}"

    num_actions = visit_counts.shape[-1]
    uniform = torch.full_like(visit_counts.float(), 1.0 / num_actions)

    if temperature == 0.0:
        # TODO: right now for 2 equally visited actions this will output 0.5, 0.5. Is this what we want? Or do we want to use F.one_hot and get a 1, 0? or 0, 1? Should we add a random tie-breaking method?
        # TODO: maybe add a dim arg instead of hardcoding the dim
        is_max = (
            visit_counts == torch.max(visit_counts, dim=-1, keepdim=True).values
        ).float()
        return is_max / is_max.sum(dim=-1, keepdim=True)

    if temperature == 1.0:
        total_visits = visit_counts.sum(dim=-1, keepdim=True)
        return torch.where(
            total_visits > 0, visit_counts.float() / total_visits, uniform
        )

    exponent = 1.0 / temperature
    scaled_visits = torch.pow(visit_counts.float(), exponent)
    total_scaled = scaled_visits.sum(dim=-1, keepdim=True)
    return torch.where(total_scaled > 0, scaled_visits / total_scaled, uniform)
