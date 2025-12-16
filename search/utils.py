# --- Reusing Gumbel Helper Logic ---
import torch


def get_completed_q_improved_policy(config, node, min_max_stats):
    completedQ = get_completed_q(node, min_max_stats)
    sigma = calculate_gumbel_sigma(
        config.gumbel_cvisit, config.gumbel_cscale, node, completedQ
    )
    eps = 1e-12
    # Ensure network policy is on the same device as sigma/completedQ or CPU
    # Usually completedQ is on device?
    # Let's move to cpu for safety if needed or keep consistency.
    logits = torch.log(node.network_policy + eps)
    pi0_logits = logits + sigma
    pi0 = torch.softmax(pi0_logits, dim=0)
    return pi0


def get_completed_q(node, min_max_stats):
    v_mix = node.get_v_mix()
    completedQ = torch.full((len(node.network_policy),), min_max_stats.normalize(v_mix))
    for action, child in node.children.items():
        if child.expanded():
            completedQ[action] = min_max_stats.normalize(
                node.get_child_q_from_parent(child)
            )
    return completedQ


def calculate_gumbel_sigma(gumbel_cvisit, gumbel_cscale, node, completedQ):
    if len(node.children) > 0:
        max_visits = max([ch.visits for ch in node.children.values()])
    else:
        max_visits = 0

    return (gumbel_cvisit + max_visits) * gumbel_cscale * completedQ
