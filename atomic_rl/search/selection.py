import torch
import torch.nn.functional as F
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional
from .qtransforms import qtransform_by_parent_and_siblings


# TODO: should we merge this with select_leaf?
def puct_score(
    tree: TensorDict,
    parent_nodes: torch.Tensor,  # [B]
    depth: int,
    *,
    pb_c_init: float = 1.25,
    pb_c_base: float = 19652.0,
    qtransform: Callable = qtransform_by_parent_and_siblings,
    **kwargs,
) -> torch.Tensor:
    """
    Computes PUCT scores matching mctx.muzero_action_selection.

    Formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    """
    batch_size = tree.batch_size[0]
    batch_range = torch.arange(batch_size, device=tree.device)

    assert tree["node_visits"].ndim == 2, (
        f"tree node buffers must be flat [B, N], got shape "
        f"{tuple(tree['node_visits'].shape)}"
    )
    assert parent_nodes.shape == (
        batch_size,
    ), f"parent_nodes shape mismatch: expected [{batch_size}], got {tuple(parent_nodes.shape)}"
    assert (
        isinstance(depth, int) and depth >= 0
    ), f"depth must be a non-negative int, got {depth!r}"
    assert pb_c_init >= 0, f"pb_c_init must be >= 0, got {pb_c_init}"
    assert pb_c_base >= 0, f"pb_c_base must be >= 0, got {pb_c_base}"

    # 1. Fetch node and child statistics
    visit_counts = tree["children_visits"][
        batch_range, parent_nodes
    ]  # [B, num_actions]
    node_visits = tree["node_visits"][batch_range, parent_nodes].unsqueeze(-1)  # [B, 1]
    prior_logits = tree["children_prior_logits"][
        batch_range, parent_nodes
    ]  # [B, num_actions]
    prior_probs = F.softmax(prior_logits, dim=-1)

    # 2. Compute PUCT exploration penalty
    pb_c = pb_c_init + torch.log((node_visits + pb_c_base + 1.0) / pb_c_base)
    policy_score = (
        torch.sqrt(node_visits.to(prior_probs.dtype))
        * pb_c
        * prior_probs
        / (visit_counts + 1.0)
    )

    # 3. Compute transformed Q-value score
    # NOTE: qtransform does the slicing for player, so return is [B, num_actions] (i think thats right?)
    value_score = qtransform(tree, parent_nodes)  # [B, num_actions]

    # 4. Add tiny uniform noise for tie breaking (matches mctx)
    noise = 1e-7 * torch.rand_like(value_score)
    scores = value_score + policy_score + noise

    # 5. Mask invalid actions at root and internal nodes
    # TODO: needed for qidentity and also vmix i think, to work for alphazero, but feels hacky. Make this simpler and less guessy. Follow simple not easy here
    legal_mask = (prior_logits > -1e7) & torch.isfinite(prior_logits)
    if depth == 0 and "root_legal_mask" in tree.keys():
        legal_mask = legal_mask & tree["root_legal_mask"]

    min_val = torch.finfo(scores.dtype).min
    scores = torch.where(legal_mask, scores, min_val)

    return scores


def puct_selection(
    tree: TensorDict,
    parent_nodes: torch.Tensor,
    depth: int,
    *,
    pb_c_init: float = 1.25,
    pb_c_base: float = 19652.0,
    qtransform: Callable = qtransform_by_parent_and_siblings,
    **kwargs,
) -> torch.Tensor:
    """Selects actions using PUCT argmax matching mctx.muzero_action_selection.

    Args:
        tree: The MCTS tree TensorDict.
        parent_nodes: Parent node indices [B].
        depth: Current tree depth.
        pb_c_init: PUCT exploration constant c1.
        pb_c_base: PUCT exploration constant c2.
        qtransform: Q-value transformation function.

    Returns:
        Selected action indices [B] from parent_nodes.
    """
    scores = puct_score(
        tree,
        parent_nodes,
        depth,
        pb_c_init=pb_c_init,
        pb_c_base=pb_c_base,
        qtransform=qtransform,
        **kwargs,
    )
    return scores.argmax(dim=-1)


# TODO: Stochastic MuZero Selection:
# In Stochastic MuZero (ICLR 2022, Antonoglou et al.) and mctx, tree search features two alternating node types:
# 1. Decision Nodes (is_chance=False): Choose an environment action a in {0, ..., A-1} via PUCT selection.
# 2. Chance Nodes (is_chance=True): Choose a chance outcome/code c in {0, ..., C-1} from afterstates.
#
# Selection at chance nodes can be performed via:
#   a) Quasi-sampling (argmax of prior / visit count):
#      code = argmax(softmax(children_prior_logits) / (children_visits + 1), dim=-1)
#   b) Categorical sampling:
#      code = torch.multinomial(F.softmax(children_prior_logits, dim=-1), num_samples=1).squeeze(-1)
#
# Helper implementation pattern to build:
# def make_stochastic_selection_fn(
#     decision_selection_fn: Callable = puct_selection,
#     chance_selection_fn: Optional[Callable] = None,
#     num_actions: Optional[int] = None,
# ) -> Callable[[TensorDict, torch.Tensor, int], torch.Tensor]:
#     """Returns a higher-order selector branching on tree['is_chance']."""
#     ...
# TODO: work with Batched MCTS, batch_mcts.pdf
# TODO: work with Vectorized MCTS
# TODO: work iwth Batched + Vectorized MCTS
# TODO: Stochastic MuZero
# TODO: stochastic muzero will require a different selection strategy based on the node type. begs the question if it should take a selection fn instead of some sort... This would also be good, as even though stochastic muzero uses the quasi sampling, if we wanted to do real sampling (for possibly dreamer, or just standard MCTS) it would have to not use an argmax
def select_leaf(
    tree: TensorDict,
    selection_fn: Callable[[TensorDict, torch.Tensor, int], torch.Tensor],
    max_depth: Optional[int] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    """Selects leaf nodes to expand by descending the search tree.

    Args:
        tree: The MCTS tree TensorDict.
        selection_fn: A callable `(tree, parent_nodes, depth) -> action [B]`
            specifying how edges are chosen at each node (e.g. `functools.partial(puct_selection, ...)`,
            a stochastic chance-aware selector, or Gumbel sequential halving).
        max_depth: Maximum number of edges from the root to descend before
            giving up on finding an unvisited node. Defaults to the number of
            search simulations (the tree width), matching mctx.

    Returns:
        leaf_parents: Parent node indices [B] where expansion occurs.
        leaf_actions: Actions taken [B] from leaf_parents.
        expansion_mask: Mask [B] indicating which batch items require expansion (True)
          vs items that hit terminal nodes or max depth (False).
        trajectory: List of (node_idx, action_idx, active_mask) tuples.
    """
    batch_size = tree.batch_size[0]
    device = tree.device
    batch_range = torch.arange(batch_size, device=device)

    assert callable(
        selection_fn
    ), f"selection_fn must be callable, got {selection_fn!r}"

    if max_depth is None:
        max_depth = tree["node_visits"].shape[-1] - 1

    assert tree["node_visits"].ndim == 2, (
        f"tree node buffers must be flat [B, N], got shape "
        f"{tuple(tree['node_visits'].shape)}"
    )
    assert (
        isinstance(max_depth, int)
        and not isinstance(max_depth, bool)
        and max_depth >= 1
    ), f"max_depth must be an int >= 1, got {max_depth!r}"

    current_node = torch.zeros(batch_size, dtype=torch.long, device=device)
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    leaf_parents = torch.zeros(batch_size, dtype=torch.long, device=device)
    leaf_actions = torch.zeros(batch_size, dtype=torch.long, device=device)
    expansion_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    trajectory = []

    for depth in range(max_depth):
        # 1. Stop descent if node is terminal
        is_term = tree["is_terminal"][batch_range, current_node]
        active_mask = active_mask & (~is_term)

        if not active_mask.any():
            break

        # 2. Select actions for active parents via selection_fn
        action = selection_fn(tree, current_node, depth)  # [B]

        # 3. Record trajectory step for active items
        trajectory.append((current_node.clone(), action.clone(), active_mask.clone()))

        # 4. Check if child is an unvisited leaf (-1)
        next_node = tree["children_index"][batch_range, current_node, action]
        is_leaf = next_node == -1

        # 5. Capture exact expansion targets for items hitting an unvisited leaf
        newly_found_leaf = active_mask & is_leaf
        if newly_found_leaf.any():
            leaf_parents = torch.where(newly_found_leaf, current_node, leaf_parents)
            leaf_actions = torch.where(newly_found_leaf, action, leaf_actions)
            expansion_mask = expansion_mask | newly_found_leaf

        # 6. Deactivate items reaching leaf
        active_mask = active_mask & (~is_leaf)

        # 7. Advance current_node for items still actively descending
        current_node = torch.where(active_mask, next_node, current_node)

    return leaf_parents, leaf_actions, expansion_mask, trajectory
