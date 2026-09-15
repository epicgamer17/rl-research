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


def quasi_sampling_selection(
    tree: TensorDict,
    parent_nodes: torch.Tensor,
    depth: int,
    *,
    qtransform: Optional[Callable] = None,
    **kwargs,
) -> torch.Tensor:
    """Selects chance outcomes using quasi-sampling argmax (Stochastic MuZero Eq. 4 / Footnote 1).

    Formula:
        c* = argmax_c [ Pr(c | as) / (N(as, c) + 1) ]

    Args:
        tree: The MCTS tree TensorDict.
        parent_nodes: Parent node indices [B].
        depth: Current tree depth.
        qtransform: Unused in chance selection, accepted for signature compatibility.

    Returns:
        Selected chance outcome indices [B] from parent_nodes.
    """
    batch_size = parent_nodes.shape[0]
    batch_range = torch.arange(batch_size, device=parent_nodes.device)

    # 1. Fetch prior logits over chance codes and child visits N(c)
    prior_logits = tree["children_prior_logits"][
        batch_range, parent_nodes
    ]  # [B, num_outcomes]
    children_visits = tree["children_visits"][
        batch_range, parent_nodes
    ]  # [B, num_outcomes]

    # 2. Convert logits to probabilities Pr(c | as)
    prior_probs = F.softmax(prior_logits, dim=-1)

    # 3. Compute quasi-sampling score: Pr(c | as) / (N(c) + 1.0)
    scores = prior_probs / (children_visits.to(prior_probs.dtype) + 1.0)

    # 4. Tiny uniform noise for tie breaking (matches mctx)
    noise = 1e-7 * torch.rand_like(scores)
    scores = scores + noise

    # 5. Mask invalid or padded branches if present
    legal_mask = kwargs.get("legal_mask", None)
    if legal_mask is None:
        legal_mask = (prior_logits > -1e7) & torch.isfinite(prior_logits)

    min_val = torch.finfo(scores.dtype).min
    scores = torch.where(legal_mask, scores, min_val)

    return scores.argmax(dim=-1)


def make_stochastic_selection_fn(
    decision_selection_fn: Callable[[TensorDict, torch.Tensor, int], torch.Tensor],
    chance_selection_fn: Callable[[TensorDict, torch.Tensor, int], torch.Tensor],
) -> Callable[[TensorDict, torch.Tensor, int], torch.Tensor]:
    r"""Creates a composite selection function routing between decision and chance nodes.

    In Stochastic MuZero (Antonoglou et al., ICLR 2022), tree search alternates between
    decision nodes (choosing environment action $a \in \mathcal{A}$) and chance / afterstate
    nodes (choosing chance code $c \in \mathcal{C}$).

    Args:
        decision_selection_fn: Callable `(tree, parent_nodes, depth, **kwargs) -> actions [B]`
            used for decision nodes (e.g. `puct_selection` or a pre-configured partial).
        chance_selection_fn: Callable `(tree, parent_nodes, depth, **kwargs) -> chance_codes [B]`
            used for chance/afterstate nodes (e.g. `quasi_sampling_selection` or a pre-configured partial).

    Returns:
        A callable selection function `(tree, parent_nodes, depth, **kwargs) -> indices [B]`
        compatible with `select_leaf` and MCTS search algorithms.
    """
    assert callable(
        decision_selection_fn
    ), f"decision_selection_fn must be callable, got {decision_selection_fn!r}"
    assert callable(
        chance_selection_fn
    ), f"chance_selection_fn must be callable, got {chance_selection_fn!r}"

    def stochastic_selection_fn(
        tree: TensorDict,
        parent_nodes: torch.Tensor,
        depth: int,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = parent_nodes.shape[0]
        device = parent_nodes.device
        batch_range = torch.arange(batch_size, device=device)

        is_chance = tree["is_chance"][batch_range, parent_nodes]
        assert is_chance.shape == (batch_size,), (
            f"is_chance mask shape mismatch: expected [{batch_size}], got {tuple(is_chance.shape)}"
        )
        assert is_chance.dtype == torch.bool, (
            f"is_chance mask must be boolean, got dtype {is_chance.dtype}"
        )

        # Fast path 1: All nodes at this step are decision nodes (common at even depths)
        if not is_chance.any():
            return decision_selection_fn(tree, parent_nodes, depth, **kwargs)

        # Fast path 2: All nodes at this step are chance nodes (common at odd depths)
        if is_chance.all():
            return chance_selection_fn(tree, parent_nodes, depth, **kwargs)

        # Heterogeneous / mixed batch: evaluate both and route per-item
        decision_actions = decision_selection_fn(tree, parent_nodes, depth, **kwargs)
        chance_actions = chance_selection_fn(tree, parent_nodes, depth, **kwargs)
        return torch.where(is_chance, chance_actions, decision_actions)

    return stochastic_selection_fn


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
