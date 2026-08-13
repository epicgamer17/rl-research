"""Regression tests for specific MCTS engine bugs.

These pin down:

1. backpropagate_ stores the *child node* value in ``children_values`` so that
   ``Q(s, a) = reward + discount * V(child)`` is reconstructed exactly, without
   double-counting the edge reward/discount.  (Regression for the nonzero
   intermediate-reward case; the mctx edge-value invariant is also checked by
   ``test_two_player_mcts``.)

2. The root node is expanded once at init (``node_visits == 1``), so the first
   selection uses the prior-exploration term instead of a pure-noise tie-break.

3. ``max_depth`` is honoured and defaults to ``num_simulations`` (mctx
   behaviour) rather than a hardcoded constant that silently stops expanding
   long games.
"""
import pytest
import torch

from atomic_rl.search import (
    backpropagate_,
    init_mcts_tree,
    mcts_search,
    select_leaf,
)
from atomic_rl.search.qtransforms import get_qvalues

pytestmark = pytest.mark.unit


# ============================================================================
# Bug 1: children_values must store the child node value, not the parent return
# ============================================================================
def test_backprop_no_double_count_with_intermediate_reward():
    """A nonzero reward on a non-terminal edge must not be applied twice.

    Q(s, a) = reward + discount * V(child). Storing the discounted return at
    the parent instead of the child value re-applies the edge and inflates Q.
    """
    B, A, S = 1, 2, 6
    tree = init_mcts_tree(
        root_embeddings=torch.zeros(B, 2),
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B, 1),
        root_to_play=torch.zeros(B, dtype=torch.long),
        num_simulations=S,
        num_actions=A,
        num_players=1,
    )
    # root --a0--> node1 (reward 0.5, discount 1.0); node1 value 2.0
    tree["children_index"][0, 0, 0] = 1
    tree["children_rewards"][0, 0, 0] = torch.tensor([[0.5]])
    tree["children_discounts"][0, 0, 0] = torch.tensor([[1.0]])
    tree["node_values"][0, 1] = torch.tensor([[2.0]])
    tree["node_visits"][0, 1] = 1
    # root --a1--> node2 (reward 0.0, discount 1.0); node2 value 3.0
    tree["children_index"][0, 0, 1] = 2
    tree["children_rewards"][0, 0, 1] = torch.tensor([[0.0]])
    tree["children_discounts"][0, 0, 1] = torch.tensor([[1.0]])
    tree["node_values"][0, 2] = torch.tensor([[3.0]])
    tree["node_visits"][0, 2] = 1

    backpropagate_(
        tree,
        [(torch.tensor([0]), torch.tensor([0]), torch.tensor([True]))],
        torch.tensor([[2.0]]),
    )
    backpropagate_(
        tree,
        [(torch.tensor([0]), torch.tensor([1]), torch.tensor([True]))],
        torch.tensor([[3.0]]),
    )

    # Edge stores the child's value, not the discounted return (2.5 / 3.0).
    torch.testing.assert_close(tree["children_values"][0, 0, 0], torch.tensor([2.0]))
    torch.testing.assert_close(tree["children_values"][0, 0, 1], torch.tensor([3.0]))

    q = get_qvalues(tree, torch.tensor([0]), torch.tensor([0]))
    assert q[0, 0].item() == pytest.approx(0.5 + 1.0 * 2.0)
    assert q[0, 1].item() == pytest.approx(0.0 + 1.0 * 3.0)
    # Correct ranking is preserved: a1 > a0. The double-count bug inflates
    # a0 to 3.0, tying (and distorting) the ranking.
    assert q[0, 1].item() > q[0, 0].item()


# ============================================================================
# Bug 2: the root is expanded once at init, so the first pick uses the prior
# ============================================================================
def test_init_root_expanded_once():
    """The root must be visited once at init (mctx update_tree_node)."""
    tree = init_mcts_tree(
        root_embeddings=torch.zeros(1, 2),
        root_logits=torch.zeros(1, 3),
        root_value=torch.zeros(1, 2),
        root_to_play=torch.zeros(1, dtype=torch.long),
        num_simulations=4,
        num_actions=3,
        num_players=2,
    )
    assert tree["node_visits"][:, 0].tolist() == [1]


@pytest.mark.parametrize("seeded", [range(5)])
def test_first_root_selection_prefers_argmax_prior(seeded):
    """With a peaked prior, the very first root selection is the argmax prior.

    Regression: when the root visit count starts at 0, sqrt(0) = 0 zeroes the
    exploration term and the first selection degenerates to a uniform-noise
    argmax instead of argmax prior. The prior must dominate the 1e-7 noise.
    """
    B, A = 1, 3
    # Peaked at different actions so a random (noise-only) selection cannot
    # consistently match: action 0, action 1, action 2.
    priors = [
        torch.tensor([[0.95, 0.03, 0.02]]),
        torch.tensor([[0.02, 0.95, 0.03]]),
        torch.tensor([[0.03, 0.02, 0.95]]),
    ]
    for expected, prior in enumerate(priors):
        for _ in seeded:
            tree = init_mcts_tree(
                root_embeddings=torch.zeros(B, 2),
                root_logits=torch.log(prior),
                root_value=torch.zeros(B, 2),
                root_to_play=torch.zeros(B, dtype=torch.long),
                num_simulations=8,
                num_actions=A,
                num_players=2,
            )
            _, leaf_actions, _, trajectory = select_leaf(tree)
            assert trajectory[0][1].item() == expected
            assert leaf_actions.item() == expected


# ============================================================================
# Bug 3: max_depth is honoured and defaults to num_simulations
# ============================================================================
def test_select_leaf_honors_max_depth():
    """A fully visited chain longer than max_depth must stop, not expand."""
    B, A, S = 1, 2, 8
    tree = init_mcts_tree(
        root_embeddings=torch.zeros(B, 2),
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B, 2),
        root_to_play=torch.zeros(B, dtype=torch.long),
        num_simulations=S,
        num_actions=A,
        num_players=2,
    )
    # Chain: 0 --a0--> 1 --a0--> 2 --a0--> 3, every edge visited.
    tree["node_visits"][0, 0] = 10
    for parent, child in ((0, 1), (1, 2), (2, 3)):
        tree["children_index"][0, parent, 0] = child
        tree["children_visits"][0, parent, 0] = 1
        tree["children_values"][0, parent, 0, 0] = 0.1

    _, _, expansion_mask, trajectory = select_leaf(tree, max_depth=2)
    # Two edges descended, then stopped at the cutoff with no expansion.
    assert len(trajectory) == 2
    assert not expansion_mask.any()


def test_select_leaf_default_max_depth_scales_with_num_simulations():
    """The default max_depth must track num_simulations, not a fixed constant.

    The tree holds num_simulations + 1 nodes, so a fully-expanded chain of
    num_simulations edges is exactly at the search horizon. With the old fixed
    512 default, a shorter horizon still let the descent run to the constant
    (trajectory length 512); the num_simulations default stops exactly at the
    tree's own horizon (trajectory length == num_simulations).
    """
    B, A, S = 1, 2, 520  # > 512 so the fixed default is distinguishable
    tree = init_mcts_tree(
        root_embeddings=torch.zeros(B, 2),
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B, 2),
        root_to_play=torch.zeros(B, dtype=torch.long),
        num_simulations=S,
        num_actions=A,
        num_players=2,
    )
    # Pre-build a fully expanded chain 0 -> 1 -> ... -> S (num_simulations edges).
    for parent, child in ((p, p + 1) for p in range(S)):
        tree["children_index"][0, parent, 0] = child
        tree["children_visits"][0, parent, 0] = 1
        tree["children_values"][0, parent, 0, 0] = 0.1
    tree["node_visits"][0, 0] = 10

    _, _, expansion_mask, trajectory = select_leaf(tree)  # default max_depth
    # Descended exactly to the tree horizon (num_simulations edges), and no
    # node was expanded beyond the allocated num_simulations + 1 slots.
    assert len(trajectory) == S
    assert not expansion_mask.any()


def test_search_linear_game_allocates_all_nodes():
    """A long non-terminal game must expand one node per simulation."""
    B, A, S = 1, 1, 64

    def recurrent_fn(actions, embeddings, to_play=None):
        logits = embeddings.new_zeros(B, A)
        value = embeddings.new_zeros(B, 2)
        reward = embeddings.new_zeros(B, 2)
        discount = embeddings.new_ones(B, 2)
        next_embeddings = embeddings + torch.ones_like(embeddings)
        next_to_play = torch.zeros(B, dtype=torch.long)
        return logits, value, reward, discount, next_embeddings, next_to_play

    _, tree = mcts_search(
        root_embeddings=torch.zeros(B, 4),
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B, 2),
        root_to_play=torch.zeros(B, dtype=torch.long),
        recurrent_fn=recurrent_fn,
        num_simulations=S,
        num_actions=A,
        num_players=2,
        legal_mask=torch.ones(B, A, dtype=torch.bool),
        dirichlet_epsilon=0.0,
    )
    assert tree["node_counts"].tolist() == [S + 1]
