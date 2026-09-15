"""Engine regression tests for specific MCTS engine bugs & invariants.

These pin down:
1. The first selection uses the prior-exploration term instead of a pure-noise tie-break.
2. max_depth is honoured and defaults to num_simulations (mctx behaviour).
3. A long non-terminal game expands one node per simulation.
"""
import pytest
import torch

from atomic_rl.search import (
    init_mcts_tree,
    mcts_search,
    puct_selection,
    select_leaf,
)

pytestmark = pytest.mark.regression


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
            _, leaf_actions, _, trajectory = select_leaf(tree, puct_selection)
            assert trajectory[0][1].item() == expected
            assert leaf_actions.item() == expected


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

    _, _, expansion_mask, trajectory = select_leaf(tree, puct_selection, max_depth=2)
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

    _, _, expansion_mask, trajectory = select_leaf(tree, puct_selection)  # default max_depth
    # Descended exactly to the tree horizon (num_simulations edges), and no
    # node was expanded beyond the allocated num_simulations + 1 slots.
    assert len(trajectory) == S
    assert not expansion_mask.any()


def test_search_linear_game_allocates_all_nodes():
    """A linear game with S simulations expands S distinct nodes."""
    B, A, S = 1, 2, 8

    def recurrent_fn(actions, embeddings, to_play):
        logits = embeddings.new_zeros(B, A)
        value = embeddings.new_zeros(B, 2)
        reward = embeddings.new_zeros(B, 2)
        discount = embeddings.new_ones(B, 2)
        next_embeddings = embeddings + torch.ones_like(embeddings)
        next_to_play = torch.zeros(B, dtype=torch.long)
        return logits, value, reward, discount, next_embeddings, next_to_play

    tree = mcts_search(
        root_embeddings=torch.zeros(B, 4),
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B, 2),
        root_to_play=torch.zeros(B, dtype=torch.long),
        recurrent_fn=recurrent_fn,
        num_simulations=S,
        num_actions=A,
        num_players=2,
        selection_fn=puct_selection,
        legal_mask=torch.ones(B, A, dtype=torch.bool),
    )
    assert tree["node_counts"].tolist() == [S + 1]
