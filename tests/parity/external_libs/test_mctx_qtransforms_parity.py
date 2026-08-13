import functools
import torch
import numpy as np
import pytest

from atomic_rl.search import init_mcts_tree
from atomic_rl.search.fpu import (
    fpu_zero,
    fpu_constant,
    fpu_parent_value,
    fpu_mixed_value,
    fpu_min_value,
)
from atomic_rl.search.qtransforms import (
    get_completed_qvalues,
    qtransform_identity,
    qtransform_by_min_max,
    qtransform_by_parent_and_siblings,
    qtransform_completed_by_mix_value,
)
from atomic_rl.search.mctx._src.tree import Tree as MctxTree
from atomic_rl.search.mctx._src.qtransforms import (
    qtransform_by_min_max as mctx_qtransform_by_min_max,
    qtransform_by_parent_and_siblings as mctx_qtransform_by_parent_and_siblings,
    qtransform_completed_by_mix_value as mctx_qtransform_completed_by_mix_value,
)


@pytest.fixture
def dummy_mcts_tree():
    """Constructs a deterministic 2-player batched MCTS tree for parity testing."""
    batch_size = 2
    num_simulations = 4
    num_actions = 4
    num_players = 2

    root_embeddings = torch.zeros((batch_size, 8), dtype=torch.float32)
    root_logits = torch.tensor(
        [[1.0, 2.0, -1.0, 0.5], [0.0, 1.5, 3.0, -2.0]], dtype=torch.float32
    )
    root_value = torch.tensor([[0.2, -0.2], [-0.5, 0.5]], dtype=torch.float32)
    root_to_play = torch.tensor([0, 1], dtype=torch.long)

    tree = init_mcts_tree(
        root_embeddings=root_embeddings,
        root_logits=root_logits,
        root_value=root_value,
        root_to_play=root_to_play,
        num_simulations=num_simulations,
        num_actions=num_actions,
        num_players=num_players,
    )

    # Populate root node (parent_node = 0) with mixed visit states
    # Batch item 0: actions 0, 1 visited; actions 2, 3 unvisited
    tree["children_visits"][0, 0] = torch.tensor([5, 3, 0, 0], dtype=torch.long)
    tree["children_rewards"][0, 0] = torch.tensor(
        [[0.0, 0.0], [0.5, -0.5], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32
    )
    tree["children_discounts"][0, 0] = torch.ones((4, 2), dtype=torch.float32)
    tree["children_values"][0, 0] = torch.tensor(
        [[0.8, -0.8], [-0.4, 0.4], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32
    )

    # Batch item 1: action 2 visited; actions 0, 1, 3 unvisited
    tree["children_visits"][1, 0] = torch.tensor([0, 0, 10, 0], dtype=torch.long)
    tree["children_rewards"][1, 0] = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [-1.0, 1.0], [0.0, 0.0]], dtype=torch.float32
    )
    tree["children_discounts"][1, 0] = torch.ones((4, 2), dtype=torch.float32)
    tree["children_values"][1, 0] = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [0.6, -0.6], [0.0, 0.0]], dtype=torch.float32
    )

    return tree


def make_mctx_tree_single(tree, batch_item_idx, node_index=0):
    """Converts a batch item of atomic_rl TensorDict tree into an unbatched mctx Tree."""
    import jax.numpy as jnp

    active_p = tree["to_play"][batch_item_idx, node_index].item()

    visits = tree["children_visits"][batch_item_idx].detach().cpu().numpy()
    rewards = tree["children_rewards"][batch_item_idx, :, :, active_p].detach().cpu().numpy()
    discounts = tree["children_discounts"][batch_item_idx, :, :, active_p].detach().cpu().numpy()
    values = tree["children_values"][batch_item_idx, :, :, active_p].detach().cpu().numpy()

    node_v = tree["node_values"][batch_item_idx, :, active_p].detach().cpu().numpy()
    raw_v = tree["raw_values"][batch_item_idx, :, active_p].detach().cpu().numpy()
    prior_logits = tree["children_prior_logits"][batch_item_idx].detach().cpu().numpy()

    return MctxTree(
        node_visits=jnp.array(tree["node_visits"][batch_item_idx].detach().cpu().numpy()),
        raw_values=jnp.array(raw_v),
        node_values=jnp.array(node_v),
        parents=jnp.array(tree["parents"][batch_item_idx].detach().cpu().numpy()),
        action_from_parent=jnp.array(
            tree["action_from_parent"][batch_item_idx].detach().cpu().numpy()
        ),
        children_index=jnp.array(
            tree["children_index"][batch_item_idx].detach().cpu().numpy()
        ),
        children_prior_logits=jnp.array(prior_logits),
        children_visits=jnp.array(visits),
        children_rewards=jnp.array(rewards),
        children_discounts=jnp.array(discounts),
        children_values=jnp.array(values),
        embeddings=jnp.array(tree["embeddings"][batch_item_idx].detach().cpu().numpy()),
        root_invalid_actions=jnp.zeros(tree.shape[-1], dtype=jnp.bool_),
        extra_data=None,
    )


def test_qtransform_by_min_max_mctx_parity(dummy_mcts_tree):
    """Verifies atomic_rl qtransform_by_min_max matches mctx when using fpu_min_value."""
    parent_nodes = torch.zeros(2, dtype=torch.long)
    min_value = -1.0
    max_value = 1.0

    atomic_res = qtransform_by_min_max(
        dummy_mcts_tree,
        parent_nodes,
        min_value=min_value,
        max_value=max_value,
        fpu_fn=functools.partial(fpu_min_value, min_value=min_value),
    )

    for b in range(2):
        mctx_tree = make_mctx_tree_single(dummy_mcts_tree, batch_item_idx=b)
        mctx_res = mctx_qtransform_by_min_max(
            mctx_tree, node_index=0, min_value=min_value, max_value=max_value
        )
        expected = torch.tensor(np.array(mctx_res), dtype=torch.float32)
        torch.testing.assert_close(atomic_res[b], expected, rtol=1e-4, atol=1e-4)


def test_qtransform_by_parent_and_siblings_mctx_parity(dummy_mcts_tree):
    """Verifies atomic_rl qtransform_by_parent_and_siblings matches mctx."""
    parent_nodes = torch.zeros(2, dtype=torch.long)

    # fpu_fn=None uses mctx dynamic min_value completion
    atomic_res = qtransform_by_parent_and_siblings(
        dummy_mcts_tree,
        parent_nodes,
    )

    for b in range(2):
        mctx_tree = make_mctx_tree_single(dummy_mcts_tree, batch_item_idx=b)
        mctx_res = mctx_qtransform_by_parent_and_siblings(mctx_tree, node_index=0)
        expected = torch.tensor(np.array(mctx_res), dtype=torch.float32)
        torch.testing.assert_close(atomic_res[b], expected, rtol=1e-4, atol=1e-4)


def test_qtransform_completed_by_mix_value_mctx_parity(dummy_mcts_tree):
    """Verifies atomic_rl qtransform_completed_by_mix_value matches mctx."""
    parent_nodes = torch.zeros(2, dtype=torch.long)

    atomic_res = qtransform_completed_by_mix_value(
        dummy_mcts_tree,
        parent_nodes,
        fpu_fn=fpu_mixed_value,
    )

    for b in range(2):
        mctx_tree = make_mctx_tree_single(dummy_mcts_tree, batch_item_idx=b)
        mctx_res = mctx_qtransform_completed_by_mix_value(mctx_tree, node_index=0)
        expected = torch.tensor(np.array(mctx_res), dtype=torch.float32)
        torch.testing.assert_close(atomic_res[b], expected, rtol=1e-4, atol=1e-4)


def test_unvisited_action_neutral_fpu(dummy_mcts_tree):
    """Verifies that fpu_zero gives unvisited actions 0.5 in [0, 1] normalized space for min_max."""
    parent_nodes = torch.zeros(2, dtype=torch.long)

    res = qtransform_by_min_max(
        dummy_mcts_tree,
        parent_nodes,
        min_value=-1.0,
        max_value=1.0,
        fpu_fn=fpu_zero,  # Unvisited actions get 0.0 raw -> (0.0 - (-1.0)) / 2.0 = 0.5 normalized
    )

    # Batch item 0: actions 2 and 3 are unvisited
    torch.testing.assert_close(res[0, 2], torch.tensor(0.5))
    torch.testing.assert_close(res[0, 3], torch.tensor(0.5))

    # Batch item 1: actions 0, 1, 3 are unvisited
    torch.testing.assert_close(res[1, 0], torch.tensor(0.5))
    torch.testing.assert_close(res[1, 1], torch.tensor(0.5))
    torch.testing.assert_close(res[1, 3], torch.tensor(0.5))
