import pytest
import torch

from atomic_rl.search import init_mcts_tree

pytestmark = pytest.mark.unit


def test_init_mcts_tree_geometry():
    """Verify that tensor buffers are pre-allocated with correct dimensions and types."""
    batch_size = 2
    num_simulations = 4
    num_actions = 3
    embedding_dim = 8
    root_embeddings = torch.randn(batch_size, embedding_dim)
    root_logits = torch.randn(batch_size, num_actions)
    root_value = torch.randn(batch_size, 2)
    root_to_play = torch.zeros(batch_size, dtype=torch.long)

    tree = init_mcts_tree(
        root_embeddings, root_logits, root_value, root_to_play, num_simulations, num_actions
    )
    max_nodes = num_simulations + 1  # 5 slots

    assert tree["embeddings"].shape == (batch_size, max_nodes, embedding_dim)
    assert tree["children_index"].shape == (batch_size, max_nodes, num_actions)
    assert tree["is_terminal"].shape == (batch_size, max_nodes)
    assert torch.all(tree["children_index"] == -1)
    assert tree["node_counts"].tolist() == [1, 1]  # Only the root is occupied initially
    torch.testing.assert_close(tree["embeddings"][:, 0], root_embeddings)


def test_init_root_expanded_once():
    """Verify that the root node is marked with node_visits == 1 upon initialization."""
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

