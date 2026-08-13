import pytest
import torch

from atomic_rl.search import backpropagate_, init_mcts_tree
from atomic_rl.search.qtransforms import get_qvalues

pytestmark = pytest.mark.unit


def test_backpropagate_single_player():
    pass


def test_backpropagate_alternating_players():
    """Verify the backward pass stores child node values on edges (mctx semantics).

    Q(root, a) = reward + discount * children_values[root, a] is reconstructed by
    get_qvalues against a hand-computed single-edge backup.
    """
    # Setup a 1-batch tree manually to isolate the math of the backward pass
    root_embeddings = torch.zeros(1, 2)
    root_logits = torch.zeros(1, 2)
    root_value = torch.zeros(1, 2)
    root_to_play = torch.zeros(1, dtype=torch.long)
    tree = init_mcts_tree(
        root_embeddings, root_logits, root_value, root_to_play, num_simulations=5, num_actions=2
    )

    # Construct a simple sequential path: Node 0 -> Node 1
    tree["children_index"][0, 0, 0] = 1
    tree["children_rewards"][0, 0, 0] = torch.tensor([0.0, 0.0])  # Reward obtained along the edge
    tree["children_discounts"][0, 0, 0] = torch.tensor([1.0, 1.0])
    # In the real search expand_node_ sets node 1's value from the model output.
    tree["node_values"][0, 1] = torch.tensor([-0.5, 0.5])
    tree["node_visits"][0, 1] = 1  # Set by expand_node_

    # Trajectory format: [(node_idx, action_idx, active_mask)]
    trajectory = [(torch.tensor([0]), torch.tensor([0]), torch.tensor([True]))]
    leaf_value = torch.tensor(
        [[-0.5, 0.5]]
    )  # Value evaluation out of Node 1 from Player 1's perspective

    backpropagate_(tree, trajectory, leaf_value)

    # Math:
    assert tree["children_visits"][0, 0, 0].item() == 1
    # The edge stores the CHILD node's value (mctx), not the discounted return.
    torch.testing.assert_close(tree["children_values"][0, 0, 0], torch.tensor([-0.5, 0.5]))
    # Q(root, a) = reward + discount * V(child) = 0.0 + 1.0 * (-0.5) = -0.5
    qvalues = get_qvalues(tree, torch.tensor([0]), torch.tensor([0]))
    assert qvalues[0, 0].item() == pytest.approx(-0.5)
    # Parent running value G = -0.5; the root was already expanded once at init
    # (mctx update_tree_node), so its mean includes the raw value 0.0 sample.
    assert tree["node_values"][0, 0, 0].item() == pytest.approx(-0.25)
    assert tree["node_visits"][0, 0].item() == 2
    # Backprop must not re-count the freshly expanded leaf node
    assert tree["node_visits"][0, 1].item() == 1