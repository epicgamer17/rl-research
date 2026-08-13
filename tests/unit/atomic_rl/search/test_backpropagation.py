import pytest
import torch

from atomic_rl.search import backpropagate_, init_mcts_tree
from atomic_rl.search.qtransforms import get_qvalues

pytestmark = pytest.mark.unit


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
        root_embeddings,
        root_logits,
        root_value,
        root_to_play,
        num_simulations=5,
        num_actions=2,
    )

    # Construct a simple sequential path: Node 0 -> Node 1
    tree["children_index"][0, 0, 0] = 1
    tree["children_rewards"][0, 0, 0] = torch.tensor(
        [0.0, 0.0]
    )  # Reward obtained along the edge
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
    torch.testing.assert_close(
        tree["children_values"][0, 0, 0], torch.tensor([-0.5, 0.5])
    )
    # Q(root, a) = reward + discount * V(child) = 0.0 + 1.0 * (-0.5) = -0.5
    qvalues = get_qvalues(tree, torch.tensor([0]), torch.tensor([0]))
    assert qvalues[0, 0].item() == pytest.approx(-0.5)
    # Parent running value G = -0.5; the root was already expanded once at init
    # (mctx update_tree_node), so its mean includes the raw value 0.0 sample.
    assert tree["node_values"][0, 0, 0].item() == pytest.approx(-0.25)
    assert tree["node_visits"][0, 0].item() == 2
    # Backprop must not re-count the freshly expanded leaf node
    assert tree["node_visits"][0, 1].item() == 1


# Analytical Oracles: (path_config, expected, num_players, discount, description)
BACKPROP_CASES = [
    (
        [(1, 0.0), (1, 1.0), (0, 0.0, 0.0)],
        [-1.0, 1.0, 0.0, 0.0],
        2,
        1.0,
        "2P: two p1s, reward for p1 on p0 turn, end on p0",
    ),
    (
        [(1, 0.0), (1, 1.0), (1, 0.0, 0.0)],
        [-1.0, 1.0, 0.0, 0.0],
        2,
        1.0,
        "2P: two p1s, reward for p1 on p0 turn, end on p1",
    ),
    (
        [(1, 0.0), (1, 1.0), (0, 1.0, 0.0)],
        [-2.0, 2.0, 1.0, 0.0],
        2,
        1.0,
        "2P: two p1s, both actions get reward, end on p0",
    ),
    (
        [(1, 0.0), (1, 1.0), (1, 1.0, 0.0)],
        [-2.0, 2.0, 1.0, 0.0],
        2,
        1.0,
        "2P: two p1s, both actions get reward, end on p1",
    ),
    (
        [(1, 1.0), (1, 1.0), (0, 0.0, 0.0)],
        [0.0, 1.0, 0.0, 0.0],
        2,
        1.0,
        "2P: Two p1 turns (but p0 got reward), end on p0",
    ),
    (
        [(1, 1.0), (1, 1.0), (1, 0.0, 0.0)],
        [0.0, 1.0, 0.0, 0.0],
        2,
        1.0,
        "2P: Two p1 turns (but p0 got reward), end on p1",
    ),
    (
        [(1, 0.0), (0, 1.0), (1, 0.0, 0.0)],
        [-1.0, 1.0, 0.0, 0.0],
        2,
        1.0,
        "2P: alt game, p1 wins on first move",
    ),
    (
        [(1, 0.0), (0, 1.0), (1, 0.0), (0, 0.0, 0.0)],
        [-1.0, 1.0, 0.0, 0.0, 0.0],
        2,
        1.0,
        "2P: alt game, p1 wins on first move (extra leaf)",
    ),
    (
        [(1, 0.0), (0, 0.0), (1, 1.0, 0.0)],
        [1.0, -1.0, 1.0, 0.0],
        2,
        1.0,
        "2P: alt game, p0 wins",
    ),
    (
        [(1, 0.0), (0, 0.0), (1, 0.0), (0, 1.0, 0.0)],
        [-1.0, 1.0, -1.0, 1.0, 0.0],
        2,
        1.0,
        "2P: alt game, p1 wins",
    ),
    (
        [(1, 0.0), (0, 0.0), (1, 0.0, 1.0)],
        [-1.0, 1.0, -1.0, 1.0],
        2,
        1.0,
        "2P: alt game with leaf value",
    ),
    (
        [(1, 0.0), (0, 0.0), (1, 0.0), (0, 0.0, 1.0)],
        [1.0, -1.0, 1.0, -1.0, 1.0],
        2,
        1.0,
        "2P: alt game with leaf value",
    ),
    (
        [(0, 1.0), (0, 1.0), (0, 1.0, 0.0)],
        [3.0, 2.0, 1.0, 0.0],
        2,
        1.0,
        "2P: All p0 turns",
    ),
    (
        [(0, 0.0), (0, 0.0), (0, 0.0, 4.0)],
        [4.0, 4.0, 4.0, 4.0],
        2,
        1.0,
        "2P: All p0 turns with leaf value",
    ),
    (
        [(1, 0.0), (1, 1.0), (0, 0.0, 4.0)],
        [3.0, -3.0, -4.0, 4.0],
        2,
        1.0,
        "2P: Two p1 turns with leaf value",
    ),
    (
        [(1, 0.0), (1, 1.0), (1, 0.0, 4.0)],
        [-5.0, 5.0, 4.0, 4.0],
        2,
        1.0,
        "2P: Two p1 turns with leaf value",
    ),
    (
        [(1, 0.0), (1, 1.0), (1, 0.0), (0, 0.0, 4.0)],
        [3.0, -3.0, -4.0, -4.0, 4.0],
        2,
        1.0,
        "2P: Two p1 turns with leaf value",
    ),
    (
        [(0, 1.0), (0, 2.0), (0, 3.0, 0.0)],
        [6.0, 5.0, 3.0, 0.0],
        1,
        1.0,
        "1P: All rewards sum up",
    ),
    (
        [(0, 1.0), (0, 0.0), (0, 0.0, 5.0)],
        [6.0, 5.0, 5.0, 5.0],
        1,
        1.0,
        "1P: Rewards + leaf value",
    ),
    (
        [(1, 0.0), (2, 0.0), (0, 1.0), (0, 0.0, 0.0)],
        [-1.0, -1.0, 1.0, 0.0, 0.0],
        3,
        1.0,
        "3P: Player 2 wins",
    ),
    # Discounting Cases (gamma < 1.0)
    ([(0, 1.0), (0, 1.0, 1.0)], [2.71, 1.9, 1.0], 1, 0.9, "1P: Discounting gamma=0.9"),
    (
        [(1, 1.0), (0, 0.0, 1.0)],
        [1.81, -0.9, 1.0],
        2,
        0.9,
        "2P: Discounting gamma=0.9, alt turns",
    ),
    (
        [(1, 0.0), (1, 1.0, 1.0)],
        [-1.71, 1.9, 1.0],
        2,
        0.9,
        "2P: Discounting gamma=0.9, non-alt turns P1->P1",
    ),
]


@pytest.mark.parametrize(
    "path_config, expected, num_players, discount, desc", BACKPROP_CASES
)
def test_muzero_multiplayer_backpropagation(
    path_config, expected, num_players, discount, desc
):
    """Verifies backpropagation against analytical oracles across 1P, 2P, 3P and discounted cases."""
    num_steps = len(path_config)
    tree = init_mcts_tree(
        root_embeddings=torch.zeros(1, 2),
        root_logits=torch.zeros(1, 2),
        root_value=torch.zeros(1, num_players),
        root_to_play=torch.tensor([0], dtype=torch.long),
        num_simulations=num_steps + 2,
        num_actions=2,
        num_players=num_players,
    )
    trajectory = []
    tree["node_visits"][0, :] = 0
    tree["node_values"][0, :] = 0.0

    for i, step in enumerate(path_config):
        to_play_next, reward = step[0], step[1]
        p_idx, c_idx = i, i + 1
        tree["children_index"][0, p_idx, 0] = c_idx
        tree["to_play"][0, c_idx] = to_play_next
        tree["parents"][0, c_idx] = p_idx
        tree["action_from_parent"][0, c_idx] = 0

        p_parent = tree["to_play"][0, p_idx].item()
        reward_vec = torch.zeros(num_players, dtype=torch.float32)
        if num_players == 1:
            reward_vec[0] = reward
        elif num_players == 2:
            reward_vec = (
                torch.tensor([reward, -reward])
                if p_parent == 0
                else torch.tensor([-reward, reward])
            )
        elif num_players == 3:
            for p in range(3):
                reward_vec[p] = reward if p == p_parent else -reward

        tree["children_rewards"][0, p_idx, 0] = reward_vec
        tree["children_discounts"][0, p_idx, 0] = torch.full((num_players,), discount)
        trajectory.append(
            (torch.tensor([p_idx]), torch.tensor([0]), torch.tensor([True]))
        )

    last_step = path_config[-1]
    leaf_val = last_step[2] if len(last_step) > 2 else 0.0
    leaf_to_play = last_step[0]
    leaf_node_idx = num_steps

    leaf_val_vec = torch.zeros(1, num_players, dtype=torch.float32)
    if num_players == 1:
        leaf_val_vec[0, 0] = leaf_val
    elif num_players == 2:
        leaf_val_vec[0] = (
            torch.tensor([leaf_val, -leaf_val])
            if leaf_to_play == 0
            else torch.tensor([-leaf_val, leaf_val])
        )
    elif num_players == 3:
        for p in range(3):
            leaf_val_vec[0, p] = leaf_val if p == leaf_to_play else -leaf_val

    tree["raw_values"][0, leaf_node_idx] = leaf_val_vec[0]
    tree["node_values"][0, leaf_node_idx] = leaf_val_vec[0]
    tree["node_visits"][0, leaf_node_idx] = 1

    backpropagate_(tree, trajectory, leaf_val_vec)

    resulting_values = [
        tree["node_values"][0, i, tree["to_play"][0, i].item()].item()
        for i in range(num_steps + 1)
    ]

    expected_tensor = torch.tensor(expected, dtype=torch.float32)
    resulting_tensor = torch.tensor(resulting_values, dtype=torch.float32)

    torch.testing.assert_close(
        resulting_tensor,
        expected_tensor,
        atol=1e-2,
        rtol=1e-2,
        msg=f"Backend Failed: {desc}",
    )
