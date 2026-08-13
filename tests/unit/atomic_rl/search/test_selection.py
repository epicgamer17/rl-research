import functools

import pytest
import torch

from atomic_rl.search import (
    init_mcts_tree,
    puct_score,
    qtransform_by_min_max,
    qtransform_identity,
    select_leaf,
)

pytestmark = pytest.mark.unit

# ==========================================
# Tests for Q-Value Normalization
# ==========================================


def test_qtrasforms_by_min_max_standard():
    """Verify standard min-max mapping to the [0, 1] interval."""
    batch_size = 2
    q_values = torch.tensor([[0.0, 5.0, 10.0], [2.0, 3.0, 4.0]])
    min_q = torch.tensor([0.0, 2.0])
    max_q = torch.tensor([10.0, 4.0])

    root_embeddings = torch.zeros(batch_size, 2)
    root_logits = torch.zeros(batch_size, 3)
    root_value = torch.zeros(batch_size, 2)
    root_to_play = torch.zeros(batch_size, dtype=torch.long)
    tree = init_mcts_tree(
        root_embeddings, root_logits, root_value, root_to_play, num_simulations=2, num_actions=3
    )
    tree["children_values"][:, 0, :, 0] = q_values
    tree["children_visits"][:, 0] = 1

    normalized = qtransform_by_min_max(
        tree,
        torch.zeros(batch_size, dtype=torch.long),
        min_value=min_q,
        max_value=max_q,
    )
    expected = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]])
    torch.testing.assert_close(normalized, expected)


def test_qtrasforms_by_min_max_division_by_zero():
    """Verify numerical safety adjustments when min_q equals max_q."""
    batch_size = 2
    q_values = torch.tensor([[5.0, 5.0], [0.0, 0.0]])
    min_q = torch.tensor([5.0, 0.0])
    max_q = torch.tensor([5.0, 0.0])  # Span is 0.0

    root_embeddings = torch.zeros(batch_size, 2)
    root_logits = torch.zeros(batch_size, 2)
    root_value = torch.zeros(batch_size, 2)
    root_to_play = torch.zeros(batch_size, dtype=torch.long)
    tree = init_mcts_tree(
        root_embeddings, root_logits, root_value, root_to_play, num_simulations=2, num_actions=2
    )
    tree["children_values"][:, 0, :, 0] = q_values
    tree["children_visits"][:, 0] = 1

    # Should safely treat span as epsilon to map (q_values - min_q) / epsilon -> 0.0
    normalized = qtransform_by_min_max(
        tree,
        torch.zeros(batch_size, dtype=torch.long),
        min_value=min_q,
        max_value=max_q,
    )
    torch.testing.assert_close(normalized, torch.zeros_like(q_values))


# ==========================================
# Tests for PUCT Scores
# ==========================================


def test_puct_score_mathematical_correctness():
    """
    Direct analytical check of the PUCT formula.
    Formula: Q_norm + c * P * sqrt(N_total) / (1 + N_action)
    where c = log((N_total + base + 1) / base) + init
    """
    # Inputs chosen for clean evaluation tracking
    q_values = torch.tensor([[2.0, 6.0]])
    priors = torch.tensor([[0.4, 0.6]])
    visits = torch.tensor([[1.0, 3.0]])
    total_visits = torch.tensor([[4.0]])

    root_embeddings = torch.zeros(1, 2)
    root_logits = torch.log(priors)  # Raw logits that softmax to [0.4, 0.6]
    root_value = torch.zeros(1, 2)
    root_to_play = torch.zeros(1, dtype=torch.long)
    tree = init_mcts_tree(
        root_embeddings, root_logits, root_value, root_to_play, num_simulations=4, num_actions=2
    )
    tree["children_values"][0, 0, :, 0] = q_values
    tree["children_visits"][0, 0] = visits.to(torch.long)
    tree["node_visits"][0, 0] = int(total_visits.item())

    # Static min/max to ensure predictable normalization output:
    #   Action 0: (2.0 - 0.0) / 8.0 = 0.25
    #   Action 1: (6.0 - 0.0) / 8.0 = 0.75
    qtransform = functools.partial(
        qtransform_by_min_max,
        min_value=torch.tensor([0.0]),
        max_value=torch.tensor([8.0]),
    )

    pb_c_base = 100.0
    pb_c_init = 2.0

    # Execute system function
    calculated_scores = puct_score(
        tree,
        torch.tensor([0]),
        depth=0,
        pb_c_base=pb_c_base,
        pb_c_init=pb_c_init,
        qtransform=qtransform,
    )

    # Manual step-by-step verification oracle
    expected_norm_q = torch.tensor([[0.25, 0.75]])

    # c = log((4 + 100 + 1) / 100) + 2.0 = log(1.05) + 2.0
    expected_c = torch.log(torch.tensor(1.05)) + 2.0

    # exploration term factor = c * sqrt(4) / (visit_counts + 1)
    exploration_factor = expected_c * torch.sqrt(total_visits) / (visits + 1)
    expected_scores = expected_norm_q + exploration_factor * priors

    torch.testing.assert_close(calculated_scores, expected_scores, atol=1e-6, rtol=1e-6)


def test_search_py_puct_scoring_math_verification():
    """
    Verify the Upper Confidence Bound calculation using the MuZero PUCT formula
    for the search_py backend.

    Setup:
    - Node visit count N(s) = 10
    - Action A: Q = 0.5, P = 0.3, N(s,a) = 5
    - Action B: Q = 0.2, P = 0.7, N(s,a) = 0

    Expected math (hand-calculated):
    - expected_score_a = 1.19773083
    - expected_score_b = 2.76823163
    """
    import math

    batch_size = 1
    num_actions = 2
    root_embeddings = torch.zeros(batch_size, 2)
    root_logits = torch.log(torch.tensor([[0.3, 0.7]]))
    root_value = torch.tensor([[0.2]])
    root_to_play = torch.zeros(batch_size, dtype=torch.long)

    tree = init_mcts_tree(
        root_embeddings, root_logits, root_value, root_to_play, num_simulations=10, num_actions=num_actions, num_players=1
    )

    tree["node_visits"][0, 0] = 10
    tree["children_visits"][0, 0] = torch.tensor([5, 0], dtype=torch.long)
    tree["children_rewards"][0, 0, :, 0] = torch.tensor([0.0, 0.0])
    tree["children_discounts"][0, 0, :, 0] = torch.tensor([1.0, 1.0])
    tree["children_values"][0, 0, :, 0] = torch.tensor([0.5, 0.2])

    qtransform = functools.partial(
        qtransform_by_min_max,
        min_value=torch.tensor([0.2]),
        max_value=torch.tensor([0.5]),
    )

    scores = puct_score(
        tree,
        torch.tensor([0]),
        depth=0,
        pb_c_init=1.25,
        pb_c_base=19652.0,
        qtransform=qtransform,
    )

    score_a = scores[0, 0].item()
    score_b = scores[0, 1].item()

    expected_score_a = 1.19773083
    expected_score_b = 2.76823163

    assert math.isclose(score_a, expected_score_a, rel_tol=1e-5), f"Action A score {score_a} != {expected_score_a}"
    assert math.isclose(score_b, expected_score_b, rel_tol=1e-5), f"Action B score {score_b} != {expected_score_b}"
    assert score_b > score_a


def test_puct_score_zero_prior_guard():
    """Verify that masked illegal root actions receive an extreme penalty and legal ones stay positive."""
    root_embeddings = torch.zeros(1, 2)
    root_logits = torch.tensor([[10.0, 5.0]])
    root_value = torch.zeros(1, 2)
    root_to_play = torch.zeros(1, dtype=torch.long)
    legal_mask = torch.tensor([[False, True]])  # Action 0 masked as illegal

    tree = init_mcts_tree(
        root_embeddings,
        root_logits,
        root_value,
        root_to_play,
        num_simulations=4,
        num_actions=2,
        legal_mask=legal_mask,
    )
    tree["node_visits"][0, 0] = 1

    scores = puct_score(tree, torch.tensor([0]), depth=0)
    assert scores[0, 0].item() == torch.finfo(torch.float32).min
    assert scores[0, 1].item() > 0.0


# ==========================================
# Tests for Selection Mechanics
# ==========================================


def test_select_leaf_early_termination():
    """Verify that leaf selection halts individual batch components when they hit unexpanded slots."""
    batch_size = 2
    root_embeddings = torch.zeros(batch_size, 2)
    root_logits = torch.zeros(batch_size, 2)
    root_value = torch.zeros(batch_size, 2)
    root_to_play = torch.zeros(batch_size, dtype=torch.long)
    tree = init_mcts_tree(
        root_embeddings, root_logits, root_value, root_to_play, num_simulations=5, num_actions=2
    )

    # Setup logits to perfectly direct the deterministic argmax choice
    tree["children_prior_logits"][:, 0, :] = torch.tensor(
        [100.0, 0.0]
    )  # Forces Action 0 at root
    tree["node_visits"][:, 0] = 1  # Activate the exploration term for determinism

    # Environment 0 has a child already expanded at index 1
    tree["children_index"][0, 0, 0] = 1
    tree["children_prior_logits"][0, 1, :] = torch.tensor(
        [0.0, 100.0]
    )  # Forces Action 1 at node 1
    tree["node_visits"][0, 1] = 1

    # Environment 1 has no children expanded at all (remains -1 at root)

    leaf_parents, leaf_actions, expansion_mask, trajectory = select_leaf(
        tree, pb_c_base=100.0, pb_c_init=1.0, max_depth=5
    )

    # Env 0 should fall deep into slot 1. Env 1 should stop immediately at root (slot 0)
    torch.testing.assert_close(leaf_parents, torch.tensor([1, 0], dtype=torch.long))

    # Validate the trajectory steps
    # Step 1: Both tracking routes were active
    assert torch.equal(trajectory[0][2], torch.tensor([True, True]))
    # Step 2: Env 1 hit a leaf node on step 1, turning inactive for depth level 2
    assert torch.equal(trajectory[1][2], torch.tensor([True, False]))


# ==========================================
# Deterministic Selection Trajectory Routing
# ==========================================


def test_select_leaf_deterministic_path():
    """
    Forces select_leaf down a strict, multi-step structural path
    by overwhelming the exploration constants with hand-crafted logits.

    Path target: Node 0 (Root) -> Action 1 -> Node 2 -> Action 0 -> Leaf (-1)
    """
    batch_size = 1
    root_embeddings = torch.zeros(batch_size, 4)
    root_logits = torch.zeros(batch_size, 2)
    root_value = torch.zeros(batch_size, 2)
    root_to_play = torch.zeros(batch_size, dtype=torch.long)

    # Allocate standard tree structure
    tree = init_mcts_tree(
        root_embeddings, root_logits, root_value, root_to_play, num_simulations=4, num_actions=2
    )

    # Establish topology linking Node 0 to Node 2 via Action 1
    tree["children_index"][0, 0, 1] = 2

    # Configure Node 0 (Root): force selection of Action 1
    tree["node_visits"][0, 0] = 1
    tree["children_prior_logits"][0, 0, :] = torch.tensor([0.0, 100.0])
    tree["children_visits"][0, 0, :] = torch.tensor([0, 0])

    # Configure Node 2: force selection of Action 0
    # Node 2's children arrays default to -1, making whatever action is selected a leaf
    tree["node_visits"][0, 2] = 1
    tree["children_prior_logits"][0, 2, :] = torch.tensor([100.0, 0.0])
    tree["children_visits"][0, 2, :] = torch.tensor([0, 0])

    # Execute selection pass
    leaf_parents, leaf_actions, expansion_mask, trajectory = select_leaf(
        tree, pb_c_base=19652, pb_c_init=1.25, max_depth=5
    )

    # 1. The search loop should break and target Node 2 as the expansion candidate
    assert leaf_parents.item() == 2

    # 2. Verify chronological sequence preservation inside trajectory tracking
    assert len(trajectory) == 2

    # Step 1 checking: At Node 0, Action 1 chosen
    node_step_1, action_step_1, mask_step_1 = trajectory[0]
    assert node_step_1.item() == 0
    assert action_step_1.item() == 1
    assert mask_step_1.item() is True

    # Step 2 checking: At Node 2, Action 0 chosen
    node_step_2, action_step_2, mask_step_2 = trajectory[1]
    assert node_step_2.item() == 2
    assert action_step_2.item() == 0
    assert mask_step_2.item() is True


def test_puct_score_internal_node_masking_with_qtransform_identity():
    """Verify that internal nodes (depth > 0) properly mask illegal actions when using qtransform_identity."""
    batch_size = 1
    num_actions = 3
    root_embeddings = torch.zeros(batch_size, 2)
    root_logits = torch.tensor([[0.0, 0.0, 0.0]])
    root_value = torch.zeros(batch_size, 2)
    root_to_play = torch.zeros(batch_size, dtype=torch.long)

    tree = init_mcts_tree(
        root_embeddings,
        root_logits,
        root_value,
        root_to_play,
        num_simulations=4,
        num_actions=num_actions,
    )

    # Set up node 1 (an internal node, depth = 1)
    tree["node_visits"][0, 1] = 5
    # Action 0 is legal, Q = -0.5
    # Action 1 is illegal, masked with -1e8
    # Action 2 is legal, Q = -0.2
    tree["children_prior_logits"][0, 1] = torch.tensor([0.0, -1e8, 0.0])
    tree["children_values"][0, 1, :, 0] = torch.tensor([-0.5, 0.0, -0.2])
    tree["children_visits"][0, 1] = torch.tensor([2, 0, 3], dtype=torch.long)

    # Evaluate PUCT score at depth 1 with qtransform_identity
    scores = puct_score(
        tree,
        torch.tensor([1]),
        depth=1,
        qtransform=qtransform_identity,
    )

    # Action 1 must be masked out to min_val despite having unvisited raw Q of 0.0
    min_val = torch.finfo(torch.float32).min
    assert scores[0, 1].item() == min_val
    # Legal actions 0 and 2 must have valid finite scores
    assert scores[0, 0].item() > min_val
    assert scores[0, 2].item() > min_val
    # Action 2 should have a higher score than Action 0 because Q(-0.2) > Q(-0.5)
    assert scores[0, 2].item() > scores[0, 0].item()