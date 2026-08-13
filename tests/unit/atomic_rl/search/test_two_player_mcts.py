"""Component-level checks for a 2-player alternating zero-sum MCTS.

The convention used throughout (and by the mp AlphaZero example) is:

  * The player to move is NOT stored in the tree; it is carried in the state
    `embeddings` (the environment supplies it).
  * Alternation + zero-sum is encoded in the edge `discount` on each transition:
      - ``discount = 1`` for a non-terminal transition,
      - ``discount = 0`` for a terminal transition (plateaus at reward).
  * ``children_values[parent, a]`` stores the value of the *child* node, so that
    ``Q(s, a) = reward + discount * children_values`` (mctx semantics).

These tests are written to isolate which component misbehaves for a 2-player game
rather than to reproduce any particular failure.
"""
import torch
import pytest

from atomic_rl.search import (
    backpropagate_,
    expand_node_,
    get_mcts_visit_policy,
    init_mcts_tree,
    mcts_search,
    puct_score,
    qtransform_by_parent_and_siblings,
    select_leaf,
)
from atomic_rl.search.qtransforms import get_qvalues
from atomic_rl.envs.functions.tictactoe import tictactoe_dynamics_fn

pytestmark = pytest.mark.unit


# ============================================================================
# Helpers
# ============================================================================
def _check_edge_child_value_invariant(tree):
    """mctx invariant: every edge must store its child node's value."""
    children_index = tree["children_index"]
    children_values = tree["children_values"]
    node_values = tree["node_values"]
    edge_exists = children_index >= 0
    assert edge_exists.any(), "tree has no expanded edges to check"
    b_idx, n_idx, a_idx = edge_exists.nonzero(as_tuple=True)
    child_idx = children_index[b_idx, n_idx, a_idx]
    child_values = node_values[b_idx, child_idx]
    edge_values = children_values[b_idx, n_idx, a_idx]
    assert torch.allclose(child_values, edge_values, atol=1e-5), (
        "children_values[parent, a] != node_values[child] for some edge. "
        "The backward pass is not storing child node values (mctx semantics)."
    )


# ============================================================================
# 1. init_mcts_tree
# ============================================================================
def test_init_tree_root_fields():
    """The root node must be populated with priors/value and neutral edges."""
    B, A, S = 2, 4, 10
    root_logits = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    tree = init_mcts_tree(
        root_embeddings=torch.zeros(B, 4),
        root_logits=root_logits,
        root_value=torch.tensor([[0.4, -0.4], [-0.6, 0.6]]),
        root_to_play=torch.zeros(B, dtype=torch.long),
        num_simulations=S,
        num_actions=A,
        num_players=2,
    )

    assert tree["node_visits"].shape == (B, S + 1)
    assert tree["children_index"].shape == (B, S + 1, A)
    assert tree["node_counts"].tolist() == [1, 1]  # root allocated only
    assert torch.equal(
        tree["children_index"][:, 1:], torch.full((B, S, A), -1)
    )  # only root occupies slot 0

    # Root priors stored exactly (logits; softmax happens at selection time)
    torch.testing.assert_close(tree["children_prior_logits"][:, 0], root_logits)
    # Root value/visits
    torch.testing.assert_close(tree["node_values"][:, 0], torch.tensor([[0.4, -0.4], [-0.6, 0.6]]))
    torch.testing.assert_close(tree["raw_values"][:, 0], torch.tensor([[0.4, -0.4], [-0.6, 0.6]]))
    assert tree["node_visits"][:, 0].tolist() == [1, 1]  # root expanded once at init (mctx)
    # Neutral edge stats: discounts default to 1.0, everything else 0/-1
    assert torch.all(tree["children_discounts"][:, 0] == 1.0)
    assert torch.all(tree["children_visits"][:, 0] == 0)
    assert torch.all(tree["children_values"][:, 0] == 0.0)


# ============================================================================
# 2. expand_node_
# ============================================================================
def test_expand_node_single_transition_two_player():
    """A single expansion must link parent->child and store model outputs."""
    B, A, S = 1, 3, 6
    tree = init_mcts_tree(
        root_embeddings=torch.zeros(B, 2),
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B, 2),
        root_to_play=torch.zeros(B, dtype=torch.long),
        num_simulations=S,
        num_actions=A,
        num_players=2,
    )
    next_emb = torch.tensor([[1.0, -1.0]])

    expand_node_(
        tree=tree,
        parent_nodes=torch.tensor([0]),
        actions_taken=torch.tensor([1]),
        policy_logits=torch.tensor([[2.0, 3.0, 0.0]]),
        value=torch.tensor([[-0.7, 0.7]]),
        rewards=torch.tensor([[0.5, -0.5]]),
        discounts=torch.tensor([[1.0, 1.0]]),
        next_embeddings=next_emb,
        players=torch.tensor([1]),
    )

    # Structural edges are recorded both directions
    assert tree["children_index"][0, 0, 1].item() == 1
    assert tree["parents"][0, 1].item() == 0
    assert tree["action_from_parent"][0, 1].item() == 1
    # Edge statistics
    assert tree["children_visits"][0, 0, 1].item() == 0
    assert torch.all(tree["children_values"][0, 0, 1] == 0.0)
    torch.testing.assert_close(tree["children_rewards"][0, 0, 1], torch.tensor([0.5, -0.5]))
    torch.testing.assert_close(tree["children_discounts"][0, 0, 1], torch.tensor([1.0, 1.0]))
    # New node stats (visit starts at 1, like mctx update_tree_node)
    assert tree["node_visits"][0, 1].item() == 1
    torch.testing.assert_close(tree["node_values"][0, 1], torch.tensor([-0.7, 0.7]))
    assert tree["is_terminal"][0, 1].item() is False
    torch.testing.assert_close(tree["embeddings"][0, 1], next_emb[0])
    torch.testing.assert_close(
        tree["children_prior_logits"][0, 1], torch.tensor([2.0, 3.0, 0.0])
    )
    assert tree["node_counts"][0].item() == 2


def test_expand_node_terminal_sets_flag_and_zeroes_prior_masking():
    """discount==0 marks the node terminal; legal_mask zeroes illegal priors."""
    B, A, S = 1, 3, 6
    tree = init_mcts_tree(
        root_embeddings=torch.zeros(B, 2),
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B, 2),
        root_to_play=torch.zeros(B, dtype=torch.long),
        num_simulations=S,
        num_actions=A,
        num_players=2,
    )

    expand_node_(
        tree=tree,
        parent_nodes=torch.tensor([0]),
        actions_taken=torch.tensor([0]),
        policy_logits=torch.tensor([[2.0, 3.0, 1.0]]),
        value=torch.tensor([[0.0, 0.0]]),
        rewards=torch.tensor([[-1.0, 1.0]]),  # player to move lost
        discounts=torch.tensor([[0.0, 0.0]]),  # terminal
        next_embeddings=torch.zeros(1, 2),
        players=torch.tensor([1]),
        legal_mask=torch.tensor([[True, True, False]]),
    )

    assert tree["is_terminal"][0, 1].item() is True
    torch.testing.assert_close(tree["children_rewards"][0, 0, 0], torch.tensor([-1.0, 1.0]))
    torch.testing.assert_close(tree["children_discounts"][0, 0, 0], torch.tensor([0.0, 0.0]))
    # Illegal action 2 prior is floored, legal ones untouched
    merged = tree["children_prior_logits"][0, 1]
    assert merged[0].item() == 2.0
    assert merged[1].item() == 3.0
    assert merged[2].item() == torch.finfo(torch.float32).min


# ============================================================================
# 3. backpropagate_
# ============================================================================
def test_backprop_alternating_chain_negamax():
    """Hand-computed 2-ply alternating backup with discount = 1 / 0."""
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

    # Manually build the pre-search tree (nodes were "expanded" earlier)
    tree["children_index"][0, 0, 0] = 1  # root -> node1 via a0
    tree["children_rewards"][0, 0, 0] = torch.tensor([0.0, 0.0])
    tree["children_discounts"][0, 0, 0] = torch.tensor([1.0, 1.0])

    tree["children_index"][0, 1, 1] = 2  # node1 -> node2 via a1
    tree["children_rewards"][0, 1, 1] = torch.tensor([1.0, -1.0])
    tree["children_discounts"][0, 1, 1] = torch.tensor([0.0, 0.0])  # terminal edge

    tree["node_visits"][0, 1] = 1
    tree["node_values"][0, 1] = torch.tensor([-0.8, 0.8])
    tree["node_visits"][0, 2] = 1
    tree["node_values"][0, 2] = torch.tensor([0.4, -0.4])

    trajectory = [
        (torch.tensor([0]), torch.tensor([0]), torch.tensor([True])),
        (torch.tensor([1]), torch.tensor([1]), torch.tensor([True])),
    ]
    backpropagate_(tree, trajectory, leaf_value=torch.tensor([[0.4, -0.4]]))

    # Edge values must equal the child node values (mctx semantics)
    torch.testing.assert_close(tree["children_values"][0, 1, 1], torch.tensor([0.4, -0.4]))
    # Node 1's visit accounting (root was expanded once at init, then backed up)
    assert tree["node_visits"][0, 0].item() == 2
    assert tree["node_visits"][0, 1].item() == 2
    assert tree["node_visits"][0, 2].item() == 1


def test_backprop_does_not_double_count_leaf():
    """A freshly expanded leaf must keep node_visits == 1 after one backup."""
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
    expand_node_(
        tree=tree,
        parent_nodes=torch.tensor([0]),
        actions_taken=torch.tensor([0]),
        policy_logits=torch.zeros(1, A),
        value=torch.tensor([[0.6, -0.6]]),
        rewards=torch.tensor([[0.0, 0.0]]),
        discounts=torch.tensor([[1.0, 1.0]]),
        next_embeddings=torch.zeros(1, 2),
        players=torch.tensor([1]),
    )
    trajectory = [(torch.tensor([0]), torch.tensor([0]), torch.tensor([True]))]
    backpropagate_(tree, trajectory, leaf_value=torch.tensor([[0.6, -0.6]]))

    assert tree["node_visits"][0, 1].item() == 1
    torch.testing.assert_close(tree["node_values"][0, 1], torch.tensor([0.6, -0.6]))
    # Root is initialized with a visit (mctx update_tree_node), then backed up once
    assert tree["node_visits"][0, 0].item() == 2


# ============================================================================
# 4. q-values / q-transform
# ============================================================================
def test_qvalues_alternating_sign_convention():
    """Q values derived from multi-player vector values."""
    B, A, S = 1, 2, 8
    tree = init_mcts_tree(
        root_embeddings=torch.zeros(B, 2),
        root_logits=torch.zeros(B, A),
        root_value=torch.tensor([[0.5, -0.5]]),
        root_to_play=torch.tensor([0]),
        num_simulations=S,
        num_actions=A,
        num_players=2,
    )
    tree["node_values"][0, 0] = torch.tensor([0.5, -0.5])
    tree["node_visits"][0, 0] = 10
    tree["children_index"][0, 0, 0] = 1
    tree["children_index"][0, 0, 1] = 2
    tree["node_values"][0, 1] = torch.tensor([-0.9, 0.9])
    tree["node_values"][0, 2] = torch.tensor([-0.2, 0.2])
    tree["children_values"][0, 0, 0] = torch.tensor([-0.9, 0.9])
    tree["children_values"][0, 0, 1] = torch.tensor([-0.2, 0.2])
    tree["children_discounts"][0, 0, 0] = torch.tensor([1.0, 1.0])
    tree["children_discounts"][0, 0, 1] = torch.tensor([1.0, 1.0])
    tree["children_visits"][0, 0, 0] = 4
    tree["children_visits"][0, 0, 1] = 4

    q = get_qvalues(tree, torch.tensor([0]), torch.tensor([0]))
    torch.testing.assert_close(q[0], torch.tensor([-0.9, -0.2]))


def test_qtransform_parent_siblings_alternating():
    """Normalized Q for multi-player scenarios."""
    B, A, S = 1, 3, 8
    tree = init_mcts_tree(
        root_embeddings=torch.zeros(B, 2),
        root_logits=torch.zeros(B, A),
        root_value=torch.tensor([[0.5, -0.5]]),
        root_to_play=torch.tensor([0]),
        num_simulations=S,
        num_actions=A,
        num_players=2,
    )
    tree["node_values"][0, 0] = torch.tensor([0.5, -0.5])
    tree["node_visits"][0, 0] = 10
    tree["children_index"][0, 0, 0] = 1
    tree["children_index"][0, 0, 1] = 2
    tree["node_values"][0, 1] = torch.tensor([-0.9, 0.9])
    tree["node_values"][0, 2] = torch.tensor([0.2, -0.2])
    tree["children_values"][0, 0, 0] = torch.tensor([-0.9, 0.9])
    tree["children_values"][0, 0, 1] = torch.tensor([0.2, -0.2])
    tree["children_discounts"][0, 0, 0] = torch.tensor([1.0, 1.0])
    tree["children_discounts"][0, 0, 1] = torch.tensor([1.0, 1.0])
    tree["children_visits"][0, 0, 0] = 4
    tree["children_visits"][0, 0, 1] = 4

    norm = qtransform_by_parent_and_siblings(tree, torch.tensor([0]))
    assert norm.shape == (B, A)


# ============================================================================
# 5. puct_score / select_leaf
# ============================================================================
def test_selection_prefers_correct_flipped_q():
    """With equal priors/visits, root picks action that maximizes active player Q."""
    B, A, S = 1, 2, 8
    tree = init_mcts_tree(
        root_embeddings=torch.zeros(B, 2),
        root_logits=torch.tensor([[0.0, 0.0]]),
        root_value=torch.tensor([[0.5, -0.5]]),
        root_to_play=torch.tensor([0]),
        num_simulations=S,
        num_actions=A,
        num_players=2,
    )
    tree["node_values"][0, 0] = torch.tensor([0.5, -0.5])
    tree["node_visits"][0, 0] = 10
    tree["children_index"][0, 0, 0] = 1
    tree["children_index"][0, 0, 1] = 2
    tree["node_values"][0, 1] = torch.tensor([-0.9, 0.9])
    tree["node_values"][0, 2] = torch.tensor([0.2, -0.2])
    tree["children_values"][0, 0, 0] = torch.tensor([-0.9, 0.9])
    tree["children_values"][0, 0, 1] = torch.tensor([0.2, -0.2])
    tree["children_discounts"][0, 0, 0] = torch.tensor([1.0, 1.0])
    tree["children_discounts"][0, 0, 1] = torch.tensor([1.0, 1.0])
    tree["children_visits"][0, 0, 0] = 4
    tree["children_visits"][0, 0, 1] = 4

    leaf_parents, _, _, trajectory = select_leaf(
        tree, pb_c_base=19652.0, pb_c_init=1.25
    )

    assert trajectory[0][1].item() == 1
    assert leaf_parents.item() == 2


def test_puct_score_obeys_root_legal_mask():
    """Illegal root actions must be floored so argmax can never select them."""
    B, A, S = 1, 3, 8
    tree = init_mcts_tree(
        root_embeddings=torch.zeros(B, 2),
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B, 2),
        root_to_play=torch.tensor([0]),
        num_simulations=S,
        num_actions=A,
        num_players=2,
        legal_mask=torch.tensor([[True, False, True]]),
    )
    tree["node_visits"][0, 0] = 1

    scores = puct_score(tree, torch.tensor([0]), depth=0)
    assert scores[0, 1].item() == torch.finfo(torch.float32).min
    assert scores[0, 0].item() > -1e8
    assert scores[0, 2].item() > -1e8


def test_search_alternating_game_preserves_edge_invariant():
    """Full search where every transition alternates."""
    B, A, S = 2, 3, 16

    def recurrent_fn(actions, embeddings, to_play=None):
        next_emb = embeddings + torch.ones_like(embeddings) * 0.01
        logits = embeddings.new_zeros(embeddings.shape[0], A)
        value = embeddings.new_full((embeddings.shape[0], 2), 0.3)
        reward = embeddings.new_zeros(embeddings.shape[0], 2)
        discount = embeddings.new_ones(embeddings.shape[0], 2)
        next_to_play = torch.zeros(embeddings.shape[0], dtype=torch.long)
        return logits, value, reward, discount, next_emb, next_to_play

    action_probs, tree = mcts_search(
        root_embeddings=torch.zeros(B, 4),
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B, 2),
        root_to_play=torch.zeros(B, dtype=torch.long),
        recurrent_fn=recurrent_fn,
        num_simulations=S,
        num_actions=A,
        num_players=2,
        dirichlet_epsilon=0.0,
    )

    assert tree["node_counts"].tolist() == [S + 1, S + 1]
    assert tree["node_visits"][:, 0].tolist() == [S + 1, S + 1]
    assert tree["children_visits"][:, 0].sum(dim=-1).tolist() == [S, S]
    torch.testing.assert_close(action_probs.sum(dim=-1), torch.ones(B))
    assert torch.all(action_probs >= 0.0)
    _check_edge_child_value_invariant(tree)


def test_search_alternating_mp_style_tictactoe():
    """The mp example's exact pattern."""
    B, A, S = 1, 9, 12
    board_embed = torch.zeros(B, 3, 3, 2)
    legal_mask = (board_embed[0, ..., 0].view(-1) == 0).unsqueeze(0)
    min_logit = torch.finfo(torch.float32).min

    def recurrent_fn(actions_taken, embeddings, to_play):
        next_embed, reward_vec, next_to_play, is_terminal, next_legal_mask = tictactoe_dynamics_fn(
            embeddings, actions_taken, to_play
        )
        logits = embeddings.new_zeros(embeddings.shape[0], A)
        logits = logits.masked_fill(~next_legal_mask, min_logit)
        v_p0 = torch.zeros(embeddings.shape[0])
        value_vec = torch.stack([v_p0, -v_p0], dim=-1)
        discount_vec = torch.where(
            is_terminal.unsqueeze(-1), torch.zeros_like(reward_vec), torch.ones_like(reward_vec)
        )
        return logits, value_vec, reward_vec, discount_vec, next_embed, next_to_play

    action_probs, tree = mcts_search(
        root_embeddings=board_embed,
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B, 2),
        root_to_play=torch.zeros(B, dtype=torch.long),
        recurrent_fn=recurrent_fn,
        num_simulations=S,
        num_actions=A,
        num_players=2,
        legal_mask=legal_mask,
        dirichlet_epsilon=0.0,
    )

    assert not torch.isnan(tree["node_values"]).any()
    assert not torch.isnan(tree["children_values"]).any()
    assert tree["node_visits"][:, 0].sum(dim=-1).item() == S + 1
    torch.testing.assert_close(action_probs.sum(dim=-1), torch.ones(B))
    _check_edge_child_value_invariant(tree)
    occupied = torch.zeros(9, dtype=torch.bool)
    occupied[0] = True
    occupied[1] = True
    occupied[2] = True
    occupied[3] = True
    occupied[5] = True
    occupied[7] = True
    board_embed[0, ..., 0] = torch.tensor(
        [[1.0, -1.0, 1.0], [-1.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    )
    legal_mask = (~occupied).unsqueeze(0)

    action_probs2, tree2 = mcts_search(
        root_embeddings=board_embed,
        root_logits=torch.zeros(B, A),
        root_value=torch.zeros(B, 2),
        root_to_play=torch.zeros(B, dtype=torch.long),
        recurrent_fn=recurrent_fn,
        num_simulations=12,
        num_actions=A,
        num_players=2,
        legal_mask=legal_mask,
        dirichlet_epsilon=0.0,
    )
    assert torch.all(tree2["children_visits"][:, 0, occupied] == 0)