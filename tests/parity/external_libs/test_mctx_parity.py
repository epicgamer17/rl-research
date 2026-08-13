import pytest
import numpy as np
import torch
import torch.nn.functional as F

mctx = pytest.importorskip("mctx")
jax = pytest.importorskip("jax")
import jax.numpy as jnp

from tensordict import TensorDict
from atomic_rl.search import (
    init_mcts_tree,
    qtransform_by_min_max,
    qtransform_by_parent_and_siblings,
    qtransform_completed_by_mix_value,
    puct_score,
    backpropagate_,
    expand_node_,
    get_mcts_visit_policy,
    mcts_search,
)


def to_jax(tensor: torch.Tensor) -> jnp.ndarray:
    return jnp.array(tensor.detach().cpu().numpy())


def to_torch(jax_array: jnp.ndarray, dtype=torch.float32, device="cpu") -> torch.Tensor:
    return torch.from_numpy(np.array(jax_array)).to(dtype=dtype, device=device)


def torch_tree_to_mctx_single(tree: TensorDict, b: int) -> mctx.Tree:
    """Extracts batch element b from an atomic_rl TensorDict search tree into an unbatched mctx.Tree."""
    raw_v = tree["raw_values"][b]
    node_v = tree["node_values"][b]
    c_rew = tree["children_rewards"][b]
    c_disc = tree["children_discounts"][b]
    c_val = tree["children_values"][b]

    if raw_v.ndim == 2 and raw_v.shape[-1] == 1:
        raw_v = raw_v.squeeze(-1)
        node_v = node_v.squeeze(-1)
    elif raw_v.ndim == 2:
        raw_v = raw_v[..., 0]
        node_v = node_v[..., 0]

    if c_rew.ndim == 3 and c_rew.shape[-1] == 1:
        c_rew = c_rew.squeeze(-1)
        c_disc = c_disc.squeeze(-1)
        c_val = c_val.squeeze(-1)
    elif c_rew.ndim == 3:
        c_rew = c_rew[..., 0]
        c_disc = c_disc[..., 0]
        c_val = c_val[..., 0]

    root_invalid = (
        ~tree["root_legal_mask"][b]
        if "root_legal_mask" in tree.keys()
        else torch.zeros(tree["children_index"].shape[-1], dtype=torch.bool)
    )

    return mctx.Tree(
        node_visits=to_jax(tree["node_visits"][b]),
        raw_values=to_jax(raw_v),
        node_values=to_jax(node_v),
        parents=to_jax(tree["parents"][b]),
        action_from_parent=to_jax(tree["action_from_parent"][b]),
        children_index=to_jax(tree["children_index"][b]),
        children_prior_logits=to_jax(tree["children_prior_logits"][b]),
        children_visits=to_jax(tree["children_visits"][b]),
        children_rewards=to_jax(c_rew),
        children_discounts=to_jax(c_disc),
        children_values=to_jax(c_val),
        embeddings=to_jax(tree["embeddings"][b]),
        root_invalid_actions=to_jax(root_invalid),
        extra_data=None,
    )


def test_qtransform_by_min_max_parity():
    """Verify qtransform_by_min_max produces identical output to mctx."""
    B, A, S = 2, 4, 5
    tree = init_mcts_tree(
        root_embeddings=torch.randn(B, 3),
        root_logits=torch.randn(B, A),
        root_value=torch.tensor([[0.5], [-0.2]]),
        root_to_play=torch.zeros(B, dtype=torch.long),
        num_simulations=S,
        num_actions=A,
        num_players=1,
    )

    tree["children_visits"][:, 0] = torch.tensor([[10, 0, 5, 2], [0, 1, 0, 8]])
    tree["children_rewards"][:, 0] = torch.tensor([[[0.1], [0.0], [0.2], [0.5]], [[0.0], [0.3], [0.0], [0.4]]])
    tree["children_discounts"][:, 0] = torch.ones(B, A, 1)
    tree["children_values"][:, 0] = torch.tensor([[[0.8], [0.0], [-0.1], [0.3]], [[0.0], [0.5], [0.0], [0.9]]])

    parent_nodes = torch.zeros(B, dtype=torch.long)
    min_val = torch.tensor([-1.0, -0.5])
    max_val = torch.tensor([1.5, 2.0])

    torch_q = qtransform_by_min_max(
        tree, parent_nodes, min_value=min_val, max_value=max_val
    )

    mctx_qs = []
    for b in range(B):
        single_tree = torch_tree_to_mctx_single(tree, b)
        mctx_q_b = mctx.qtransform_by_min_max(
            single_tree, 0, min_value=float(min_val[b]), max_value=float(max_val[b])
        )
        mctx_qs.append(np.array(mctx_q_b))

    mctx_q = np.stack(mctx_qs, axis=0)
    np.testing.assert_allclose(torch_q.numpy(), mctx_q, rtol=1e-5, atol=1e-5)


def test_qtransform_by_parent_and_siblings_parity():
    """Verify qtransform_by_parent_and_siblings matches mctx implementation."""
    B, A, S = 2, 3, 5
    tree = init_mcts_tree(
        root_embeddings=torch.randn(B, 4),
        root_logits=torch.tensor([[1.0, 0.5, -0.2], [0.0, 2.0, 1.0]]),
        root_value=torch.tensor([[0.25], [-0.5]]),
        root_to_play=torch.zeros(B, dtype=torch.long),
        num_simulations=S,
        num_actions=A,
        num_players=1,
    )

    tree["children_visits"][:, 0] = torch.tensor([[3, 0, 1], [0, 5, 2]])
    tree["children_rewards"][:, 0] = torch.tensor([[[0.5], [0.0], [0.1]], [[0.0], [1.0], [-0.2]]])
    tree["children_discounts"][:, 0] = torch.ones(B, A, 1)
    tree["children_values"][:, 0] = torch.tensor([[[0.4], [0.0], [0.9]], [[0.0], [-0.3], [0.7]]])

    parent_nodes = torch.zeros(B, dtype=torch.long)

    torch_q = qtransform_by_parent_and_siblings(tree, parent_nodes)

    mctx_qs = []
    for b in range(B):
        single_tree = torch_tree_to_mctx_single(tree, b)
        mctx_q_b = mctx.qtransform_by_parent_and_siblings(single_tree, 0)
        mctx_qs.append(np.array(mctx_q_b))

    mctx_q = np.stack(mctx_qs, axis=0)
    np.testing.assert_allclose(torch_q.numpy(), mctx_q, rtol=1e-5, atol=1e-5)


def test_qtransform_completed_by_mix_value_parity():
    """Verify qtransform_completed_by_mix_value matches mctx implementation."""
    B, A, S = 2, 4, 5
    tree = init_mcts_tree(
        root_embeddings=torch.randn(B, 2),
        root_logits=torch.tensor([[0.5, 1.2, -0.3, 0.1], [1.0, 0.0, 0.0, 0.5]]),
        root_value=torch.tensor([[0.1], [0.4]]),
        root_to_play=torch.zeros(B, dtype=torch.long),
        num_simulations=S,
        num_actions=A,
        num_players=1,
    )

    tree["children_visits"][:, 0] = torch.tensor([[5, 0, 2, 0], [1, 3, 0, 0]])
    tree["children_rewards"][:, 0] = torch.tensor([[[0.1], [0.0], [0.3], [0.0]], [[0.0], [0.2], [0.0], [0.0]]])
    tree["children_discounts"][:, 0] = torch.ones(B, A, 1)
    tree["children_values"][:, 0] = torch.tensor([[[0.6], [0.0], [0.4], [0.0]], [[0.8], [0.5], [0.0], [0.0]]])

    parent_nodes = torch.zeros(B, dtype=torch.long)

    torch_q = qtransform_completed_by_mix_value(
        tree, parent_nodes, value_scale=0.1, maxvisit_init=50.0, rescale_values=True, use_mixed_value=True
    )

    mctx_qs = []
    for b in range(B):
        single_tree = torch_tree_to_mctx_single(tree, b)
        mctx_q_b = mctx.qtransform_completed_by_mix_value(
            single_tree, 0, value_scale=0.1, maxvisit_init=50.0, rescale_values=True, use_mixed_value=True
        )
        mctx_qs.append(np.array(mctx_q_b))

    mctx_q = np.stack(mctx_qs, axis=0)
    np.testing.assert_allclose(torch_q.numpy(), mctx_q, rtol=1e-5, atol=1e-5)


def test_action_selection_parity():
    """Verify action selection under PUCT produces identical top action choices."""
    B, A, S = 2, 4, 10
    tree = init_mcts_tree(
        root_embeddings=torch.randn(B, 3),
        root_logits=torch.tensor([[2.0, 1.0, 0.0, -1.0], [0.5, 2.5, 0.1, 0.0]]),
        root_value=torch.tensor([[0.0], [0.0]]),
        root_to_play=torch.zeros(B, dtype=torch.long),
        num_simulations=S,
        num_actions=A,
        num_players=1,
    )

    tree["children_visits"][:, 0] = torch.tensor([[10, 5, 1, 0], [2, 12, 0, 1]])
    tree["children_rewards"][:, 0] = torch.tensor([[[0.2], [0.1], [0.0], [0.0]], [[0.0], [0.5], [0.0], [0.1]]])
    tree["children_discounts"][:, 0] = torch.ones(B, A, 1)
    tree["children_values"][:, 0] = torch.tensor([[[0.7], [0.6], [0.1], [0.0]], [[0.3], [0.9], [0.0], [0.2]]])

    parent_nodes = torch.zeros(B, dtype=torch.long)

    rng_key = jax.random.PRNGKey(42)
    mctx_actions = []
    for b in range(B):
        single_tree = torch_tree_to_mctx_single(tree, b)
        mctx_act_b = mctx.muzero_action_selection(
            rng_key, single_tree, node_index=0, depth=0
        )
        mctx_actions.append(np.array(mctx_act_b))

    mctx_action = np.stack(mctx_actions)

    visit_counts = tree["children_visits"][:, 0]
    node_visits = tree["node_visits"][:, 0].unsqueeze(-1)
    prior_probs = F.softmax(tree["children_prior_logits"][:, 0], dim=-1)
    pb_c = 1.25 + torch.log((node_visits + 19652.0 + 1.0) / 19652.0)
    policy_score = torch.sqrt(node_visits.float()) * pb_c * prior_probs / (visit_counts + 1.0)
    value_score = qtransform_by_parent_and_siblings(tree, parent_nodes)
    deterministic_scores = value_score + policy_score
    torch_action = deterministic_scores.argmax(dim=-1)

    np.testing.assert_array_equal(torch_action.numpy(), mctx_action)


def test_get_mcts_visit_policy_parity():
    """Verify get_mcts_visit_policy matches mctx visit probabilities summary."""
    visit_counts = torch.tensor([[10, 5, 0, 15], [0, 0, 0, 0]], dtype=torch.float32)

    # Temperature tau = 1.0
    torch_probs = get_mcts_visit_policy(visit_counts, temperature=1.0)
    mctx_probs = np.array([
        [10/30, 5/30, 0.0, 15/30],
        [0.25, 0.25, 0.25, 0.25],
    ])
    np.testing.assert_allclose(torch_probs.numpy(), mctx_probs, rtol=1e-5, atol=1e-5)

    # Temperature tau = 0.0
    torch_probs_max = get_mcts_visit_policy(torch.tensor([[10, 5, 0, 15]]), temperature=0.0)
    np.testing.assert_allclose(torch_probs_max.numpy(), [[0.0, 0.0, 0.0, 1.0]])


def test_full_mcts_search_parity():
    """Verify full mcts_search matches mctx.muzero_policy end-to-end on a deterministic task."""
    B = 1
    A = 3
    S = 5

    root_embeddings = torch.tensor([[1.0, 2.0]])
    root_logits = torch.tensor([[0.5, 2.5, -0.5]])
    root_value = torch.tensor([[0.1]])
    root_to_play = torch.tensor([0], dtype=torch.long)

    # Recurrent functions with clear logits preference for action 1
    def py_recurrent_fn(actions, parent_embeddings, parent_to_play):
        b = actions.shape[0]
        logits = torch.tensor([[0.1, 2.5, 0.1]]).repeat(b, 1)
        value = torch.ones(b, 1) * 0.5
        reward = actions.float().unsqueeze(-1) * 0.2
        discount = torch.ones(b, 1)
        next_embeddings = parent_embeddings + actions.float().unsqueeze(-1)
        next_to_play = torch.zeros(b, dtype=torch.long)
        return logits, value, reward, discount, next_embeddings, next_to_play

    def jax_recurrent_fn(params, rng_key, action, embedding):
        del params, rng_key
        batch_size = action.shape[0]
        logits = jnp.tile(jnp.array([[0.1, 2.5, 0.1]]), (batch_size, 1))
        value = jnp.ones((batch_size,)) * 0.5
        reward = action.astype(jnp.float32) * 0.2
        discount = jnp.ones((batch_size,))
        next_embedding = embedding + action[:, None].astype(jnp.float32)
        recurrent_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return recurrent_output, next_embedding

    # Run atomic_rl search
    action_probs, tree = mcts_search(
        root_embeddings=root_embeddings,
        root_logits=root_logits,
        root_value=root_value,
        root_to_play=root_to_play,
        recurrent_fn=py_recurrent_fn,
        num_simulations=S,
        num_actions=A,
        num_players=1,
        dirichlet_epsilon=0.0,
        temperature=1.0,
    )

    # Run mctx muzero_policy search
    root = mctx.RootFnOutput(
        prior_logits=to_jax(root_logits),
        value=to_jax(root_value.squeeze(-1)),
        embedding=to_jax(root_embeddings),
    )

    rng_key = jax.random.PRNGKey(0)
    policy_output = mctx.muzero_policy(
        params=None,
        rng_key=rng_key,
        root=root,
        recurrent_fn=jax_recurrent_fn,
        num_simulations=S,
        invalid_actions=None,
    )

    mctx_tree = policy_output.search_tree

    torch_visits = tree["children_visits"][:, 0].numpy()
    mctx_visits = np.array(mctx_tree.children_visits[:, 0])
    np.testing.assert_array_equal(torch_visits, mctx_visits)

    np.testing.assert_allclose(action_probs.numpy(), np.array(policy_output.action_weights), rtol=1e-5, atol=1e-5)

    torch_root_v = tree["node_values"][:, 0].squeeze(-1).numpy()
    mctx_root_v = np.array(mctx_tree.node_values[:, 0])
    np.testing.assert_allclose(torch_root_v, mctx_root_v, rtol=1e-5, atol=1e-5)

    torch_child_v = tree["children_values"][:, 0].squeeze(-1).numpy()
    mctx_child_v = np.array(mctx_tree.children_values[:, 0])
    np.testing.assert_allclose(torch_child_v, mctx_child_v, rtol=1e-5, atol=1e-5)
