import torch
from tensordict import TensorDict
from typing import Tuple, Callable, Optional
from ..utils import add_dirichlet_noise
from .backpropagation import backpropagate_
from .expansion import expand_node_
from .selection import select_leaf
from .tree import init_mcts_tree
from .qtransforms import qtransform_by_parent_and_siblings
from .policies import get_mcts_visit_policy


# TODO: remember we eventually want gumbel sequential halving and possibly other search methods too.
# TODO: dont hard code dirichlet params, pass em in as optional arguments
# TODO: avoid flags like is_zero_sum
# TODO: make it easier to do stochastic muzero and stochastic alphazero.
# TODO: should this just return the tree?
# TODO: what about sampled muzero?
# TODO: make this less hardcoded. I don't mind if search orchestration goes in the orchestration code, if it gives users more freedom.


def mcts_search(
    root_embeddings: torch.Tensor,
    root_logits: torch.Tensor,
    root_value: torch.Tensor,  # [B, num_players]
    root_to_play: torch.Tensor,  # [B]
    recurrent_fn: Callable[
        [torch.Tensor, torch.Tensor],
        Tuple[
            torch.Tensor,  # logits: [B, num_actions]
            torch.Tensor,  # value: [B, num_players]
            torch.Tensor,  # reward: [B, num_players]
            torch.Tensor,  # discount: [B, num_players]
            torch.Tensor,  # next_embeddings: [B, D...]
            torch.Tensor,  # next_to_play: [B]
        ],
    ],
    num_simulations: int,
    num_actions: int,
    num_players: int,
    *,
    legal_mask: Optional[torch.Tensor] = None,
    qtransform: Callable = qtransform_by_parent_and_siblings,
    dirichlet_epsilon: float = 0.0,
    dirichlet_alpha: float = 0.3,
    pb_c_base: float = 19652.0,
    pb_c_init: float = 1.25,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, TensorDict]:
    """Runs batched N-player MuZero MCTS search matching mctx semantics."""
    device = root_embeddings.device
    assert (
        isinstance(num_simulations, int)
        and not isinstance(num_simulations, bool)
        and num_simulations >= 1
    ), f"num_simulations must be an int >= 1, got {num_simulations!r}"
    assert (
        isinstance(num_actions, int) and not isinstance(num_actions, bool)
    ) and num_actions >= 1, f"num_actions must be an int >= 1, got {num_actions!r}"
    assert (
        isinstance(num_players, int) and num_players >= 1
    ), f"num_players must be an int >= 1, got {num_players!r}"
    assert (
        dirichlet_epsilon >= 0
    ), f"dirichlet_epsilon must be >= 0, got {dirichlet_epsilon}"
    assert dirichlet_alpha >= 0, f"dirichlet_alpha must be >= 0, got {dirichlet_alpha}"
    assert pb_c_base >= 0, f"pb_c_base must be >= 0, got {pb_c_base}"
    assert pb_c_init >= 0, f"pb_c_init must be >= 0, got {pb_c_init}"
    assert temperature >= 0, f"temperature must be >= 0, got {temperature}"

    assert (
        root_embeddings.ndim >= 2
    ), f"root_embeddings must be at least 2D [B, D...], got shape {tuple(root_embeddings.shape)}"
    batch_size = root_embeddings.shape[0]

    assert root_logits.shape == (batch_size, num_actions), (
        f"root_logits shape mismatch: expected [{batch_size}, {num_actions}], "
        f"got {tuple(root_logits.shape)}"
    )
    assert root_value.shape == (batch_size, num_players), (
        f"root_value shape mismatch: expected [{batch_size}, {num_players}], "
        f"got {tuple(root_value.shape)}"
    )
    assert root_to_play.shape == (
        batch_size,
    ), f"root_to_play shape mismatch: expected [{batch_size}], got {tuple(root_to_play.shape)}"

    if legal_mask is not None:
        assert legal_mask.shape == (batch_size, num_actions), (
            f"legal_mask shape mismatch: expected [{batch_size}, {num_actions}], "
            f"got {tuple(legal_mask.shape)}"
        )

    batch_range = torch.arange(batch_size, device=device)

    # 1. Apply Dirichlet noise to root logits
    curr_root_logits = root_logits.clone()
    if dirichlet_epsilon > 0.0:
        curr_root_logits = add_dirichlet_noise(
            curr_root_logits,
            dirichlet_epsilon,
            dirichlet_alpha,
            mask=legal_mask,
        )

    # 2. Initialize N-Player Tree State
    tree = init_mcts_tree(
        root_embeddings=root_embeddings,
        root_logits=curr_root_logits,
        root_value=root_value,
        root_to_play=root_to_play,
        num_simulations=num_simulations,
        num_actions=num_actions,
        num_players=num_players,
        legal_mask=legal_mask,
    )

    # 3. Simulation Loop
    for _ in range(num_simulations):
        # A. Selection Phase
        leaf_parents, leaf_actions, expansion_mask, trajectory = select_leaf(
            tree,
            pb_c_base=pb_c_base,
            pb_c_init=pb_c_init,
            qtransform=qtransform,
        )

        if not trajectory:
            break

        b_idx = batch_range[expansion_mask]
        leaf_value = tree["node_values"].new_zeros(batch_size, num_players)

        if b_idx.numel() > 0:
            # B. Model Dynamics Step
            parent_embeddings = tree["embeddings"][b_idx, leaf_parents[b_idx]]
            (
                prior_logits,
                value,
                reward,
                discount,
                next_embeddings,
                next_to_play,
            ) = recurrent_fn(leaf_actions[b_idx], parent_embeddings)

            leaf_value[b_idx] = value

            # Scatter model outputs into full-batch arrays
            prior_logits_full = root_logits.new_zeros(batch_size, num_actions)
            value_full = tree["node_values"].new_zeros(batch_size, num_players)
            reward_full = tree["children_rewards"].new_zeros(batch_size, num_players)
            discount_full = tree["children_discounts"].new_zeros(
                batch_size, num_players
            )
            next_embeddings_full = root_embeddings.new_zeros(
                (batch_size, *root_embeddings.shape[1:])
            )
            players_full = tree["to_play"].new_zeros(batch_size)

            prior_logits_full[b_idx] = prior_logits
            value_full[b_idx] = value
            reward_full[b_idx] = reward
            discount_full[b_idx] = discount
            next_embeddings_full[b_idx] = next_embeddings
            players_full[b_idx] = next_to_play

            # C. Expansion Phase
            expand_node_(
                tree=tree,
                parent_nodes=leaf_parents,
                actions_taken=leaf_actions,
                policy_logits=prior_logits_full,
                value=value_full,
                rewards=reward_full,
                discounts=discount_full,
                next_embeddings=next_embeddings_full,
                players=players_full,
                legal_mask=None,
                masks=expansion_mask,
            )

        # D. Backpropagation Phase
        backpropagate_(
            tree=tree,
            trajectory=trajectory,
            leaf_value=leaf_value,
        )

    # 4. Extract Visit Policy and Select Actions
    root_visits = tree["children_visits"][:, 0]  # [B, A]
    action_probs = get_mcts_visit_policy(root_visits, temperature=temperature)

    if temperature == 0.0:
        selected_action = action_probs.argmax(dim=-1)
    else:
        selected_action = torch.multinomial(action_probs, num_samples=1).squeeze(-1)

    return selected_action, action_probs, tree
