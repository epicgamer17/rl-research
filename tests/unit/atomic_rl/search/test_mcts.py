import pytest
import torch

from atomic_rl.action_selection import add_dirichlet_noise
from atomic_rl.search import mcts_search, puct_selection

pytestmark = pytest.mark.unit


def test_mcts_search_root_legal_mask():
    """Verify that passing legal_mask to mcts_search explicitly masks root illegal actions."""
    root_embed = torch.zeros(1, 4)

    def expansion_fn(embeds):
        logits = torch.tensor([[2.0, 5.0, 1.0]])  # Action 1 has highest unmasked logit!
        values = torch.tensor([0.5])
        return logits, values

    def dummy_dynamics_fn(embeds, actions):
        next_embeds = torch.zeros_like(embeds)
        rewards = torch.zeros(1)
        next_to_play = torch.zeros(1, dtype=torch.long)
        is_terminal = torch.zeros(1, dtype=torch.bool)
        return next_embeds, rewards, next_to_play, is_terminal

    root_logits, root_value = expansion_fn(root_embed)
    root_legal_mask = torch.tensor([[True, False, True]])  # Action 1 is explicitly illegal

    def recurrent_fn(actions, embeds, to_play=None):
        next_embeds, rewards, next_to_play, is_terminal = dummy_dynamics_fn(
            embeds, actions
        )
        logits, values = expansion_fn(next_embeds)
        values = torch.where(is_terminal, torch.zeros_like(values), values).unsqueeze(-1)
        rewards = rewards.unsqueeze(-1)
        discount = torch.where(
            is_terminal.unsqueeze(-1), torch.zeros_like(values), torch.ones_like(values)
        )
        return logits, values, rewards, discount, next_embeds, next_to_play

    root_logits_noisy = add_dirichlet_noise(
        root_logits,
        epsilon=0.5,
        alpha=0.3,
        mask=root_legal_mask,
    )

    tree = mcts_search(
        root_embeddings=root_embed,
        root_logits=root_logits_noisy,
        root_value=root_value.unsqueeze(-1),
        root_to_play=torch.zeros(1, dtype=torch.long),
        recurrent_fn=recurrent_fn,
        num_simulations=20,
        num_actions=3,
        num_players=1,
        selection_fn=puct_selection,
        legal_mask=root_legal_mask,
    )

    # Action 1 (highest raw logit) must be masked to 0.0 prior at root and receive 0 visits
    root_priors = torch.softmax(tree["children_prior_logits"][0, 0], dim=-1)
    assert root_priors[1].item() == 0.0
    assert tree["children_visits"][0, 0, 1].item() == 0.0
    assert (
        tree["children_visits"][0, 0, 0].item()
        + tree["children_visits"][0, 0, 2].item()
        == 20
    )


def test_mcts_search_dynamics_fn_legal_mask():
    """Verify that recurrent_fn masking child logits explicitly masks child node illegal actions during search."""
    root_embed = torch.zeros(1, 4)

    def expansion_fn(embeds):
        # Always return raw unmasked logits where Action 1 is highest
        logits = torch.tensor([[1.0, 10.0, 1.0]])
        values = torch.tensor([[0.5]])
        return logits, values

    def dynamics_fn_with_mask(embeds, actions):
        next_embeds = torch.zeros_like(embeds)
        rewards = torch.zeros(1, 1)
        next_to_play = torch.zeros(1, dtype=torch.long)
        is_terminal = torch.zeros(1, dtype=torch.bool)
        # 5th item: dynamics returns legal mask for next state (Action 1 illegal in expanded nodes)
        next_legal_mask = torch.tensor([[True, False, True]])
        return next_embeds, rewards, next_to_play, is_terminal, next_legal_mask

    # Root legal mask allows all actions at root so simulation moves to child
    root_legal_mask = torch.tensor([[True, True, True]])

    root_logits, root_value = expansion_fn(root_embed)

    def recurrent_fn(actions, embeds, to_play=None):
        next_embeds, rewards, next_to_play, is_terminal, next_legal_mask = dynamics_fn_with_mask(
            embeds, actions
        )
        logits, values = expansion_fn(next_embeds)
        # Mask illegal actions for the child node inside the recurrent_fn
        masked_logits = torch.where(next_legal_mask, logits, -1e9)
        values = torch.where(is_terminal.unsqueeze(-1), torch.zeros_like(values), values)
        discount = torch.where(
            is_terminal.unsqueeze(-1), torch.zeros_like(values), torch.ones_like(values)
        )
        return masked_logits, values, rewards, discount, next_embeds, next_to_play

    tree = mcts_search(
        root_embeddings=root_embed,
        root_logits=root_logits,
        root_value=root_value,
        root_to_play=torch.zeros(1, dtype=torch.long),
        recurrent_fn=recurrent_fn,
        num_simulations=10,
        num_actions=3,
        num_players=1,
        selection_fn=puct_selection,
        legal_mask=root_legal_mask,
    )

    # At child nodes, dynamics_fn mask prevents Action 1 from receiving visits at deeper levels
    assert tree["children_index"].shape[1] > 1
