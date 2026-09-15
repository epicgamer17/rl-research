"""
Target Network Parameter Update Utilities.

Stateful in-place functions for soft (EMA) and hard updating target network weights.
"""

import torch
import torch.nn as nn


# NOTE: This is a stateful function which is not ideal for the functional paradigm but follows standard pytorch and ml conventions.
def soft_update_target_network_(
    model: nn.Module, target_model: nn.Module, tau: float = 0.005
) -> None:
    """
    Soft update of target network parameters.
    target_params = (1 - tau) * target_params + tau * params

    Args:
        model (nn.Module): The model to update the target network from.
        target_model (nn.Module): The target network to update.
        tau (float): The soft update coefficient.
    """
    assert 0.0 <= tau <= 1.0, f"tau must be in [0, 1], got {tau}"
    with torch.no_grad():
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            assert (
                target_param.shape == param.shape
            ), f"Parameter shape mismatch: {target_param.shape} vs {param.shape}"
            # TODO: why not             target_param.copy_(ema_update(target_param, param, tau))? Is this not cleaner?
            target_param.copy_((1.0 - tau) * target_param + tau * param)


# NOTE: This is a stateful function which is not ideal for the functional paradigm but follows standard pytorch and ml conventions.
def hard_update_target_network_(model: nn.Module, target_model: nn.Module) -> None:
    """
    Hard update of target network parameters.
    target_params = params

    Args:
        model (nn.Module): The model to update the target network from.
        target_model (nn.Module): The target network to update.

    This function is created in contrast to soft_update_target_network_.
    """
    target_model.load_state_dict(model.state_dict())
