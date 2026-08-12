import torch
import torch.nn as nn
import numpy as np
import random
from typing import Tuple, List, Callable, Optional, Union
from tensordict import TensorDict


# NOTE: DONT LET THIS FILE BUILD UP AND HAVE A LOT FUNCTIONS, THAT IS A SIGN OF BAD ORGANIZATION.


# TODO: do we need the inplace and not inplace operations?
def ema_update(
    old_ema: torch.Tensor, new_value: torch.Tensor, alpha: float
) -> torch.Tensor:
    """
    Calculates the exponential moving average (EMA).
    Formula: (1 - alpha) * old_ema + alpha * new_value
    """
    assert (
        old_ema.shape == new_value.shape
    ), f"EMA shape mismatch: {old_ema.shape} vs {new_value.shape}"

    return (1.0 - alpha) * old_ema + alpha * new_value


def ema_update_(
    old_ema: torch.Tensor, new_value: torch.Tensor, alpha: float
) -> torch.Tensor:
    """
    In-place exponential moving average (EMA).
    Formula: (1 - alpha) * old_ema + alpha * new_value
    """
    assert (
        old_ema.shape == new_value.shape
    ), f"EMA shape mismatch: {old_ema.shape} vs {new_value.shape}"

    # Use optimized in-place kernels: old = old * (1-a) + new * a
    # This is rearranged for .add_ usage: old = old - a*old + a*new => old += a*(new - old)
    return old_ema.lerp_(new_value, alpha)


def to_tensor(
    numpy_array: Union[np.ndarray, int, float, bool],
    device: torch.device = torch.device("cpu"),
    dtype: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    """
    Converts a numpy array (or primitive) to a PyTorch tensor on the specified device.

    Args:
        numpy_array: The numpy array or primitive to convert.
        device: The target device.
        dtype: The target data type. Defaults to float32.

    Returns:
        A PyTorch tensor.
    """
    return torch.as_tensor(numpy_array, dtype=dtype, device=device)


# TODO: does this handle pendulum?
def to_numpy_action(action_tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch action tensor to a numpy array for Gymnasium consumption.
    Handles detaching, moving to CPU, and flattening for discrete spaces.

    Args:
        action_tensor: The action tensor from the model.

    Returns:
        A numpy array of actions.
    """
    res = action_tensor.detach().cpu().numpy()
    # For discrete actions (LongTensor/IntTensor), we cast to int32
    if action_tensor.dtype in [torch.long, torch.int]:
        # Flatten if it's a single action per batch entry (e.g., [B, 1] -> [B])
        # This is the standard for Gymnasium VectorEnv consumption of discrete actions.
        if res.ndim > 1 and res.shape[-1] == 1:
            return res.flatten().astype(np.int32)
        return res.astype(np.int32)
    # For continuous actions, we keep the original shape and dtype (float32)
    return res
