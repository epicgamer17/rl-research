import torch
import torch.nn as nn
import numpy as np
import random
from typing import Tuple, List, Callable, Optional, Union
from tensordict import TensorDict


# NOTE: DONT LET THIS FILE BUILD UP AND HAVE A LOT FUNCTIONS, THAT IS A SIGN OF BAD ORGANIZATION.


def _allocate_tensordict(
    shapes: dict, batch_size: List[int], device: str = "cpu"
) -> TensorDict:
    """Allocates a zeroed TensorDict of any arbitrary geometry."""
    data = TensorDict({}, batch_size=batch_size, device=device)
    for key, shape in shapes.items():
        dtype = torch.long if "action" in key else torch.float32
        data.set(key, torch.zeros((*batch_size, *shape), dtype=dtype, device=device))
    return data


# TODO: do we need the inplace and not inplace operations?
# TODO/NOTE: should we remove and just use torch.lerp? For me personally, ema update is a much more semantically meaningful than lerp.
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
    # TODO: can also use torch.lerp here
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


def scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Scales gradients passing through tensor during backpropagation (MuZero Appendix G).

    In the forward pass, this acts as an identity function: output == tensor.
    In the backward pass, incoming gradients are scaled by `scale`:
        dL/d(tensor) = scale * dL/d(output)

    Used in MuZero to scale gradients entering the dynamics function by 0.5,
    and to scale individual loss terms by 1/K.

    Args:
        tensor: Input tensor of arbitrary shape.
        scale: Scaling coefficient applied to backward gradients.

    Returns:
        Tensor identical in value and shape to input tensor, with scaled gradients during backprop.
    """
    assert isinstance(
        scale, (int, float)
    ), f"scale must be a float or int, got {type(scale)}"

    return tensor * scale + tensor.detach() * (1 - scale)
