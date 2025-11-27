import torch
import torch.nn.init as init
from torch import nn, Tensor


def support_to_scalar(
    probabilities: torch.Tensor, support_size: int, eps: float = 0.001
):
    """
    Convert categorical probabilities over the support [-support_size .. +support_size]
    into scalar(s) using the inverse of the MuZero transformation.

    Args:
        probabilities: Tensor of shape (L,) or (B, L) where L == 2*support_size + 1.
        support_size: integer support size.
        eps: small epsilon used by MuZero (default 0.001).

    Returns:
        Tensor of shape () (scalar) if input was 1D, or (B,) for batched input.
    """
    if probabilities.dim() == 1:
        probs = probabilities.unsqueeze(0)  # shape (1, L)
        squeeze_out = True
    else:
        probs = probabilities
        squeeze_out = False

    batch, L = probs.shape
    assert L == 2 * support_size + 1, "probabilities length must equal 2*support_size+1"

    device = probs.device
    dtype = probs.dtype

    support = torch.arange(
        -support_size, support_size + 1, device=device, dtype=dtype
    ).unsqueeze(
        0
    )  # (1, L)
    z = torch.sum(
        probs * support, dim=1
    )  # expected value on transformed scale, shape (B,)

    # inverse transform from MuZero appendix:
    # f^{-1}(z) = sign(z) * ( ((sqrt(1 + 4*eps*(|z| + 1 + eps)) - 1) / (2*eps))^2 - 1 )
    sign = torch.sign(z)
    abs_z = torch.abs(z)
    inner = 1.0 + 4.0 * eps * (abs_z + 1.0 + eps)
    inv = sign * (((torch.sqrt(inner) - 1.0) / (2.0 * eps)) ** 2 - 1.0)

    if squeeze_out:
        return inv.squeeze(0)
    return inv  # shape (B,)


def scalar_to_support(x: torch.Tensor | float, support_size: int, eps: float = 0.001):
    """
    Convert scalar(s) into a categorical (2*support_size+1)-vector on the MuZero transformed scale.

    Args:
        x: scalar (float or 0-dim tensor) or 1D tensor of shape (B,).
        support_size: integer support size.
        eps: small epsilon used by MuZero (default 0.001).

    Returns:
        Tensor of shape (L,) (if input scalar) or (B, L) for batched inputs,
        where L == 2*support_size + 1. Values are non-negative and sum to 1 per row.
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    x = x.to(dtype=torch.float32)
    squeeze_input = False
    if x.dim() == 0:
        x = x.unsqueeze(0)  # (1,)
        squeeze_input = True

    device = x.device
    batch = x.shape[0]
    L = 2 * support_size + 1

    # forward transform from MuZero appendix:
    # f(x) = sign(x) * (sqrt(|x| + 1) - 1) + eps * x
    sign = torch.sign(x)
    abs_x = torch.abs(x)
    x_trans = sign * (torch.sqrt(abs_x + 1.0) - 1.0) + eps * x

    # clamp into support range
    x_trans = torch.clamp(x_trans, -support_size, support_size)

    floor = torch.floor(x_trans)
    prob = x_trans - floor  # fractional part, in [0,1)

    out = torch.zeros((batch, L), device=device, dtype=torch.float32)

    idx_lower = (floor + support_size).long()  # index for floor
    idx_upper = idx_lower + 1  # index for floor+1 (may be out of bounds)

    batch_idx = torch.arange(batch, device=device)

    # assign (1 - frac) to floor bin
    out[batch_idx, idx_lower] = 1.0 - prob

    # assign frac to upper bin only if upper bin is within range
    valid_upper_mask = idx_upper <= (L - 1)
    if valid_upper_mask.any():
        valid_batch_idx = batch_idx[valid_upper_mask]
        valid_upper_idx = idx_upper[valid_upper_mask]
        valid_prob = prob[valid_upper_mask]
        out[valid_batch_idx, valid_upper_idx] = valid_prob

    # for cases where floor == support_size (i.e., at upper boundary), all mass already on lower bin
    if squeeze_input:
        return out.squeeze(0)  # shape (L,)
    return out  # shape (B, L)


# def support_to_scalar(probabilities, support_size):
#     """
#     Transform a categorical representation (1D probs vector)
#     into a scalar.
#     """
#     assert probabilities.dim() == 1, "probabilities must be 1D"
#     assert probabilities.shape[0] == 2 * support_size + 1, "shape mismatch"

#     # support values from -support_size to +support_size
#     support = torch.arange(
#         -support_size,
#         support_size + 1,
#         device=probabilities.device,
#         dtype=torch.float32,
#     )
#     x = torch.sum(support * probabilities)

#     # Invert the scaling (from paper appendix)
#     x = torch.sign(x) * (
#         ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
#         ** 2
#         - 1
#     )
#     return x


# def scalar_to_support(x, support_size):
#     # raise NotImplementedError  # "this is bugged and outputs a uniform instead of a 2-hot"
#     """
#     Transform a scalar (float or 0D tensor)
#     into categorical probs (1D tensor of length 2*support_size+1).
#     """
#     if not torch.is_tensor(x):
#         x = torch.tensor(x, dtype=torch.float32)

#     # Reduce the scale
#     x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

#     # Clamp into support range
#     x = torch.clamp(x, -support_size, support_size)

#     floor = torch.floor(x)
#     prob = x - floor

#     probabilities = torch.zeros(2 * support_size + 1, device=x.device)

#     # distribute mass between floor and floor+1
#     probabilities[int(floor + support_size)] = 1 - prob
#     if floor + 1 <= support_size:
#         probabilities[int(floor + support_size + 1)] = prob

#     # probabilities = torch.softmax(logits, dim=0)
#     return probabilities


# FOR BATCHED INPUTS
# def support_to_scalar(probabilities, support_size):
#     """
#     Transform a categorical representation to a scalar
#     See paper appendix Network Architecture
#     """
#     # Decode to a scalar
#     # print(probabilities.shape)
#     support = (
#         torch.tensor([x for x in range(-support_size, support_size + 1)])
#         .expand(probabilities.shape)
#         .float()
#         .to(device=probabilities.device)
#     )
#     x = torch.sum(support * probabilities, dim=1, keepdim=True)

#     # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
#     x = torch.sign(x) * (
#         ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
#         ** 2
#         - 1
#     )
#     return x


# def scalar_to_support(x, support_size):
#     """
#     Transform a scalar to a categorical representation with (2 * support_size + 1) categories
#     See paper appendix Network Architecture
#     """
#     # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
#     x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

#     # Encode on a vector
#     x = torch.clamp(x, -support_size, support_size)
#     floor = x.floor()
#     prob = x - floor
#     logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
#     logits.scatter_(
#         2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
#     )
#     indexes = floor + support_size + 1
#     prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
#     indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
#     logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
#     probabilities = logits.softmax(dim=2)
#     # print(probabilities.shape)
#     return probabilities


def scale_gradient(tensor, scale):
    """
    Scales the gradient for the backward pass without changing the forward pass.
    Args:
        tensor (torch.Tensor): The input tensor.
        scale (float): The scaling factor for the gradient.
    """
    return tensor * scale + tensor.detach() * (1 - scale)


def zero_weights_initializer(m: nn.Module) -> None:
    """Initializes the weights and biases of a layer to zero."""
    if hasattr(m, "weight") and m.weight is not None:
        init.constant_(m.weight, 0.0)
    if hasattr(m, "bias") and m.bias is not None:
        init.constant_(m.bias, 0.0)
