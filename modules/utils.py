import torch

import torch


def support_to_scalar(probabilities, support_size):
    """
    Transform a categorical representation (1D probs vector)
    into a scalar.
    """
    assert probabilities.dim() == 1, "probabilities must be 1D"
    assert probabilities.shape[0] == 2 * support_size + 1, "shape mismatch"

    # support values from -support_size to +support_size
    support = torch.arange(
        -support_size,
        support_size + 1,
        device=probabilities.device,
        dtype=torch.float32,
    )
    x = torch.sum(support * probabilities)

    # Invert the scaling (from paper appendix)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x


def scalar_to_support(x, support_size):
    # raise NotImplementedError  # "this is bugged and outputs a uniform instead of a 2-hot"
    """
    Transform a scalar (float or 0D tensor)
    into categorical probs (1D tensor of length 2*support_size+1).
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    # Reduce the scale
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Clamp into support range
    x = torch.clamp(x, -support_size, support_size)

    floor = torch.floor(x)
    prob = x - floor

    probabilities = torch.zeros(2 * support_size + 1, device=x.device)

    # distribute mass between floor and floor+1
    probabilities[int(floor + support_size)] = 1 - prob
    if floor + 1 <= support_size:
        probabilities[int(floor + support_size + 1)] = prob

    # probabilities = torch.softmax(logits, dim=0)
    return probabilities


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
