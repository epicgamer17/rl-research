import torch

_epsilon = 1e-12


def _is_low_precision(tensor: torch.Tensor):
    """Returns True if the tensor is in low precision or autocast is enabled."""
    return (
        tensor.dtype in [torch.float16, torch.bfloat16] or torch.is_autocast_enabled()
    )


def categorical_crossentropy(predicted: torch.Tensor, target: torch.Tensor, axis=-1):
    if not _is_low_precision(predicted):
        assert torch.allclose(
            torch.sum(predicted, dim=axis, keepdim=True),
            torch.ones_like(torch.sum(predicted, dim=axis, keepdim=True)),
            atol=1e-2,
        ), f"Predicted probabilities do not sum to 1: sum = {torch.sum(predicted, dim=axis, keepdim=True)}, for predicted = {predicted}"
    assert predicted.shape == target.shape, f"{predicted.shape} = { target.shape}"

    predicted = (predicted + _epsilon) / torch.sum(
        predicted + _epsilon, dim=axis, keepdim=True
    )
    log_prob = torch.log(predicted)
    return -torch.sum(log_prob * target, axis=axis)


class CategoricalCrossentropyLoss:
    def __init__(self, from_logits=False, axis=-1):
        self.from_logits = from_logits
        self.axis = axis

    def __call__(self, predicted, target):
        return categorical_crossentropy(predicted, target, self.axis)

    def __eq__(self, other):
        if not isinstance(other, CategoricalCrossentropyLoss):
            return False
        return self.from_logits == other.from_logits and self.axis == other.axis


def kl_divergence(predicted: torch.Tensor, target: torch.Tensor, axis=-1):
    assert predicted.shape == target.shape, f"{predicted.shape} = { target.shape}"
    if not _is_low_precision(predicted):
        assert torch.allclose(
            torch.sum(predicted, dim=axis, keepdim=True),
            torch.ones_like(torch.sum(predicted, dim=axis, keepdim=True)),
            atol=1e-2,
        ), f"Predicted probabilities do not sum to 1: sum = {torch.sum(predicted, dim=axis, keepdim=True)}, for predicted = {predicted}"
    if not _is_low_precision(target):
        assert torch.allclose(
            torch.sum(target, dim=axis, keepdim=True),
            torch.ones_like(torch.sum(target, dim=axis, keepdim=True)),
            atol=1e-2,
        ), f"Target probabilities do not sum to 1: sum = {torch.sum(target, dim=axis, keepdim=True)}, for target = {target}"
    # 1. Add epsilon prevents 0/0 errors
    # 2. Normalize BOTH to ensure they sum to 1.0
    predicted = (predicted + _epsilon) / torch.sum(
        predicted + _epsilon, dim=axis, keepdim=True
    )
    target = (target + _epsilon) / torch.sum(target + _epsilon, dim=axis, keepdim=True)

    # 3. Compute KL: sum(target * log(target / predicted))
    # Splitting the log is numerically more stable: target * (log(target) - log(predicted))
    return torch.sum(target * (torch.log(target) - torch.log(predicted)), dim=axis)


class KLDivergenceLoss:
    def __init__(self, from_logits=False, axis=-1):
        self.from_logits = from_logits
        self.axis = axis

    def __call__(self, predicted, target):
        return kl_divergence(predicted, target, self.axis)

    def __eq__(self, other):
        if not isinstance(other, KLDivergenceLoss):
            return False
        return self.from_logits == other.from_logits and self.axis == other.axis


def huber(predicted: torch.Tensor, target: torch.Tensor, axis=-1, delta: float = 1.0):
    assert predicted.shape == target.shape, f"{predicted.shape} = { target.shape}"
    diff = torch.abs(predicted - target)
    return torch.where(
        diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta)
    ).view(-1)


class HuberLoss:
    def __init__(self, axis=-1, delta: float = 1.0):
        self.axis = axis
        self.delta = delta

    def __call__(self, predicted, target):
        return huber(predicted, target, axis=self.axis, delta=self.delta)

    def __eq__(self, other):
        if not isinstance(other, HuberLoss):
            return False
        return self.axis == other.axis and self.delta == other.delta


def mse(predicted: torch.Tensor, target: torch.Tensor):
    assert predicted.shape == target.shape, f"{predicted.shape} = { target.shape}"
    return (predicted - target) ** 2


class MSELoss:
    def __init__(self):
        pass

    def __call__(self, predicted, target):
        return mse(predicted, target)

    def __eq__(self, other):
        return isinstance(other, MSELoss)
