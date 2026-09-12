import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Sets random seeds for reproducibility across random, numpy, and torch (CPU, CUDA, MPS).

    Args:
        seed: The integer seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
