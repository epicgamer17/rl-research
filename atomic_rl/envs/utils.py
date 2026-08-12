"""
Environment Utilities.

Helper routines for parsing vector environment info structures and observation resets.
"""

import numpy as np
from typing import Tuple


def extract_vector_env_final_obs(info) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the true final observations from a Gymnasium Vector Environment's info dict.
    Safely handles auto-reset hidden states.

    Args:
        info: The info object returned by `envs.step()`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - env_indices: 1D array of environment indices that terminated or truncated.
            - final_obs: Stacked array of the true final observations for those environments.
    """
    pass
