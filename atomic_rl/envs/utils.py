"""
Environment Utilities.

Helper routines for parsing vector environment info structures and observation resets.
"""

import numpy as np
from typing import Tuple


# TODO: messy, hard to read, and doesnt work reliably or at least not tested reliably on both gym vector envs and pufferlib's vector envs.
def extract_vector_env_final_obs(info) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the true final observations from a Gymnasium Vector Environment's info dict.
    Safely handles the auto-reset hidden states.

    Some vector-env wrappers (notably PufferLib's `pufferlib.vector`) deliver
    `info` as a list rather than a Gymnasium-style dict, and do not preserve
    the truncation-state observation at all. In those cases this returns
    empty arrays — the caller should not bootstrap through the truncated
    next-state since `V(s_truncated)` is not recoverable.

    Args:
        info: The info object returned by `envs.step()`. Expected to be a dict
            with `final_observation` / `_final_observation` keys (Gymnasium
            vector envs). Any other type (list, None, etc.) is treated as
            "no final observations available" and yields empty arrays.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - env_indices: 1D array of environment indices that terminated or truncated.
            - final_obs: Stacked array of the true final observations for those environments.
              Returns (empty_array, empty_array) if no environments ended or if the
              info format does not expose final observations.
    """
    # Note: If `info` is not a dict (e.g. PufferLib lists), this will explicitly crash,
    # which is intended under the fail-fast philosophy unless handled explicitly.
    # TODO: implement correct behaviour for pufferlib

    # Handle None or non-dict inputs gracefully
    if not isinstance(info, dict):
        return np.array([]), np.array([])

    # Vector envs only add this key if at least one environment ended
    if "final_observation" not in info:
        return np.array([]), np.array([])

    # Gymnasium uses "_final_observation" as the boolean mask for which envs actually ended
    mask = info.get("_final_observation")
    if mask is None:
        return np.array([]), np.array([])

    env_indices = np.where(mask)[0]

    if len(env_indices) == 0:
        return np.array([]), np.array([])

    # Use explicit list comprehension to extract items using the NumPy array indices.
    # This avoids the TypeError from "fancy indexing" a Python list and avoids
    # allocating a slow intermediate np.array(..., dtype=object).
    valid_observations = [info["final_observation"][i] for i in env_indices]
    true_final_obs = np.stack(valid_observations)

    return env_indices, true_final_obs
