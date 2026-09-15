from . import metaoptimizers
from .obgd import obgd_update_, obgd_td_update_, ObGD
from .adaptive_obgd import (
    adaptive_obgd_update_,
    adaptive_obgd_td_update_,
    AdaptiveObGD,
)

# NOTE: The Objects in here like ObGD, AdaptiveObGD, K1, and K2, IBDB, are prime examples of the functional core OOP orchestration design philosophy in our CONTRIBUTING.md
