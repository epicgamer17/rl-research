from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("atomic-rl")
except PackageNotFoundError:
    __version__ = "0.1.0"

from . import action_selection
from . import bptt
from . import buffers
from . import evaluation
from . import losses
from . import metrics
from . import networks
from . import optimization
from . import returns
from . import schedules
from . import search
from . import td
from . import utils

__all__ = [
    "action_selection",
    "bptt",
    "buffers",
    "evaluation",
    "losses",
    "metrics",
    "networks",
    "optimization",
    "returns",
    "schedules",
    "search",
    "td",
    "utils",
]



