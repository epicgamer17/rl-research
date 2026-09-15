from .linear import get_linear_schedule
from .exponential import get_exponential_schedule
from .ape_x_epsilon import get_ape_x_epsilon
from .stepwise import get_stepwise_schedule

__all__ = [
    "get_linear_schedule",
    "get_exponential_schedule",
    "get_ape_x_epsilon",
    "get_stepwise_schedule",
]

# NOTE: The schedules folder and implementations here are a great example of our semantic meaning design philosophy. A lot of these are one line equations, but an equation is not as easy to understand and semantically understandable as get_linear_schedule() which is very clear.
