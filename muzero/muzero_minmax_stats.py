import collections
from typing import Optional

MAXIMUM_FLOAT_VALUE = float("inf")
KnownBounds = collections.namedtuple("KnownBounds", ["min", "max"])


class MinMaxStats(object):
    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.max = known_bounds.max if known_bounds else MAXIMUM_FLOAT_VALUE
        self.min = known_bounds.min if known_bounds else -MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.max = max(self.max, value)
        self.min = min(self.min, value)

    def normalize(self, value: float) -> float:
        if self.max > self.min:
            # We normalize only when we have at a max and min value
            return (value - self.min) / (self.max - self.min)
        return value

    def __repr__(self):
        return f"min: {self.min}, max: {self.max}"
