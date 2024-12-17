import torch
import math


class Healer:
    def __init__(self):
        pass

    def __call__(self, grid):
        self.heal(grid)

    def heal(self, grid):
        pass


# "Initial conditions"
# flow both the periodic and
# aperiodic network states with unidirectional velocity inputs,
# corresponding to a velocity of 0.8 m/s, in three different directions
# (0,pi/5,pi/2-pi/5) for 250 ms each to heal any strain and defects in
# the formed pattern.
class BurakHealer(Healer):
    def __init__(self, heal_directions=None, device=None):
        if heal_directions is None:
            self.heal_directions = [
                torch.tensor([0.8, 0], device=device, dtype=torch.float32),
                torch.tensor(
                    [0.8 * math.cos(torch.pi / 5), 0.8 * math.sin(torch.pi / 5)],
                    device=device,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    [
                        0.8 * math.cos(torch.pi / 2 - math.pi / 5),
                        0.8 * math.sin(torch.pi / 2 - math.pi / 5),
                    ],
                    device=device,
                    dtype=torch.float32,
                ),
            ]
        else:
            self.heal_directions = heal_directions

    def heal(self, grid):
        dt = grid.dt
        steps = int(250 // dt)

        for direction in self.heal_directions:
            for i in range(steps):
                grid.step(direction)
