from nd_scaffold import *
from typing import Tuple
import matplotlib
from matplotlib import pyplot as plt


UP = torch.tensor([0, 1])
DOWN = torch.tensor([0, -1])
LEFT = torch.tensor([-1, 0])
RIGHT = torch.tensor([1, 0])
# ROTATE_LEFT = torch.tensor([0, 0, 1])
# ROTATE_RIGHT = torch.tensor([0, 0, -1])


def get_action():
    a = None
    while a is None:
        action = input(
            "Enter 'w' to move up, 's' to move down, "
            "'a' to move left, 'd' to move right, 'q' to move forward, "
            "'q' to rotate left, 'e' to rotate right, 'quit' to quit: "
        )
        if action == "quit":
            return None
        elif action == "w":
            a = UP
        elif action == "s":
            a = DOWN
        elif action == "a":
            a = LEFT
        elif action == "d":
            a = RIGHT
        # elif action == 'q':
        #   a = ROTATE_LEFT
        # elif action == 'e':
        #   a = ROTATE_RIGHT
        else:
            print("Invalid action, type quit to exit")

    return a
