import numpy as np

GRID_HEIGHT = 6
GRID_WIDTH = 9

START = np.array([5, 3], dtype=int)

ACTION_LABELS = ["up", "down", "left", "right"]
ACTIONS = {
    "up": np.array([-1, 0], dtype=int),
    "down": np.array([1, 0], dtype=int),
    "left": np.array([0, -1], dtype=int),
    "right": np.array([0, 1], dtype=int)
}

REWARD_GOAL = 1
REWARD_NON_GOAL = 0

STOP = 'STOP'
