import numpy as np
from environment import Environment

from constants import REWARD_GOAL, REWARD_NON_GOAL


def test_update_state():
    env = Environment()
    env._update_state("up")
    # test state update on move
    assert np.all([4, 3] == env.state)
    # test state update on obstacle
    env._update_state("up")
    assert np.all([4, 3] == env.state)

    env = Environment()
    env._update_state("down")
    # test state update on boundary
    assert np.all([5, 3] == env.state)


def test_reward():
    env = Environment()
    _, reward = env.step("up")
    assert reward == REWARD_NON_GOAL

    for _ in range(5):
        _, reward = env.step("right")
        assert reward == REWARD_NON_GOAL
    for _ in range(3):
        _, reward = env.step("up")
        assert reward == REWARD_NON_GOAL

    state, reward = env.step("up")
    assert reward == REWARD_GOAL

    # also test reset
    assert np.all([5, 3] == env.state)
