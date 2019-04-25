import numpy as np

from constants import ACTIONS
from constants import GRID_HEIGHT, GRID_WIDTH, START

from constants import REWARD_GOAL, REWARD_NON_GOAL

GOAL = np.array([0, 8], dtype=int)

INITIAL_OBSTACLES = [np.array([3, i], dtype=int) for i in range(0, 8)]
NEW_OBSTACLES = [np.array([3, i], dtype=int) for i in range(1, 9)]
OBSTACLE_CHANGE = 8000



class Environment:

    def __init__(self):
        self.height = GRID_HEIGHT
        self.width = GRID_WIDTH

        self.start = START
        self.goal = GOAL

        self.actions = ACTIONS

        self.obstacles = INITIAL_OBSTACLES
        self.obstacle_flag = False

        self.state = START
        self.step_num = 0

    def step(self, action):
        """
        Step function to provide reward to agent and update its state given an action
        Reward is 0 for non-goal tiles, 1 for goal-tile
        :param action: {"up", "down", "left", "right"} action chosen by agent
        :return: updated_state, reward
        """
        # introduce obstacle change
        if not self.obstacle_flag and self.step_num > OBSTACLE_CHANGE:
            self.obstacles = NEW_OBSTACLES
            self.obstacle_flag = True
        self.step_num += 1

        self._update_state(action)
        # if agent reaches the goal then reinitialize it
        if np.all(self.state == self.goal):
            updated_state = np.copy(self.start)
            reward = REWARD_GOAL
        else:
            updated_state = np.copy(self.state)
            reward = REWARD_NON_GOAL
        self.state = updated_state
        return updated_state, reward

    def _update_state(self, action_direction):
        """
        Helper function to update agent state
        If action is not allowed (outside bounds or obstacle) then the state is not updated
        :param action_direction: {"up", "down", "left", "right"} proposed action direction
        :return: None
        """
        desired_state = self.state + self.actions[action_direction]
        if 0 <= desired_state[0] < self.height \
                and 0 <= desired_state[1] < self.width \
                and list(desired_state) not in [list(obstacle) for obstacle in self.obstacles]:
            self.state = desired_state
