import numpy as np

from constants import ACTIONS, ACTION_LABELS
from constants import GRID_HEIGHT, GRID_WIDTH, START

GAMMA = 0.95

EPSILON = 0.25
ALPHA = 0.3


class Agent:

    def __init__(self, environment):
        self.environment = environment

        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.alpha = ALPHA

        self.num_actions = len(ACTIONS)
        self.action_labels = ACTION_LABELS
        self.actions = ACTIONS

        self.state = START
        self.policy = np.zeros((GRID_HEIGHT, GRID_WIDTH, len(ACTIONS)))

    def step(self):
        choice = self._choose_action()
        action = self.action_labels[choice]
        updated_state, reward = self.environment.step(action)
        self.update_policy(choice, updated_state, reward)
        self.state = updated_state
        return reward

    def _choose_action(self):
        """
        Helper function to choose the next action based on a greedy-epsilon policy
        :return: choice of action
        """
        random_action = np.random.randint(self.num_actions)
        best_action = np.argmax(self.policy[self.state[0], self.state[1]])

        choice = np.random.choice([random_action, best_action], p=[self.epsilon, 1 - self.epsilon])
        return choice

    def update_policy(self, choice, updated_state, reward):
        """
        Helper function to update the policy based on the given reward
        Q(s,a) <- Q(s,a) + alpha * [reward + gamma * max_a' Q(s', a') - Q(s,a)]
        :param choice: "a" choice of action
        :param updated_state: "s'" new state after action
        :param reward: reward of action
        :return: None
        """
        next_expected_reward = np.max(self.policy[updated_state[0], updated_state[1]])
        current_policy = self.policy[self.state[0], self.state[1], choice]
        self.policy[self.state[0], self.state[1], choice] += self.alpha * (
                    reward + self.gamma * next_expected_reward - current_policy)
