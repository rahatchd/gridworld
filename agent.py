from multiprocessing import Process, Value, Array, Queue, Lock

import numpy as np

from environment import Environment

from constants import ACTIONS, ACTION_LABELS
from constants import GRID_HEIGHT, GRID_WIDTH, START
from constants import REWARD_GOAL
from constants import STOP

GAMMA = 0.95

EPSILON = 0.25
ALPHA = 0.2

I_ASYNC_UPDATE = 5
T_MAX = 4 * 10 ** 4


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
        self._update_policy(choice, updated_state, reward)
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

    def _update_policy(self, choice, updated_state, reward):
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


class AsyncAgent(Agent):

    def __init__(self, environment, lock, global_policy, global_step_num):
        Agent.__init__(self, environment)
        self.lock = lock
        self.global_policy = global_policy
        self.global_step_num = global_step_num

        self.step_num = 0
        self.async_update = I_ASYNC_UPDATE

    def _choose_action(self):
        random_action = np.random.randint(self.num_actions)
        best_action = np.argmax(self.global_policy[self.state[0], self.state[1]])

        choice = np.random.choice([random_action, best_action], p=[self.epsilon, 1 - self.epsilon])
        return choice

    def _update_policy(self, choice, updated_state, reward):
        next_expected_reward = np.max(self.global_policy[updated_state[0], updated_state[1]])
        current_policy = self.global_policy[self.state[0], self.state[1], choice]
        self.policy[self.state[0], self.state[1], choice] += reward + self.gamma * next_expected_reward - current_policy

    def step(self):
        reward = Agent.step(self)
        self.step_num += 1
        if (self.step_num % self.async_update) == 0:
            with self.lock:
                self.global_step_num.value = self.global_step_num.value + self.async_update
                self.global_policy += self.alpha * self.policy
            self.policy.fill(0)
        return reward


class AgentManager:

    def __init__(self, num_agents):
        self.num_agents = num_agents

        self.lock = Lock()
        self.ep_lock = Lock()
        global_policy = Array('d', GRID_HEIGHT * GRID_WIDTH * len(ACTIONS), lock=self.lock)
        self.global_policy = np.frombuffer(global_policy.get_obj(), dtype='d').reshape(GRID_HEIGHT,
                                                                                       GRID_WIDTH,
                                                                                       len(ACTIONS))
        self.global_step_num = Value('i', 0)
        self.global_step_max = T_MAX
        self.episodes = Queue()

        self.processes = [Process(target=self.instantiate_agent) for _ in range(num_agents)]
        for process in self.processes:
            process.start()
        for process in self.processes:
            process.join()
        self.episodes.put(STOP)

    def instantiate_agent(self):
        env = Environment()
        agent = AsyncAgent(env, self.lock, self.global_policy, self.global_step_num)

        episode_length = 0
        while self.global_step_num.value < self.global_step_max:
            reward = agent.step()
            episode_length += 1
            if reward == REWARD_GOAL:
                with self.ep_lock:
                    self.episodes.put(episode_length)
                episode_length = 0
