from environment import Environment
from agent import Agent, AgentManager

from constants import REWARD_GOAL
from constants import STOP

import matplotlib.pyplot as plt


def demonstrate_learning():
    env = Environment()
    agent = Agent(env)

    episodes = []

    episode_length = 0
    for n in range(1 * 10 ** 4):
        reward = agent.step()
        episode_length += 1
        if reward == REWARD_GOAL:
            episodes.append(episode_length)
            episode_length = 0

    plt.plot(episodes)
    plt.title('Q-Learning Results')
    plt.xlabel('Episode Number')
    plt.ylabel('Steps to Goal')
    plt.show()


def demonstrate_adaptation():
    env = Environment()
    agent = Agent(env)

    episodes = []

    episode_length = 0
    for n in range(4 * 10 ** 4):
        reward = agent.step()
        episode_length += 1
        if reward == REWARD_GOAL:
            episodes.append(episode_length)
            episode_length = 0

    plt.plot(episodes)
    plt.title('Q-Learning Adaptation to Obstacle Change')
    plt.xlabel('Episode Number')
    plt.ylabel('Steps to Goal')
    plt.show()


def demonstrate_async_speedup():
    manager = AgentManager(1)
    episodes = []
    for i in iter(manager.episodes.get, STOP):
        episodes.append(i)

    manager = AgentManager(5)
    episodes1 = []
    for i in iter(manager.episodes.get, STOP):
        episodes1.append(i)

    manager = AgentManager(10)
    episodes2 = []
    for i in iter(manager.episodes.get, STOP):
        episodes2.append(i)

    plt.subplot(311)
    plt.plot(episodes)
    plt.title('Async Q-Learning (y-axis is steps to goal)')
    plt.ylabel('1 Agent')
    plt.subplot(312)
    plt.plot(episodes1)
    plt.ylabel('5 Agents')
    plt.subplot(313)
    plt.plot(episodes2)
    plt.ylabel('10 Agents')
    plt.xlabel('Episode Number')
    plt.show()


if __name__ == "__main__":
    demonstrate_async_speedup()
