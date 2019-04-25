from environment import Environment
from agent import Agent, AgentManager

from constants import REWARD_GOAL

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
    am = AgentManager(5)
    print am.global_policy


if __name__ == "__main__":
    demonstrate_async_speedup()
