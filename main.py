from environment import Environment
from agent import Agent

import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = Environment()
    agent = Agent(env)

    episodes = []

    episode_length = 0
    for n in range(4 * 10 ** 4):
        reward = agent.step()
        episode_length += 1
        if reward == 1:
            episodes.append(episode_length)
            episode_length = 0

    plt.plot(episodes)
    plt.show()
