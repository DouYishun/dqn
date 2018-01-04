import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.wrappers import Monitor
import DQN
import DDQN


def my_test(agent, env, episodes, step):
    history_reward = []
    for i_episode in range(episodes):
        sum_reward = 0
        state = env.reset()
        state = np.reshape(state, [1, agent.state_size])
        for t in range(step):
            action = np.argmax(agent.model.predict(state)[0])
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_size])
            sum_reward += reward
            state = next_state
            if done or t == step - 1:
                print("episode: {}/{}, score: {}".format(i_episode, episodes, t))
                break
        history_reward.append(sum_reward)

    plot_reward(history_reward)


def plot_reward(history_reward):
    num = len(history_reward) // 10
    history_reward = history_reward[:num * 10]
    splited_reward = [history_reward[i:i + 10] for i in range(0, len(history_reward), 10)]
    mean = np.mean(splited_reward, axis=1)
    std = np.std(splited_reward, axis=1)
    x = list(range(num))
    plt.errorbar(x, mean, std, linestyle='None', marker='o')
    plt.show()


def test_cartpole():
    dqn_model_filename = "model/dqn_pro.h5"
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    env = Monitor(env, "video", force=True)
    episodes = 30
    step = 20000

    agent = DQN.Cartpole().agent
    agent.load(dqn_model_filename)

    my_test(agent, env, episodes, step)


def main():
    test_cartpole()


if __name__ == '__main__':
    main()

