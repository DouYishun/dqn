# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def nn(self):
        model = Sequential()
        model.add(Dense(4, input_dim=self.input_size, activation='relu'))
        model.add(Dense(4, activation='relu'))
        model.add(Dense(self.output_size, activation='linear'))
        return model


class CNN:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    def model(self):
        model = Sequential()
        model.add(Conv2D(16, (8, 8), subsample=(4, 4), input_shape=(1, 1, self.input_size),
                         border_mode='same', activation='relu'))
        model.add(Conv2D(32, (4, 4), subsample=(2, 2), border_mode='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.output_size, activation='linear'))


class ReplyMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, state_size, action_size, alpha, gamma, epsilon, epsilon_decay, epsilon_min):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha  # learning rate
        self.model = self.build_model()

        self.memory = ReplyMemory(2000)

    def build_model(self):
        model = MLP(self.state_size, self.action_size).nn()
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        print(model.summary())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        batch_size = min(batch_size, self.memory.__len__())
        history_loss = []
        mini_batch = self.memory.sample(batch_size=batch_size)
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            history = self.model.fit(state, target_f, epochs=1, verbose=0)  # feeds input and output pairs to the model
            history_loss.extend(history.history['loss'])
        return np.mean(history_loss)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def run(agent, env, episodes, step, penalty):
    history_loss = []
    history_reward = []
    batch_size = 32
    for i_episode in range(episodes):
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        state = env.reset()
        sum_reward = 0
        state = np.reshape(state, [1, agent.state_size])
        for t in range(step):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            sum_reward += reward
            reward = reward if not done else penalty
            next_state = np.reshape(next_state, [1, agent.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            # agent.replay(batch_size)
            if done or t == step - 1:
                print("episode: {}/{}, score: {}, epsilon: {:.2}".format(i_episode, episodes, t, agent.epsilon), end='')
                break
        history_loss.append(agent.replay(batch_size))
        history_reward.append(sum_reward)
        print(", avg_loss: {:.2f}, reward: {:.2f}".format(history_loss[i_episode], history_reward[i_episode]))

        if i_episode % 50 == 0 and i_episode > 0:
            agent.save("./model/dqn.h5")

    plot(history_loss, history_reward)


def plot(history_loss, history_reward):
    num = len(history_loss) // 10

    history_loss = history_loss[:num * 10]
    splited_loss = [history_loss[i:i + 10] for i in range(0, len(history_loss), 10)]
    mean_loss = np.mean(splited_loss, axis=1)

    history_reward = history_reward[:num * 10]
    splited_reward = [history_reward[i:i + 10] for i in range(0, len(history_reward), 10)]
    mean_reward = np.mean(splited_reward, axis=1)

    x = [i * 10 for i in list(range(num))]

    plt.subplot(2, 1, 1)
    plt.plot(x, mean_loss, '.-')
    plt.title('Training Loss and Reward')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(x, mean_reward, '.-')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')

    plt.show()


class Cartpole:
    def __init__(self):
        self.episodes, self.step, self.penalty = 1000, 20000, -100
        self.env = gym.make('CartPole-v0')
        self.env = self.env.unwrapped

        alpha, gamma, epsilon, epsilon_decay, epsilon_min = 0.001, 0.95, 1.0, 0.995, 0.01
        self.agent = DQNAgent(state_size=self.env.observation_space.shape[0], action_size=self.env.action_space.n,
                              alpha=alpha, gamma=gamma, epsilon=epsilon,
                              epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)

    def run_(self):
        run(self.agent, self.env, self.episodes, self.step, self.penalty)


def main():
    cartpole = Cartpole()
    cartpole.run_()


if __name__ == "__main__":
    main()
