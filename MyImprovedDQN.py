# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Flatten, Input
from keras.optimizers import Adam, Adagrad
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from functools import reduce


class MLP:
    def __init__(self, input_size, output_size, hidden_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

    def nn(self):
        model = Sequential()
        model.add(Dense(self.hidden_size, input_dim=self.input_size, activation='relu'))
        model.add(Dense(self.hidden_size, activation='relu'))
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

        self.hidden_size = 24
        self.model = self.build_model()

        self.memory = ReplyMemory(2000)

        # self.tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    def build_model(self):
        model = MLP(self.state_size, self.action_size, self.hidden_size).nn()
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
        """
        Invoke every episode.
        :param batch_size:
        :return:
        """
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
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return np.mean(history_loss)


def run_cartpole():
    alpha, gamma, epsilon, epsilon_decay, epsilon_min, penalty = 0.001, 0.95, 1.0, 0.995, 0.01, -10
    episodes, step = 1000, 20000
    avg_loss = []
    episode_reward = []
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    state_size = env.observation_space.shape[0]
    print(state_size)
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size, alpha=alpha, gamma=gamma, epsilon=epsilon,
                     epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    batch_size = 64

    for i_episode in range(episodes):
        state = env.reset()
        sum_reward = 0
        state = np.reshape(state, [1, state_size])
        for t in range(step):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            sum_reward += reward
            reward = reward if not done else -100
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, epsilon: {:.2}".format(i_episode, episodes, t, agent.epsilon), end='')
                break
        avg_loss.append(agent.replay(batch_size))
        episode_reward.append(sum_reward)
        print(", avg_loss: {:.2f}, reward: {:.2f}".format(avg_loss[i_episode], episode_reward[i_episode]))

    # Get training and test loss histories

    # Create count of the number of epochs
    # Visualize loss history
    print(avg_loss)
    plt.plot(avg_loss)
    plt.legend(['Training Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def main():
    run_cartpole()


if __name__ == "__main__":
    main()
