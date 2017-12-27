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


class DQNAgent:
    def __init__(self, state_size, action_size, alpha, gamma, epsilon, epsilon_decay, epsilon_min):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma    # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha  # learning rate
        self.model = self.build_model()
        self.history_loss = []

        self.tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                                      write_graph=True, write_images=True)

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(16, (8, 8), subsample=(4, 4), input_shape=(1, 1, self.state_size),
                         border_mode='same', activation='relu'))
        model.add(Conv2D(32, (4, 4), subsample=(2, 2), border_mode='same', activation='relu'))
        #model.add(Conv2D(32, (3, 3), subsample=(1, 1), border_mode='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

        print(model.summary())
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            history = self.model.fit(state, target_f, epochs=1, verbose=0)  # feeds input and output pairs to the model.
            self.history_loss.extend(history.history['loss'])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def run_cartpole():
    alpha, gamma, epsilon, epsilon_decay, epsilon_min, penalty = 0.001, 0.95, 1.0, 0.995, 0.01, -10
    episodes, step = 1000, 20000
    env = gym.make('CartPole-v1')
    env = env.unwrapped
    state_size = env.observation_space.shape[0]
    print(state_size)
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size, alpha=alpha, gamma=gamma, epsilon=epsilon,
                     epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    batch_size = 32

    for i_episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, 1, 1, state_size])
        for t in range(step):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, 1, 1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, epsilon: {:.2}".format(i_episode, episodes, t, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    # Get training and test loss histories
    training_loss = agent.history_loss

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.legend(['Training Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def main():
    run_cartpole()


if __name__ == "__main__":
    main()
