# Osman Furkan Kınlı - S002969
# Department of Computer Science @ Özyeğin University
# CS545 Homework-2: Discretization for Deep Q-Learning

import gym
import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, optimizers

"""
    This class is the base class for deep Q-learning agent.
    Environment: Mountain Car Continuous (version-0)
    Discretization level: 9
    Action space: [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    Model architecture:
        # Fully connected layer with 128 units
        # Rectified Linear Unit activation
        # Fully connected layer with 32 units
        # Rectified Linear Unit activation
        # Output layer with 9 units linearly activated
    Loss: Huber loss || Mean-squared error
    Optimizer: Adam
    Learning rate: 1e-3
"""


class Agent:
    def __init__(self):
        self.env = gym.make('MountainCarContinuous-v0')
        self.state_size = self.env.observation_space.shape[0]
        self.step_size = 9
        self.d_action_space = np.linspace(-1, 1, self.step_size)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self, name="model"):
        model = models.Sequential(name=name)
        model.add(layers.Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(self.step_size, activation='linear'))
        model.summary()
        model.compile(loss="mse", optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.d_action_space)
        else:
            action = self.d_action_space[np.argmax(self.model.predict(state)[0])]

        return [action]

    def test(self, num_episode=100, max_step=500, plot_file=None):
        reward_lst = []
        for i in range(num_episode):
            state = self.env.reset()
            state = np.reshape(state, newshape=[1, self.state_size])
            total_reward = 0

            for time in range(max_step):
                self.env.render()

                action = self.act(state)

                _, reward, done, _ = self.env.step(action)
                total_reward += reward

                if done:
                    print("Episode {}/{}, Score: {}".format(i + 1, num_episode, time))
                    break

            reward_lst.append(total_reward)

            print("Average reward in {}.episode: {:.3f} \t"
                  "Maximum reward so far: {:.3f}".format(i + 1, np.mean(reward_lst), np.max(reward_lst)))

        if plot_file is not None:
            self.plot(reward_lst, plot_file)

        self.close()

        return np.mean(reward_lst), np.max(reward_lst)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def plot(self, rewards, to_file):
        plt.plot(rewards)
        plt.savefig(to_file)

    def load(self, file):
        self.model.load_weights(file)

    def save(self, to_file):
        self.model.save_weights(to_file)

    def close(self):
        self.env.close()
