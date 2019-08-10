# Osman Furkan Kınlı - S002969
# Department of Computer Science @ Özyeğin University
# CS545 Homework-2: Discretization for Deep Q-Learning

from DQNAgent import DQNAgent
from collections import deque
import numpy as np
import random
import os

"""
    This class is Experience Replay deep Q-Learning agent. (ReplayDQNAgent)
    
    Memory: 2000 experience at one level
    Start iter: After some experience, ready to sample and train
"""


class ReplayDQNAgent(DQNAgent):
    def __init__(self):
        self.name = "Furkan_Kinli_2.h5f"
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        super(ReplayDQNAgent, self).__init__()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        batch_x = random.sample(self.memory, self.batch_size)
        replay_x = np.zeros((self.batch_size, self.state_size))
        replay_y = np.zeros((self.batch_size, self.step_size))

        for i, (state, action, reward, next_state, done) in enumerate(batch_x):
            out = self.model.predict(state)[0]

            if not done:
                out[np.where(self.d_action_space == action)] = \
                    reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            else:
                out[np.where(self.d_action_space == action)] = reward

            replay_x[i] = state
            replay_y[i] = out

        self.model.fit(replay_x, replay_y, epochs=1, verbose=0)

    def train(self, num_episode, max_step, plot_file):
        reward_lst = []
        for i in range(num_episode):
            state = self.env.reset()
            state = np.reshape(state, newshape=[1, self.state_size])
            total_reward = 0

            for time in range(max_step):
                self.env.render()

                action = self.act(state)

                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, newshape=[1, self.state_size])
                total_reward += reward

                self.remember(state, action, reward, next_state, done)

                if len(self.memory) > self.batch_size:
                    self.replay()

                state = next_state

                if done:
                    print("Episode {}/{}, Score: {}".format(i+1, num_episode, time))
                    break

            self.update_epsilon()
            reward_lst.append(total_reward)

            print("Average reward in {}.episode: {:.3f} - {:.3f} \t"
                  "Epsilon: {:.3f} \t"
                  "Maximum reward so far: {:.3f}".format(i + 1, np.average(reward_lst), np.mean(reward_lst),
                                                         self.epsilon, np.max(reward_lst)))

        self.plot(reward_lst, plot_file)
        self.save(os.path.join("./results", self.name))
        self.close()
