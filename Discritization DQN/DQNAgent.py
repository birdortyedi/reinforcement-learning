# Osman Furkan Kınlı - S002969
# Department of Computer Science @ Özyeğin University
# CS545 Homework-2: Discretization for Deep Q-Learning

from Agent import Agent
import numpy as np
import os

"""
    This class is Vanilla deep Q-learning agent. (DQNAgent)
"""


class DQNAgent(Agent):
    def __init__(self):
        self.name = "Furkan_Kinli_1.h5f"
        super(DQNAgent, self).__init__()

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

                self.learn(state, action, next_state, reward, done)

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

    def learn(self, state, action, next_state, reward, done):
        out = self.model.predict(state)

        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

        out[0][np.where(self.d_action_space == action)] = target
        self.model.fit(state, out, epochs=3, verbose=0)
