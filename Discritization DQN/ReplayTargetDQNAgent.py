# Osman Furkan Kınlı - S002969
# Department of Computer Science @ Özyeğin University
# CS545 Homework-2: Discretization for Deep Q-Learning

from ReplayDQNAgent import ReplayDQNAgent
import numpy as np
import random

"""
    This class is Experience Replay with Target Network deep Q-Learning agent. 
    (ReplayTargetDQNAgent)
    
    Memory: 1000 experience at one level
    Start iter: After 1000 experience, ready to sample and train
    Target model architecture: Same architecture with DQNAgent
"""


class ReplayTargetDQNAgent(ReplayDQNAgent):
    def __init__(self):
        super(ReplayTargetDQNAgent, self).__init__()
        self.name = "Furkan_Kinli_3.h5f"
        self.target_model = self._build_model(name="target")
        self.update_target_model()
        self.update_step = 10

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def replay(self):
        batch_x = random.sample(self.memory, self.batch_size)

        replay_x = np.zeros((self.batch_size, self.state_size))
        replay_y = np.zeros((self.batch_size, self.step_size))

        for i, (state, action, reward, next_state, done) in enumerate(batch_x):
            out = self.model.predict(state)[0]

            if not done:
                out[np.where(self.d_action_space == action)] = \
                    reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            else:
                out[np.where(self.d_action_space == action)] = reward

            replay_x[i] = state
            replay_y[i] = out

        self.model.fit(replay_x, replay_y, epochs=1, verbose=0)
