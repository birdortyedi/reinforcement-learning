# Osman Furkan Kınlı - S002969
# Department of Computer Science @ Özyeğin University
# CS545 Homework-2: Discretization for Deep Q-Learning

from DQNAgent import DQNAgent
from ReplayDQNAgent import ReplayDQNAgent
from ReplayTargetDQNAgent import ReplayTargetDQNAgent
import os

EPISODES_V = 150  # After 125, vanilla agent stucks on low-ground
EPISODES = 1000
MAX_STEP = 500


if __name__ == '__main__':
    vanilla_agent = DQNAgent()
    vanilla_agent.train(EPISODES_V, MAX_STEP, os.path.join("./results", "vanilla_rewards.png"))

    replay_agent = ReplayDQNAgent()
    replay_agent.train(EPISODES, MAX_STEP, os.path.join("./results", "replay_rewards.png"))

    replay_target_agent = ReplayTargetDQNAgent()
    replay_target_agent.train(EPISODES, MAX_STEP, os.path.join("./results", "replay__target_rewards.png"))
