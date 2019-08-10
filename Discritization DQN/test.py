# Osman Furkan Kınlı - S002969
# Department of Computer Science @ Özyeğin University
# CS545 Homework-2: Discretization for Deep Q-Learning

from DQNAgent import DQNAgent
from ReplayDQNAgent import ReplayDQNAgent
from ReplayTargetDQNAgent import ReplayTargetDQNAgent
import os
import numpy as np

RUN = 10
EPISODES = 100
MAX_STEP = 500

if __name__ == '__main__':
    vanilla_agent = DQNAgent()
    vanilla_agent.load(os.path.join("./results", "Furkan_Kinli_1.h5f"))

    replay_agent = ReplayDQNAgent()
    replay_agent.load(os.path.join("./results", "Furkan_Kinli_2.h5f"))

    replay_target_agent = ReplayTargetDQNAgent()
    replay_target_agent.load(os.path.join("./results", "Furkan_Kinli_3.h5f"))
    replay_target_agent.update_target_model()

    vanilla_results, replay_results, replay_target_results = [], [], []

    for i in range(RUN):
        v_avg, v_max = vanilla_agent.test()
        vanilla_results.append((v_avg, v_max))
        print("Vanilla DQN: \n\tEpisode-{}: Avg: {} - Max: {}".format(i, v_avg, v_max))

        r_avg, r_max = replay_agent.test()
        replay_results.append((r_avg, r_max))
        print("Experience Replay DQN: \n\tEpisode-{}: Avg: {} - Max: {}".format(i, r_avg, r_max))

        r_t_avg, r_t_max = replay_target_agent.test()
        replay_target_results.append((r_t_avg, r_t_max))
        print("Experience Replay with Target Network DQN: \n"
              "\tEpisode-{}: Avg: {} - Max: {}".format(i, r_t_avg, r_t_max))

    print("For vanilla agent: \n "
          "\t Avg. reward after 10 runs: {} \n"
          "\t Max. reward after 10 runs: {} \n"
          "\n For experience replay agent: \n"
          "\t Avg. reward after 10 runs: {} \n"
          "\t Max. reward after 10 runs: {} \n"
          "\n For experience replay with target network agent: \n"
          "\t Avg. reward after 10 runs: {} \n"
          "\t Max. reward after 10 runs: {}".format(np.mean(np.asarray([r[0] for r in vanilla_results])),
                                                    np.max(np.asarray([r[1] for r in vanilla_results])),
                                                    np.mean(np.asarray([r[0] for r in replay_results])),
                                                    np.max(np.asarray([r[1] for r in replay_results])),
                                                    np.mean(np.asarray([r[0] for r in replay_target_results])),
                                                    np.max(np.asarray([r[1] for r in replay_target_results]))))
