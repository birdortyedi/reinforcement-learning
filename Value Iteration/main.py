# Osman Furkan Kınlı - S002969
# Department of Computer Science @ Özyeğin University
# CS545 Homework-1: Value Iteration

from Environment import Environment
import os
import numpy as np
import pickle as pkl


def value_iter(envr, gamma=1.0, epsilon=0.01):
    # To initialize values arbitrarily (e.g zeros)
    values = np.zeros((envr.grid_size, envr.grid_size))

    while True:
        delta = 0
        # For each state in the grid
        for r in range(envr.grid_size):
            for c in range(envr.grid_size):
                v = values[r][c]
                vs = []

                # Find the action that has the maximum value
                for action in [0, 1, 2, 3]:
                    envr.set_current_state(r * envr.grid_size + c)
                    new_state, reward, _ = envr.move(action)

                    # Formula:
                    # new_v of that action in that state: reward + discount factor * current v of new state
                    v_ = reward + gamma * values[new_state // envr.grid_size][new_state % envr.grid_size]
                    vs.append(v_)

                # Assign the maximum one to the value
                values[r][c] = max(vs)
                # Calculate the amount of change in values
                error = abs(v-values[r][c])
                # Assign delta of system with a certain condition
                delta = max(delta, error)

        # print(values)
        if delta < epsilon: # Stop condition
            break

    return values

if __name__ == '__main__':
    # list of files in the current directory
    file_list = os.listdir(os.getcwd())
    # To find the files starting with "Grid"
    files = [file for file in file_list if file.startswith("Grid")]

    values_list = dict()

    for f in files:
        envr = Environment(f)
        values = value_iter(envr)
        values_list[f] = values

    # print(values_list)

    # To save the solutions
    with open('Solutions.pkl', 'wb') as f:
        pkl.dump(values_list, f, protocol=pkl.HIGHEST_PROTOCOL)

    # # To read the solutions
    # with open('Solutions.pkl', 'rb') as f:
    #     values_list = pkl.load(f)
    # print(values_list)

