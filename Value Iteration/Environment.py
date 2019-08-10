import pickle as pkl
import copy


class Environment:
    def __init__(self, grid_file):
        """
        The constructor of this class takes the name of grid_file as input and loads the file.
        Afterwards, it extracts the corresponding data from the file. The file contains a dictionary
        object which include the grid and the start position. Grid is a 2d String list and start position
        is a 1D integer list with size 2. You need to give the absolute or the relative path of the file.
        The files are included in the zip folder on LMS. To learn more about loading and saving python objects,
        see the python library called "pickle"

        Args:
            grid_file (String): The absolute/relative path of the file containing the grid
        """
        _f = open(grid_file, "rb")
        _grid_data = pkl.load(_f)
        self.grid = _grid_data["grid"]
        self.start = _grid_data["start"]

        self.grid_size = len(self.grid)
        self.limits = [0, self.grid_size - 1]
        self.current_position = copy.deepcopy(self.start)

    def reset(self):
        """
        Resets the environment To the starting position
        """
        self.current_position = copy.deepcopy(self.start)
        return self._to_state(self.current_position)

    def _to_state(self, position):
        """
        DO NOT CALL THIS FUNCTION
        This function gets the position coordinate as input and returns it in terms of an integer.

        Args:
            position(list): Current position of the agent
        Returns:
            state(int): The index of the position
        """
        return position[0] * self.grid_size + position[1]

    def set_current_state(self, state_index):
        """
        This function takes the index of a state as input and sets the current state of the system with
        that index.

        Args:
            state_id(int): Position to set
        """
        state_row = state_index // self.grid_size
        state_col = state_index % self.grid_size
        self.current_position = [state_row, state_col]

    def move(self, action):
        """
        This function gets your action as input and checks whether your action is vertical or
        horizontal and calls the corresponding function. The actions are defined as:
        UP = 0
        LEFT = 1
        DOWN = 2
        RIGHT = 3

        Args:
            action (int): action taken
        Returns:
            state(int) : New state formed after the action is taken
            reward (int): Corresponding reward if action is legal else 0
            done(bool): If the current episode is done or not
        """
        if self.grid[self.current_position[0]][self.current_position[1]] == "G":
            return self._to_state(self.current_position), 0, True
        if action in [0, 2]:
            return self._move_vertical(action)
        else:
            return self._move_horizontal(action)

    def _move_vertical(self, action):
        """
        DON'T CALL THIS FUNCTION
        The function that controls the vertical actions for the environment. It first checks
        whether the action is a legal action or not(Wall collusions) and then executes the move.
        If the action taken is illegal, nothing happens and the systems returns the current state
        New Position is calculated as [old_position_x + (action-1), old_position_y]
        The (action -1) comes from the fact that actions are 0 and 2 and when you subtract 1 from
        them, they become -1 and 1 which is incrementation for vertical position.

        Args:
            action (int): action taken
        Returns:
            state(int) : New state the agent is in after the action is taken
            reward (int): Corresponding reward if action is legal else 0
            done(bool): If the current episode is done or not
        """
        curr_y = self.current_position[0]
        if (curr_y == 0 and action == 0) or (curr_y == self.grid_size - 1 and action == 2):
            return self._to_state(self.current_position), -1, False
        else:
            new_position = [self.current_position[0] + (action - 1), self.current_position[1]]
            rew, done = self._get_reward(self.current_position, new_position)
            self.current_position = new_position
            if done:
                final_position = copy.deepcopy(self.current_position)
                self.reset()
                return self._to_state(final_position), rew, done
            return self._to_state(self.current_position), rew, done

    def _move_horizontal(self, action):
        """
        DON'T CALL THIS FUNCTION
        See details of _move_vertical(self, action) function for more information. The only difference
        is that this controls the horizontal move and since left is 1 and right is 3, we calculate
        (action - 2) instead of action -1 for horizontal incrementation -1 and 1.
        Args:
            action (int): action taken
        Returns:
            state(int) : New state formed after the action is taken
            reward (int): Corresponding reward if action is legal else 0
            done(bool): If the current episode is done or not
        """

        curr_x = self.current_position[1]
        if (curr_x == 0 and action == 1) or (curr_x == self.grid_size - 1 and action == 3):
            return self._to_state(self.current_position), -1, False
        else:
            new_position = [self.current_position[0], self.current_position[1] + (action - 2)]
            rew, done = self._get_reward(self.current_position, new_position)
            self.current_position = new_position
            if done:
                final_position = copy.deepcopy(self.current_position)
                self.reset()
                return self._to_state(final_position), rew, done
            return self._to_state(self.current_position), rew, done

    def _get_reward(self, p_p, n_p):
        """
        DON'T CALL THIS FUNCTION
        Function that returns the reward generated from the coordinate formed after the
        action is taken. The correlation is as follows:

        Floor -> Floor = -1
        Floor -> Mountain = -3
        Mountain -> Mountain = -2
        Mountain -> Floor = -1
        (Any) -> Goal = 10

        Args:
            p_p (list): previous coordinate
            n_p (list): New coordinate after the action

        Returns:
            int: The reward
        """
        if self.grid[n_p[0]][n_p[1]] == "G":
            return 10, True
        if self.grid[n_p[0]][n_p[1]] == self.grid[p_p[0]][p_p[1]]:
            return (-1, False) if self.grid[n_p[0]][n_p[1]] == "F" else (-2, False)
        if self.grid[n_p[0]][n_p[1]] == "M" and self.grid[p_p[0]][p_p[1]] == "F":
            return -3, False
        else:
            return -1, False
