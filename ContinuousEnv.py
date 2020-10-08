import numpy as np


class ContinuousEnv:
    """
    A simple bit flip environment
    Bit of the current state flips as an action
    Reward of -1 for each step
    """
    def __init__(self, size):
        self.size = size
        self.state = size * (2 * np.random.random(2) - 1)
        self.goal = size * (2 * np.random.random(2) - 1)
        self.threshold = 0.5

    def reset_env(self):
        """
        Resets the environment with new state and goal
        """
        self.state = self.size * (2 * np.random.random(2) - 1)
        self.goal = self.size * (2 * np.random.random(2) - 1)

    def take_step(self, action):
        """
        Returns updated_state, reward, and done for the step taken
        """
        self.state += (action / 4)
        good_done = np.linalg.norm(self.goal) <= self.threshold
        bad_done = np.max(np.abs(self.state)) > self.size
        if good_done:
            reward = 0
        else:
            reward = -1
        return np.copy(self.state / self.size), reward, good_done or bad_done

    def print_state(self):
        """
        Prints the current state
        """
        print('Current State:', self.state)
