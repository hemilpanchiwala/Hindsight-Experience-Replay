import numpy as np


class BitFlipEnv:
    """
    A simple bit flip environment
    Bit of the current state flips as an action
    Reward of -1 for each step
    """
    def __init__(self, n_bits):
        self.n_bits = n_bits
        self.state = np.random.randint(2, size=self.n_bits)
        self.goal = np.random.randint(2, size=self.n_bits)

    def reset_env(self):
        """
        Resets the environment with new state and goal
        """
        self.state = np.random.randint(2, size=self.n_bits)
        self.goal = np.random.randint(2, size=self.n_bits)

    def take_step(self, action):
        """
        Returns updated_state, reward, and done for the step taken
        """
        self.state[action] = self.state[action] ^ 1
        done = False
        if np.array_equal(self.state, self.goal):
            done = True
            reward = 0
        else:
            reward = -1
        return np.copy(self.state), reward, done

    def print_state(self):
        """
        Prints the current state
        """
        print('Current State:', self.state)
