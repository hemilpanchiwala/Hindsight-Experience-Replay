import numpy as np


class ExperienceReplayMemory(object):
    def __init__(self, memory_size, input_dims):
        super(ExperienceReplayMemory, self).__init__()
        self.max_mem_size = memory_size
        self.counter = 0

        # initializes the state, next_state, action, reward, and terminal experience memory
        print(type(input_dims))
        self.state_memory = np.zeros((memory_size, *input_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((memory_size, *input_dims), dtype=np.float32)
        self.reward_memory = np.zeros(memory_size, dtype=np.float32)
        self.action_memory = np.zeros(memory_size, dtype=np.int64)
        self.terminal_memory = np.zeros(memory_size, dtype=bool)

    def add_experience(self, state, action, reward, next_state, done):
        """
        Adds new experience to the memory.
        """
        curr_index = self.counter % self.max_mem_size

        self.state_memory[curr_index] = state
        self.action_memory[curr_index] = action
        self.reward_memory[curr_index] = reward
        self.next_state_memory[curr_index] = next_state
        self.terminal_memory[curr_index] = done

        self.counter += 1

    def get_random_experience(self, batch_size):
        """
        Returns any random memory from the experience replay memory.
        """
        rand_index = np.random.choice(min(self.counter, self.max_mem_size), batch_size)

        rand_state = self.state_memory[rand_index]
        rand_action = self.action_memory[rand_index]
        rand_reward = self.reward_memory[rand_index]
        rand_next_state = self.next_state_memory[rand_index]
        rand_done = self.terminal_memory[rand_index]

        return rand_state, rand_action, rand_reward, rand_next_state, rand_done
