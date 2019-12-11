import numpy as np


class her_replay_buffer(object):
    def __init__(self, num_steps, input_shape, n_actions):
        self.mem_size = num_steps
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def populate_her_buffer(self, index, state, action, reward, state_, done):
        #index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

    def get_params(self, index):
        return self.state_memory[index], self.action_memory[index], self.reward_memory[index], self.new_state_memory[index], self.terminal_memory[index]







