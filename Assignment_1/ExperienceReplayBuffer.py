import random
import numpy as np
from collections import deque

class ExperienceReplayBuffer():
    def __init__(self):
        self.max_capacity = 1000000
        self.batch_size = 128
        # self.memories = np.array([])
        self.memories = deque(maxlen = self.max_capacity)

    # Saves a trajectory to the buffer
    def save_trajectory(self, state0, action, reward, state1, terminal):
        # if len(self.memories) == self.max_capacity:
        #     self.memories[:-1] = self.memories[1:]
        #     self.memories[-1] = (state0, action, reward, state1, terminal)
        # else:
        #     self.memories = np.append(self.memories, (state0, action, reward, state1, terminal))
        self.memories.append((state0, action, reward, state1, terminal))
    
    # Returns a batch for training from the buffer
    def get_train_batch(self):
        if len(self.memories) < self.batch_size:
            return None
        batch = random.sample(self.memories, self.batch_size)

        
        state0_batch = []
        action_batch = []
        reward_batch = []
        state1_batch = []
        terminal_batch = []

        for trajetory in batch:
            state0_batch.append(trajetory[0])
            action_batch.append(trajetory[1])
            reward_batch.append(trajetory[2])
            state1_batch.append(trajetory[3])
            terminal_batch.append(trajetory[4])

        return np.array(state0_batch), np.array(action_batch), np.array(reward_batch), np.array(state1_batch), np.array(terminal_batch)
