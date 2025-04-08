# Add memory buffer
import torch
import random
class MMbuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0

    def add(self, state, action, reward, next_state, log_probability, done):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(torch.cat((state.view(-1), action.view(-1), reward.view(-1), next_state.view(-1), log_probability.view(-1), done.view(-1))))
        else:
            self.buffer[self.index] = (torch.cat((state.view(-1), action.view(-1), reward.view(-1), next_state.view(-1), log_probability.view(-1), done.view(-1))))
        self.index = (self.index + 1) % self.buffer_size

    def getmemories(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)
    
    def reset(self):
        self.buffer = []
        self.index = 0