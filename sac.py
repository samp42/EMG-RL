import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from tqdm import tqdm

class ReplayBuffer():
    def __init__(self, buffer_size=1000000):
        self.buffer_size = buffer_size
        self.buffer = []
        self.position = 0

    def add(self, transition):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.buffer_size

    def sample(self, batch_size):
        return np.random.choice(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class SAC():
    def __init__(self, env_id="Pendulum-v1", num_envs=8, horizon=2048, epochs=10, mini_batch_size=256,):
        pass

    def train(self):
        pass

    def learn(self):
        pass



if __name__ == "__main__":
    sac = SAC()
    sac.train()
