import gymnasium as gym
import torch
import numpy

distribution = torch.tensor([0.1,0.5,0.4])
a = torch.distributions.Categorical(distribution)

print(a.sample())