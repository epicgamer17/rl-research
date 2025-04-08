import torch
from typing import Tuple
from torch import nn

class CFRNetwork(nn.Module):
    def __init__(self,
                 config=None,
                 input_shape=Tuple[int],
                 output_shape=int,
    ):
        self.config = config
        super(CFRNetwork, self).__init__()
        self.values = [PolicyNetwork(config=config["value"]) for _ in range(config["num_players"])]
        self.policy = ValueNetwork(config=config["policy"])

    
class ValueNetwork(nn.Module):
    def __init__(self,
                 config=None,
    ):
        self.config = config
        self.optimizer = self.config["optimizer"]
        super(ValueNetwork, self).__init__()
        self.layers = torch.nn.ModuleList([
            layer for layer in self.config.layers])
    
    def forward(self, x):
        """
        Forward pass through the network.
        :param x: Input tensor.
        :return: Output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def reset(self):
        """
        Reset the network parameters.
        """
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def learn(self, batch):
        """
        Given a batch of experiences, update the network params
        :param batch: Batch of experiences Tuple of form(iteration_num, state, regret).
        """
        self.optimizer.zero_grad()
        outputs = self.forward(batch[1])
        loss = (batch[0] * ((outputs - batch[2]) ** 2)).mean()
        loss.backward()
        self.optimizer.step()




class PolicyNetwork(nn.Module):
    def __init__(self,
                 config=None,
    ):
        self.config = config
        super(PolicyNetwork, self).__init__()
        self.layers = torch.nn.ModuleList([
            layer for layer in self.config.layers])
    
    def forward(self, x):
        """
        Forward pass through the network.
        :param x: Input tensor.
        :return: Output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def reset(self):
        """
        Reset the network parameters.
        """
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
