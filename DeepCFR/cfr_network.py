import torch
from typing import Tuple
from torch import nn

class CFRNetwork(nn.Module):
    def __init__(self,
                 config=None,
                 input_shape=Tuple[int],
                 output_shape=int,
                 device=None,
    ):
        self.config = config
        super(CFRNetwork, self).__init__()
        self.values = [ValueNetwork(config=config["value"]) for _ in range(config["num_players"])]
        self.policy = PolicyNetwork(config=config["policy"])
        
    def forward(self, x):
        """
        Forward pass through the network.
        :param x: Input tensor.
        :return: Output tensor.
        """
        for i in range(len(self.layers)):
            if i<=3:
                x = torch.nn.functional.relu(self.layers[i](x))
            else:
                x = self.layers[i](x)
        return x
    
class ValueNetwork(nn.Module):
    def __init__(self,
                 config=None,
    ):
        self.config = config
        super(ValueNetwork, self).__init__()
        layer1 = torch.nn.Linear(self.config["input_shape"], self.config["hidden_size"])
        layer2 = torch.nn.Linear(self.config["hidden_size"], self.config["hidden_size"])
        layer3 = torch.nn.Linear(self.config["hidden_size"], self.config["hidden_size"])
        layer4 = torch.nn.LayerNorm(self.config["hidden_size"])
        layer5 = torch.nn.Linear(self.config["hidden_size"], self.config["output_shape"])
        self.layers = torch.nn.ModuleList([
            layer1,
            layer2,
            layer3,
            layer4,
            layer5,
        ])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])


    
    def forward(self, x):
        """
        Forward pass through the network.
        :param x: Input tensor.
        :return: Output tensor.
        """
        for i in range(len(self.layers)):
            if i<3:
                if self.layers[i].in_features == self.layers[i].out_features:
                    x = torch.nn.functional.relu(self.layers[i](x) + x)
                else:
                    x = torch.nn.functional.relu(self.layers[i](x))
            else:
                x = self.layers[i](x)
        return x
    
    def reset(self):
        """
        Reset the network parameters.
        """
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def learn(self,iteration_num, state, regret):
        """
        Given a batch of experiences, update the network params
        :param batch: Batch of experiences Tuple of form(iteration_num, state, regret).
        """
        self.optimizer.zero_grad()
        outputs = self.forward(state)
        # optional : weight each loss by iteration num batch[0].unsqueeze(1) * 
        loss = (((outputs - regret.detach()) ** 2)).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()




class PolicyNetwork(nn.Module):
    def __init__(self,
                 config=None,
    ):
        self.config = config
        super(PolicyNetwork, self).__init__()
        layer1 = torch.nn.Linear(self.config["input_shape"], self.config["hidden_size"])
        layer2 = torch.nn.Linear(self.config["hidden_size"], self.config["hidden_size"])
        layer3 = torch.nn.Linear(self.config["hidden_size"], self.config["hidden_size"])
        layer4 = torch.nn.LayerNorm(self.config["hidden_size"])
        layer5 = torch.nn.Linear(self.config["hidden_size"], self.config["output_shape"])
        self.layers = torch.nn.ModuleList([
            layer1,
            layer2,
            layer3,
            layer4,
            layer5,
        ])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config["learning_rate"])
    
    def forward(self, x):
        """
        Forward pass through the network.
        :param x: Input tensor.
        :return: Output tensor.
        """
        for i in range(len(self.layers)):
            if i<3:
                if self.layers[i].in_features == self.layers[i].out_features:
                    x = torch.nn.functional.relu(self.layers[i](x) + x)
                else:
                    x = torch.nn.functional.relu(self.layers[i](x))
            else:
                x = self.layers[i](x)
        return x
    
    def reset(self):
        """
        Reset the network parameters.
        """
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def learn(self, iteration_num, state, regret, linear=False):
        """
        Given a batch of experiences, update the network params
        :param batch: Batch of experiences Tuple of form(iteration_num, state, regret).
        """
        self.optimizer.zero_grad()
        outputs = self.forward(state)
        reshaped_iternum = iteration_num.view(-1,1).detach()
        # optional : weight each loss by iteration num batch[0].unsqueeze(1) * 
        if linear:
            loss = (reshaped_iternum * ((outputs - regret.detach()) ** 2)).mean()
        else:
            loss = (((outputs - regret.detach()) ** 2)).mean()
        loss.backward()
        self.optimizer.step()
        return loss.item()