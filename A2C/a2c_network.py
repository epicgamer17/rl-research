import torch.nn as nn
import torch
import time


class A2CNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.jointlay = config["jointlay"]
        # if want joint layers then 1 if not then 0
        if self.jointlay == 1:
            self.linearstack = nn.Sequential(*config["stack"])
            self.actorstack = nn.Sequential(
                *config["actorhead"]
                # nn.Linear(10, config["actionspace"]),
                # nn.Softmax(dim=0)
            )
            self.criticstack = nn.Sequential(*config["critichead"])

            self.actorstack = ActorStack(config)
            self.criticstack = CriticStack(config)

    def forward(self, x):
        # differing forward function based on wether the layers are joint or not
        if self.jointlay == 1:
            a = self.linearstack(x)
            return self.actorstack(a), self.criticstack(a)
        else:
            return self.actorstack(x), self.criticstack(x)


# If seperate parameters define the actor and critic separately
class ActorStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stack = nn.Sequential(*config["actorstack"])

    def forward(self, x):
        a = self.stack(x)
        return a


class CriticStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stack = nn.Sequential(*config["criticstack"])

    def forward(self, x):
        a = self.stack(x)
        return a
