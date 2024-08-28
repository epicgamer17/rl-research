import torch.nn as nn
import torch
import time

class NeuralNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        # if want joint layers then 1 if not then 0
        if (config["jointlay"]==1):
            self.linearstack = nn.Sequential(*config["stack"])
            self.actorstack = nn.Sequential(*config["actorhead"]
                #nn.Linear(10, config["actionspace"]),
                #nn.Softmax(dim=0)
            )
            self.criticstack = nn.Sequential(*config["critichead"])
        else:
            self.actorstack = ActorStack(config)
            self.criticstack = CriticStack(config)
        
    def forward(self, x):
        #differing forward function based on wether the layers are joint or not
        if (config["jointlay"]==1):
            a = self.linearstack(x)
            return self.actorstack(a), self.criticstack(a)
        else:
            return self.actorstack(x), self.criticstack(x) 




#If seperate parameters define the actor and critic separately
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


start = time.time()
### Your code .......

def loss(advantage,logprob):
    return logprob*advantage

if (1):
    config = {"criticstack": [nn.Linear(625,100), nn.ReLU(), nn.Linear(100,25),nn.ReLU(),nn.Linear(25,1)],
            "actorstack": [nn.Linear(625,100), nn.ReLU(), nn.Linear(100,25),nn.ReLU(),nn.Linear(25,4),nn.Softmax(dim=0)],
            "inputsize": 625,
            "jointlay": 1,
            "actorhead": [nn.Linear(25, 4)],
            "critichead": [nn.Linear(25, 1)],
            "stack": [nn.Linear(625,100), nn.ReLU(), nn.Linear(100,25),nn.ReLU()]
            }
    network = NeuralNetwork(config=config)
    #print(network)

    x=torch.flatten(torch.rand(25,25))
    optimizer = torch.optim.SGD(network.parameters(), lr=1e-3)
    for i in range(1):
        a = network(x)
        print(a)
        if (config["jointlay"]):
            print(network.parameters)
            b = loss(a[0][1], torch.tensor(2))
            b.backward(retain_graph=True)
            c = loss(a[1], torch.tensor(5))
            c.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(network.parameters)
        else:
            b = loss(a[0][1], torch.tensor(2))
            b.backward()
            optimizer.step()
            optimizer.zero_grad()
            c = loss(a[1], torch.tensor(5))
            c.backward()
            optimizer.step()
            optimizer.zero_grad()

print(a)


#nn.Linear(config["inputsize"], 100),
#nn.ReLU(),
#nn.Linear(100,50),
#nn.ReLU(),
#nn.Linear(50,10),
#nn.ReLU(),