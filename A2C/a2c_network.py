import torch.nn as nn
import torch

class NeuralNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linearstack = nn.Sequential(
            nn.Linear(config["inputsize"], 100),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.ReLU(),
            nn.Linear(50,10),
            nn.ReLU(),
        )
        self.actorstack = nn.Sequential(
            nn.Linear(10, config["actionspace"]),
            nn.Softmax(dim=0)
        )
        self.criticstack = nn.Linear(10, 1)
    
    def forward(self, x):
        a = self.linearstack(x)
        return self.actorstack(a), self.criticstack(a)

config = {"inputsize": 625, "actionspace": 4}
network = NeuralNetwork(config=config)
print(network)
x=torch.flatten(torch.rand(25,25))
torch.autograd.set_detect_anomaly(True)
optimizer = torch.optim.SGD(network.parameters(), lr=1e-3)
for i in range(5000):
    a = network(x)
    loss = torch.nn.MSELoss()
    b = loss(a[0], torch.tensor([0,0,0,1.0]))
    b.backward(retain_graph=True)
    c = loss(a[1], torch.tensor(1.0))
    c.backward()
    optimizer.step()
    optimizer.zero_grad()


print(a)