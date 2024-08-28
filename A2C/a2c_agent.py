import torch
import torch.nn as nn
from a2c_network import NeuralNetwork

config = {"discountfactor": 0.9}

class A2CAgent():
    def __init__(self,config,name):
        #make models and whatnot
        self.name=name
        self.Nepisodes = config["epsisodes"]
        self.Dfactor = config["discountfactor"]
        self.t_max = config["t_max"]
        self.model = NeuralNetwork(config)
        #add config to choose optimizer
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)


    def select_action(self, distribution):
        a = torch.distributions.Categorical(distribution)
        #from actor softmax prob distribution sample an action
        return a.sample(), distribution[a.sample().item()].item()

    def step(self, state, env):
        #get action, input action to env, return array under format : [ state, nextstate, action selected,reward, done, logprob of action for learning, value of state ]
        output = self.model(state)
        #represents predictions from actor and critic
        action, logprob = self.select_action(output[0])
        nstate, r, done = 1, 2 ,3
        #env(action)
        return [state, nstate, r, done, action, logprob, output[1]]

    def predict(self, inputs):
        #get critic and actor outputs for select state, dependent on inputs, softmax for actor value for critic
        return self.model(inputs)

    def actorloss(advantage,logprob):
        return logprob*advantage

    def train(self,env):
        # Training loop
        for i in range(self.NEpisodes):
            #For a number of episodes
            t = 0
            done = False
            episode = []
            state = env.first
            #reset time step counter, done flag, epsiode list, state 
            while t<self.t_max or done!=True:
                #since monte carlo sort of alg need to finish the episode before learning
                a = self.step(state, env)
                # [ state, nextstate, action selected,reward, done, logprob of action for learning, value of state ]
                done = a[-2]
                episode.append(a)
                state = a[1]
                t = t+1
            #Then learn from episode
            rewardsum = 0
            for i in range(len(episode)-1, -1, -1):

                # For each episode backward : do backprop for critic, calculated by ∑Rt - V(St), so Rt-->Rtmax, sum of discounted rewards mean square error with V(s)
                # Then update actor with : Log prob t * (Rt + ∂V(s')-V(s)), the parentheses represents the Advantage of the function, rewards possible vs rewards gotten
                # [ state, nextstate, action selected,reward, done, logprob of action for learning, value of state ]
                currentep = episode[i]
                rewardsum = rewardsum + currentep[3]
                Vloss = torch.nn.MSELoss()
                a = Vloss(torch.tensor(rewardsum), currentep[-1])
                if (i==(len(episode)-1)):
                    #If terminal state, then V(S') = 0
                    advantage = currentep[3] - currentep[-1]
                else:
                    advantage = currentep[3] + self.Dfactor * episode[i+1][-1] - currentep[-1] 
            
                b = self.actorloss(currentep[-2], advantage)

                #If the layers are joint then we have to accumulate the gradients and do it that way. 
                # IF GRADIENTS ACCUMULATE, THEN HOW DOES BACKWARD WORK FOR THE HEADS? I guess it does it all at once with the backward graph
                if (config["jointlay"]):
                    a.backward(retain_graph=True)
                    b.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                else:
                    a.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    b.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                





config = {"criticstack": [nn.Linear(625,100), nn.ReLU(), nn.Linear(100,25),nn.ReLU(),nn.Linear(25,1)],
            "actorstack": [nn.Linear(625,100), nn.ReLU(), nn.Linear(100,25),nn.ReLU(),nn.Linear(25,4),nn.Softmax(dim=0)],
            "inputsize": 625,
            "jointlay": 1,
            "actorhead": [nn.Linear(25, 4)],
            "critichead": [nn.Linear(25, 1)],
            "stack": [nn.Linear(625,100), nn.ReLU(), nn.Linear(100,25),nn.ReLU()],
            "epsisodes": 1000,
            "discountfactor": 0.99,
            "t_max": 100
            }

Agent = A2CAgent(config, 'James')

x=torch.flatten(torch.rand(25,25))
print(Agent.predict(x))