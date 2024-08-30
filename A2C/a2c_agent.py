import torch
import torch.nn as nn
from a2c_network import NeuralNetwork
import numpy as np
import gymnasium as gym

config = {"discountfactor": 0.9}

class A2CAgent():
    def __init__(self,config,name):
        #make models and whatnot
        torch.autograd.set_detect_anomaly(True)
        self.name=name
        self.observationshape = config["observationshape"]
        self.actionspace = config["actionspace"]
        self.Nepisodes = config["epsisodes"]
        self.Dfactor = config["discountfactor"]
        self.t_max = config["t_max"]
        self.model = NeuralNetwork(config)
        #add config to choose optimizer
        if (config["jointlay"]):
            self.actoroptimizer = torch.optim.SGD(self.model.actorstack.parameters(), lr=2e-3)
            self.criticoptimizer = torch.optim.SGD(self.model.criticstack.parameters(), lr=2e-3)
        else:
            self.actoroptimizer = torch.optim.SGD(self.model.actorstack.parameters(), lr=2e-3)
            self.criticoptimizer = torch.optim.SGD(self.model.criticstack.parameters(), lr=2e-3)
        


    def select_action(self, distribution, output):
        a = torch.distributions.Categorical(distribution)
        #from actor softmax prob distribution sample an action
        return a.sample(), output[a.sample().item()]

    def step(self, state, env):
        #get action, input action to env, return array under format : [ state, nextstate, action selected,reward, done, logprob of action for learning, value of state ]
        output = self.model(state)
        c = output[0].detach()
        #represents predictions from actor and critic
        action, logprob = self.select_action(c,output[0])
        #(observation, reward, terminated, truncated, info)
        nextstate = env.step(action.item())
        nstate, r, done = torch.from_numpy(nextstate[0]), nextstate[1] , (nextstate[2] or nextstate[3])
        #env(action)
        return [state, nstate, action, r, done, logprob, output[1]]

    def predict(self, inputs):
        #get critic and actor outputs for select state, dependent on inputs, softmax for actor value for critic
        return self.model(inputs)

    def batchactorloss(self, advantages,logprobs):
        #minimizing the loss
        sumloss = 0
        print(torch.tensor(logprobs))
        print(advantages)
        print(logprobs * advantages)
        loss = torch.mean(logprobs * advantages)
        print("LOSS ")
        print(loss)
        sumloss += loss
        loss.backward()
        self.actoroptimizer.step()
        self.actoroptimizer.zero_grad()

    
    def batchcriticloss(self, sums, values):
        mseloss = torch.nn.HuberLoss()
        for i in range(len(values)):
            loss = mseloss(sums[i], values[i])/len(values)
            loss.backward()
        self.criticoptimizer.step()
        self.criticoptimizer.zero_grad()


    def train(self,env):
        # Training loop
        for i in range(10000):
            print(i)
            #get new init state
            state = torch.tensor(env.reset()[0])
            #For a number of episodes
            t = 0
            done = False
            episode = []
            #reset time step counter, done flag, epsiode list, state 
            while  (t<self.t_max and done!=True):
                #since monte carlo sort of alg need to finish the episode before learning
                a = self.step(state, env)
                # [ state, nextstate, action selected,reward, done,action,  logprob of action for learning, value of state ] 

                done = a[4]
                episode.append(a)
                state = a[1]
                t = t+1
            #Then learn from episode
            rewardsum = 0
            rew = []
            vs = []
            adv = []
            log = []
            for o in range(len(episode)-1):

                # For each episode backward : do backprop for critic, calculated by ∑Rt - V(St), so Rt-->Rtmax, sum of discounted rewards mean square error with V(s)
                # Then update actor with : Log prob t * (Rt + ∂V(s')-V(s)), the parentheses represents the Advantage of the function, rewards possible vs rewards gotten
                # [ state, nextstate, action selected,reward, done, logprob of action for learning, value of state ]
                currentep = episode[o]
                rewardsum = rewardsum + currentep[3]
               
                """if (o==len(episode)-1):
                    #If terminal state, then V(S') = 0
                    advantage = currentep[3] - currentep[-1]
                else:"""
                advantage = currentep[3] + (self.Dfactor * episode[o+1][-1]) - currentep[-1] 
                log.append(torch.log(currentep[-2]))
                adv.append(advantage[0])
                rew.append(rewardsum)
                vs.append(currentep[-1])

                print("REWARD SUM AND VALUE ESTIMATION")
                print(rewardsum)
                print(currentep[-1].item())
                print("ADVANTAGE, REWARD AND PREDICTIONS")
                print(advantage.item())
                print(currentep[3])
                print(currentep[-1].item())
                if (i==len(episode)-2):
                    print("NAN")
                else:
                   print(episode[o+1][-1].item())

 
            self.batchcriticloss(torch.tensor(rew), vs)
            self.batchactorloss(torch.tensor(adv), log)
                #If the layers are joint then we have to accumulate the gradients and do it that way. 
                # IF GRADIENTS ACCUMULATE, THEN HOW DOES BACKWARD WORK FOR THE HEADS? I guess it does it all at once with the backward
            







                

env = gym.make("CartPole-v1", render_mode="human")




config = {"criticstack": [nn.Linear(4,20), nn.ReLU(), nn.Linear(20,25),nn.ReLU(),nn.Linear(25,1)],
            "actorstack": [nn.Linear(4,20), nn.ReLU(), nn.Linear(20,25),nn.ReLU(),nn.Linear(25,2),nn.ReLU(),nn.Softmax(dim=0)],
            "jointlay": 0,
            "actorhead": [nn.Linear(25, 2),nn.ReLU(),nn.LayerNorm,nn.Softmax(dim=0)],
            "critichead": [nn.Linear(25, 1)],
            "stack": [nn.Linear(4,100), nn.ReLU(), nn.Linear(100,25),nn.ReLU()],
            "epsisodes": 10000,
            "discountfactor": 0.99,
            "t_max": 500,
            "observationshape": env.observation_space.shape[0],
            "actionspace": env.action_space.n
            }


Agent = A2CAgent(config, 'James')
Agent.train(env)
env.render()