from base_agent.agent import BaseAgent
import torch
import torch.nn as nn

config = {"discountfactor": 0.9}

class A2CAgent():
    def __init__(self,config,name):
        #make models and whatnot
        self.Nepisodes = 2
        self.Dfactor = config["discountfactor"]
        self.t_max = config["t_max"]


    def select_action(self, distribution):
        return distribution.sample()

    def step(self, state, env):
        action = self.select_action(self.model.actor(state))
        nstate, r, done = env(action)
        return [state, nstate, action, done]

    def learn(self,env):
        # for however many episodes
        for i in range(self.NEpisodes):
            t = 0
            done = False
            episode = []
            state = env.first
            while t<self.t_max or done!=True:
                a = self.step(state, env)
                done = a[-1]
                episode.append(a)
                state = a[1]
                t = t+1
            for i in range(len(episode), 0):
                #do stuff to get R
                print("hi")
                #
