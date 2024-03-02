import time
import gymnasium as gym
environments_list = gym.make("CartPole-v1", render_mode="rgb_array")
GAME_ACTIONS = environments_list.action_space.n
GAME_OBS = environments_list.observation_space.shape[0]
from copy import deepcopy
from math import log, sqrt, inf
import random

class Node:
    def __init__(self, game, done, parent, observation, action_to_get_here):
        self.game = game
        self.done = done
        self.parent = parent
        self.observation = observation
        self.action__to_get_here = action_to_get_here
        self.children = {}
        self.total_rollouts = 0
        self.visits = 0
        self.action_left_to_try = list(range(GAME_ACTIONS))

    def get_UCB_score(self):
        if self.visits == 0:
            return inf
        return self.total_rollouts / self.visits + sqrt(log(self.parent.visits) / self.visits)
    
    def create_children(self):
        if self.done:
            return None 
        for action in range(GAME_ACTIONS):
            new_game = deepcopy(self.game)
            observation, reward, done, info = new_game.step(action)
            self.children[action] = Node(new_game, done, self, observation, action)


    def explore(self):
        current = self
        while current.children:
            child = current.children
            max_U = max(c.get_UCB_score() for c in child.values())
            actions = [ a for a,c in child.items() if c.get_UCB_score() == max_U ]                  
            action_selected = random.choice(actions)
            current = child[action_selected]
                
        if current.visits ==0:
            current.total_rollouts += current.rollout()
        else:
            current.create_children()
            if current.children:
                current = random.choice(current.children)
            current.total_rollouts += current.rollout()
    
        current.visits += 1      
        dad = current
            
        while dad.parent:
            dad = dad.parent
            dad.visits += 1
            dad.total_rollouts += current.total_rollouts

    def rollout(self):
        if self.done:
            return 0        
        v = 0
        done = False
        new_game = deepcopy(self.game)
        while not done:
            action = new_game.action_space.sample()
            observation, reward, done, info = new_game.step(action)
            v += reward
            if done:
                new_game.reset()
                new_game.close()
                return v          

    def choose_next(self):
        if self.done:
            raise ValueError("game has ended")
        if not self.children:
            raise ValueError('no children found and game has not ended')
        max_N = max(node.N for node in self.children.values())
        max_children = [ c for a,c in self.children.items() if c.N == max_N ]   
        max_child = random.choice(max_children)
        return max_child, max_child.action_to_get_here