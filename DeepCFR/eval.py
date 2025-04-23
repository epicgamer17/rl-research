import torch
import numpy as np

class ModelEvaluator:
    def __init__(self, model, type="HH", model2=None, env=None, blinds=None, num_episodes=50000):
        """
        Evaluate the model on the given environment.
        :param model: The model to evaluate.
        :param type: The type of evaluation to perform.
        :param env: The environment to evaluate the model on.
        :param blinds: The blinds for the environment. [small blind, big blind]
        :param num_episodes: The number of episodes to run.
        :return: The average reward and the number of episodes.
        """
        assert type in ["HH", "LBR"], "Invalid type. Must be Head to Head, Learned Best Response"
        # TOTE is LBR for each player just run this a bunch
        self.model = model
        self.type = type
        self.model2 = model2
        self.env = env
        self.blinds = blinds
        self.num_episodes = num_episodes
        if self.type == "HH":
            self.model.eval()
            self.model2.eval()
        elif self.type == "LBR":
            self.model.eval()
        
    
    def evaluate(self):
        if self.type == "LBR":
            self.model2.train(self.env)
            self.model2.eval()
        rewards = {"player_0": [], "player_1": []}
        for i in range(self.num_episodes):
            rew1 = 0
            rew2 = 0
            self.env.reset()
            self.model2.eval()
            while not self.env.done:
                state, reward, terminated, truncated, info = self.env.last()
                if self.env.current_player == 0:
                    action = self.model.get_action(state)
                else:
                    action = self.model2.get_action(state)
                next_state, reward, terminated, truncated, next_info= self.env.step(action)
                rew1 += reward[0]
                rew2 += reward[1]
            rewards["player_0"].append(rew1/self.blinds[1])
            rewards["player_1"].append(rew2/self.blinds[1])
        
        return rewards["player_0"], rewards["player_1"]