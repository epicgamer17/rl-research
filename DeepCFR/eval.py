import torch
import numpy as np

class ModelEvaluator:

    def __init__(model, type="HH", model2=None, env=None):
        """
        Evaluate the model on the given environment.
        :param model: The model to evaluate.
        :param type: The type of evaluation to perform.
        :param env: The environment to evaluate the model on.
        :return: The average reward and the number of episodes.
        """
        # Set the model to evaluation mode
        model.eval()
        assert env is not None, "Environment must be provided for evaluation."
        assert type in ["HH", "LBR"]
    
    def step(self):
        """
        Perform a step in the evaluation process.
        :return: The average reward and the number of episodes.
        """
        action = self.model.act(self.env.last())
        self.env.step(action)
        return self.env, self.env.reward, self.env.done
    
    
    def evaluate(self, num_episodes=10000):
        """
        Evaluate the model on the given environment.
        :param num_episodes: The number of episodes to evaluate the model on.
        :return: The average reward and the number of episodes.
        """
        rewards_p_1 = np.zeros(num_episodes)
        rewards_p_0 = np.zeros(num_episodes)
        for episode in range(num_episodes):
            termination = False
            while not termination or truncation:
                observation, reward, termination, truncation, info = self.env.last()
                if termination or truncation:
                    # IF TERMINATED THEN PASS UP VALUE TO PARET (THIS IS A RECURSIVE CALL)
                    return reward #IE PAYOFF only for activate player
                else:
                    if self.env.agent_selection[-1] == 0:
                        action = self.model.select_actions(self.model.predict_policy(observation))
                    else:
                        action = self.model2.select_actions(self.model2.predict_policy(observation))
                self.env.step(action)
            
            rewards_p_1.append(self.env.rewards["player_1"])  # dict of {agent_0: r0, agent_1: r1}
            rewards_p_0.append(self.env.rewards["player_0"])
        exploitability = self.exploitability(rewards_p_1, rewards_p_0)
        return exploitability

    def exploitability(self, rewards_p_1, rewards_p_0):
        """
        Calculate the exploitability of the model.
        :param rewards_p_1: The rewards for player 1.
        :param rewards_p_0: The rewards for player 0.
        :return: The exploitability of the model.
        """
        exploitability = np.mean(rewards_p_1) - np.mean(rewards_p_0)
        return exploitability