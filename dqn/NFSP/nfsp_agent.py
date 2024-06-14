# baseline for NFSP, should work for any model type (paper uses DQN)
# Initialize game Γ and execute an agent via RUNAGENT for each player in the game
# function RUNAGENT(Γ)
# Initialize replay memories MRL (circular buffer) and MSL (reservoir)
# Initialize average-policy network Π(s, a | θΠ) with random parameters θΠ
# Initialize action-value network Q(s, a | θQ) with random parameters θQ
# Initialize target network parameters θQ0 ← θQ
# Initialize anticipatory parameter η
# for each episode do
# Set policy σ ← {
# epsilon-greedy (Q), with probability η
# Π, with probability 1 − η
# }
# Observe initial information state s1 and reward r1
# for t = 1, T do
# Sample action at from policy σ
# Execute action at in game and observe reward rt+1 and next information state st+1
# Store transition (st, at, rt+1, st+1) in reinforcement learning memory MRL
# if agent follows best response policy σ = epsilon-greedy (Q) then
# Store behaviour tuple (st, at) in supervised learning memory MSL
# end if
# Update θΠ with stochastic gradient descent on loss
# L(θΠ) = E(s,a) from MSL [log Π(s, a | θΠ)]
# Update θQ with stochastic gradient descent on loss
# LθQ = E(s,a,r,s0) from MRL [r + maxa0 Q(s0, a0| θQ0) − Q(s, a | θQ)^2]
# Periodically update target network parameters θ   Q0 ← θQ
# end for
# end for
# end function

import sys
from time import time

from agent_configs import NFSPDQNConfig

import torch
from utils import get_legal_moves, current_timestamp, update_per_beta


sys.path.append("../../")

from dqn.rainbow.rainbow_agent import RainbowAgent


import random
from base_agent.agent import BaseAgent
from agent_configs.dqn.nfsp_config import NFSPDQNConfig
from imitation_learning.policy_imitation_agent import PolicyImitationAgent


class NFSPDQN(BaseAgent):
    def __init__(
        self,
        env,
        config: NFSPDQNConfig,
        device: torch.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            # MPS is sometimes useful for M2 instances, but only for large models/matrix multiplications otherwise CPU is faster
            else (
                torch.device("mps")
                if torch.backends.mps.is_available() and torch.backends.mps.is_built()
                else torch.device("cpu")
            )
        ),
    ) -> None:
        super().__init__(env, config, "NFSPDQN")
        rl_configs = self.config.rl_configs
        sl_configs = self.config.sl_configs
        self.nfsp_agents = [
            NFSPDQNAgent(env, rl_configs[i], sl_configs[i], f"NFSPAgent_{i}", device)
            for i in range(self.config.num_players)
        ]

        self.stats = {
            "rl_loss": [],
            "sl_loss": [],
            "test_score": [],
        }

        self.targets = {
            "test_score": self.env.spec.reward_threshold,
        }

    def select_agent_policies(self):
        for p in range(self.config.num_players):
            self.nfsp_agents[p].select_policy(self.config.anticipatory_param)

    def predict(self, state, info):
        return self.nfsp_agents[info["player"]].predict(state)

    def select_actions(self, predicted, info) -> torch.Tensor:
        return self.nfsp_agents[info["player"]].select_actions(predicted, info)

    def learn(self, current_player):
        return self.nfsp_agents[current_player].learn()

    def train(self):
        training_time = time()
        training_step = 0
        while (
            training_step < self.training_steps
        ):  # change to training steps and just make it so it ends a game even if the training steps are done
            self.select_agent_policies()
            state, info = self.env.reset()
            done = False

            while not done:
                with torch.no_grad():
                    current_player = info["player"]
                    prediction = self.predict(state, info)
                    action = self.select_actions(
                        prediction,
                        info,
                    )
                    # should we store this for SL agent if it as sl agent action?

                    target_policy = torch.zeros(self.num_actions)
                    target_policy[action] = 1.0
                    # print(current_player)
                    self.nfsp_agents[current_player].sl_agent.replay_buffer.store(
                        state, target_policy
                    )  # Store best moves in SL Memory

                    self.nfsp_agents[current_player].rl_agent.replay_buffer.set_beta(
                        update_per_beta(
                            self.nfsp_agents[
                                current_player
                            ].rl_agent.replay_buffer.beta,
                            self.nfsp_agents[
                                current_player
                            ].rl_agent.config.per_beta_final,
                            self.training_steps,
                        )
                    )

                    # self.nfsp_agents[0].rl_agent.transition = [state_p1, action_p1]
                    next_state, rewards, terminated, truncated, info = self.env.step(
                        action
                    )
                    state = next_state
                    done = terminated or truncated

                    training_step += 1
                for minibatch in range(self.config.num_minibatches):
                    rl_loss, sl_loss = self.learn(current_player)

                if (
                    training_step
                    % self.nfsp_agents[current_player].rl_agent.config.transfer_interval
                    == 0
                ):
                    self.nfsp_agents[current_player].rl_agent.update_target_model()

                if training_step % self.checkpoint_interval == 0:
                    for p in range(self.config.num_players):
                        self.nfsp_agents[p].save_checkpoint(
                            training_step,
                            training_step * self.config.replay_interval,
                            time() - training_time,
                        )
                    # CALL CHECKPOINTING ON THE INDIVIDUAL AGENTS

                if not done:
                    self.nfsp_agents[current_player].store_transition(
                        action, state, rewards[current_player], done
                    )  # stores experiences

            for p in range(self.config.num_players):
                self.nfsp_agents[p].store_transition(action, state, rewards[p], done)
        self.env.close()

    def test(self, num_trials):
        policies = [self.nfsp_agents[p].policy for p in range(self.config.num_players)]
        for p in range(self.config.num_players):
            self.nfsp_agents[p].policy = "best_response"
        self.nfsp_agents[0].policy = "average_strategy"
        test_score = 0

        for _ in range(num_trials):
            print("Trial ", _)
            state, info = self.test_env.reset()
            done = False
            while not done:
                for p in range(self.config.num_players):
                    prediction = self.predict(state, info)
                    action = self.select_actions(prediction, info)
                    next_state, reward, terminated, truncated, info = (
                        self.test_env.step(action)
                    )
                    done = terminated or truncated
                    state = next_state
                    if done:
                        break
            test_score += 1 if reward[0] >= 0 else 0

        for p in range(self.config.num_players):
            self.nfsp_agents[p].policy = policies[p]
        return 1 - (
            test_score / num_trials
        )  # I THINK THIS IS NOT THE RIGHT THING FOR EXPLOITABILITY


class NFSPDQNAgent(BaseAgent):
    def __init__(
        self,
        env,
        rl_config,
        sl_config,
        name=f"dqn_nfsp_{current_timestamp():.1f}",
        device: torch.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            # MPS is sometimes useful for M2 instances, but only for large models/matrix multiplications otherwise CPU is faster
            else (
                torch.device("mps")
                if torch.backends.mps.is_available() and torch.backends.mps.is_built()
                else torch.device("cpu")
            )
        ),
    ) -> None:
        super().__init__(env, rl_config, name)
        self.rl_agent = RainbowAgent(env, rl_config, name, device)
        self.sl_agent = PolicyImitationAgent(env, sl_config, name, device)
        self.policy = "best_response"  # "average_strategy" or "best_response
        # transition = [state, action, reward, next_state, done]
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = None

    def store_transition(self, action, state, reward, done):
        self.transition = [self.previous_state, self.previous_action]
        if (
            self.previous_action is not None
            and self.previous_state is not None
            and self.previous_reward is not None
        ):
            self.transition += [self.previous_reward, state, done]
            # only do this if it is average policy? (open spiel)
            self.rl_agent.replay_buffer.store(*self.transition)
        if not done:
            self.previous_state = state
            self.previous_action = action
            self.previous_reward = reward
        else:
            self.previous_state = None
            self.previous_action = None
            self.previous_reward = None

    def select_policy(self, anticipatory_param):
        if random.random() < anticipatory_param:
            return "best_response"
        else:
            return "average_strategy"

    def predict(self, state):
        if self.policy == "average_strategy":
            prediction = self.sl_agent.predict(state)
        else:
            prediction = self.rl_agent.predict(state)
        return prediction

    def select_actions(self, prediction, info):
        if self.policy == "average_strategy":
            action = self.sl_agent.select_actions(prediction, info)
        else:
            action = self.rl_agent.select_actions(prediction, info)

        return action

    def learn(self):
        rl_loss = 0
        sl_loss = 0
        if (
            self.rl_agent.replay_buffer.size
            > self.rl_agent.config.min_replay_buffer_size
        ):  # only do if not in rl agent mode? (open spiel)
            rl_loss = self.rl_agent.learn()
        if (
            self.sl_agent.replay_buffer.size
            > self.sl_agent.config.min_replay_buffer_size
        ):
            sl_loss = self.sl_agent.learn()
        return rl_loss, sl_loss
