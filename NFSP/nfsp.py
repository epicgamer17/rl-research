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


import random
from time import time
from NFSP.supervised_agent import AverageStrategyAgent
from base_agent.agent import BaseAgent


class NFSPAgent(BaseAgent):
    def __init__(self, env, config: NFSPConfig, name, agent_type) -> None:
        super().__init__(env, config, name)
        rl_config = self.config.rl_config
        self.rl_agents = [
            agent_type(env, rl_config, name) for player in self.config.num_players
        ]
        sl_config = self.config.sl_config
        self.sl_agents = [
            AverageStrategyAgent(env, sl_config, name)
            for player in self.config.num_players
        ]
        self.current_agent = 0

    def train(self):
        training_time = time()
        self.is_test = False
        stats = {
            "score": [],
            "rl_loss": [],
            "sl_loss": [],
            "test_score": [],
        }
        targets = {
            "score": self.env.spec.reward_threshold,
            "test_score": self.env.spec.reward_threshold,
        }

        state, _ = self.env.reset()
        score = 0
        for training_step in range(self.training_steps * self.config.num_players):
            for _ in range(self.config.replay_interval):
                action = self.select_action(state)

                next_state, reward, terminated, truncated, info = self.step(
                    action
                )  # Stores RL Experiences in step function
                done = terminated or truncated
                state = next_state
                score += reward
                self.config.per_beta = min(
                    1.0,
                    self.config.per_beta
                    + (1 - self.config.per_beta)
                    / self.training_steps,  # per beta increase
                )

                if done:
                    state, _ = self.env.reset()
                    stats["score"].append(score)  # might be irrelevant for NFSP
                    score = 0

            for minibatch in range(self.config.num_minibatches):
                rl_loss, sl_loss = self.experience_replay()
                stats["rl_loss"].append(
                    rl_loss
                )  # may want to average since it could be noisy between the different agents
                stats["sl_loss"].append(sl_loss)

            if training_step % self.config.transfer_interval == 0:
                self.rl_agents[self.current_agent].update_target_model(
                    training_step
                )  # Update target model for the current RL agent

            if training_step % self.checkpoint_interval == 0 and training_step > 0:
                self.save_checkpoint(
                    stats,
                    targets,
                    5,
                    training_step,
                    training_step * self.config.replay_interval,
                    time() - training_time,
                )

            self.current_agent = (self.current_agent + 1) % self.config.num_players

        self.save_checkpoint(
            stats,
            targets,
            5,
            training_step,
            training_step * self.config.replay_interval,
            time() - training_time,
        )
        self.env.close()

    def select_action(self, state, legal_moves=None):
        if random.random() < self.config.anticipatory_param:
            action = self.rl_agents[self.current_agent].select_action(
                state, legal_moves
            )
            self.sl_agents[self.current_agent].replay_buffer.store(
                state, action
            )  # Store best moves in SL Memory
        else:
            action = self.sl_agents[self.current_agent].select_action(
                state, legal_moves
            )

        return action

    def experience_replay(self):
        rl_loss = self.rl_agents[self.current_agent].experience_replay()
        sl_loss = self.sl_agents[self.current_agent].experience_replay()
        return rl_loss, sl_loss
