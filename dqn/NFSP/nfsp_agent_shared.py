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

import gc
import os
from pathlib import Path
import pickle
import sys
from time import time

from agent_configs import NFSPDQNConfig

from numpy import average
import torch
from utils import get_legal_moves, current_timestamp, update_per_beta, plot_graphs
import dill

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
        name: str = f"nfsp_dqn_{current_timestamp():.1f}",
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
        super().__init__(env, config, name)
        rl_config = self.config.rl_configs[0]
        sl_config = self.config.sl_configs[0]

        self.rl_agent: RainbowAgent = RainbowAgent(
            env, rl_config, f"rl_agent_{name}", device
        )
        self.sl_agent: PolicyImitationAgent = PolicyImitationAgent(
            env, sl_config, f"sl_agent_{name}", device
        )

        self.policies = ["average_strategy"] * self.config.num_players

        self.previous_states = [None] * self.config.num_players
        self.previous_infos = [None] * self.config.num_players
        self.previous_actions = [None] * self.config.num_players

        self.stats = {
            "rl_loss": [],
            "sl_loss": [],
            "exploitability": [],
        }

        self.targets = {
            "exploitability": 0,
        }

    def select_agent_policies(self):
        for p in range(self.config.num_players):
            # print("Selecting policy")
            if random.random() < self.config.anticipatory_param:
                # print("best_response")
                self.policies[p] = "best_response"
            else:
                # print("average_strategy")
                self.policies[p] = "average_strategy"

    def predict(self, state, info):
        if self.policies[info["player"]] == "average_strategy":
            prediction = self.sl_agent.predict(state, info)
        else:
            prediction = self.rl_agent.predict(state)
        return prediction

    def select_actions(self, prediction, info: dict) -> torch.Tensor:
        if self.policies[info["player"]] == "average_strategy":
            # print("selecting with average strategy")
            action = self.sl_agent.select_actions(prediction)
        else:
            # print("selecting with best response")
            action = self.rl_agent.select_actions(prediction, info)

        return action

    def learn(self):
        rl_losses = []
        sl_losses = []
        if (
            self.rl_agent.replay_buffer.size
            > self.rl_agent.config.min_replay_buffer_size
        ):  # only do if not in rl agent mode? (open spiel)
            rl_loss = self.rl_agent.learn().mean()
            rl_losses.append(rl_loss)
        if (
            self.sl_agent.replay_buffer.size
            > self.sl_agent.config.min_replay_buffer_size
        ):
            sl_loss = self.sl_agent.learn()
            sl_losses.append(sl_loss)
        average_rl_loss = (
            sum(rl_losses) / len(rl_losses) if len(rl_losses) > 0 else None
        )
        average_sl_loss = (
            sum(sl_losses) / len(sl_losses) if len(sl_losses) > 0 else None
        )
        return average_rl_loss, average_sl_loss

    def store_transition(self, action, state, info, reward, done, player):
        previous_state = self.previous_states[player]
        previous_info = self.previous_infos[player]
        previous_action = self.previous_actions[player]

        if (
            previous_action is not None
            and previous_state is not None
            and previous_info is not None
        ):
            assert (
                self.previous_infos["player"] == info["player"] or done == True
            ), "Players don't match, {} != {}".format(
                self.previous_infos["player"], info["player"]
            )

            transition = [
                previous_state,
                previous_info,
                previous_action,
                reward,
                state,
                info,
                done,
            ]
            # only do this if it is average policy? (open spiel)
            self.rl_agent.replay_buffer.store(*transition)
        if not done:
            self.previous_states[player] = state
            self.previous_infos[player] = info
            self.previous_actions[player] = action
        else:
            self.previous_states[player] = None
            self.previous_infos[player] = None
            self.previous_actions[player] = None

    def fill_replay_buffers(self):
        pass

    def train(self):
        training_time = time()

        self.select_agent_policies()
        state, info = self.env.reset()
        rewards = [None] * self.config.num_players
        done = False

        for training_step in range(self.start_training_step, self.training_steps):
            with torch.no_grad():
                for _ in range(self.config.replay_interval):
                    current_player: int = info["player"]
                    prediction = self.predict(state, info)
                    action = self.select_actions(
                        prediction,
                        info,
                    ).item()
                    # print("Action", action)

                    # only average strategy mode? (open spiel)
                    self.store_transition(
                        action,
                        state,
                        info,
                        rewards[current_player],
                        done,
                        info["player"],
                    )  # stores experiences

                    target_policy = torch.zeros(self.num_actions)
                    target_policy[action] = 1.0
                    # print(current_player)
                    if self.policies[current_player] == "best_response":
                        self.sl_agent.replay_buffer.store(
                            state, info, target_policy
                        )  # Store best moves in SL Memory

                    next_state, rewards, terminated, truncated, next_info = (
                        self.env.step(action)
                    )
                    done = terminated or truncated
                    state = next_state
                    info = next_info

                    # terminal so store experiences for all agents based on terminal state
                    if done:
                        # print("Rewards", rewards)
                        for p in range(self.config.num_players):
                            # could only do if policy is average strategy mode
                            self.store_transition(
                                action, state, info, rewards[p], done, p
                            )
                        self.select_agent_policies()
                        state, info = self.env.reset()
                        rewards = [
                            None
                        ] * self.config.num_players  # ugly but makes code cleaner for storing in rl
                        done = False  # ugly but makes code cleaner for storing in rl

                if training_step % self.rl_agent.config.transfer_interval == 0:
                    self.rl_agent.update_target_model()

                self.rl_agent.replay_buffer.set_beta(
                    update_per_beta(
                        self.rl_agent.replay_buffer.beta,
                        self.rl_agent.config.per_beta_final,
                        self.training_steps,
                    )
                )

            for minibatch in range(self.config.num_minibatches):
                rl_loss, sl_loss = self.learn()
                # print("Losses", rl_loss, sl_loss)
                if rl_loss is not None:
                    self.stats["rl_loss"].append({"loss": rl_loss})
                if sl_loss is not None:
                    self.stats["sl_loss"].append({"loss": sl_loss})

            if training_step % self.checkpoint_interval == 0 and training_step > 0:
                self.save_checkpoint(
                    training_step,
                    training_step * self.config.replay_interval,
                    time() - training_time,
                )

        self.save_checkpoint(
            training_step,
            training_step * self.config.replay_interval,
            time() - training_time,
        )
        self.env.close()

    def save_checkpoint(
        self,
        training_step,
        frames_seen,
        time_taken,
    ):
        exploitability = 0

        dir = Path("checkpoints", self.model_name)
        training_step_dir = Path(dir, f"step_{training_step}")
        os.makedirs(dir, exist_ok=True)
        os.makedirs(Path(dir, "graphs"), exist_ok=True)
        os.makedirs(Path(dir, "configs"), exist_ok=True)
        if self.env.render_mode == "rgb_array":
            os.makedirs(Path(training_step_dir, "videos"), exist_ok=True)
        if self.config.save_intermediate_weights:
            weights_path = str(Path(training_step_dir, f"model_weights"))
            os.makedirs(Path(training_step_dir, "model_weights"), exist_ok=True)
            os.makedirs(Path(training_step_dir, "optimizers"), exist_ok=True)
            os.makedirs(Path(training_step_dir, "replay_buffers"), exist_ok=True)
            os.makedirs(Path(training_step_dir, "graphs_stats"), exist_ok=True)

            # save the model weights
            torch.save(
                self.rl_agent.model.state_dict(),
                f"{weights_path}/{self.rl_agent.model_name}_weights.keras",
            )
            torch.save(
                self.sl_agent.model.state_dict(),
                f"{weights_path}/{self.sl_agent.model_name}_weights.keras",
            )

            # save optimizer (pickle doesn't work but dill does)
            with open(
                Path(
                    training_step_dir,
                    f"optimizers/{self.rl_agent.model_name}_optimizer.dill",
                ),
                "wb",
            ) as f:
                dill.dump(self.rl_agent.optimizer, f)

            with open(
                Path(
                    training_step_dir,
                    f"optimizers/{self.sl_agent.model_name}_optimizer.dill",
                ),
                "wb",
            ) as f:
                dill.dump(self.sl_agent.optimizer, f)

            # save replay buffer
            self.rl_agent.save_replay_buffers(training_step_dir)
            self.sl_agent.save_replay_buffers(training_step_dir)

        for p in range(self.config.num_players):
            exploitability += self.test(
                self.checkpoint_trials, p, training_step, training_step_dir
            )

        # save config
        self.config.dump(f"{dir}/configs/config.yaml")

        # exploitability /= self.config.num_players
        self.stats["exploitability"].append({"exploitability": exploitability})

        # save the graph stats and targets
        stats_path = Path(training_step_dir, f"graphs_stats", exist_ok=True)
        os.makedirs(stats_path, exist_ok=True)
        with open(Path(training_step_dir, f"graphs_stats/stats.pkl"), "wb") as f:
            pickle.dump(self.stats, f)
        with open(Path(training_step_dir, f"graphs_stats/targets.pkl"), "wb") as f:
            pickle.dump(self.targets, f)

        gc.collect()

        # plot the graphs (and save the graph)
        plot_graphs(
            self.stats,
            self.targets,
            training_step,
            frames_seen,
            time_taken,
            self.model_name,
            f"{dir}/graphs",
        )

    def test(self, num_trials, player, step, dir="./checkpoints"):
        if num_trials == 0:
            return
        with torch.no_grad():
            training_policies = [
                self.policies[p] for p in range(self.config.num_players)
            ]
            for p in range(self.config.num_players):
                self.policies[p] = "best_response"
            self.policies[player] = "average_strategy"
            test_score = 0
            if self.test_env.render_mode == "rgb_array":
                self.test_env.episode_trigger = lambda x: (x + 1) % num_trials == 0
                self.test_env.video_folder = "{}/videos/{}/{}".format(
                    dir, f"self.model_name_{player}", step
                )
                if not os.path.exists(self.test_env.video_folder):
                    os.makedirs(self.test_env.video_folder)

            for _ in range(num_trials):
                print("Trial ", _)
                state, info = self.test_env.reset()
                done = False
                while not done:
                    for p in range(self.config.num_players):
                        prediction = self.predict(state, info)
                        print("Prediction", prediction)
                        action = self.select_actions(prediction, info).item()
                        action_string = (
                            "call"
                            if action == 0
                            else (
                                "raise"
                                if action == 1
                                else "fold" if action == 2 else "check"
                            )
                        )
                        print(action_string)
                        next_state, reward, terminated, truncated, info = (
                            self.test_env.step(action)
                        )
                        done = terminated or truncated
                        state = next_state
                        average_strategy_reward = reward[player]
                        total_reward = sum(reward)
                        test_score += (total_reward - average_strategy_reward) / (
                            self.config.num_players - 1
                        )
                        if done:
                            break

            for p in range(self.config.num_players):
                self.policies[p] = training_policies[p]
            return test_score / num_trials  # is this correct for exploitability?

    def load_from_checkpoint(self, dir: str, training_step):
        dir = Path("checkpoints", self.model_name)
        training_step_dir = Path(dir, f"step_{training_step}")
        self.config = self.config.__class__.load(Path(dir, "configs/config.yaml"))
        self.rl_config = self.config.rl_configs[0]
        self.sl_config = self.config.sl_configs[0]
        weights_path = str(Path(training_step_dir, f"model_weights"))
        # save the model weights
        self.rl_agent.load_model_weights(
            f"{weights_path}/{self.rl_agent.model_name}_weights.keras"
        )
        self.sl_agent.load_model_weights(
            f"{weights_path}/{self.sl_agent.model_name}_weights.keras"
        )
        self.rl_agent.config = self.rl_config
        self.sl_agent.config = self.sl_config

        with open(
            Path(
                training_step_dir,
                f"optimizers/{self.rl_agent.model_name}_optimizer.dill",
            ),
            "rb",
        ) as f:
            self.rl_agent.optimizer = dill.load(f)

        with open(
            Path(
                training_step_dir,
                f"optimizers/{self.sl_agent.model_name}_optimizer.dill",
            ),
            "rb",
        ) as f:
            self.sl_agent.optimizer = dill.load(f)

        # save replay buffer
        self.rl_agent.load_replay_buffers(training_step_dir)
        self.sl_agent.load_replay_buffers(training_step_dir)

        # save the graph stats and targets
        with open(Path(training_step_dir, f"graphs_stats/stats.pkl"), "rb") as f:
            self.stats = pickle.load(f)
        with open(Path(training_step_dir, f"graphs_stats/targets.pkl"), "rb") as f:
            self.targets = pickle.load(f)

        self.start_training_step = training_step
