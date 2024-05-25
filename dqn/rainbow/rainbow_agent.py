from typing import NamedTuple
from time import time
import datetime
import torch

from agent_configs import RainbowConfig
from utils import update_per_beta, action_mask, get_legal_moves
import numpy as np

import sys

sys.path.append("../../")

from base_agent.agent import BaseAgent
from replay_buffers.prioritized_n_step_replay_buffer import PrioritizedNStepReplayBuffer
from dqn.rainbow.rainbow_network import RainbowNetwork


class Sample(NamedTuple):
    ids: np.ndarray
    indices: np.ndarray
    actions: np.ndarray
    observations: np.ndarray
    weights: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray


class RainbowAgent(BaseAgent):
    def __init__(
        self,
        env,
        config: RainbowConfig,
        device: torch.device,
        name=datetime.datetime.now().timestamp(),
    ):
        super(RainbowAgent, self).__init__(env, config, name)
        self.config = config
        self.model = RainbowNetwork(
            config=config,
            output_size=self.num_actions,
            input_shape=self.observation_dimensions,
        ).to(device)
        self.target_model = RainbowNetwork(
            config=config,
            output_size=self.num_actions,
            input_shape=self.observation_dimensions,
        ).to(device)

        self.model.initialize(self.config.kernel_initializer)
        self.target_model.load_state_dict(self.model.state_dict())
        optimizer = self.config.optimizer(
            params=self.model.parameters,
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
        )

        self.replay_buffer = PrioritizedNStepReplayBuffer(
            observation_dimensions=self.observation_dimensions,
            max_size=self.config.replay_buffer_size,
            batch_size=self.config.minibatch_size,
            max_priority=1.0,
            alpha=self.config.per_alpha,
            beta=self.config.per_beta,
            # epsilon=config["per_epsilon"],
            n_step=self.config.n_step,
            gamma=self.config.discount_factor,
        )

        # could use a MuZero min-max config and just constantly update the suport size (would this break the model?)
        # self.v_min = self.config.v_min
        # self.v_max = self.config.v_max

        self.support = torch.linspace(
            self.config.v_min,
            self.config.v_max,
            self.config.atom_size,
            device=device,
        )
        """row vector Tensor(atom_size)
        """

        self.transition = list()
        self.stats = {
            "score": [],
            "loss": [],
            "test_score": [],
        }
        self.targets = {
            "score": self.env.spec.reward_threshold,
            "test_score": self.env.spec.reward_threshold,
        }

    def predict_single(self, state, legal_moves=None):
        with torch.no_grad():
            state_input = self.preprocess(state)

        q_distribution: torch.Tensor = self.model(inputs=state_input)

        # (B, output_size, atom_size) *
        # (                atom_size)
        # is valid broadcasting
        q_values = q_distribution * self.support

        q_values = action_mask(
            actions=q_values,
            legal_moves=legal_moves,
            num_actions=self.num_actions,
            mask_value=-torch.inf,
        )
        return q_values

    def select_action(self, state, legal_moves=None):
        q_values = self.predict_single(state, legal_moves)
        # print("Q Values ", q_values)
        selected_action = np.argmax(q_values)
        # print("Selected Action ", selected_action)
        if not self.is_test:
            self.transition = [state, selected_action]
        return selected_action

    def step(self, action):
        # print("Action ", action)
        if not self.is_test:
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.transition += [reward, next_state, done]

            self.replay_buffer.store(*self.transition)

        else:
            next_state, reward, terminated, truncated, info = self.test_env.step(action)

        return next_state, reward, terminated, truncated, info

    def learn(self) -> np.ndarray:
        losses = np.zeros(self.config.training_iterations)
        for i in range(self.config.training_iterations):
            samples = self.replay_buffer.sample()
            weights, indices, observations, actions = (
                torch.Tensor(samples["weights"]),
                samples["indices"],
                samples["observations"],
                samples["actions"],
            )
            discount_factor = self.config.discount_factor**self.config.n_step
            inputs = self.preprocess(observations)

            # (B, atom_size)
            target_distributions = self.compute_target_distributions(
                samples, discount_factor
            )

            self.config.optimizer.zero_grad()
            # (B, output_size, atom_size)
            initial_distributions = self.model(inputs)

            # (B, outputs, atom_size) -[index by [0..B-1, a_0..a_B-1]]> (B, atom_size)
            predicted_distribution = initial_distributions[
                range(self.config.minibatch_size), actions
            ]

            loss = self.config.loss_function(
                predicted_distribution, target_distributions
            )
            assert np.all(loss) >= 0, "Elementwise Loss: {}".format(loss)
            loss.backwards()
            self.optimizer.step()
            self.replay_buffer.update_priorities(indices, loss.numpy())
            self.model.reset_noise()
            self.target_model.reset_noise()
            losses[i] = loss.detach().numpy()
        return losses

    def compute_target_distributions(self, samples, discount_factor):
        with torch.no_grad():
            inputs, next_inputs, rewards, dones = (
                self.preprocess(samples["observations"]),
                self.preprocess(samples["next_observations"]),
                torch.Tensor(samples["rewards"]).reshape(-1, 1),
                torch.Tensor(samples["dones"]).reshape(-1, 1),
            )
            dist: torch.Tensor = self.model(inputs)
            # (B, outputs, atom_size) -[sum(2)] -> (B, outputs) -[argmax(1)]-> (B)
            next_actions = (dist * self.support).sum(dim=2).argmax(dim=1)

            next_dist: torch.Tensor = self.target_model(next_inputs)

            # (B, outputs, atom_size) -[index by [0..B-1, a_0..a_B-1]]> (B, atom_size)
            p = next_dist[range(self.config.minibatch_size), next_actions]

            # (B, 1) + k(B, atom_size) * (B, atom_size) -> (B, atom_size)
            Tz = (rewards + self.config.discount_factor * dones * self.support).clamp_(
                self.config.v_min, self.config.v_max
            )

            # all elementwise
            b: torch.Tensor = (
                (Tz - self.config.v_min)
                * (self.config.atom_size - 1)
                / (self.config.v_max - self.config.v_min)
            )
            l, u = b.floor().long(), b.ceil().long()

            m = torch.zeros_like(p)
            m.scatter_add_(dim=1, index=l, src=p * (u.float()) - b)
            m.scatter_add_(dim=1, index=u, src=p * (b - u.float()))

            # print("Target Distributions ", m)
            return m

    def fill_replay_buffer(self):
        state, _ = self.env.reset()
        for i in range(self.config.min_replay_buffer_size):
            action = self.env.action_space.sample()
            self.transition = [state, action]

            next_state, reward, terminated, truncated, info = self.step(action)
            done = terminated or truncated
            state = next_state
            if done:
                state, _ = self.env.reset()

    def update_target_model(self):
        if self.config.soft_update:
            for wt, wp in zip(self.target_model.parameters(), self.model.parameters()):
                wt.copy_(self.config.ema_beta * wt + (1 - self.config.ema_beta) * wp)
        else:
            self.target_model.load_state_dict(self.model.state_dict())

    def train(self):
        training_time = time()
        self.is_test = False
        self.fill_replay_buffer()
        state, info = self.env.reset()
        score = 0
        target_model_updated = (False, False)  # (score, loss)
        self.training_steps += self.start_training_step

        for training_step in range(self.start_training_step, self.training_steps):
            for _ in range(self.config.replay_interval):
                action = self.select_action(state, get_legal_moves(info))
                next_state, reward, terminated, truncated, info = self.step(action)
                done = terminated or truncated
                state = next_state
                score += reward
                self.replay_buffer.beta = update_per_beta(
                    self.replay_buffer.beta, 1.0, self.training_steps
                )

                if done:
                    state, info = self.env.reset()
                    self.stats["score"].append(
                        {
                            "score": score,
                            "target_model_updated": target_model_updated[0],
                        }
                    )
                    target_model_updated = (False, target_model_updated[1])
                    score = 0

            for minibatch in range(self.config.num_minibatches):
                losses = self.learn()
                # could do things other than taking the mean here
                self.stats["loss"].append(
                    {
                        "loss": losses.mean(),
                        "target_model_updated": target_model_updated[1],
                    }
                )
                target_model_updated = (target_model_updated[0], False)

            if training_step % self.config.transfer_interval == 0:
                target_model_updated = (True, True)
                # stats["test_score"].append(
                #     {"target_model_weight_update": training_step}
                # )
                self.update_target_model(training_step)

            if training_step % self.checkpoint_interval == 0 and training_step > 0:
                self.save_checkpoint(
                    5,
                    training_step,
                    training_step * self.config.replay_interval,
                    time() - training_time,
                )
        self.save_checkpoint(
            5,
            training_step,
            training_step * self.config.replay_interval,
            time() - training_time,
        )
        self.env.close()

    def load_model_weights(self, weights_path: str):
        state_dict = torch.load(weights_path)
        self.model.load_state_dict(state_dict)
        self.target_model.load_state_dict(self.model.state_dict())
