import gc
from time import time
from pkg_resources import get_distribution
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from agent_configs import RainbowConfig
from utils import update_per_beta, get_legal_moves, current_timestamp

import sys

from utils.utils import epsilon_greedy_policy

sys.path.append("../../")

from base_agent.agent import BaseAgent
from replay_buffers.prioritized_n_step_replay_buffer import PrioritizedNStepReplayBuffer
from dqn.rainbow.rainbow_network import RainbowNetwork


class RainbowAgent(BaseAgent):
    def __init__(
        self,
        env,
        config: RainbowConfig,
        name=f"rainbow_{current_timestamp():.1f}",
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
    ):
        super(RainbowAgent, self).__init__(env, config, name, device=device)
        self.model = RainbowNetwork(
            config=config,
            output_size=self.num_actions,
            input_shape=(self.config.minibatch_size,) + self.observation_dimensions,
        )
        self.target_model = RainbowNetwork(
            config=config,
            output_size=self.num_actions,
            input_shape=(self.config.minibatch_size,) + self.observation_dimensions,
        )

        if not self.config.kernel_initializer == None:
            self.model.initialize(self.config.kernel_initializer)

        self.model.to(device)
        self.target_model.to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer: torch.optim.Optimizer = self.config.optimizer(
            params=self.model.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
        )

        self.replay_buffer = PrioritizedNStepReplayBuffer(
            observation_dimensions=self.observation_dimensions,
            observation_dtype=self.env.observation_space.dtype,
            max_size=self.config.replay_buffer_size,
            batch_size=self.config.minibatch_size,
            max_priority=1.0,
            alpha=self.config.per_alpha,
            beta=self.config.per_beta,
            # epsilon=config["per_epsilon"],
            n_step=self.config.n_step,
            gamma=self.config.discount_factor,
            compressed_observations=(
                self.env.lz4_compress if hasattr(self.env, "lz4_compress") else False
            ),
        )

        # could use a MuZero min-max config and just constantly update the suport size (would this break the model?)
        # self.v_min = self.config.v_min
        # self.v_max = self.config.v_max

        self.support = torch.linspace(
            self.config.v_min,
            self.config.v_max,
            self.config.atom_size,
            device=device,
        ).to(device)
        """row vector Tensor(atom_size)
        """

        self.stats = {
            "score": [],
            "loss": [],
            "test_score": [],
        }
        self.targets = {
            "score": self.env.spec.reward_threshold,
            "test_score": self.env.spec.reward_threshold,
        }

    def predict(self, states) -> torch.Tensor:
        # could change type later
        state_input = self.preprocess(states)
        q_distribution: torch.Tensor = self.model(state_input)
        masked_q_distribution = q_distribution
        return q_distribution

    def predict_target(self, states) -> torch.Tensor:
        # could change type later
        state_input = self.preprocess(states)
        q_distribution: torch.Tensor = self.target_model(state_input)
        return q_distribution

    def select_actions(self, distribution):
        q_values = distribution * self.support
        selected_actions = q_values.sum(2, keepdim=False).argmax(1, keepdim=False)
        return selected_actions

    def learn(self) -> np.ndarray:
        losses = np.zeros(self.config.training_iterations)
        for i in range(self.config.training_iterations):
            samples = self.replay_buffer.sample()
            loss = self.learn_from_sample(samples)
            losses[i] = loss
        return losses

    def learn_from_sample(self, samples: dict):
        observations, weights, actions = (
            samples["observations"],
            samples["weights"],
            torch.from_numpy(samples["actions"]).to(self.device).long(),
        )
        # print("actions", actions)

        # print(observations[0])

        # (B, outputs, atom_size) -[index action dimension by actions]> (B, atom_size)
        online_distributions = self.predict(observations)[
            range(self.config.minibatch_size), actions
        ]
        # (B, atom_size)
        target_distributions = self.compute_target_distributions(samples)
        # print("predicted", online_distributions)
        # print("target", target_distributions)

        weights_cuda = torch.from_numpy(weights).to(torch.float32).to(self.device)
        # (B)
        elementwise_loss = self.config.loss_function(
            online_distributions, target_distributions
        )
        assert torch.all(elementwise_loss) >= 0, "Elementwise Loss: {}".format(
            elementwise_loss
        )
        assert (
            elementwise_loss.shape == weights_cuda.shape
        ), "Loss Shape: {}, Weights Shape: {}".format(
            elementwise_loss.shape, weights_cuda.shape
        )
        loss = elementwise_loss * weights_cuda
        self.optimizer.zero_grad()
        loss.mean().backward()
        if self.config.clipnorm > 0:
            # print("clipnorm", self.config.clipnorm)
            clip_grad_norm_(self.model.parameters(), self.config.clipnorm)

        self.optimizer.step()
        self.update_replay_priorities(
            samples=samples,
            priorities=elementwise_loss.detach().to("cpu").numpy()
            + self.config.per_epsilon,
        )
        self.model.reset_noise()
        self.target_model.reset_noise()
        return loss.detach().to("cpu").mean().item()

    def update_replay_priorities(self, samples, priorities):
        self.replay_buffer.update_priorities(samples["indices"], priorities)

    def compute_target_distributions(self, samples):
        with torch.no_grad():
            discount_factor = self.config.discount_factor**self.config.n_step
            delta_z = (
                (self.config.v_max - self.config.v_min) / (self.config.atom_size - 1)
                if self.config.atom_size > 1
                else (self.config.v_max - self.config.v_min)
            )
            next_observations, rewards, dones = (
                samples["next_observations"],
                torch.from_numpy(samples["rewards"]).to(self.device).view(-1, 1),
                torch.from_numpy(samples["dones"]).to(self.device).view(-1, 1),
            )
            online_distributions = self.predict(next_observations)
            target_distributions = self.predict_target(next_observations)
            next_actions = self.select_actions(
                online_distributions
            )  # {} is the info but we are not doing action masking yet
            # (B, outputs, atom_size) -[index by [0..B-1, a_0..a_B-1]]> (B, atom_size)
            probabilities = target_distributions[
                range(self.config.minibatch_size), next_actions
            ]
            # print(probabilities)

            # (B, 1) + k(B, atom_size) * (B, atom_size) -> (B, atom_size)
            Tz = (rewards + discount_factor * (~dones) * self.support).clamp(
                self.config.v_min, self.config.v_max
            )
            # print("Tz", Tz)

            # all elementwise
            b: torch.Tensor = (Tz - self.config.v_min) / delta_z
            l, u = (
                torch.clamp(b.floor().long(), 0, self.config.atom_size - 1),
                torch.clamp(b.ceil().long(), 0, self.config.atom_size - 1),
            )
            # print("b", b)
            # print("l", l)
            # print("u", u)

            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.config.atom_size - 1)) * (l == u)] += 1
            # print("fixed l", l)
            # print("fixed u", u)

            m = torch.zeros_like(probabilities)
            m.scatter_add_(dim=1, index=l, src=probabilities * ((u.float()) - b))
            m.scatter_add_(dim=1, index=u, src=probabilities * ((b - l.float())))
            return m

    def fill_replay_buffer(self):
        with torch.no_grad():
            state, info = self.env.reset()
            for i in range(self.config.min_replay_buffer_size + self.config.n_step - 1):
                print("filling replay buffer", i)
                # dist = self.predict(state)
                # action = self.select_actions(dist).item()
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, next_info = self.env.step(
                    action
                )
                done = terminated or truncated
                # print(state)
                self.replay_buffer.store(
                    state, info, action, reward, next_state, next_info, done
                )
                # print(self.replay_buffer.observation_buffer[0])
                state = next_state
                info = next_info
                if done:
                    state, info = self.env.reset()
                # gc.collect()

    def update_target_model(self):
        if self.config.soft_update:
            for wt, wp in zip(self.target_model.parameters(), self.model.parameters()):
                wt.copy_(self.config.ema_beta * wt + (1 - self.config.ema_beta) * wp)
        else:
            self.target_model.load_state_dict(self.model.state_dict())

    def train(self):
        start_time = time()
        score = 0
        target_model_updated = (False, False)  # (score, loss)

        self.fill_replay_buffer()
        state, info = self.env.reset()

        # self.training_steps += self.start_training_step
        for training_step in range(self.start_training_step, self.training_steps):
            print("training step", training_step)
            with torch.no_grad():
                for _ in range(self.config.replay_interval):
                    values = self.predict(state)
                    action = epsilon_greedy_policy(
                        values,
                        self.config.eg_epsilon,
                        wrapper=lambda values: self.select_actions(values).item(),
                        range=self.num_actions,
                    ).item()
                    next_state, reward, terminated, truncated, next_info = (
                        self.env.step(action)
                    )
                    done = terminated or truncated
                    # print(state)
                    self.replay_buffer.store(
                        state, info, action, reward, next_state, next_info, done
                    )
                    state = next_state
                    info = next_info
                    score += reward
                    self.replay_buffer.set_beta(
                        update_per_beta(
                            self.replay_buffer.beta,
                            self.config.per_beta_final,
                            self.training_steps,
                        )
                    )

                    if done:
                        state, info = self.env.reset()
                        score_dict = {
                            "score": score,
                            "target_model_updated": target_model_updated[0],
                        }
                        self.stats["score"].append(score_dict)
                        target_model_updated = (False, target_model_updated[1])
                        score = 0

                if training_step % self.config.transfer_interval == 0:
                    target_model_updated = (True, True)
                    # stats["test_score"].append(
                    #     {"target_model_weight_update": training_step}
                    # )
                    self.update_target_model()

            for minibatch in range(self.config.num_minibatches):
                losses = self.learn()
                # print(losses)
                loss_mean = losses.mean()
                # could do things other than taking the mean here
                self.stats["loss"].append(
                    {"loss": loss_mean, "target_model_updated": target_model_updated[1]}
                )
                target_model_updated = (target_model_updated[0], False)

            if training_step % self.checkpoint_interval == 0 and training_step > 0:
                # print(self.stats["score"])
                self.save_checkpoint(
                    training_step,
                    training_step * self.config.replay_interval,
                    time() - start_time,
                )
            # gc.collect()

        self.save_checkpoint(
            training_step,
            training_step * self.config.replay_interval,
            time() - start_time,
        )
        self.env.close()

    def load_model_weights(self, weights_path: str):
        state_dict = torch.load(weights_path)
        self.model.load_state_dict(state_dict)
        self.target_model.load_state_dict(self.model.state_dict())
