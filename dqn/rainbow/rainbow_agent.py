import math
from time import time
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.sgd import SGD
from torch.optim.adam import Adam
import numpy as np
from agent_configs import RainbowConfig
from utils import (
    update_per_beta,
    get_legal_moves,
    current_timestamp,
    action_mask,
    epsilon_greedy_policy,
    CategoricalCrossentropyLoss,
    HuberLoss,
    KLDivergenceLoss,
    MSELoss,
    update_inverse_sqrt_schedule,
    update_linear_schedule,
)

import sys

from utils.utils import epsilon_greedy_policy

sys.path.append("../../")

from base_agent.agent import BaseAgent
from replay_buffers.prioritized_n_step_replay_buffer import PrioritizedNStepReplayBuffer
from dqn.rainbow.rainbow_network import RainbowNetwork
from stats.stats import PlotType, StatTracker


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
        from_checkpoint=False,
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

        if self.config.optimizer == Adam:
            self.optimizer: torch.optim.Optimizer = self.config.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == SGD:
            print("Warning: SGD does not use adam_epsilon param")
            self.optimizer: torch.optim.Optimizer = self.config.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )

        self.replay_buffer = PrioritizedNStepReplayBuffer(
            observation_dimensions=self.observation_dimensions,
            observation_dtype=self.observation_dtype,
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
            num_players=self.config.game.num_players,
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

        self.eg_epsilon = self.config.eg_epsilon

        self.stats = StatTracker(
            model_name=self.model_name,
            stat_keys=[
                "score",
                "loss",
                "test_score",
            ],
            target_values={
                "score": (
                    self.env.spec.reward_threshold
                    if hasattr(self.env, "spec") and self.env.spec.reward_threshold
                    else None
                ),
                "test_score": (
                    self.env.spec.reward_threshold
                    if hasattr(self.env, "spec") and self.env.spec.reward_threshold
                    else None
                ),
            },
            use_tensor_dicts={
                "test_score": ["score", "max_score", "min_score"],
            },
        )
        self.stats.add_plot_types(
            "score",
            PlotType.ROLLING_AVG,
            PlotType.BEST_FIT_LINE,
            rolling_window=100,
        )
        self.stats.add_plot_types("test_score", PlotType.BEST_FIT_LINE)
        self.stats.add_plot_types("loss", PlotType.ROLLING_AVG, rolling_window=100)

    def checkpoint_model_weights(self, checkpoint):
        checkpoint = super().checkpoint_model_weights(checkpoint)
        checkpoint["target_model"] = self.target_model.state_dict()
        return checkpoint

    def load_model_weights(self, checkpoint):
        self.model.load_state_dict(checkpoint["model"])
        self.target_model.load_state_dict(checkpoint["target_model"])
        self.target_model.eval()

    def predict(self, states, *args, **kwargs) -> torch.Tensor:
        # could change type later
        state_input = self.preprocess(states)
        q_distribution: torch.Tensor = self.model(state_input)
        return q_distribution

    def predict_target(self, states) -> torch.Tensor:
        # could change type later
        state_input = self.preprocess(states)
        q_distribution: torch.Tensor = self.target_model(state_input)
        return q_distribution

    def select_actions(
        self,
        distribution,
        info: dict = None,
    ):

        if self.config.atom_size > 1:
            q_values = distribution * self.support
            q_values = q_values.sum(2, keepdim=False)
        else:
            q_values = distribution
        if "legal_moves" in info:
            legal_moves = get_legal_moves(info)
            q_values = action_mask(
                q_values, legal_moves, mask_value=-float("inf"), device=self.device
            )
        # print("Q Values", q_values)
        # q_values with argmax ties
        # selected_actions = torch.stack(
        #     [
        #         torch.tensor(np.random.choice(np.where(x.cpu() == x.cpu().max())[0]))
        #         for x in q_values
        #     ]
        # )
        # print(selected_actions)
        selected_actions = q_values.argmax(1, keepdim=False)
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
            # torch.from_numpy(samples["actions"]).to(self.device).long(),
            samples["actions"].to(self.device).long(),
        )
        # print("actions", actions)

        # print("Observations", observations)
        # (B, outputs, atom_size) -[index action dimension by actions]> (B, atom_size)
        online_predictions = self.predict(observations)[
            range(self.config.minibatch_size), actions
        ]
        # for param in self.model.parameters():
        #     print(param)
        # print(self.predict(observations))
        # print(online_predictions)
        # (B, atom_size)
        if self.config.atom_size > 1:
            # print("using categorical dqn loss")
            assert isinstance(
                self.config.loss_function, KLDivergenceLoss
            ) or isinstance(
                self.config.loss_function, CategoricalCrossentropyLoss
            ), "Only KLDivergenceLoss and CategoricalCrossentropyLoss are supported for atom_size > 1, recieved {}".format(
                self.config.loss_function
            )
            target_predictions = self.compute_target_distributions(samples)
        else:
            # print("using default dqn loss")
            assert isinstance(self.config.loss_function, HuberLoss) or isinstance(
                self.config.loss_function, MSELoss
            ), "Only HuberLoss or MSELoss are supported for atom_size = 1, recieved {}".format(
                self.config.loss_function
            )
            # next_observations, rewards, dones = (
            #     torch.from_numpy(samples["next_observations"]).to(self.device),
            #     torch.from_numpy(samples["rewards"]).to(self.device),
            #     torch.from_numpy(samples["dones"]).to(self.device),
            # )
            next_observations, rewards, dones = (
                samples["next_observations"].to(self.device),
                samples["rewards"].to(self.device),
                samples["dones"].to(self.device),
            )

            next_action_masks = samples["next_action_masks"].to(self.device)
            next_infos = [
                {"legal_moves": torch.nonzero(mask).squeeze().tolist()}
                for mask in next_action_masks
            ]
            # next_infos = samples["next_infos"].to(self.device)
            target_predictions = self.predict_target(next_observations)  # next q values
            # print("Next q values", target_predictions)
            # print("Current q values", online_predictions)
            # print(self.predict(next_observations))
            next_actions = self.select_actions(
                self.predict(next_observations),  # current q values
                info=next_infos,
            )

            # print("RL Learning Summary")
            # print("Online Predictions:", online_predictions)
            # print("Target Predictions:", target_predictions)
            # print("Next Actions:", next_actions)
            # print("Rewards:", rewards)
            # print("Dones:", dones)

            target_predictions = target_predictions[
                range(self.config.minibatch_size), next_actions
            ]  # this might not work
            # print(target_predictions)
            target_predictions = (
                rewards + self.config.discount_factor * (~dones) * target_predictions
            )
            # print(target_predictions)

        # print("predicted", online_distributions)
        # print("target", target_distributions)

        # weights_cuda = torch.from_numpy(weights).to(torch.float32).to(self.device)
        weights_cuda = weights.to(torch.float32).to(self.device)
        # (B)
        elementwise_loss = self.config.loss_function(
            online_predictions, target_predictions
        )
        # print("Loss", elementwise_loss.mean())
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
        # print("computing target distributions")
        with torch.no_grad():
            discount_factor = self.config.discount_factor**self.config.n_step
            delta_z = (self.config.v_max - self.config.v_min) / (
                self.config.atom_size - 1
            )
            # next_observations, rewards, dones = (
            #     samples["next_observations"],
            #     torch.from_numpy(samples["rewards"]).to(self.device).view(-1, 1),
            #     torch.from_numpy(samples["dones"]).to(self.device).view(-1, 1),
            # )
            next_observations, rewards, dones = (
                samples["next_observations"],
                samples["rewards"].to(self.device).view(-1, 1),
                samples["dones"].to(self.device).view(-1, 1),
            )

            online_distributions = self.predict(next_observations)
            target_distributions = self.predict_target(next_observations)

            # print(samples["next_infos"])
            # recreate legal moves from action masks samples
            next_action_masks = samples["next_action_masks"].to(self.device)
            next_infos = [
                {"legal_moves": torch.nonzero(mask).squeeze().tolist()}
                for mask in next_action_masks
            ]
            next_actions = self.select_actions(
                online_distributions,
                info=next_infos,
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
            # dones = dones.squeeze()
            # masked_probs = torch.ones_like(probabilities) / self.config.atom_size
            # masked_probs[~dones] = probabilities[~dones]

            m = torch.zeros_like(probabilities)
            m.scatter_add_(dim=1, index=l, src=probabilities * ((u.float()) - b))
            m.scatter_add_(dim=1, index=u, src=probabilities * ((b - l.float())))
            # print("old_m", (m * self.support).sum(-1))

            # projected_distribution = torch.zeros_like(probabilities)
            # projected_distribution.scatter_add_(
            #     dim=1, index=l, src=masked_probs * (u.float() - b)
            # )
            # projected_distribution.scatter_add_(
            #     dim=1, index=u, src=masked_probs * (b - l.float())
            # )
            # print("m", (projected_distribution * self.support).sum(-1))
            return m

    def fill_replay_buffer(self):
        print("replay buffer size:", self.replay_buffer.size)
        with torch.no_grad():
            state, info = self.env.reset()
            target_size = self.config.min_replay_buffer_size
            while self.replay_buffer.size < target_size:
                if (self.replay_buffer.size % (math.ceil(target_size / 100))) == 0:
                    print(
                        f"filling replay buffer: {self.replay_buffer.size} / ({target_size})"
                    )
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

    def update_eg_epsilon(self, training_step):
        if self.config.eg_epsilon_decay_type == "linear":
            # print("decaying eg epsilon linearly")
            self.eg_epsilon = update_linear_schedule(
                self.config.eg_epsilon_final,
                self.config.eg_epsilon_final_step,
                self.config.eg_epsilon,
                training_step,
            )
        elif self.config.eg_epsilon_decay_type == "inverse_sqrt":
            self.eg_epsilon = update_inverse_sqrt_schedule(
                self.config.eg_epsilon,
                training_step,
            )
        else:
            raise ValueError(
                "Invalid epsilon decay type: {}".format(
                    self.config.eg_epsilon_decay_type
                )
            )

    def train(self):
        super().train()
        start_time = time() - self.training_time
        score = 0
        self.fill_replay_buffer()

        state, info = self.env.reset()

        while self.training_step < self.config.training_steps:
            if self.training_step % self.config.print_interval == 0:
                self.print_training_progress()

            with torch.no_grad():
                for _ in range(self.config.replay_interval):
                    values = self.predict(state)
                    # print(values)
                    action = epsilon_greedy_policy(
                        values,
                        info,
                        self.eg_epsilon,
                        wrapper=lambda values, info: self.select_actions(
                            values, info
                        ).item(),
                    )
                    # print("Action", action)
                    # print("Epislon Greedy Epsilon", self.eg_epsilon)
                    next_state, reward, terminated, truncated, next_info = (
                        self.env.step(action)
                    )
                    done = terminated or truncated
                    # print("State", state)
                    self.replay_buffer.store(
                        state, info, action, reward, next_state, next_info, done
                    )
                    state = next_state
                    info = next_info
                    score += reward
                    if done:
                        state, info = self.env.reset()
                        self.stats.append("score", score)
                        score = 0
            self.replay_buffer.set_beta(
                update_per_beta(
                    self.replay_buffer.beta,
                    self.config.per_beta_final,
                    self.training_steps,
                    self.config.per_beta,
                )
            )

            self.update_eg_epsilon(self.training_step + 1)
            # print("replay buffer size", len(self.replay_buffer))
            for minibatch in range(self.config.num_minibatches):
                if len(self.replay_buffer) < self.config.min_replay_buffer_size:
                    break
                losses = self.learn()
                # print(losses)
                loss_mean = losses.mean()
                # could do things other than taking the mean here
                self.stats.append("loss", loss_mean)

            if self.training_step % self.config.transfer_interval == 0:
                self.update_target_model()

            if self.training_step % self.checkpoint_interval == 0:
                self.stats.set_time_elapsed(time() - start_time)
                self.stats.increment_steps(
                    self.training_step * self.config.replay_interval
                )
                self.save_checkpoint()
            # gc.collect()
            self.training_step += 1

        self.stats.set_time_elapsed(time() - start_time)
        self.stats.increment_steps(self.training_step * self.config.replay_interval)
        self.save_checkpoint()
        # self.env.close()
