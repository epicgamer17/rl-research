import torch
from utils import action_mask, normalize_policy, current_timestamp, get_legal_moves
from base_agent.agent import BaseAgent
from torch.nn.utils import clip_grad_norm_

from imitation_learning.supervised_network import SupervisedNetwork
from replay_buffers.nfsp_reservoir_buffer import NFSPReservoirBuffer


class PolicyImitationAgent(BaseAgent):
    def __init__(
        self,
        env,
        config,
        name=f"policy_imitation_{current_timestamp():.1f}",
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
        super().__init__(env, config, name, device)

        self.replay_buffer = NFSPReservoirBuffer(
            observation_dimensions=self.observation_dimensions,
            observation_dtype=self.env.observation_space.dtype,
            max_size=self.config.replay_buffer_size,
            num_actions=self.num_actions,
            batch_size=self.config.minibatch_size,
        )

        self.model = SupervisedNetwork(
            config,
            self.num_actions,
            (self.config.minibatch_size,) + self.observation_dimensions,
        )

        self.optimizer: torch.optim.Optimizer = self.config.optimizer(
            params=self.model.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
        )

    def select_actions(self, predictions, info):
        distribution = torch.distributions.Categorical(probs=predictions)

        selected_action = distribution.sample()
        return selected_action

    def predict(self, state, info):
        state_input = self.preprocess(state)
        policy = self.model(inputs=state_input)
        # policy = action_mask(
        #     policy, get_legal_moves(info), self.num_actions, mask_value=0
        # )
        # policy = normalize_policy(policy)

        return policy

    def learn(self):
        for training_iteration in range(self.config.training_iterations):
            sample = self.replay_buffer.sample()
            observations = sample["observations"]
            targets = torch.from_numpy(sample["targets"]).to(self.device)

            policy = self.predict(observations, {})
            # LEGAL MOVE MASKING?
            loss = self.config.loss_function(targets, policy).mean()

            self.optimizer.zero_grad()
            loss.backward()
            print(self.config.clipnorm)
            if self.config.clipnorm > 0:
                clip_grad_norm_(self.model.parameters(), self.config.clipnorm)
            self.optimizer.step()

            # RESET NOISE IF IM DOING THAT
        return loss.detach()
