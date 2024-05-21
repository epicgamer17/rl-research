import tensorflow as tf
from NFSP.supervised_network import SupervisedNetwork
from base_agent.agent import BaseAgent
from replay_buffers.nfsp_reservoir_buffer import ReservoirBuffer
import tensorflow_probability as tfp


class BaseImitationAgent(BaseAgent):
    def __init__(self, env, config, name, replay_buffer, network, loss_function):
        super().__init__(env, config, name)
        self.replay_buffer = replay_buffer
        self.model = network
        self.loss_function = loss_function

    def experience_replay(self):
        for training_iteration in range(self.config.training_iterations):
            with tf.GradientTape() as tape:
                sample = self.replay_buffer.sample()
                observations = sample["observations"]
                targets = sample["targets"]

                state_input = self.prepare_states(observations)
                policy = self.model(inputs=state_input)
                # LEGAL MOVE MASKING?
                loss = self.loss_function(
                    targets, policy
                )  # catergorical cross entropy for policies

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.config.optimizer.apply_gradients(
                    grads_and_vars=zip(gradients, self.model.trainable_variables)
                )

                # RESET NOISE IF IM DOING THAT
        return loss

    def select_action(self, state, legal_moves=None):
        raise NotImplementedError

    def predict_single(self, state, legal_moves=None):
        raise NotImplementedError
