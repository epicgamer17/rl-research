import sys
import torch
from agent_configs import CFRConfig
from cfr_network import CFRNetwork
from replay_buffers import nfsp_reservoir_buffer
import datetime
sys.path.append("../")
from base_agent.agent import BaseAgent
import time

class CFRAgent(BaseAgent):
    def __init__(
            self,
            env,
            config=CFRConfig,
            name=datetime.datetime.now().timestamp(),
            device: torch.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            # MPS is sometimes useful for M2 instances, but only for large models/matrix multiplications otherwise CPU is faster
            # else (
            #     torch.device("mps")
            #     if torch.backends.mps.is_available() and torch.backends.mps.is_built()
            else torch.device("cpu")
            # )
        ),
        from_checkpoint=False
    ):
        super(CFRAgent, self).__init__(env, config, name=name, device=device, from_checkpoint=from_checkpoint)
        self.config = config
        self.players = config["num_players"]
        self.network = CFRNetwork(
            config=config["network"],
            input_shape=env.observation_space.shape,
            output_shape=env.action_space.n,
        )

        self.value_buffer = nfsp_reservoir_buffer.NFSPReservoirBuffer(
            observation_dimensions=env.observation_space.shape,
            observation_dtype=torch.float32,
            max_size=config["replay_buffer_size"],
            num_actions=env.action_space.n,
            batch_size=config["minibatch_size"],
            compressed_observations=False)
        
        self.policy_buffer = nfsp_reservoir_buffer.NFSPReservoirBuffer(
            observation_dimensions=env.observation_space.shape,
            observation_dtype=torch.float32,
            max_size=config["replay_buffer_size"],
            num_actions=env.action_space.n,
            batch_size=config["minibatch_size"],
            compressed_observations=False)

    def predict(self, state, player_id):
        """
        Predict the action probabilities for a given state.
        :param state: The current state of the environment.
        :return: Action probabilities.
        """
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            action_probs = self.network.values[player_id](state)
            return action_probs.cpu().numpy()
    

    def select_actions(self, predicted, info, mask_actions=False):
        """
        Select action based on predicted regret valeus.

        """
        positive_regret = torch.clamp(predicted, min=0)
        regret_sum = positive_regret.sum(dim=1)
        if regret_sum == 0:
            action = torch.randint(0, predicted.shape[1], (predicted.shape[0],))
        else:
            action = torch.multinomial(positive_regret / regret_sum, 1)
        action = action.squeeze(1).cpu().numpy()
        return action

    def learn(self, player_id):
        """
        For each CFR iteration, update traverser's value network from scratch (batches number of SGD mini-batches).
        """
        # GET BATCHES FROM VALUE BUFFER
        samples = self.value_buffer.sample()
        # LOOP THROUGH FOR CONFIG NUMBER OF SGD ITERS
        self.network.values[player_id].reset()
        for sample in samples:
            self.network.values[player_id].learn(sample)
        
    def policy_learn(self):
        """
        Final training cycle, updating policy network on average strategy of players in past t iterations.
        """
        # GET BATCHES FROM POLICY BUFFER
        samples = self.policy_buffer.sample()
        # LOOP THROUGH FOR CONFIG NUMBER OF FINAL POLICY ITERS
        self.network.policy.reset()
        for sample in samples:
            self.network.policy.learn(sample)

    def train(self):
        super().train()

        start_time = time() - self.training_time
        state, info = self.env.reset()
        for i in range(self.config.training_steps):
            # FOR training_steps CFR ITERATIONS
            for p in range(self.players):
                # FOR EACH PLAYER, DO T TRAVERSALS
                for t in range(self.config.traversals):
                    # FOR EACH TRAVERSAL, COLLECT GAME DATA AND PUT IN REPLAY BUFFER
                    # TRAVERSE (PARAMS, REPLAY BUFFERS, OTHER POLICIES)
                    # RECURSIVE CALLS TO TRAVERSE
                    print(f"Player {p} Traversal {t}")
                self.learn(p)
            print(f"Iteration {i} done")
        
        self.policy_learn()

        self.training_time = time() - start_time
        self.total_environment_steps = self.config.training_steps * self.config.steps_per_epoch
        self.save_checkpoint()
        self.env.close()

        return self.network.policy







