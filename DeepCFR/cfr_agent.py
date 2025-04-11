import sys
import torch
from agent_configs import CFRConfig
from cfr_network import CFRNetwork
from replay_buffers import nfsp_reservoir_buffer
import datetime
sys.path.append("../")
from base_agent.agent import BaseAgent
import time
import numpy as np
import copy

class CFRAgent(): # BaseAgent):
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
        # super(CFRAgent, self).__init__(env, config, name=name, device=device, from_checkpoint=from_checkpoint)
        self.name = name
        self.env = env
        self.device = device
        self.config = config
        self.players = config.num_players
        self.network = CFRNetwork(
            config=config.network,
            input_shape=env.observation_space.shape,
            output_shape=env.action_space.n,
        )

        self.value_buffer = [nfsp_reservoir_buffer.NFSPReservoirBuffer(
            observation_dimensions=env.observation_space.shape,
            observation_dtype=torch.float32,
            max_size=config.replay_buffer_size,
            num_actions=env.action_space.n,
            batch_size=config.minibatch_size,
            compressed_observations=False) for _ in range(self.players)]
      
        self.policy_buffer = nfsp_reservoir_buffer.NFSPReservoirBuffer(
            observation_dimensions=env.observation_space.shape,
            observation_dtype=torch.float32,
            max_size=config.replay_buffer_size,
            num_actions=env.action_space.n,
            batch_size=config.minibatch_size,
            compressed_observations=False)
   
        self.traversals = config.traversals
        self.steps_per_epoch = config.steps_per_epoch
        self.training_steps = config.training_steps

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
    

    def select_actions(self, predicted, info, mask_actions=False, traverser=False):
        """
        Select action based on predicted regret valeus.

        """
        positive_regret = torch.clamp(predicted, min=0)
        regret_sum = positive_regret.sum(dim=1)
        sample = None
        policy = None
        if regret_sum == 0:
            sample = torch.randint(0, predicted.shape[1], (predicted.shape[0],))
            policy = np.array([1/predicted.shape[1]]*predicted.shape[1])
        else:
            sample = torch.multinomial(positive_regret / regret_sum, 1)
            policy = np.array(predicted.cpu().numpy() / regret_sum.cpu().numpy())

        sample = sample.squeeze(1).cpu().numpy()
        return sample, policy

    def learn(self, player_id):
        """
        For each CFR iteration, update traverser's value network from scratch (batches number of SGD mini-batches).
        """
        # GET BATCHES FROM VALUE BUFFER
        samples = self.value_buffer[player_id].sample()
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
        for i in range(self.config.training_steps):
            # FOR training_steps CFR ITERATIONS         
            for p in range(self.players):
                # FOR EACH PLAYER, DO T TRAVERSALS
                for t in range(self.config.traversals):
                    # FOR EACH TRAVERSAL, RESET ENVIRONEMENT (DEBATABLE STEP) BUT RESET WITH SET SEED FOR RECREATION
                    random_seed = np.random.randint(0, 2**32 - 1)
                    self.env.seed(random_seed)
                    observation, reward, termination, truncation, info = self.env.reset()
                    traverse_history = [] # for recreation

                    self.traverse(history=traverse_history, iteration_T=i, seed=random_seed, game=self.env, active_player=self.env.agent_selection, traverser=p)
                    print(f"Player {p} Traversal {t}")
                # NOW LEARN FROM THE PLAYER'S VALUE BUFFER
                self.learn(p)
            print(f"Iteration {i} done")
        
        self.policy_learn()

        self.training_time = time() - start_time
        self.total_environment_steps = self.config.training_steps * self.config.steps_per_epoch
        self.save_checkpoint()
        self.env.close()

        return self.network.policy


    @torch.no_grad()
    def traverse(self, history, iteration_T, seed, game, active_player, traverser):
        """
        Traverse the game tree recursive call.
        :param history: The current history of the game (for recreation purposes).
        :param iteration_T: The current iteration of the CFR algorithm (for linear cfr weighting).
        :param seed: The current seed of the game (for recreation purposes).
        :param env: The environment object.
        :param active_player: The current active player.
        """
        # GET CURRENT STATE
        observation, reward, termination, truncation, info = game.last()
        if termination or truncation:
            # IF TERMINATED THEN PASS UP VALUE TO PARET (THIS IS A RECURSIVE CALL)
            return reward #IE PAYOFF only for activate player
        elif active_player == traverser:
            predictions = self.predict(observation["observation"], active_player)
            sample, policy = self.select_actions(predictions, info, mask_actions=observation["mask"], traverser=True) # MASKING NOT YET IMPLEMENTED
            # if active player, branch off and traverse
            v_policy = [0] * len(policy)
            reg = [0] * len(policy)
            for i in range(len(policy)):
                recreate_env = self.recreate_env(game, history, seed) # UGGGG
                recreate_env.step(i)
                v_policy[i] = self.traverse(history, iteration_T, seed, recreate_env, active_player.next(), traverser) # RAWRRRR RECURSIVE DO WE NEED THIS???????
            for i in range(len(policy)):
                reg[i] = v_policy[i] - torch.sum(torch.tensor(v_policy) * torch.tensor(policy)).item()
            # ADD TO ACTIVE PLAYER'S VALUE BUFFER
            self.value_buffer[active_player].add(observation, reg, iteration_T)
        else:
            predictions = self.predict(observation["observation"], active_player)
            sample, policy = self.select_actions(predictions, info, mask_actions=observation["mask"], traverser=False) # MASKING NOT YET IMPLEMENTED
            self.policy_buffer.add(observation, policy, iteration_T)
            game.step(sample)
            history.append(sample)
            return self.traverse(history, iteration_T, seed, game, active_player.next(), traverser)


    def recreate_env(self, env, history, seed):
        """
        Recreate the environment from the history and seed.
        :param env: The environment object.
        :param history: The current history of the game (for recreation purposes).
        :param seed: The current seed of the game (for recreation purposes).
        :return: The recreated environment.
        """
        # RECREATE ENVIRONMENT
        env = copy.deepcopy(env)
        env.seed(seed)
        env.reset()
        for action in history:
            env.step(action)
        return env
