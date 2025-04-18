import sys
import torch
from cfr_network import CFRNetwork
from agent_configs import CFRConfig

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
        self.observation_space = config.observation_space
        self.action_space = config.action_space
        self.name = name
        self.env = env
        self.device = device
        self.config = config
        self.players = config.num_players
        self.active_player_obj = config.active_player_obj
        self.network = CFRNetwork(
            config=config.network,
            input_shape=self.observation_space,
            output_shape=self.action_space,
        )

        self.value_buffer = [nfsp_reservoir_buffer.NFSPReservoirBuffer(
            observation_dimensions=self.observation_space,
            observation_dtype=torch.float32,
            max_size=config.replay_buffer_size,
            num_actions=self.action_space,
            batch_size=config.minibatch_size,
            compressed_observations=True) for _ in range(self.players)]
      
        self.policy_buffer = nfsp_reservoir_buffer.NFSPReservoirBuffer(
            observation_dimensions=self.observation_space,
            observation_dtype=torch.float32,
            max_size=config.replay_buffer_size,
            num_actions=self.action_space,
            batch_size=config.minibatch_size,
            compressed_observations=True)
   
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
        positive_regret = torch.clamp(torch.from_numpy(predicted), min=0)
        info = info.type(torch.float)
        positive_regret = positive_regret* info
        regret_sum = positive_regret.sum(dim=-1)
        sample = None
        policy = None
        if regret_sum == 0:
            if mask_actions:
                # if all regrets are zero, sample uniformly from action space

                sample = torch.multinomial(info, 1)
                policy = np.array([1/torch.sum(info)]*predicted.shape[0])
                policy[info == 0] = 0
            else:
                sample = torch.randint(0, predicted.shape[0]-1, 1)
                policy = np.array([1/predicted.shape[0]]*predicted.shape[0])
        else:
            if mask_actions:
                sample = torch.multinomial((positive_regret / regret_sum)*info, 1)
                policy = np.array(positive_regret.cpu().numpy() / regret_sum.cpu().numpy())
                policy[info == 0] = 0

        sample = sample.squeeze(0).cpu().numpy()
        return sample, policy
    
    def learn(self, player_id):
        """
        For each CFR iteration, update traverser's value network from scratch (batches number of SGD mini-batches).
        """
        # GET BATCHES FROM VALUE BUFFER
        self.network.values[player_id].reset()

        for i in range(self.config.steps_per_epoch):
            print("PLAYER ID", player_id)
            print("LEARNING ITERATION", i)
            samples = self.value_buffer[player_id].sample()
            # get num of obs
            num_samples = len(samples["observations"])
            # LOOP THROUGH FOR CONFIG NUMBER OF SGD ITERS
            observations = torch.tensor(np.array([samples["observations"][sample]["observation"] for sample in range(num_samples)]))
            target_policy = torch.tensor(np.array([samples["targets"][sample] for sample in range(num_samples)]))
            iteration = torch.tensor([samples["infos"][sample]["iteration"] for sample in range(num_samples)])
            loss = self.network.values[player_id].learn(batch=[iteration, observations, target_policy])
            print("LOSS", loss)
        
    def policy_learn(self):
        """
        Final training cycle, updating policy network on average strategy of players in past t iterations.
        """
        for i in range(self.config.steps_per_epoch):
            print("LEARNING ITERATION", i)
            samples = self.policy_buffer.sample()
            # get num of obs
            num_samples = len(samples["observations"])
            # LOOP THROUGH FOR CONFIG NUMBER OF SGD ITERS
            observations = torch.tensor(np.array([samples["observations"][sample]["observation"] for sample in range(num_samples)]))
            target_policy = torch.tensor(np.array([samples["targets"][sample] for sample in range(num_samples)]))
            iteration = torch.tensor([samples["infos"][sample]["iteration"] for sample in range(num_samples)])
            loss = self.network.policy.learn(batch=[iteration, observations, target_policy])
            print("POLICY LOSS", loss)

    def train(self):
        start_time = time.time()
        for i in range(self.config.training_steps):
            # FOR training_steps CFR ITERATIONS         
            for p in range(self.players):
                # FOR EACH PLAYER, DO T TRAVERSALS
                for t in range(self.config.traversals):
                    # FOR EACH TRAVERSAL, RESET ENVIRONEMENT (DEBATABLE STEP) BUT RESET WITH SET SEED FOR RECREATION
                    random_seed = np.random.randint(0, 2**32 - 1)
                    self.env.reset(seed=random_seed)
                    active_player = self.env.agent_selection[-1]
                    self.active_player_obj.set_active_player(int(active_player))
                    observation, reward, termination, truncation, info = self.env.last()
                    traverse_history = [] # for recreation
                    self.traverse(history=traverse_history, iteration_T=i, seed=random_seed, game=self.env, active_player=self.active_player_obj.get_active_player(), traverser=p)
                    print(f"Player {p} Traversal {t}")
                # NOW LEARN FROM THE PLAYER'S VALUE BUFFER
                self.learn(p)
            print(f"Iteration {i} done")
        
        self.policy_learn()

        self.training_time = time.time() - start_time
        self.total_environment_steps = self.config.training_steps * self.config.steps_per_epoch
        self.save_checkpoint()
        return self.network.policy

    def save_checkpoint(self):
        """
        Save the model checkpoint.
        """
        torch.save(self.network.state_dict(), f"checkpoints/{self.name}.pt")
        print(f"Checkpoint saved at {self.name}.pt")

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
        traverser_id = traverser
        self.active_player_obj.set_active_player(active_player)
        # GET CURRENT STATE
        observation, reward, termination, truncation, info = game.last()
        if termination or truncation:
            # IF TERMINATED THEN PASS UP VALUE TO PARET (THIS IS A RECURSIVE CALL)
            return reward #IE PAYOFF only for activate player
        elif active_player == traverser_id:
            predictions = self.predict(observation["observation"], active_player)
            sample, policy = self.select_actions(predictions, info=torch.from_numpy(observation["action_mask"]), mask_actions=True, traverser=traverser_id) # MASKING NOT YET IMPLEMENTED
            # if active player, branch off and traverse
            v_policy = [0] * len(policy)
            reg = [0] * len(policy)
            for i in range(len(policy)):
                if policy[i] == 0:
                    continue
                recreate_env = self.recreate_env(game, history, seed) # UGGGG
                recreate_env.step(i)
                history.append(i)
                v_policy[i] = self.traverse(copy.deepcopy(history), iteration_T, seed, recreate_env,self.active_player_obj.next(), traverser=traverser_id) # RAWRRRR RECURSIVE DO WE NEED THIS???????
                history.pop()
            for i in range(len(policy)):
                reg[i] = v_policy[i] - torch.sum(torch.tensor(v_policy) * torch.tensor(policy)).item()
            # ADD TO ACTIVE PLAYER'S VALUE BUFFER
            self.value_buffer[active_player].store(observation, target_policy=reg, iteration=iteration_T, info={"iteration": iteration_T+1})
            return torch.sum(torch.tensor(v_policy) * torch.tensor(policy)).item() # RETURN VALUE FOR ACTIVE PLAYER #### ALTERNATIVELY JUST RETURN REWARD OF SAMPLED ACTION
        else:
            predictions = self.predict(observation["observation"], active_player)
            sample, policy = self.select_actions(predictions, info=torch.from_numpy(observation["action_mask"]), mask_actions=True, traverser=traverser_id) # MASKING NOT YET IMPLEMENTED
            self.policy_buffer.store(observation, target_policy=policy, iteration=iteration_T, info={"iteration": iteration_T+1})
            game.step(sample)
            history.append(sample)
            return self.traverse(copy.deepcopy(history), iteration_T, seed, game, self.active_player_obj.next(), traverser=traverser_id)


    def recreate_env(self, env, history, seed):
        """
        Recreate the environment from the history and seed.
        :param env: The environment object.
        :param history: The current history of the game (for recreation purposes).
        :param seed: The current seed of the game (for recreation purposes).
        :return: The recreated environment.
        """
        # RECREATE ENVIRONMENT
        new_env = copy.deepcopy(env)
        new_env.reset(seed=seed)
        for action in history:
            new_env.step(action)
        return new_env

    def evaluate(self, const_policy, tested_agent, eval_games):
        """
        Evaluate the model on the test set.
        # MUST APPROXIMATE BEST RESPONSE LOSS FROM THE POLICY NETWORK
        :return: The evaluation loss.
        # add new policy network, one new value buffer
        """
        rewards = []
        for i in range(eval_games):
            # FOR EACH EVAL GAME, RESET ENVIRONEMENT (DEBATABLE STEP) BUT RESET WITH SET SEED FOR RECREATION
            random_seed = np.random.randint(0, 2**32 - 1)
            self.env.reset(seed=random_seed)
            active_player = self.env.agent_selection[-1]
            self.active_player_obj.set_active_player(int(active_player))
            observation, reward, termination, truncation, info = self.env.last()
            while not termination or not truncation:
                # GET CURRENT STATE
                observation, reward, termination, truncation, info = self.env.last()
                active_player = self.active_player_obj.get_active_player()
                if active_player == 1:
                    predictions = self.predict(observation["observation"], active_player)
                    sample, policy = self.select_actions(predictions, info=torch.from_numpy(observation["action_mask"]), mask_actions=True)
                else:
                    sample = 2
                # if active player, branch off and traverse
                self.env.step(sample)
                self.active_player_obj.next()
            if reward!=0:
                rewards.append(reward)
        return np.mean(rewards), np.std(rewards)


       
