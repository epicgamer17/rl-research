from agent_configs import CFRConfig
from active_player import ActivePlayer
from cfr_agent import CFRAgent
import torch
from cfr_network import CFRNetwork
import numpy as np
import os
import copy
import tensorflow as tf

def evaluatebots(agent1, agent2, num_of_eval_games, mini_env, config, in_size):
    modelselect = CFRAgent(env=mini_env, config=config)
    eval_games = num_of_eval_games
    rewards_player_1 = []
    rewards_player_2  = []
    for i in range(eval_games):
        # FOR EACH EVAL GAME, RESET ENVIRONEMENT (DEBATABLE STEP) BUT RESET WITH SET SEED FOR RECREATION

        modelselect.env.reset()
        observation, reward, termination, truncation, infos =  modelselect.env.last()
        init_starting_player = np.random.randint(0, 2)
        modelselect.active_player_obj.set_active_player(init_starting_player)
        while not termination and not truncation:
            # GET CURRENT STATE
            observation, reward, termination, truncation, infos =  modelselect.env.last()
            if termination or truncation:
                break
            active_player =  modelselect.active_player_obj.get_active_player()
            if active_player == 0:
                predictions = agent1.policy(torch.tensor(observation['observation'], dtype=torch.float32).reshape(1,in_size)).detach().numpy()[0]

                sample, policy = modelselect.select_actions(predictions, info=torch.from_numpy(observation["action_mask"]).type(torch.float), mask_actions=True)
            else:
                # predictions = np.ones(4) / 4
                # sample, policy = modelselect.select_actions(predictions, info=torch.from_numpy(observation["action_mask"]).type(torch.float), mask_actions=True)
                predictions = agent2.policy(torch.tensor(observation['observation'], dtype=torch.float32).reshape(1,in_size)).detach().numpy()[0]
                sample, policy = modelselect.select_actions(predictions, info=torch.from_numpy(observation["action_mask"]).type(torch.float), mask_actions=True)
            # if active player, branch off and traverse
            modelselect.env.step(sample)
            # print("STATE", observation)
            # print("ACTIVE PLAYER: ", active_player)
            # print("ACTION: ", sample)
            # print("REWARD: ", reward)
            # print("TERMINATION: ", termination)
            # print("TRUNCATION: ", truncation)
            # print("INFO: ", infos)
            # print("NEXT STATE: ", modelselect.env.state)
            # print("REWAS", modelselect.env.state.rewards())
            modelselect.active_player_obj.next()
        if init_starting_player == 0:
            final_rewards_p_1 = modelselect.env.state.rewards()[0]
            final_rewards_p_2 = modelselect.env.state.rewards()[1]
        else:
            final_rewards_p_1 = modelselect.env.state.rewards()[1]
            final_rewards_p_2 = modelselect.env.state.rewards()[0]
        rewards_player_1.append(final_rewards_p_1)
        rewards_player_2.append(final_rewards_p_2)
    return rewards_player_1, rewards_player_2
    print("PLAYER 1 REW MEAN: ", np.mean(rewards_player_1))
    print("PLAYER 1 REW STD: ", np.std(rewards_player_1))
    print("PLAYER 2 REW MEAN: ", np.mean(rewards_player_2))
    print("PLAYER 2 REW STD: ", np.std(rewards_player_2))


class NFSPWrapper:
    def __init__(self,env):
        self.game = env
        self.state = self.game.new_initial_state()
        self.observations = {"info_state":[self.state.observation_tensor(_) for _ in range(self.state.num_players())], "legal_actions":[self.state.legal_actions(_) for _ in range(self.state.num_players())]}
        self.agent_selection = str(self.state.current_player())
        self.traverser = None
        self.rewards = [0 for _ in range(self.state.num_players())]
    
    def is_simultaneous_move(self):
        return self.state.is_simultaneous_node()


    def last(self):
        return self.state.is_terminal()

    def current_player(self):
        return self.state.current_player()
    
    def step(self, action):
        if self.state.is_chance_node():
            while self.state.is_chance_node():
                self.state.apply_action(np.random.choice(self.state.legal_actions()))
        else:
            self.state.apply_action(action)
            if self.state.is_chance_node():
                while self.state.is_chance_node():
                    self.state.apply_action(np.random.choice(self.state.legal_actions()))
        
        if not self.state.is_terminal():
            while self.state.is_chance_node():
                self.state.apply_action(np.random.choice(self.state.legal_actions()))
            self.agent_selection =  str(self.state.current_player())

        return self.obs()
    
    def reset(self, seed=None):
        self.state = self.game.new_initial_state()
        while self.state.is_chance_node():
            self.state.apply_action(np.random.choice(self.state.legal_actions()))
        self.agent_selection =  str(self.state.current_player())

        return self.obs()
    
    
    def obs(self):
        if not self.state.is_terminal():
            self.observations["info_state"][self.state.current_player()] = self.state.observation_tensor(self.state.current_player())
            self.observations["legal_actions"][self.state.current_player()] = np.stack(self.state.legal_actions(self.state.current_player()))
            # not_current_player = 1 - self.state.current_player()
            # self.observations["info_state"][not_current_player] = self.state.observation_tensor(not_current_player)
            # self.observations["legal_actions"][not_current_player] = np.stack(self.state.legal_actions_mask(not_current_player))

        if not self.state.is_chance_node():
            self.rewards = self.state.rewards()
        else:
            self.rewards = [0 for _ in range(self.state.num_players())]

        if self.state.is_terminal():
            return {"observation":self.state.observation_tensor(int(self.agent_selection)), "action_mask":np.stack([0,1,2,3])}, self.state.player_reward(self.traverser) if self.traverser is not None else self.state.player_return(int(self.agent_selection)), self.state.is_terminal(), False, "OPENSPIEL"
        return {"observation":self.state.observation_tensor(self.state.current_player()), "action_mask":np.stack(self.state.legal_actions(self.state.current_player()))}, self.state.player_reward(self.traverser) if self.traverser is not None else self.state.player_return(self.state.current_player()), self.state.is_terminal(), False, "OPENSPIEL"
    


def nfsptrain(agents, env, max_nodes, game_string):
    nodes = 0
    checkpoint = 0.1
    while nodes<=max_nodes:
        env.reset()
        player0 = np.random.randint(0, 2)
        templist = [agents[player0], agents[1-player0]]
        templist[0].player_id = 0
        templist[0]._rl_agent.player_id = 0
        templist[1].player_id = 1
        templist[1]._rl_agent.player_id = 1
        while not env.last():
            current_player = env.current_player()
            action, probs = templist[current_player].step(copy.deepcopy(env))
            env.step(action)
            nodes += 1
        print("Nodes:", nodes)
        if nodes >= checkpoint * max_nodes:
            for i in range(len(agents)):
                print("Checkpoint reached: ", checkpoint)
                if not os.path.exists("checkpoints/" + game_string + "/"):
                    os.makedirs("checkpoints/" + game_string + "/")
                if not os.path.exists("checkpoints/"+ game_string + "/nfsp/"):
                    os.makedirs("checkpoints/"+ game_string + "/nfsp/")
                if not os.path.exists("checkpoints/"+ game_string + "/nfsp/" + str(i)):
                    os.makedirs("checkpoints/"+ game_string + "/nfsp/" + str(i))
                if not os.path.exists("checkpoints/"+ game_string + "/nfsp/"+str(i)+ "/" + str(nodes)):
                    os.makedirs("checkpoints/"+ game_string + "/nfsp/"+str(i)+ "/" + str(nodes))
                agents[i].save("checkpoints/"+ game_string + "/nfsp/"+str(i)+ "/" + str(nodes))
            checkpoint += 0.1
    for i in range(len(agents)):
        print("Checkpoint reached: ", checkpoint)
        if not os.path.exists("checkpoints/" + game_string + "/"):
            os.makedirs("checkpoints/" + game_string + "/")
        if not os.path.exists("checkpoints/"+ game_string + "/nfsp/"):
            os.makedirs("checkpoints/"+ game_string + "/nfsp/")
        if not os.path.exists("checkpoints/"+ game_string + "/nfsp/" + str(i)):
            os.makedirs("checkpoints/"+ game_string + "/nfsp/" + str(i))
        if not os.path.exists("checkpoints/"+ game_string + "/nfsp/"+str(i)+ "/" + str(nodes)):
            os.makedirs("checkpoints/"+ game_string + "/nfsp/"+str(i)+ "/" + str(nodes))
        agents[i].save("checkpoints/"+ game_string + "/nfsp/"+str(i)+ "/" + str(nodes))


class WrapperEnv:
    def __init__(self,game):
        self.game= game
        self.state = game.new_initial_state()
        self.agent_selection = str(self.state.current_player())
        self.traverser = None
    
    def reset(self, seed=None):
        self.state = self.game.new_initial_state()
        while self.state.is_chance_node():
            self.state.apply_action(np.random.choice(self.state.legal_actions()))
        self.agent_selection =  str(self.state.current_player())
        return self.obs()
    
    def step(self, action):
        if self.state.is_chance_node():
            while self.state.is_chance_node():
                self.state.apply_action(np.random.choice(self.state.legal_actions()))

        else:
            self.state.apply_action(action)
            if self.state.is_chance_node():
                while self.state.is_chance_node():
                    self.state.apply_action(np.random.choice(self.state.legal_actions()))
        
        if self.state.is_terminal():
            return self.obs()
        else:
            # store = copy.deepcopy(self.state)
            while self.state.is_chance_node():
                self.state.apply_action(np.random.choice(self.state.legal_actions()))
            # print("3")
            # print(self.state.is_terminal())
            # if self.state.is_terminal():
            #     print("store:", store)
            # print(self.state)
            # print(self.state.legal_actions_mask(int(self.agent_selection)))
            # print("3")
            self.agent_selection =  str(self.state.current_player())

            return self.obs()
    
    def last(self):
        return self.obs()
    
    def obs(self):
        return {"observation":self.state.observation_tensor(int(self.agent_selection)), "action_mask":np.stack(self.state.legal_actions_mask(int(self.agent_selection)))}, self.state.player_reward(self.traverser) if self.traverser is not None else self.state.player_return(int(self.agent_selection)), self.state.is_terminal(), False, "OPENSPIEL"
    

def load_agents(path1, path2, p_v_networks, num_players):
    agent1_state = torch.load(path1)
    agent2_state = torch.load(path2)

    agent1 = CFRNetwork(
        config = {'policy': p_v_networks, 'value': p_v_networks, 'num_players':num_players}
    )
    agent1.policy.load_state_dict(agent1_state)
    agent2 = CFRNetwork(
        config = {'policy': p_v_networks, 'value': p_v_networks, 'num_players':num_players}
    )
    agent2.policy.load_state_dict(agent2_state)

    agent1.policy.eval()
    agent2.policy.eval()
    return agent1, agent2


class EvalWrapper():
    def __init__(self, game, agent, in_size, out_size, select_model):
        """
        game : assume game already wrapper for evaled agent either wrapper or nfsp    
        """
        self.game = game
        self.render_mode = None
        self.action_space = ActionSpaceSample(game, out_size)
        self.agent = agent
        self.select_model = select_model
        self.observation_shape = np.array([i for i in range(in_size)], dtype=np.float32) # shape[0] with be in_size for agent checking
        self.observation_space = np.array([i for i in range(in_size)], dtype=np.float32)  # shape[0] with be in_size for agent checking
        self.insize= in_size
        self.starting_player = np.random.randint(0, 2)
        self.spec = EmptySpec()
        # 1 is agent, 0 is evaluator
    
    def reset(self,seed=None):
        # should reset, and return state and info
        self.starting_player = np.random.randint(0, 2)
        obs_mask, rew, terminal, truncated, info = self.game.reset()
        if self.starting_player == 0:
            return obs_mask["observation"], {"legal_moves":self.game.state.legal_actions()}
        else:
            obs_mask["action_mask"][0] = 0 # can't fold as player 1 during cards delt must fix, must give players an id.
            preds = self.agent.policy(torch.tensor(obs_mask['observation'], dtype=torch.float32).reshape(1,self.insize)).detach().numpy()[0]
            action, policy = self.select_model.select_actions(preds, info=torch.from_numpy(obs_mask["action_mask"]).type(torch.float), mask_actions=True)
            obs_mask, rew, terminal, truncated, info = self.game.step(action)
            return obs_mask["observation"], {"legal_moves":self.game.state.legal_actions()}
    
    def step(self,action):
        """
        action : action to take
        """
        # apply action to game
        # return obs, reward, done, info
        obs_mask, rew, terminal, truncated, info = self.game.step(action)
        reward = 0
        if not self.game.state.is_terminal():
            preds = self.agent.policy(torch.tensor(obs_mask['observation'], dtype=torch.float32).reshape(1,self.insize)).detach().numpy()[0]
            action, policy = self.select_model.select_actions(preds, info=torch.from_numpy(obs_mask["action_mask"]).type(torch.float), mask_actions=True)
            obs_mask, rew, terminal, truncated, info = self.game.step(action)
        
        if self.starting_player == 0:
            reward = self.game.state.player_reward(0)
        else:
            reward = self.game.state.player_reward(1)
        return obs_mask["observation"], reward, terminal, truncated, {"legal_moves":self.game.state.legal_actions()}
        

class EmptyConf():
    def __init__(self):
        self.num_players = 1
        self.has_legal_moves = True
        self.is_discrete = True
        self.min_score = -1200
        self.max_score = 1200

class EmptySpec():
    def __init__(self):
        self.reward_threshold = 1200


class ActionSpaceSample():
    def __init__(self, game, out_size):
        self.game = game
        self.shape = torch.tensor([i for i in range(out_size)]).shape # shape[0] with be out_size for agent checking
    
    def sample(self):
        print(self.game.state.legal_actions())
        return np.random.choice(self.game.state.legal_actions())
    



class NFSPEvalWrapper():
    def __init__(self, game, agent, in_size, out_size):
        """
        game : assume game already wrapper for evaled agent either wrapper or nfsp    
        """
        self.game = game
        self.render_mode = None
        self.action_space = ActionSpaceSample2(game, out_size)
        self.agent = agent
        self.observation_shape = np.array([i for i in range(in_size)], dtype=np.float32) # shape[0] with be in_size for agent checking
        self.observation_space = np.array([i for i in range(in_size)], dtype=np.float32)  # shape[0] with be in_size for agent checking
        self.insize= in_size
        self.starting_player = np.random.randint(0, 2)
        self.spec = EmptySpec()

        if self.starting_player == 0:
            self.agent.player_id = 1
            self.agent._rl_agent.player_id = 1
        else:
            self.agent.player_id = 0
            self.agent._rl_agent.player_id = 0
        # 1 is agent, 0 is evaluator
    
    def reset(self,seed=None):
        # should reset, and return state and info
        self.starting_player = np.random.randint(0, 2)
        if self.starting_player == 0:
            self.agent.player_id = 1
            self.agent._rl_agent.player_id = 1
        else:
            self.agent.player_id = 0
            self.agent._rl_agent.player_id = 0
            

        obs_mask, rew, terminal, truncated, info = self.game.reset()
        if self.starting_player == 0:
            return obs_mask["observation"], {"legal_moves":self.game.state.legal_actions()}
        else:
            construct = copy.deepcopy(self.game) # can't fold as player 1 during cards delt must fix, must give players an id.

            action, probs = self.agent.step(construct, is_evaluation=True)
            obs_mask, rew, terminal, truncated, info = self.game.step(action)
            if self.game.state.is_terminal():
                legal_moves = [0,1,2,3]
            else:
                legal_moves = self.game.state.legal_actions()
            return obs_mask["observation"], {"legal_moves":legal_moves}
    
    def step(self,action):
        """
        action : action to take
        """
        # apply action to game
        # return obs, reward, done, info
        if self.starting_player == 0:
            reward = self.game.state.player_reward(0)
        else:
            reward = self.game.state.player_reward(1)
        if self.game.state.is_terminal():
            return self.game.state.observation_tensor(0), reward, self.game.state.is_terminal(), False, {"legal_moves":self.game.state.legal_actions()} 
        obs_mask, rew, terminal, truncated, info = self.game.step(action)
        reward = 0
        if not self.game.state.is_terminal() and not terminal:
            if self.game.state.current_player() == self.agent.player_id:
                    construct = copy.deepcopy(self.game)
                    action, probs = self.agent.step(construct, is_evaluation=True)
                    obs_mask, rew, terminal, truncated, info = self.game.step(action)
            else:
                return obs_mask["observation"], reward, terminal, truncated, {"legal_moves":self.game.state.legal_actions()}
        
        return obs_mask["observation"], reward, terminal, truncated, {"legal_moves":self.game.state.legal_actions()}
        

class EmptyConf():
    def __init__(self):
        self.num_players = 1
        self.has_legal_moves = True
        self.is_discrete = True
        self.min_score = -1200
        self.max_score = 1200

class EmptySpec():
    def __init__(self):
        self.reward_threshold = 1200


class ActionSpaceSample2():
    def __init__(self, game, out_size):
        self.game = game
        self.shape = torch.tensor([i for i in range(out_size)]).shape # shape[0] with be out_size for agent checking
    
    def sample(self):
        if self.game.state.legal_actions() == []:
            return 0
        action = np.random.choice(self.game.state.legal_actions())
        return action
    

def LoadNFSPAgent(path, agent, pid):
    q_network_path = path + "q_network_pid" + str(pid)
    avg_network_path = path + "avg_network_pid" + str(pid)
    Qvars = tf.train.list_variables(q_network_path)
    Avars = tf.train.list_variables(avg_network_path)

    for v in Qvars:
        print(v)
    qvar_dict = {}
    for v in Qvars:
        name = v[0]
        shape = v[1]
        var = tf.Variable(tf.zeros(shape), name=name)
        qvar_dict[name] = var
    Qsaver = tf.compat.v1.train.Saver(qvar_dict)

    avar_dict = {}
    for v in Avars:
        name = v[0]
        shape = v[1]
        var = tf.Variable(tf.zeros(shape), name=name)
        avar_dict[name] = var
    Asaver = tf.compat.v1.train.Saver(avar_dict)

    agent._savers[0] = ("q_network", Qsaver)
    agent._savers[1] = ("avg_network", Asaver)