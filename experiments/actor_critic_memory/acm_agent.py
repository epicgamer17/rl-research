import torch
import torch.nn as nn
from acm_network import ACMNetwork, MHA
from acm_memory import MMbuffer
from ppo.ppo_agent import PPOAgent
from ACMconfig import ACModelconfig, MHABconfig
import time
class ACMAgent:
    def __init__(self, configu=ACModelconfig, device="cpu"):
        """
        Initialize the ACM agent
        Args:
            config (Config): Configuration object
            device (str): Device to use ("cpu" or "cuda")
        """
        # Parse config
        
        self.PPOconfig = configu.PPOconfig
        self.MHABconfig = configu.MHABconfig
        self.Buffconfig = configu.Buffconfig
        self.config = configu.ACMconfig
        self.env = self.config.env
        self.device = device
        self.stats = {
            "score": [],
            "actor_loss": [],
            "critic_loss": [],
            "ACM_loss": [],
            "test_score": [],
        }

        # Initialize memory buffer
        self.memory_buffer = MMbuffer(self.Buffconfig.buffer_size)

        # Initialize network
        # self.network = ACMNetwork(config=self.ACMNetconfig, device=self.device)
        self.network = MHA(config=self.MHABconfig, device=self.device, logging=False)

        # Init PPO
        self.PPO = PPOAgent(env=self.env, config=self.PPOconfig, device=self.device)

        if self.config.logging:
            self.log()

    def preprocess(self, state):
        """
        Preprocess the state
        Args:
            state (torch.Tensor): State tensor
        Returns:
            torch.Tensor: Preprocessed state tensor
        """
        # Normalize the state
        preprocessedstate = self.PPO.preprocess(state)
        # Get history from memory buffer
        if len(self.memory_buffer) > 0:
            history = self.memory_buffer.getmemories(self.Buffconfig.buffer_size)
            history.to(self.device)
            print(history)
            augmented_state = self.network(preprocessedstate, history)
        else:
            augmented_state = preprocessedstate
        return augmented_state
    
    def act(self, state):
        """
        Act based on the state
        Args:
            state (torch.Tensor): State tensor
        Returns:
            torch.Tensor: Action tensor
        """
        # Preprocess the state
        preprocessedstate = self.preprocess(state)
        # Get action from PPO
        value = self.PPO.model.critic(inputs=preprocessedstate)
        if self.PPO.discrete_action_space:
            policy = self.PPO.model.actor(inputs=preprocessedstate)[0]
            distribution = torch.distributions.Categorical(probs=policy)
        else:
            mean, std = self.PPO.model.actor(inputs=preprocessedstate)
            distribution = torch.distributions.Normal(mean, std)
        return self.PPO.select_actions([distribution, value]).item(), distribution, value
    
    def learn(self, state, action, reward, next_state, done):
        """
        Learn from the experience
        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor
            reward (float): Reward
            next_state (torch.Tensor): Next state tensor
            done (bool): Done flag
        """
        # Add experience to memory buffer
        self.memory_buffer.add(state, action, reward, next_state, done)
        # Learn from PPO
        critic_loss, actor_loss = self.PPO.learn(state, action, reward, next_state, done)
        # Learn from ACM network
        self.network.learn(critic_loss)
        self.network.learn(actor_loss)
        self.stats["actor_loss"].append(actor_loss)
        self.stats["critic_loss"].append(critic_loss)
        self.stats["ACM_loss"].append(self.network.loss)
        self.stats["test_score"].append(self.PPO.test_score)



    def save_checkpoint(self):
        """
        Save the checkpoint
        """
        torch.save(self.PPO.model.state_dict(), self.config.checkpoint_path)
        torch.save(self.network.state_dict(), self.config.checkpoint_path.replace(".pt", "_network.pt"))
        print("Checkpoint saved at {}".format(self.config.checkpoint_path))



    def load_checkpoint(self):
        """
        Load the checkpoint
        """
        self.PPO.model.load_state_dict(torch.load(self.config.checkpoint_path))
        self.network.load_state_dict(torch.load(self.config.checkpoint_path.replace(".pt", "_network.pt")))
        print("Checkpoint loaded from {}".format(self.config.checkpoint_path))
    


    def print_training_progress(self):
        """
        Print the training progress
        """
        print(
            "Training step: {}, Time: {:.2f}s, Total Environment Steps: {}, Score: {}".format(
                self.training_step,
                self.training_time,
                self.total_environment_steps,
                self.stats["score"][-1]["score"],
            )
        )



    def train(self):
        start_time = time() - self.training_time
        state, info = self.env.reset()

        while self.training_step < self.config.training_steps:
            with torch.no_grad():
                if self.training_step % self.config.print_interval == 0:
                    self.print_training_progress()
                num_episodes = 0
                score = 0
                for timestep in range(self.config.steps_per_epoch):
                    action, distribution, value = self.act(state)

                    next_state, reward, terminated, truncated, next_info = self.env.step(
                        action
                    )
                    log_probability = distribution.log_prob(torch.tensor(action))
                    value = value[0][0]
                    self.PPO.replay_buffer.store(
                        state, info, action, value, log_probability, reward
                    )
                    self.memory_buffer.add(state, action, reward, next_state, terminated or truncated)
                    done = terminated or truncated
                    state = next_state
                    info = next_info
                    score += reward

                    if done or timestep == self.config.steps_per_epoch - 1:
                        last_value = (
                            0 if done else self.PPO.model.critic(self.preprocess(next_state))
                        )
                        self.PPO.replay_buffer.finish_trajectory(last_value)
                        num_episodes += 1
                        state, info = self.env.reset()
                        score_dict = {"score": score}
                        self.stats["score"].append(score_dict)
                        score = 0

            self.learn()

            if self.training_step % self.checkpoint_interval == 0:
                self.training_time = time() - start_time
                self.total_environment_steps += self.config.steps_per_epoch
                self.save_checkpoint()
            self.training_step += 1

        self.training_time = time() - start_time
        self.total_environment_steps = self.config.training_steps * self.config.steps_per_epoch
        self.save_checkpoint()
        self.env.close()
        


    def log(self):
        """
        Log the configuration
        """
        print("ACM Agent Configuration:")
        print("Environment: {}".format(self.env))
        print("Checkpoint Path: {}".format(self.config.checkpoint_path))
        print("Training Steps: {}".format(self.config.training_steps))
        print("Print Interval: {}".format(self.config.print_interval))
        print("Steps per Epoch: {}".format(self.config.steps_per_epoch))
        print("Checkpoint Interval: {}".format(self.config.checkpoint_interval))
        print("PPO Configuration:")
        # Critic : adam_epsilon, learning_rate, clip_norm, optimizer
        # Actor : adam_epsilon, learning_rate, clip_norm, optimizer
        # PPO : TO ADD LATER
        # print("ACM Network Configuration:")
        # print("Input Shape: {}".format(self.ACMNetconfig.input_dim))
        # print("Output Shape: {}".format(self.ACMNetconfig.output_dim))
        # print("Embedding Shapes: {}".format(self.ACMNetconfig.embed_shapes))
        # print("Optimizer: {}".format(self.ACMNetconfig.optimizer))
        # print("Learning Rate: {}".format(self.ACMNetconfig.learning_rate))
        print("MultiHeadAttentionBlock config:")
        print("Config: {}".format(self.MHABconfig))
        print("Memory Buffer Configuration:")
        print("Buffer Size: {}".format(self.Buffconfig.buffer_size))