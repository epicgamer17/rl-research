# FOR TRAINING ON MIMI
from alphazero_agent import AlphaZeroAgent
import gymnasium as gym
import tensorflow as tf
import numpy as np
import gym_envs


env = gym.make("gym_envs/TicTacToe-v0", render_mode="rgb_array")


# MODEL SEEMS TO BE UNDERFITTING SO TRY AND GET IT TO OVERFIT THEN FIND A HAPPY MEDIUM
# 1. INCREASE THE NUMBER OF RESIDUAL BLOCKS
# 2. INCREASE THE NUMBER OF FILTERS
# 3. DECREASE REGULARIZATION
# 4. TRY DECREASING LEARNING RATE (maybe its that whole thing where the policy goes to like 1 0 0 0 0... etc and then goes back on the third training step, so maybe the learning rate is too high)
# 5. TO OVERFIT USE LESS DATA (but that is probably just a bad idea)
config = {
    "activation": "relu",
    "kernel_initializer": "glorot_uniform",
    "optimizer": tf.keras.optimizers.legacy.Adam,
    "min_learning_rate": 0.001,  # 0.0001 # 0.00001 could maybe increase by a factor of 10 or 100 and try to do some weights regularization
    "max_learning_rate": 0.001,  # 0.0001
    "number_of_lr_cycles": 1,  # this will determine the step size based on training steps
    # STILL ADD A SCHEDULE FOR BASE LEARNING RATE (MIN LEARNING RATE)
    "adam_epsilon": 3.25e-6,
    "clipnorm": None,
    # NORMALIZATION?
    # REWARD CLIPPING
    "training_steps": 40,  # alpha zero did 700,000, the lessons from alpha zero did 40 generations but 1000 batches per generation, so 40,000 batches (they just had a cyclical learning rate per generation (also they trained twice on the same data every generation))
    "num_filters": 256,
    "kernel_size": 3,
    "stride": 1,
    "num_res_blocks": 20,
    "critic_conv_filters": 32,  # 1
    "critic_conv_layers": 1,
    "critic_dense_size": 256,
    "critic_dense_layers": 1,
    "actor_conv_filters": 32,  #
    "actor_conv_layers": 1,
    "actor_dense_size": 0,
    "actor_dense_layers": 0,
    "replay_buffer_size": 800,  # IN GAMES
    "replay_batch_size": 560,  # SHOULD BE ROUGHLY SAME AS AVERAGE MOVE PER GENERATION (SO LIKE 7 TIMES NUMBER OF GAMES PLAYED PER GENERATION) <- what was used in the original paper (they played 44M games, 50 moves per game and sampled 700,000 minibatches of size 4096 (so thats like sampling 1 time per move roughly but this was also happening with parrallel data collection i believe))
    "games_per_generation": 10,  # times 8 from augmentation
    "root_dirichlet_alpha": 0.75,  # Less than 1 more random, greater than one more flat # 2 in theory? # 0.3 in alphazero for chess
    "root_exploration_fraction": 0.25,  # 0.25 in paper
    "pb_c_base": 5000,  # Seems unimportant to be honest (increases puct the more simulations there are)
    "pb_c_init": 2.0,  # 1.25 in paper
    "num_simulations": 200,  # INCREASE THIS
    # 'two_player': True,
    "weight_decay": 0.00,  # could try setting this to something other than 0 and increasing learning rate
    "num_sampling_moves": 4,
    "initial_temperature": 1,
    "exploitation_temperature": 0.1,
    "value_loss_factor": 1,  # could try setting this to something other than 1
}

agent = AlphaZeroAgent(env, config=config, name="alphazero")
agent.train()
