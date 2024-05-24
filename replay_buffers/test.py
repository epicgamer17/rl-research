import dill
import tensorflow

tensorflow.random.set_seed(0)

import gymnasium as gym

import sys


sys.path.append("..")
from rainbow.rainbow_agent import RainbowAgent

import numpy as np
import random

np.random.seed(0)
random.seed(0)


def new_std_rainbow():
    with open("rainbow_config.dill", mode="rb") as f:
        config = dill.load(f)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    agent = RainbowAgent(env, config, "RainbowDQN-{}".format(env.unwrapped.spec.id))
    return agent


def make_transitions(n: int):
    agent = new_std_rainbow()
    agent.is_test = False

    transitions = list()

    state, _ = agent.env.reset()
    for training_step in range(n):
        action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = agent.step(action)

        done = terminated or truncated
        t = (state, action, reward, next_state, done)
        transitions.append(t)

        state = next_state
        agent.config.per_beta = min(
            1.0,
            agent.config.per_beta
            + (1 - agent.config.per_beta)
            / agent.config.training_steps,  # per beta increase
        )

        if done:
            state, _ = agent.env.reset()

    return transitions


def test_ten_steps_old():
    agent = new_std_rainbow()
    with open("standard_rainbow_agent.dill", mode="rb") as f:
        agent: RainbowAgent = dill.load(f)

    agent.is_test = False

    state, _ = agent.env.reset()
    for training_step in range(10):
        action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = agent.step(action)

        done = terminated or truncated
        state = next_state

        if done:
            state, _ = agent.env.reset()

    print(f"n_step={agent.config.n_step}, steps=10")
    print("Prioritized observation buffer:")
    for i in range(10):
        print(agent.replay_buffer.observation_buffer[i])

    print("NStep observation buffer:")
    for i in range(10):
        print(agent.n_step_replay_buffer.observation_buffer[i])

    print("Prioritized next observation buffer:")
    for i in range(10):
        print(agent.replay_buffer.next_observation_buffer[i])

    print("NStep next observation buffer:")
    for i in range(10):
        print(agent.n_step_replay_buffer.next_observation_buffer[i])


from replay_buffers.n_step_replay_buffer import ReplayBuffer as OldNStep
from replay_buffers.prioritized_replay_buffer import (
    PrioritizedReplayBuffer as OldPrioritized,
)
from replay_buffers.prioritized_nstep import ReplayBuffer as NewNStep


def test_n_step():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    shared = dict(
        observation_dimensions=env.observation_space.shape,
        max_size=123,
        batch_size=8,
        n_step=7,
    )
    oldN = OldNStep(**shared)
    oldP = OldPrioritized(**shared)
    new = NewNStep(**shared)
    transitions = make_transitions(400)

    for t in transitions:
        print(*t)

        new.store(*t)

        oldN.store(*t)
        oldP.store(*t)

        assert np.all(new.observation_buffer == oldP.observation_buffer)
        assert np.all(new.next_observation_buffer == oldP.next_observation_buffer)
        assert np.all(new.action_buffer == oldP.action_buffer)
        assert np.all(new.reward_buffer == oldP.reward_buffer)
        assert np.all(new.done_buffer == oldP.done_buffer)

    assert oldP.max_size == new.max_size

    tree_capacity = 1
    while tree_capacity < new.max_size:
        tree_capacity *= 2

    for i in range(tree_capacity):
        assert oldP.sum_tree[i] == new.sum_tree[i]
        assert oldP.min_tree[i] == new.min_tree[i]


test_n_step()

# test_ten_step()
