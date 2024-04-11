import os

from agent_configs import RainbowConfig, ApeXActorConfig
from game_configs import CartPoleConfig


def test_dumping_1():
    dict = {"activation": "relu", "kernel_initializer": "orthogonal"}
    conf = RainbowConfig(dict, game_config=CartPoleConfig())
    conf.dump("rainbow.yaml")

    loaded: RainbowConfig = RainbowConfig.load("rainbow.yaml")

    # skip optimizer field
    test_skipped = set(["optimizer"])

    for k in dir(conf):
        if k.startswith("__") or test_skipped.__contains__(k):
            continue

        if not callable(getattr(conf, k)):
            expected = getattr(conf, k)
            actual = getattr(loaded, k)
            assert (
                expected == actual
            ), f"dump/load failed on key {k}. Expected {expected}, actual {actual}"

    os.remove("rainbow.yaml")


def test_dumping_2():
    distributed_config = {
        "storage_hostname": "127.0.0.1",
        "storage_username": "ezra",
        "storage_password": "aaa",
        "storage_port": 5556,
        "actor_replay_port": 5554,
        "replay_addr": "127.0.0.1",
    }

    rainbow_config = {
        "width": 512,
        # "loss_function": losses.CategoricalCrossentropy(),
        "activation": "relu",
        "kernel_initializer": "orthogonal",
        "adam_epsilon": 0.0003125,
        "transfer_interval": 100,
        "dense_layers": 2,
        "per_epsilon": 0.001,
        "per_alpha": 0.5,
        "per_beta": 0.4,
        "clipnorm": None,
    }

    actor_config = {
        "actor_buffer_size": 128,  # sets minibatch size and replay buffer size
        "poll_params_interval": 128,
    }

    conf = {**rainbow_config, **distributed_config, **actor_config}

    conf = ApeXActorConfig(conf, CartPoleConfig())
    conf.dump("apex_actor.yaml")

    loaded: ApeXActorConfig = ApeXActorConfig.load("apex_actor.yaml")

    # skip optimizer field
    test_skipped = set(["optimizer"])

    for k in dir(conf):
        if k.startswith("__") or test_skipped.__contains__(k):
            continue

        if not callable(getattr(conf, k)):
            expected = getattr(conf, k)
            actual = getattr(loaded, k)
            assert (
                expected == actual
            ), f"dump/load failed on key {k}. Expected {expected}, actual {actual}"

    os.remove("apex_actor.yaml")
