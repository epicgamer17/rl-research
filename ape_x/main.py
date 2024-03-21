import subprocess
import tensorflow as tf
import gym
import numpy as np
import threading
import copy

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("main.log", mode="w")
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(fh)
logger.addHandler(ch)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[fh, ch],
    format="%(asctime)s %(name)s %(threadName)s %(levelname)s: %(message)s",
)


def make_cartpole_env():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    return env


def main():

    learner_process = subprocess.Popen(
        ["python", "main_learner.py"],
    )

    num_actors = 1
    processes = list()
    for i in range(num_actors):
        id = i
        process = subprocess.Popen(
            ["python", "main_actor.py", str(id)],
        )
        processes.append(process)

    for process in processes:
        process.wait()

    learner_process.wait()

    print("====FINISHED====")


if __name__ == "__main__":
    main()
