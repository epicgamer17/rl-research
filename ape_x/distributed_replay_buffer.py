import pickle
import gymnasium as gym
import numpy as np
from compress_utils import compress, decompress
import entities.replayMemory_capnp as replay_memory_capnp


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler("replay_buffer.log", mode="w")
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(message)s"))
fh.setFormatter(
    logging.Formatter("%(asctime)s %(name)s %(threadName)s %(levelname)s: %(message)s")
)

logger.addHandler(fh)
logger.addHandler(ch)


import time
import sys

sys.path.append("../")
from refactored_replay_buffers.prioritized_nstep import ReplayBuffer


import zmq

import message_codes


def make_sample(replay_memory: ReplayBuffer):
    try:
        samples = replay_memory.__sample__()
    except AssertionError as e:
        # if the buffer does not have enough samples, return empty samples
        return b""
    except Exception as e:
        logger.exception(f"sample error: {e}")
        raise e

    # logger.info(f"samples: {samples}")

    # convert to capnp types
    ids = list()

    n = len(samples.ids)
    for i in range(n):
        ids.append(samples.ids[i])

    builder = replay_memory_capnp.TransitionBatch.new_message()
    builder.ids = ids
    builder.observations = compress(samples.observations)
    builder.nextObservations = compress(samples.next_observations)
    builder.actions = samples.actions.astype(int).tolist()
    builder.rewards = samples.rewards.astype(float).tolist()
    builder.dones = samples.dones.astype(bool).tolist()
    builder.indices = samples.indices.astype(int).tolist()
    builder.weights = samples.weights.astype(float).tolist()

    return builder.to_bytes_packed()


def main(learner_port, actors_port):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    replay_memory = ReplayBuffer(
        observation_dimensions=env.observation_space.shape,
        max_size=20000,
        batch_size=2**7,
        max_priority=1.0,
        alpha=0.5,  # config["per_alpha"],
        # epsilon=config["per_epsilon"],
        n_step=3,  # config["n_step"],
        gamma=0.01,  # config["discount_factor"],
    )

    ctx = zmq.Context()

    actors_socket = ctx.socket(zmq.PULL)
    actors_socket.bind(f"tcp://*:{actors_port}")
    learners_socket = ctx.socket(zmq.REP)
    learners_socket.bind(f"tcp://*:{learner_port}")

    logger.info(
        f"replay buffer started on ports {actors_port} (actors) and {learner_port} (learner)"
    )

    poller = zmq.Poller()
    poller.register(actors_socket, zmq.POLLIN)
    poller.register(learners_socket, zmq.POLLIN)

    while True:
        logger.info("waiting for message")

        try:
            socks = dict(poller.poll())
            logger.info(f"socks: {socks}")
        except KeyboardInterrupt:
            break

        if actors_socket in socks:
            logger.info("actor request")
            msg = actors_socket.recv()
            logger.info(f"received: {msg}")
            msg = actors_socket.recv()
            batch = replay_memory_capnp.TransitionBatch.from_bytes_packed(msg)
            t_i = time.time()
            n = len(batch.ids)
            logger.info(f"adding {n} transitions to buffer")

            ids = batch.ids
            observations = decompress(batch.observations)
            nextObservations = decompress(batch.nextObservations)
            actions = batch.actions
            rewards = batch.rewards
            dones = batch.dones
            priorities = batch.priorities

            for i in range(len(ids)):
                replay_memory.store(
                    observations[i],
                    actions[i],
                    rewards[i],
                    nextObservations[i],
                    dones[i],
                    id=ids[i],
                    priority=priorities[i],
                )

            dt = time.time() - t_i
            logger.info(
                f"adding transitions took {dt}. New buffer size: {replay_memory.size}",
            )

        if learners_socket in socks:
            logger.info("learner request")
            msg = learners_socket.recv()
            logger.info(f"received: {msg}")
            if msg == message_codes.LEARNER_REQUESTS_BATCH:
                batch = make_sample(replay_memory)
                learners_socket.send(batch)
            elif msg == message_codes.LEARNER_UPDATE_PRIORITIES:
                res = learners_socket.recv()
                learners_socket.send(b"")
                updates = replay_memory_capnp.PriorityUpdate.from_bytes(res)
                replay_memory.update_priorities(
                    np.array(updates.indices), np.array(updates.priorities), list(ids)
                )


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a distributed Ape-X replay buffer"
    )
    parser.add_argument("learner_port", type=str, default="5554")
    parser.add_argument("actors_port", type=str, default="5555")
    args = parser.parse_args()
    main(args.learner_port, args.actors_port)
