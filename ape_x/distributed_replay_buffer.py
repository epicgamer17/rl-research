import gymnasium as gym
import numpy as np
from compress_utils import compress, decompress
import entities.replayMemory_capnp as replay_memory_capnp
import zmq
import message_codes
import time
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

import sys

sys.path.append("../")
from refactored_replay_buffers.prioritized_nstep import ReplayBuffer


class SaveableReplayBuffer:
    def __init__(self, load=False, env: gym.Env = None):
        if load:
            self.load("replay_memory.xz")
        else:
            if env is None:
                raise ValueError("env must be provided if load is False")
            self.replay_memory = ReplayBuffer(
                observation_dimensions=env.observation_space.shape,
                max_size=50000,
                batch_size=128,
                max_priority=1.0,
                alpha=0.5,  # config["per_alpha"],
                # epsilon=config["per_epsilon"],
                n_step=1,  # config["n_step"],   # we don't need n-step because the actors give n-step transitions already
                gamma=0.99,  # config["discount_factor"],
            )

    def load(self, path):
        with open(path, "rb") as file:
            self.replay_memory = decompress(file.read())

    def save(self, path):
        with open(path, "wb") as file:
            file.write(compress(self.replay_memory))

    def sample(self):
        return self.replay_memory.__sample__()

    def store(self, *args, **kwargs):
        self.replay_memory.store(*args, **kwargs)


class ReplayServer:
    def __init__(self, replay_memory: SaveableReplayBuffer, learner_port, actors_port):
        self.replay_memory = replay_memory
        self.ctx = zmq.Context()

        self.actors_socket = self.ctx.socket(zmq.PULL)
        self.actors_socket.bind(f"tcp://*:{actors_port}")
        self.learners_socket = self.ctx.socket(zmq.REP)
        self.learners_socket.bind(f"tcp://*:{learner_port}")

        self.poller = zmq.Poller()
        self.poller.register(self.actors_socket, zmq.POLLIN)
        self.poller.register(self.learners_socket, zmq.POLLIN)

    def make_sample(self):
        try:
            samples = self.replay_memory.sample()
        except AssertionError as e:
            # if the buffer does not have enough samples, return empty samples
            return b""
        except Exception as e:
            logger.exception(f"sample error: {e}")
            raise e

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

    def run(self):
        while True:
            try:
                socks = dict(self.poller.poll())
            except KeyboardInterrupt:
                break

            if self.actors_socket in socks:
                msg = self.actors_socket.recv()
                msg = self.actors_socket.recv()
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
                    self.replay_memory.store(
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
                    f"adding transitions took {dt}. New buffer size: {self.replay_memory.replay_memory.size}",
                )

            if self.learners_socket in socks:
                msg = self.learners_socket.recv()
                if msg == message_codes.LEARNER_REQUESTS_BATCH:
                    batch = self.make_sample()
                    self.learners_socket.send(batch)
                elif msg == message_codes.LEARNER_UPDATE_PRIORITIES:
                    res = self.learners_socket.recv()
                    self.learners_socket.send(b"")
                    updates = replay_memory_capnp.PriorityUpdate.from_bytes_packed(res)
                    indices = np.array(updates.indices)
                    ids = list(updates.ids)
                    losses = np.array(updates.losses)

                    self.replay_memory.replay_memory.update_priorities(
                        indices=indices, priorities=losses, ids=ids
                    )
        self.replay_memory.save("replay_memory.xz")


def main(learner_port, actors_port, load=False):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    replay_memory = SaveableReplayBuffer(load=load, env=env)
    server = ReplayServer(replay_memory, learner_port, actors_port)
    logger.info(
        f"replay buffer started on ports {actors_port} (actors) and {learner_port} (learner)"
    )

    server.run()


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a distributed Ape-X replay buffer"
    )
    parser.add_argument("learner_port", type=str, default="5554")
    parser.add_argument("actors_port", type=str, default="5555")
    args = parser.parse_args()
    main(args.learner_port, args.actors_port)
