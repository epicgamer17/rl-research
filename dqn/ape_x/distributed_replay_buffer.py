import argparse
import signal
import numpy as np
import entities.replayMemory_capnp as replay_buffer_capnp
import zmq
import message_codes
import time
import logging
from agent_configs import ReplayBufferConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
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
from replay_buffers.prioritized_n_step_replay_buffer import PrioritizedNStepReplayBuffer
from storage.compress_utils import compress, decompress


class SaveableReplayBuffer:
    def __init__(self, config: ReplayBufferConfig, load=False):
        if load:
            self.load("replay_buffer.xz")
        else:
            self.replay_buffer = PrioritizedNStepReplayBuffer(
                observation_dimensions=config.observation_dimensions,
                max_size=config.max_size,
                min_size=config.min_size,
                batch_size=config.batch_size,
                max_priority=config.max_priority,
                alpha=config.per_alpha,  # config["per_alpha"],
                beta=config.per_beta,  # config["per_beta"],
                n_step=config.n_step,
                gamma=config.discount_factor,
            )

    def load(self, path):
        with open(path, "rb") as file:
            self.replay_buffer = decompress(file.read())

    def save(self, path):
        with open(path, "wb") as file:
            file.write(compress(self.replay_buffer))

    def sample(self):
        return self.replay_buffer.sample()

    def store(self, *args, **kwargs):
        self.replay_buffer.store(*args, **kwargs)


class ReplayServer:
    def __init__(
        self, replay_buffer: SaveableReplayBuffer, learner_port, actors_port, model_name
    ):
        self.replay_buffer = replay_buffer
        self.ctx = zmq.Context()

        self.actors_socket = self.ctx.socket(zmq.PULL)
        self.actors_socket.bind(f"tcp://*:{actors_port}")
        self.learners_socket = self.ctx.socket(zmq.REP)
        self.learners_socket.bind(f"tcp://*:{learner_port}")

        self.poller = zmq.Poller()
        self.poller.register(self.actors_socket, zmq.POLLIN)
        self.poller.register(self.learners_socket, zmq.POLLIN)

        self.exit = False
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)

        self.current_episode = 0
        self.model_name = model_name

    def make_sample(self, beta):
        try:
            self.replay_buffer.beta = beta
            samples = self.replay_buffer.sample()
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

        builder = replay_buffer_capnp.TransitionBatch.new_message()
        builder.ids = ids
        builder.observations = compress(samples.observations)
        builder.nextObservations = compress(samples.next_observations)
        builder.actions = samples.actions.astype(int).tolist()
        builder.rewards = samples.rewards.astype(float).tolist()
        builder.dones = samples.dones.astype(bool).tolist()
        builder.indices = samples.indices.astype(int).tolist()
        builder.weights = samples.weights.astype(float).tolist()

        return builder.to_bytes_packed()

    def cleanup(self):
        self.replay_buffer.save(f"{self.model_name}_episode_{self.current_episode}.xz")
        self.exit = True

    def run(self):
        while not self.exit:
            try:
                socks = dict(self.poller.poll())
            except KeyboardInterrupt:
                break

            if self.actors_socket in socks:
                msg = self.actors_socket.recv()
                msg = self.actors_socket.recv()
                batch = replay_buffer_capnp.TransitionBatch.from_bytes_packed(msg)
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
                    self.replay_buffer.store(
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
                    f"adding transitions took {dt}. New buffer size: {self.replay_buffer.replay_buffer.size}",
                )

            if self.learners_socket in socks:
                msg = self.learners_socket.recv()
                if msg == message_codes.LEARNER_REQUESTS_BATCH:
                    self.replay_buffer.replay_buffer.beta = float(
                        self.learners_socket.recv_string()
                    )
                    batch = self.make_sample()
                    self.learners_socket.send(batch)
                elif msg == message_codes.LEARNER_UPDATE_PRIORITIES:
                    res = self.learners_socket.recv()
                    self.learners_socket.send(b"")
                    updates = replay_buffer_capnp.PriorityUpdate.from_bytes_packed(res)
                    indices = np.array(updates.indices)
                    ids = list(updates.ids)
                    losses = np.array(updates.losses)

                    self.replay_buffer.replay_buffer.update_priorities(
                        indices=indices, priorities=losses, ids=ids
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Run a distributed Ape-X replay buffer"
    )
    parser.add_argument("--learner_port", type=str, default="5554")
    parser.add_argument("--actors_port", type=str, default="5555")
    parser.add_argument("--load", default=False, action="store_true")
    parser.add_argument("--model_name", type=str, default="learner")
    # TODO pass in this command line argument wherever this is called (ie hyperopt.go)
    parser.add_argument(
        "--config_file", type=str, default="configs/replay_config_example.yaml"
    )

    args = parser.parse_args()
    config = ReplayBufferConfig.load(args.config_file)
    replay_buffer = SaveableReplayBuffer(config, args.load)
    server = ReplayServer(
        replay_buffer, args.learner_port, args.actors_port, args.model_name
    )
    logger.info(
        f"replay buffer started on ports {args.actors_port} (actors) and {args.learner_port} (learner)"
    )

    server.run()


if __name__ == "__main__":
    main()
