import pickle
import socket
import asyncio
import gymnasium as gym
import capnp
import numpy as np
from compress_utils import compress, decompress

capnp.remove_import_hook()

replay_memory_capnp = capnp.load("./entities/replayMemory.capnp")

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


class ReplayMemoryImpl(replay_memory_capnp.ReplayMemory.Server):
    def __init__(self, replay_buffer: ReplayBuffer, shared_dict: dict):
        self.replay_memory = replay_buffer
        self.shared_dict = shared_dict

    def addTransitionBatch(self, batch, _context):
        t_i = time.time()
        n = len(batch.ids)
        logger.info(f"adding {n} transitions to buffer")

        ids = batch.ids
        observations = pickle.loads(batch.observations)
        nextObservations = pickle.loads(batch.nextObservations)
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
            f"adding transitions took {dt}. New buffer size: {self.replay_memory.size}",
        )

    def sample(self, batchSize, _context):
        logger.info(f"sample, batchSize: {batchSize}")
        try:
            samples = self.replay_memory.__sample__()
        except AssertionError as e:
            # if the buffer does not have enough samples, return empty samples
            return {
                "ids": [],
                "observations": bytes(),
                "actions": [],
                "rewards": [],
                "nextObservations": bytes(),
                "dones": [],
                "indices": [],
            }
        except Exception as e:
            logger.exception(f"sample error: {e}")
            raise e

        # logger.info(f"samples: {samples}")

        # convert to capnp types
        ids = list()
        actions = list()
        rewards = list()
        dones = list()
        indices = list()
        weights = list()

        n = len(samples.ids)
        for i in range(n):
            ids.append(samples.ids[i])
            actions.append(int(samples.actions[i]))
            rewards.append(float(samples.rewards[i]))
            dones.append(bool(samples.dones[i]))
            indices.append(int(samples.indices[i]))
            weights.append(float(samples.weights[i]))

        ret = {
            "ids": ids,
            "observations": pickle.dumps(samples.observations, protocol=5),
            "nextObservations": pickle.dumps(samples.next_observations, protocol=5),
            "actions": samples.actions.tolist(),
            "rewards": rewards,
            "dones": dones,
            "indices": indices,
            "weights": weights,
        }
        return ret

    def updatePriorities(self, indices, ids, priorities, _context):
        # logger.info( f"updatePriorities - indices: {indices}, ids: {ids}, priorities: {priorities}")
        logger.info("updatePriorities")
        self.replay_memory.update_priorities(
            np.array(list(indices)), np.array(list(priorities)), list(ids)
        )

    def removeOldExperiences(self, _context):
        # not necessary as replay buffer has a fixed size
        logger.info("removeOldExperiences")

    def getWeights(self, _context):
        # logger.info(f"getWeights {self.shared_dict['weights']}")
        logger.info(f"getWeights ")
        return self.shared_dict["weights"]

    def setWeights(self, weights: bytes, _context):
        logger.info(f"setWeights")
        self.shared_dict["weights"] = weights

    def ping(self, _context):
        logger.info("ping")


class Server:
    async def _reader(self):
        while self.retry:
            try:
                # Must be a wait_for so we don't block on read()
                data = await asyncio.wait_for(self.reader.read(2**16), timeout=1)
            except asyncio.TimeoutError:
                logger.debug("myreader timeout.")
                continue
            except Exception as err:
                logger.exception("Unknown myreader err: %s", err)
                return False
            await self.server.write(data)
        logger.debug("myreader done.")
        return True

    async def _writer(self):
        while self.retry:
            try:
                # Must be a wait_for so we don't block on read()
                data = await asyncio.wait_for(self.server.read(2**16), timeout=1)
                self.writer.write(data.tobytes())
            except asyncio.TimeoutError:
                logger.debug("mywriter timeout.")
                continue
            except Exception as err:
                logger.exception("Unknown mywriter err: %s", err)
                return False
        logger.debug("mywriter done.")
        return True

    async def start(self, reader, writer, replay_memory, shared_dict):
        # Start TwoPartyServer using TwoWayPipe (only requires bootstrap)

        self.server = capnp.TwoPartyServer(
            bootstrap=ReplayMemoryImpl(
                replay_buffer=replay_memory,
                shared_dict=shared_dict,
            )
        )
        self.reader = reader
        self.writer = writer
        self.retry = True

        # Assemble reader and writer tasks, run in the background
        coroutines = [self._reader(), self._writer()]
        tasks = asyncio.gather(*coroutines, return_exceptions=True)

        while True:
            self.server.poll_once()
            # Check to see if reader has been sent an eof (disconnect)
            if self.reader.at_eof():
                self.retry = False
                break
            await asyncio.sleep(0.01)

        # Make wait for reader/writer to finish (prevent possible resource leaks)
        await tasks


def new_connection_with_replay_memory(replay_memory, shared_dict):
    async def new_connection(reader, writer):
        server = Server()
        await server.start(reader, writer, replay_memory, shared_dict)

    return new_connection


async def main(addr="localhost", port=60000):
    host = "localhost"
    port = "60000"

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
    shared_dict = {
        "weights": compress(None),
    }

    # Handle both IPv4 and IPv6 cases
    try:
        logger.info("Try IPv4")
        server = await asyncio.start_server(
            new_connection_with_replay_memory(replay_memory, shared_dict),
            host,
            port,
            family=socket.AF_INET,
        )
    except Exception:
        logger.info("Try IPv6")
        server = await asyncio.start_server(
            new_connection_with_replay_memory(replay_memory, shared_dict),
            host,
            port,
            family=socket.AF_INET6,
        )

    async with server:
        await server.serve_forever()


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a distributed Ape-X replay buffer"
    )
    parser.add_argument("capnp_conn", type=str, default="localhost:60000")
    args = parser.parse_args()

    addr = args.capnp_conn.split(":")[0]
    port = int(args.capnp_conn.split(":")[1])
    asyncio.run(main(addr, port))
