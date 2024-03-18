import pickle
import socket
import asyncio
import gym
import capnp

capnp.remove_import_hook()

replay_buffer_capnp = capnp.load("./entities/replayMemory.capnp")

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
from replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer


class ReplayMemoryImpl(replay_buffer_capnp.ReplayMemory.Server):
    def __init__(self, replay_memory):
        self.replay_memory = replay_memory

        self.weights = bytes()

    def addTransitionBatch(self, batch, _context):
        initial_time = time.time()
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
            to_store = {
                "observation": observations[i],
                "action": actions[i],
                "reward": rewards[i],
                "next_observation": nextObservations[i],
                "done": dones[i],
                "priority": priorities[i],
                "id": ids[i],
            }
            self.replay_memory.store_with_priority_exact(**to_store)

        delta_t = time.time() - initial_time
        logger.info(
            f"adding transitions took {delta_t}. New buffer size: {self.replay_memory.size}",
        )

    def sample(self, batchSize, _context):
        logger.info("sample", _context)
        return self.replay_memory.sample()

    def updatePriorities(self, indices, ids, priorities, _context):
        logger.info("updatePriorities", indices, ids, priorities, _context)
        self.replay_memory.update_priorities(indices, priorities, ids)

    def removeOldExperiences(self, _context):
        # not necessary as replay buffer has a fixed size
        logger.info("removeOldExperiences", _context)

    def getWeights(self, _context):
        logger.info("getWeights", _context)
        return self.weights

    def setWeights(self, weights, _context):
        logger.info("setWeights", weights, _context)
        self.weights = weights

    def ping(self):
        logger.info("ping")


class Server:
    async def _reader(self):
        while self.retry:
            try:
                # Must be a wait_for so we don't block on read()
                data = await asyncio.wait_for(self.reader.read(4096), timeout=1)
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
                data = await asyncio.wait_for(self.server.read(4096), timeout=1)
                self.writer.write(data.tobytes())
            except asyncio.TimeoutError:
                logger.debug("mywriter timeout.")
                continue
            except Exception as err:
                logger.exception("Unknown mywriter err: %s", err)
                return False
        logger.debug("mywriter done.")
        return True

    async def start(self, reader, writer, replay_memory):
        # Start TwoPartyServer using TwoWayPipe (only requires bootstrap)

        self.server = capnp.TwoPartyServer(
            bootstrap=ReplayMemoryImpl(
                replay_memory=replay_memory,
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


def new_connection_with_replay_memory(replay_memory):
    async def new_connection(reader, writer):
        server = Server()
        await server.start(reader, writer, replay_memory)

    return new_connection


async def main():
    host = "localhost"
    port = "60000"

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    replay_memory = PrioritizedReplayBuffer(
        observation_dimensions=env.observation_space.shape,
        max_size=10000,
        batch_size=100,
        max_priority=1.0,
        alpha=0.5,  # config["per_alpha"],
        # epsilon=config["per_epsilon"],
        n_step=3,  # config["n_step"],
        gamma=0.01,  # config["discount_factor"],
    )

    # Handle both IPv4 and IPv6 cases
    try:
        logger.info("Try IPv4")
        server = await asyncio.start_server(
            new_connection_with_replay_memory(replay_memory),
            host,
            port,
            family=socket.AF_INET,
        )
    except Exception:
        logger.info("Try IPv6")
        server = await asyncio.start_server(
            new_connection_with_replay_memory(replay_memory),
            host,
            port,
            family=socket.AF_INET6,
        )

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
