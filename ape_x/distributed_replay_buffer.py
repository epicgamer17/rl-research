import socket
import asyncio
import gym
import capnp

capnp.remove_import_hook()

replay_buffer_capnp = capnp.load("./entities/replayMemory.capnp")

import sys

sys.path.append("../")
from replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer


class ReplayMemoryImpl(replay_buffer_capnp.ReplayMemory.Server):
    def __init__(self, *args, **kwargs):
        self.replay_memory = PrioritizedReplayBuffer(*args, **kwargs)

    def addTransitionBatch(self, batch, _context):
        print("addTransitionBatch", batch, _context)
        return None

    def sample(self, _context):
        print("sample", _context)
        return None

    def updatePriorities(self, indices, priorities, _context):
        print("updatePriorities", indices, priorities, _context)
        return None

    def removeOldExperiences(self, _context):
        print("removeOldExperiences", _context)
        return None


class Server:
    async def _reader(self):
        while self.retry:
            try:
                # Must be a wait_for so we don't block on read()
                data = await asyncio.wait_for(self.reader.read(4096), timeout=0.1)
            except asyncio.TimeoutError:
                print("myreader timeout.")
                continue
            except Exception as err:
                print("Unknown myreader err: %s", err)
                return False
            await self.server.write(data)
        print("myreader done.")
        return True

    async def _writer(self):
        while self.retry:
            try:
                # Must be a wait_for so we don't block on read()
                data = await asyncio.wait_for(self.server.read(4096), timeout=0.1)
                self.writer.write(data.tobytes())
            except asyncio.TimeoutError:
                print("mywriter timeout.")
                continue
            except Exception as err:
                print("Unknown mywriter err: %s", err)
                return False
        print("mywriter done.")
        return True

    async def start(self, reader, writer):
        # Start TwoPartyServer using TwoWayPipe (only requires bootstrap)
        env = gym.make("CartPole-v1", render_mode="rgb_array")

        self.server = capnp.TwoPartyServer(
            bootstrap=ReplayMemoryImpl(
                observation_dimensions=env.observation_space.shape,
                max_size=10000,
                batch_size=100,
                max_priority=1.0,
                alpha=0.5,  # config["per_alpha"],
                # epsilon=config["per_epsilon"],
                n_step=3,  # config["n_step"],
                gamma=0.01,  # config["discount_factor"],
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


async def new_connection(reader, writer):
    server = Server()
    await server.start(reader, writer)


async def main():
    host = "localhost"
    port = "60000"

    # Handle both IPv4 and IPv6 cases
    try:
        print("Try IPv4")
        server = await asyncio.start_server(
            new_connection, host, port, family=socket.AF_INET
        )
    except Exception:
        print("Try IPv6")
        server = await asyncio.start_server(
            new_connection, host, port, family=socket.AF_INET6
        )

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
