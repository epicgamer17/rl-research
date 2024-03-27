import asyncio
import logging
import socket
from abc import ABC

import capnp

capnp.remove_import_hook()
replay_memory_capnp = capnp.load("./entities/replayMemory.capnp")

logger = logging.getLogger(__name__)


class RPCClient(ABC):
    def __init__(self, bootstrap_as, addr="localhost", port=60000):
        self.bootstrap_as = bootstrap_as
        self.addr = addr
        self.port = port
        self._timeout = 30
        self._heartbeat_interval = 30

        self.running = False
        self.connected = False

        self.reader = None
        self.writer = None
        self.client = None
        self.rpc = None

        self.tasks = []
        self.extra_coroutines = []

    async def _reader(self):
        while self.running:
            logger.debug("socketreader running.")
            try:
                data = await asyncio.wait_for(
                    self.reader.read(2**16), timeout=self._timeout
                )
            except asyncio.TimeoutError:
                # read timeout - either there is no data or the connection is lost
                logger.debug("socketreader timeout.")
                continue
            except Exception as err:
                # any other error - finish the task
                logger.exception(f"socketreader err: {err}")
                return False

            self.client.write(data)

        logger.debug("socketreader done.")
        return True

    async def _writer(self):
        while self.running:
            logger.debug("socketwriter running.")
            try:
                data = await asyncio.wait_for(
                    self.client.read(2**16), timeout=self._timeout
                )
                self.writer.write(data.tobytes())
                await self.writer.drain()
            except asyncio.TimeoutError:
                # write timeout - either there is no data or the connection is lost
                logger.debug("socketwriter timeout.")
                continue
            except Exception as err:
                # any other error - finish the task
                logger.exception(f"socketwriter err: {err}")
                return False
        logger.debug("socketwriter done.")
        return True

    async def _heartbeat(self):
        while self.running:
            logger.debug("heartbeat running.")
            promise = self.rpc.ping()
            try:
                logger.debug("ping sent.")
                await asyncio.wait_for(
                    promise.a_wait(), timeout=self._heartbeat_interval - 1
                )
                logger.debug("ping response received.")
                await asyncio.sleep(self._heartbeat_interval)
            except asyncio.TimeoutError:
                # ping timeout - connection lost, trigger stop sequence and finish the task
                self.stop()
                return False
            except Exception as err:
                logger.exception(f"heartbeat err: f{err}")
                return False

        logger.debug("heartbeat done.")
        return True

    async def _open_connection(self):
        try:
            logger.debug("Try IPv4")
            self.reader, self.writer = await asyncio.open_connection(
                self.addr, self.port, family=socket.AF_INET
            )
            self.connected = True
            logger.info(f"connection to {self.addr}:{self.port} opened.")
            return True
        except Exception as e:
            logger.exception(f"failed to open connection: {e}")
            return False

    def _init_rpc(self):
        logger.debug("initializing rpc client and bootstrapping")
        self.client = capnp.TwoPartyClient()
        self.rpc = self.client.bootstrap().cast_as(self.bootstrap_as)

    async def _run(self):
        logger.debug("socket run.")

        # open connection
        connection_attempts = 0

        while not self.connected:
            if connection_attempts < 2:
                opened = await self._open_connection()
                if opened:
                    break
                await asyncio.sleep(1)
            if connection_attempts < 5:
                opened = await self._open_connection()
                if opened:
                    break
                await asyncio.sleep(5)
            if connection_attempts < 10:
                opened = await self._open_connection()
                if opened:
                    break
                await asyncio.sleep(10)
            else:
                break

            connection_attempts += 1

        if not self.connected:
            logger.warning("failed to connect to rpc server")
            return False

        # initialize rpc client
        self._init_rpc()

        # start background tasks and wait for them to finish
        task_results = []
        async with asyncio.TaskGroup() as tg:
            self.running = True
            logger.debug("backgrounding socket reader and writer functions")
            task_results.append(tg.create_task(self._reader()))
            task_results.append(tg.create_task(self._writer()))
            logger.debug("backgrounding heartbeat")
            task_results.append(tg.create_task(self._heartbeat()))
            logger.debug("backgrounding extra coroutines:")

            for coroutine in self.extra_coroutines:
                logger.debug(f"extra coroutine: {coroutine}")
                task_results.append(tg.create_task(coroutine))

        if self.running:
            # if the tasks finished and the client is still running, something went wrong
            logger.error("background tasks finished unexpectedly")
            return False
        else:
            logger.info("background tasks finished")

        # check if any of the tasks failed. if so, retry the connection
        task_failed = False
        for result in task_results:
            if result is False:
                task_failed = True
                break

        # clean up
        self.reader = None
        self.writer = None
        self.client = None
        self.rpc = None
        self.connected = False

        logger.debug("_run done.")
        return task_failed

    def stop(self):
        logger.info("stopping...")
        self.running = False

    async def start(self):
        logger.info(f"starting up...")

        attempts = 0
        retry = True

        while retry and attempts < 5:
            failed = await self._run()
            if not failed:
                break

            if attempts < 2:
                await asyncio.sleep(5)
            elif attempts < 5:
                await asyncio.sleep(10)

            attempts += 1

        logger.info("start exited.")

    def get_rpc(self):
        return self.rpc


class ReplayMemoryClient(RPCClient):
    def __init__(self, addr="localhost", port=60000):
        super().__init__(
            bootstrap_as=replay_memory_capnp.ReplayMemory, addr=addr, port=port
        )
