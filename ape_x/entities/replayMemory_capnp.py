"""This is an automatically generated stub for `replayMemory.capnp`."""
import os

import capnp  # type: ignore

capnp.remove_import_hook()
here = os.path.dirname(os.path.abspath(__file__))
module_file = os.path.abspath(os.path.join(here, "replayMemory.capnp"))
TransitionBatch = capnp.load(module_file).TransitionBatch
TransitionBatchBuilder = TransitionBatch
TransitionBatchReader = TransitionBatch
PriorityUpdate = capnp.load(module_file).PriorityUpdate
PriorityUpdateBuilder = PriorityUpdate
PriorityUpdateReader = PriorityUpdate
