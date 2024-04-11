# not named _test to avoid getting autotested by pytest
storage_config = StorageConfig(
    "127.0.0.1", 5553, "ezra", "EA00E05EC7592F8A4B41FED9B30A3D26"
)

import numpy as np

import sys

sys.path.append("..")
from storage.storage import Storage, StorageConfig
from storage.compress_utils import compress, decompress


def test_storage():
    client = Storage(storage_config, True)

    assert client.get_weights() == None

    assert client.get_latest_weights_id() == None

    weights = np.arange(1000000)

    client.store_weights(compress(weights))
    weights_id = client.get_latest_weights_id()

    print(weights_id)
    assert weights_id != None

    new_weights = decompress(client.get_weights())
    print(new_weights)

    assert np.all(weights == new_weights)

    weights = weights * 2

    client.store_weights(compress(weights))
    new_weights_id = client.get_latest_weights_id()

    print(weights_id)
    assert weights_id != new_weights_id and new_weights_id is not None

    new_weights = decompress(client.get_weights())
    print(new_weights)
    assert np.all(weights == new_weights)


test_storage()
