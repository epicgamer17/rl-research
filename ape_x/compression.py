import pickle
import bz2


def pickle_and_compress(data):
    data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    compressed_data = bz2.compress(data)
    return compressed_data


def decompress_and_unpickle(compressed_data):
    data = bz2.decompress(compressed_data)
    return pickle.loads(data)
