import pickle
import lzma


def compress(o):
    bytes = pickle.dumps(o, protocol=5)
    compressor = lzma.LZMACompressor()
    ret = compressor.compress(bytes)
    ret += compressor.flush()
    return ret


def decompress(o):
    decompressor = lzma.LZMADecompressor()
    ret = decompressor.decompress(o)
    return pickle.loads(ret)
