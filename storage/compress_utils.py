import io
import lzma


def compress(buffer: io.BytesIO) -> bytes:
    compressor = lzma.LZMACompressor()
    ret = compressor.compress(buffer.read())
    ret += compressor.flush()
    return ret


def decompress(o: bytes) -> io.BytesIO:
    decompressor = lzma.LZMADecompressor()
    ret = decompressor.decompress(o)
    buffer = io.BytesIO(ret)
    return buffer
