import io
import lzma


def compress_buffer(buffer: io.BytesIO) -> bytes:
    """Compress a buffer using lz4, returning the compressed bytes.
    Don't forget to close the buffer once you are done.

    Args:
        buffer (io.BytesIO): A buffer

    Returns:
        bytes: The compressed bytes
    """
    return compress_bytes(buffer.read())


def compress_bytes(b: bytes) -> bytes:
    compressor = lzma.LZMACompressor()
    ret = compressor.compress(b)
    ret += compressor.flush()
    return ret


def decompress_bytes(o: bytes) -> bytes:
    decompressor = lzma.LZMADecompressor()
    ret = decompressor.decompress(o)
    return ret
