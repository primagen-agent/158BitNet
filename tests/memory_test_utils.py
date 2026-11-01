"""Current snapshot inspection and local HTTP test helpers."""

from pathlib import Path
import hashlib
import socket
import struct
import subprocess
import zlib

MEMORY_INPUT = 4096
MEMORY_HIDDEN = 2048
MEMORY_WIDTH = 64
MEMORY_IDENTITY = 2112
MEMORY_TENSORS = 91
MEMORY_READ_CONTEXT_WIDTH = 16
MEMORY_RAW_BOUNDARY_OUTPUTS = 2
MEMORY_RAW_BOUNDARY_CONTEXT = 3
MEMORY_SOURCE_CONTEXT_HIDDEN = 32
MEMORY_SOURCE_CONTEXT_INPUT = 2 * MEMORY_WIDTH + MEMORY_HIDDEN
MEMORY_QUERY_HEADS = 4
MEMORY_SEMANTIC = MEMORY_WIDTH * MEMORY_QUERY_HEADS
MEMORY_QUERY = MEMORY_SEMANTIC + MEMORY_WIDTH


def validate_memory_network(path, gguf=None):
    """Check the sole 1B header and binding before starting an evaluation.

    Validate the raw residual, source-context network, independent scope query
    and raw query-context geometry. The C loader validates all sizes and CRCs.
    """
    with Path(path).open("rb") as source:
        header = source.read(56)
        declarations = source.read(MEMORY_TENSORS * 8)
    geometry = (1, MEMORY_INPUT, MEMORY_WIDTH, MEMORY_TENSORS)
    if (
        len(header) != 56
        or header[:8] != b"BNSMODEL"
        or struct.unpack_from("<4I", header, 8) != geometry
    ):
        raise ValueError("not a current 1B slot-memory model")
    if (
        len(declarations) != MEMORY_TENSORS * 8
        or struct.unpack_from("<I", declarations, 75 * 8)[0]
        != MEMORY_RAW_BOUNDARY_OUTPUTS * MEMORY_RAW_BOUNDARY_CONTEXT * MEMORY_INPUT * 4
        or any(
            struct.unpack_from("<I", declarations, index * 8)[0] != elements * 4
            for index, elements in (
                (76, MEMORY_WIDTH * (MEMORY_INPUT + 2)),
                (77, MEMORY_WIDTH),
                (78, MEMORY_SEMANTIC),
                (79, MEMORY_QUERY_HEADS),
                (80, MEMORY_WIDTH + MEMORY_RAW_BOUNDARY_CONTEXT * MEMORY_INPUT),
                (81, MEMORY_SOURCE_CONTEXT_HIDDEN * MEMORY_SOURCE_CONTEXT_INPUT),
                (82, MEMORY_SOURCE_CONTEXT_HIDDEN),
                (83, MEMORY_SOURCE_CONTEXT_HIDDEN),
                (84, 1),
                (85, 3 * 2 * MEMORY_HIDDEN),
                (86, 3),
                (87, MEMORY_READ_CONTEXT_WIDTH * 2 * MEMORY_HIDDEN),
                (88, MEMORY_READ_CONTEXT_WIDTH * MEMORY_SEMANTIC),
                (89, MEMORY_WIDTH * MEMORY_READ_CONTEXT_WIDTH),
                (90, MEMORY_WIDTH * 4 * MEMORY_INPUT),
            )
        )
    ):
        raise ValueError("not a current raw-boundary/source-context/scope-query/query-context geometry")
    if gguf is not None:
        with Path(gguf).open("rb") as source:
            identity = hashlib.file_digest(source, "sha256").digest()
        if header[24:56] != identity:
            raise ValueError("memory model is bound to a different backbone")
    return dict(
        version=1,
        input=MEMORY_INPUT,
        width=MEMORY_WIDTH,
        identity=MEMORY_IDENTITY,
        tensors=MEMORY_TENSORS,
        query_heads=MEMORY_QUERY_HEADS,
        semantic=MEMORY_SEMANTIC,
        query=MEMORY_QUERY,
        raw_boundary_shape=(MEMORY_RAW_BOUNDARY_OUTPUTS, MEMORY_RAW_BOUNDARY_CONTEXT * MEMORY_INPUT),
        raw_membership_shape=(1, MEMORY_WIDTH + MEMORY_RAW_BOUNDARY_CONTEXT * MEMORY_INPUT),
        source_context_hidden_shape=(MEMORY_SOURCE_CONTEXT_HIDDEN, MEMORY_SOURCE_CONTEXT_INPUT),
        source_context_output_shape=(1, MEMORY_SOURCE_CONTEXT_HIDDEN),
        scope_query_encoder_shape=(MEMORY_WIDTH, MEMORY_INPUT + 2),
        scope_query_pool_shape=(MEMORY_QUERY_HEADS, MEMORY_WIDTH),
        read_context_memory_shape=(MEMORY_READ_CONTEXT_WIDTH, 2 * MEMORY_HIDDEN),
        read_context_query_shape=(MEMORY_READ_CONTEXT_WIDTH, MEMORY_SEMANTIC),
        read_context_delta_shape=(MEMORY_WIDTH, MEMORY_READ_CONTEXT_WIDTH),
        query_context_shape=(MEMORY_WIDTH, 4 * MEMORY_INPUT),
        backbone_sha256=header[24:56].hex(),
    )


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def stop_server(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def assert_no_kv(response):
    session = response.get("bitnet_session") or {}
    if session.get("cached_tokens") != 0 or session.get("reused_tokens") != 0:
        raise AssertionError(f"KV cache was reused: {session}")


def read_slots(path):
    raw = Path(path).read_bytes()
    assert raw[:8] == b"BNSSTATE"
    assert struct.unpack_from("<I", raw, 8)[0] == 2
    size, crc = struct.unpack_from("<II", raw, 44)
    body = raw[52:]
    assert len(body) == size and zlib.crc32(body) == crc
    offset = 0

    def take(fmt):
        nonlocal offset
        width = struct.calcsize(fmt)
        assert offset + width <= len(body), "truncated snapshot field"
        value = struct.unpack_from(fmt, body, offset)
        offset += width
        return value

    def string():
        nonlocal offset
        count = take("<I")[0]
        assert offset + count <= len(body), "truncated snapshot string"
        value = body[offset : offset + count].decode()
        offset += count
        return value

    count, _ = take("<IQ")
    rows = []
    for _ in range(count):
        identifier, _, _, status = take("<QQQI")
        content, source, reference = string(), string(), string()
        speaker, observed_at, source_id = string(), string(), string()
        start, end, tokens = take("<QQI")
        if status != 2:
            width = (MEMORY_IDENTITY + tokens * MEMORY_WIDTH + MEMORY_READ_CONTEXT_WIDTH) * 4
            assert offset + width <= len(body), "truncated 1B snapshot representations"
            offset += width
        rows.append(
            dict(
                id=identifier,
                status=status,
                content=content,
                source=source,
                reference=reference,
                speaker=speaker,
                observed_at=observed_at,
                source_id=source_id,
                source_start=start,
                source_end=end,
            )
        )
    assert offset == len(body)
    return rows
