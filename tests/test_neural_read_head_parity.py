#!/usr/bin/env python3
"""C/Python parity for the learned read-decision head artifact.

The binary layout is decoded here independently of the C loader, then both
implementations score the same feature rows. This checks layer shapes, weight
ordering, standardization, activations and the backbone binding; it does not
re-derive the head's training signal.
"""

import argparse
import hashlib
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np

FEATURES = 11
MAGIC = b"NSRDHEAD"


def read_artifact(path):
    data = Path(path).read_bytes()
    if data[:8] != MAGIC:
        raise ValueError("bad read-head magic")
    version, width, layers = struct.unpack_from("<III", data, 8)
    if version != 1 or width != FEATURES or layers != 3:
        raise ValueError("unexpected read-head header")
    offset = 20
    stack = []
    for _ in range(layers):
        out_features, in_features = struct.unpack_from("<II", data, offset)
        offset += 8
        weight = np.frombuffer(data, "<f4", out_features * in_features, offset).reshape(
            out_features, in_features
        )
        offset += 4 * out_features * in_features
        bias = np.frombuffer(data, "<f4", out_features, offset)
        offset += 4 * out_features
        stack.append((weight.copy(), bias.copy()))
    mean = np.frombuffer(data, "<f4", width, offset).copy()
    offset += 4 * width
    std = np.frombuffer(data, "<f4", width, offset).copy()
    offset += 4 * width
    reader = data[offset : offset + 32]
    offset += 32
    backbone = data[offset : offset + 32]
    offset += 32
    if offset != len(data):
        raise ValueError("trailing read-head bytes")
    return stack, mean, std, reader, backbone


def forward(stack, mean, std, values):
    out = (values - mean) / std
    for index, (weight, bias) in enumerate(stack):
        out = out @ weight.T + bias
        if index + 1 < len(stack):
            out = np.maximum(out, 0.0)
    return out[:, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checker")
    parser.add_argument("gguf")
    parser.add_argument("artifact")
    parser.add_argument("head")
    args = parser.parse_args()

    stack, mean, std, reader, backbone = read_artifact(args.head)
    expected = hashlib.sha256(Path(args.gguf).read_bytes()).digest()
    assert backbone == expected, "head is bound to a different backbone"
    assert len(reader) == 32

    rng = np.random.default_rng(7)
    rows = np.empty((257, FEATURES), dtype=np.float32)
    rows[:, 0] = rng.normal(0.0, 6.0, len(rows))          # reader logit
    rows[:, 1:4] = rng.uniform(-1.0, 1.0, (len(rows), 3))  # s1, s2, s3
    for column in range(4, 7):                             # within-row offsets
        rows[:, column] = -rng.uniform(0.0, 0.5, len(rows))
    rows[:, 7] = rng.normal(0.0, 0.2, len(rows))
    rows[:, 8:10] = rows[:, 0:1] * rows[:, 4:6]
    rows[:, 10] = rows[:, 1] * rows[:, 2]
    assert np.isfinite(rows).all()

    expected_logits = forward(stack, mean, std, rows)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, output = root / "features", root / "logits"
        with source.open("wb") as handle:
            handle.write(struct.pack("<I", len(rows)))
            handle.write(rows.astype("<f4").tobytes())
        subprocess.run(
            [args.checker, args.artifact, args.gguf, args.head, str(source), str(output)],
            check=True,
        )
        with output.open("rb") as handle:
            count = struct.unpack("<I", handle.read(4))[0]
            assert count == len(rows)
            actual = np.frombuffer(handle.read(4 * count), "<f4")
            assert not handle.read()
    np.testing.assert_allclose(actual, expected_logits, atol=2e-5, rtol=2e-5)
    print(f"PASS: {len(rows)} C/Python read-decision head parity cases")


if __name__ == "__main__":
    sys.exit(main())
