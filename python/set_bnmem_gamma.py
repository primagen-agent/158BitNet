#!/usr/bin/env python3
"""Copy a BNMEM1 checkpoint while replacing only its fusion gamma."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


GAMMA_OFFSET = 8 + 4 + 4 + 32 * 4 + 4 * 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("gamma", type=float)
    args = parser.parse_args()
    if not 0.0 <= args.gamma <= 1.0:
        parser.error("gamma must be in [0, 1]")

    data = bytearray(Path(args.input).read_bytes())
    if data[:8] != b"BNMEM1\x00\x00":
        raise ValueError("input is not BNMEM1")
    if len(data) < GAMMA_OFFSET + 4:
        raise ValueError("truncated BNMEM1 header")
    old_gamma = struct.unpack_from("<f", data, GAMMA_OFFSET)[0]
    struct.pack_into("<f", data, GAMMA_OFFSET, args.gamma)
    Path(args.output).write_bytes(data)
    print(
        f"copied {args.input} -> {args.output}; "
        f"gamma {old_gamma:.6g} -> {args.gamma:.6g}"
    )


if __name__ == "__main__":
    main()
