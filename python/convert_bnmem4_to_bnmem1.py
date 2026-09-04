#!/usr/bin/env python3
"""Convert a legacy full-query BNMEM4 file to unified BNMEM1."""

import argparse
import struct
from pathlib import Path


def read_exact(handle, size):
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("truncated BNMEM file")
    return data


def u32(data, offset):
    return struct.unpack_from("<I", data, offset)[0]


def convert(source, destination):
    source = Path(source)
    destination = Path(destination)
    with source.open("rb") as src:
        # Through denominator mode: magic/version/layers/ids/dims/scalars
        # plus the v4 denom field.
        fixed = read_exact(src, 192)
        if fixed[:8] != b"BNMEM4\x00\x00" or u32(fixed, 8) != 4:
            raise ValueError("source is not a BNMEM4 version-4 file")

        n_layers = u32(fixed, 12)
        d_model, kv_dim, q_dim, head_dim = struct.unpack_from(
            "<IIII", fixed, 144)
        if not 1 <= n_layers <= 32:
            raise ValueError(f"invalid layer count {n_layers}")

        bias_bytes = 2 * n_layers * 4
        biases = read_exact(src, bias_bytes)
        payload_bytes = (
            2 * n_layers * kv_dim * d_model * 4
            + 3 * n_layers * d_model * 4
            + n_layers * q_dim * 4
            + n_layers * head_dim * 4
            + n_layers * q_dim * d_model * 4
        )

        old_manifest = 192 + bias_bytes + payload_bytes
        src.seek(old_manifest)
        count = struct.unpack("<I", read_exact(src, 4))[0]
        entries = {}
        for _ in range(count):
            name_len = struct.unpack("<I", read_exact(src, 4))[0]
            name = read_exact(src, name_len)
            rows, cols, offset, crc = struct.unpack(
                "<IIQI", read_exact(src, 20))
            entries[name.decode()] = (rows, cols, offset, crc)
        if src.read(1):
            raise ValueError("trailing bytes after legacy manifest")

        order = [
            "wk", "wv", "w_agg", "gdu_aw", "gdu_bw", "mem_norm",
            "query_norm", "query_proj",
        ]
        if set(entries) != set(order):
            raise ValueError(
                f"unexpected legacy tensors: {sorted(entries)}")

        temp = destination.with_suffix(destination.suffix + ".tmp")
        try:
            with temp.open("wb") as dst:
                dst.write(b"BNMEM1\x00\x00")
                dst.write(struct.pack("<I", 1))
                dst.write(fixed[12:])
                dst.write(struct.pack("<I", 0))  # full query projection
                dst.write(biases)

                rewritten = []
                for name in order:
                    rows, cols, old_offset, crc = entries[name]
                    size = rows * cols * 4
                    new_offset = dst.tell()
                    src.seek(old_offset)
                    remaining = size
                    while remaining:
                        chunk = read_exact(src, min(1024 * 1024, remaining))
                        dst.write(chunk)
                        remaining -= len(chunk)
                    rewritten.append(
                        (name.encode(), rows, cols, new_offset, crc))

                dst.write(struct.pack("<I", len(rewritten)))
                for name, rows, cols, offset, crc in rewritten:
                    dst.write(struct.pack("<I", len(name)))
                    dst.write(name)
                    dst.write(struct.pack(
                        "<IIQI", rows, cols, offset, crc))
            temp.replace(destination)
        except Exception:
            temp.unlink(missing_ok=True)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    convert(args.source, args.destination)
    print(destination if (destination := args.destination) else "")


if __name__ == "__main__":
    main()
