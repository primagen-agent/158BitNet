#!/usr/bin/env python3
"""Attach the learned operation head to its exact frozen writer binary."""
import argparse
from pathlib import Path
import struct
import zlib
import torch
from typed_memory_training import file_fingerprint
from export_typed_writer_model import tensor_payload


def attach(base_path, operation_path, output_path):
    operation = torch.load(operation_path, map_location="cpu", weights_only=True)
    if (operation.get("format") != "TYPED_CONTEXT_OPERATION_V1" or
        operation.get("writer_binary_sha256") != file_fingerprint(base_path)):
        raise ValueError("context head is not bound to this writer binary")
    base = Path(base_path).read_bytes()
    version, hidden, rank, bands, layers, maximum, count, _, _ = struct.unpack("<IIIIIIIff", base[8:44])
    if base[:8] != b"BNTWRITE" or version != 1 or count != 39:
        raise ValueError("expected the frozen V1 writer")
    sha_offset = 44 + bands * 8
    if base[sha_offset:sha_offset + 32].hex() != operation["backbone_sha256"]:
        raise ValueError("context head backbone mismatch")
    offset = sha_offset + 32 + count * 8
    if offset > len(base):
        raise ValueError("truncated writer header")
    payloads = [tensor_payload(operation["state_dict"][name], shape, name)
                for name, shape in (("0.weight", (rank, hidden * 2)), ("0.bias", (rank,)),
                                    ("2.weight", (2, rank)), ("2.bias", (2,)))]
    header = bytearray(base[:offset])
    struct.pack_into("<I", header, 8, 2)
    struct.pack_into("<I", header, 32, 43)
    for payload in payloads:
        header += struct.pack("<II", len(payload), zlib.crc32(payload) & 0xffffffff)
    Path(output_path).write_bytes(header + base[offset:] + b"".join(payloads))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base"); parser.add_argument("operation"); parser.add_argument("output")
    args = parser.parse_args()
    attach(args.base, args.operation, args.output)
