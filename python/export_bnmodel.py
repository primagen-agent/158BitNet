"""Write and inspect single-file .bnmodel weight containers.

Layout (little-endian, mirrored by the loader in src/neural_memory.c):
  [4B "BNMW"] [4B version] [4B n_entries] [4B index_size]
  index: one compact JSON line per tensor {"n":name,"s":[dims],"o":offset}
  [1B newline]
  contiguous float32 payload; offsets are relative to the payload start.

append_tensors() is the EC-001 deployment step: it re-exports a v2 base file
plus new tensors as v3, keeping the base payload bytes at identical offsets.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"BNMW"


def _index_line(name: str, shape: tuple[int, ...], offset: int) -> bytes:
    payload = json.dumps({"n": name, "s": list(shape), "o": offset},
                         ensure_ascii=False, separators=(",", ":"))
    return payload.encode() + b"\n"


def _as_f32(tensor: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(tensor)
    if array.dtype != "<f4":
        array = array.astype("<f4")
    return array.reshape(array.shape if array.ndim else (1,))


def write_bnmodel(path, tensors: dict, version: int) -> None:
    if version not in (2, 3):
        raise ValueError(f"unsupported bnmodel version {version}")
    names = list(tensors)
    prepared = {name: _as_f32(tensors[name]) for name in names}

    offsets, cursor = {}, 0
    for name in names:
        offsets[name] = cursor
        cursor += prepared[name].size * 4

    index = b"".join(
        _index_line(name, prepared[name].shape, offsets[name]) for name in names)
    blob = b"".join(prepared[name].tobytes() for name in names)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", version, len(names), len(index)))
        f.write(index)
        f.write(b"\n")
        f.write(blob)


def read_bnmodel_index(path) -> dict:
    """Return {name: (shape_tuple, byte_offset_into_payload)}."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError("not a bnmodel file")
        version, count, index_size = struct.unpack("<III", f.read(12))
        index = f.read(index_size).decode()

    result = {}
    for line in index.splitlines():
        if not line:
            continue
        entry = json.loads(line)
        result[entry["n"]] = (tuple(entry["s"]), entry["o"])
    if len(result) != count:
        raise ValueError("index entry count mismatch")
    return result


def append_tensors(base_path, new_tensors: dict, out_path) -> dict:
    """Copy base tensors byte-identically and append new ones, version + 1."""
    with open(base_path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError("not a bnmodel file")
        base_version, count, index_size = struct.unpack("<III", f.read(12))
        index = f.read(index_size).decode()
        f.read(1)  # newline
        payload = f.read()

    entries = [json.loads(line) for line in index.splitlines() if line]
    if len(entries) != count:
        raise ValueError("index entry count mismatch")
    for entry in entries:
        if entry["n"] in new_tensors:
            raise ValueError(f"tensor {entry['n']} already exists in base")

    base_end = 0
    for entry in entries:
        n_floats = 1
        for dim in entry["s"]:
            n_floats *= dim
        base_end = max(base_end, entry["o"] + n_floats * 4)
    if base_end != len(payload):
        raise ValueError("base payload size mismatch")

    new_names = list(new_tensors)
    prepared = {name: _as_f32(new_tensors[name]) for name in new_names}
    cursor = base_end
    for name in new_names:
        shape, offset = prepared[name].shape, cursor
        entries.append({"n": name, "s": list(shape), "o": offset})
        cursor += prepared[name].size * 4

    new_index = b"".join(
        _index_line(entry["n"], tuple(entry["s"]), entry["o"]) for entry in entries)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", base_version + 1, len(entries), len(new_index)))
        f.write(new_index)
        f.write(b"\n")
        f.write(payload)
        f.write(b"".join(prepared[name].tobytes() for name in new_names))
    return read_bnmodel_index(out_path)


def replace_tensors(base_path, replacements: dict, out_path, new_version=None) -> dict:
    """Rewrite the base file with tensors replaced (same name) or added.
    Untouched payloads are copied verbatim. new_version defaults to the
    base version (use to bump v2 -> v3 when adding gated tensors)."""
    with open(base_path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError("not a bnmodel file")
        version, count, index_size = struct.unpack("<III", f.read(12))
        index = f.read(index_size).decode()
        f.read(1)
        payload = f.read()

    entries = [json.loads(line) for line in index.splitlines() if line]
    if len(entries) != count:
        raise ValueError("index entry count mismatch")

    # materialize untouched payloads by their base offsets
    for entry in entries:
        n_floats = 1
        for dim in entry["s"]:
            n_floats *= dim
        entry["data"] = payload[entry["o"]:entry["o"] + n_floats * 4]

    by_name = {e["n"]: e for e in entries}
    out_version = new_version if new_version is not None else version
    for name, tensor in replacements.items():
        array = _as_f32(tensor)
        if name in by_name:
            by_name[name]["s"] = list(array.shape)
            by_name[name]["data"] = array.tobytes()
        else:
            entry = {"n": name, "s": list(array.shape), "data": array.tobytes()}
            entries.append(entry)
            by_name[name] = entry

    offset = 0
    for entry in entries:
        entry["o"] = offset
        offset += len(entry["data"])

    names = [e["n"] for e in entries]
    new_index = b"".join(
        _index_line(e["n"], tuple(e["s"]), e["o"]) for e in entries)
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", out_version, len(entries), len(new_index)))
        f.write(new_index)
        f.write(b"\n")
        for entry in entries:
            f.write(entry["data"])
    return read_bnmodel_index(out_path)
