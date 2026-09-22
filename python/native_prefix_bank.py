"""Frozen C prefix activations for offline training, never a serving memory store."""
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import torch

from native_memory_encoder import BACKBONE_SHA256, HIDDEN, VOCAB, sha_file
from prepare_neural_memory_protocol import digest


ROW_BYTES = 12 + 4 * (HIDDEN + VOCAB)


def validate_prefix(ids):
    if type(ids) is not tuple or not 2 <= len(ids) <= 128 or any(type(i) is not int or not 0 <= i < VOCAB for i in ids):
        raise ValueError("full immutable native prefix required")


def extract_prefix_batch(probe, gguf, prefixes, directory):
    if not 1 <= len(prefixes) <= 10000: raise ValueError("invalid batch count")
    for ids in prefixes: validate_prefix(ids)
    root = Path(directory); root.mkdir(parents=True, exist_ok=False)
    source, target, log = (root/n for n in ("input.bin", "output.bin", "native.log"))
    with source.open("xb") as stream:
        stream.write(b"BNPI0001" + struct.pack("<I", len(prefixes)))
        for ids in prefixes: stream.write(struct.pack("<I",len(ids)) + np.asarray(ids,dtype="<i4").tobytes())
    env = {k:v for k,v in os.environ.items() if not k.startswith("BITNET_")}
    env.update(BITNET_NUM_THREADS="4",BITNET_QUIET="0",BITNET_METAL_I2S="0",BITNET_METAL_OUTPUT="0")
    with log.open("x") as stream:
        subprocess.run([str(probe),str(gguf),str(source),str(target)],check=True,stdout=stream,stderr=stream,env=env,timeout=3600)
    lines = log.read_text().splitlines()
    if "[bitnet] cpu tier: arm_neon" not in lines or f"BNP_TRACE rows={len(prefixes)} fresh_contexts={len(prefixes)}" not in lines:
        raise ValueError("unqualified native dispatch/fresh context trace")
    return read_prefix_batch(target,prefixes)


def read_prefix_batch(path, prefixes):
    expected = 20 + len(prefixes)*ROW_BYTES
    if Path(path).stat().st_size != expected: raise ValueError("prefix artifact length mismatch")
    raw = np.memmap(path,dtype=np.uint8,mode="r")
    if bytes(raw[:8]) != b"BNPO0001" or struct.unpack("<III",bytes(raw[8:20])) != (len(prefixes),HIDDEN,VOCAB):
        raise ValueError("prefix artifact header mismatch")
    hidden = np.ndarray((len(prefixes),HIDDEN),dtype="<f4",buffer=raw,offset=32,strides=(ROW_BYTES,4))
    logits = np.ndarray((len(prefixes),VOCAB),dtype="<f4",buffer=raw,offset=32+4*HIDDEN,strides=(ROW_BYTES,4))
    for i,ids in enumerate(prefixes):
        validate_prefix(ids)
        if struct.unpack("<III",bytes(raw[20+i*ROW_BYTES:32+i*ROW_BYTES])) != (len(ids),0,len(ids)):
            raise ValueError("prefix position mismatch or KV reuse")
    if not np.isfinite(hidden).all() or not np.isfinite(logits).all(): raise ValueError("nonfinite native features")
    return hidden,logits


class PrefixBank:
    def __init__(self, root, *, expected_manifest_sha256):
        self.root = Path(root)
        self.manifest = json.loads((self.root/"manifest.json").read_text())
        if digest(self.manifest) != expected_manifest_sha256 or self.manifest["backbone_sha256"] != BACKBONE_SHA256:
            raise ValueError("prefix bank identity mismatch")
        self.rows = {}; self.arrays = []
        for batch in self.manifest["batches"]:
            path=self.root/batch["path"]
            if path.resolve().parent.parent != self.root.resolve() or sha_file(path) != batch["sha256"]:
                raise ValueError("prefix batch path or hash mismatch")
            prefixes=[tuple(ids) for ids in batch["prefixes"]]
            arrays=read_prefix_batch(path,prefixes); self.arrays.append(arrays)
            for index,ids in enumerate(prefixes):
                key=digest(ids)
                if key in self.rows: raise ValueError("duplicate prefix bank entry")
                self.rows[key]=(len(self.arrays)-1,index)

    def __call__(self, prefix):
        validate_prefix(prefix)
        key=digest(prefix)
        if key not in self.rows: raise ValueError("prefix absent from offline bank; not a generation cache")
        batch,index=self.rows[key]
        return tuple(torch.from_numpy(array[index].copy()) for array in self.arrays[batch])
