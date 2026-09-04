#!/usr/bin/env python3
"""Trainable backbone LoRA projections and BNLORA1 bundle I/O."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


OUTPUT_BLOCK_INDEX = 0xFFFFFFFF
LAYER_IDS = {
    "q": 0,
    "k": 1,
    "v": 2,
    "o": 3,
    "gate": 4,
    "up": 5,
    "down": 6,
    "output": 7,
}


class LoRAProjection(nn.Module):
    def __init__(
        self,
        block_index: int,
        layer_id: int,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
        device: str = "cuda",
    ):
        super().__init__()
        if rank < 1 or rank > 256:
            raise ValueError("LoRA rank must be in [1, 256]")
        self.block_index = int(block_index)
        self.layer_id = int(layer_id)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.a = nn.Parameter(torch.empty(
            rank, in_dim, device=device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(
            out_dim, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, hidden):
        low = F.linear(hidden.float(), self.a)
        return F.linear(low, self.b) * self.scale


class BackboneLoRA(nn.Module):
    def __init__(
        self,
        cfg,
        blocks,
        targets=("q", "v", "o"),
        rank: int = 8,
        alpha: float = 16.0,
        device: str = "cuda",
    ):
        super().__init__()
        dims = {
            "q": (cfg.hidden, cfg.q_dim),
            "k": (cfg.hidden, cfg.kv_dim),
            "v": (cfg.hidden, cfg.kv_dim),
            "o": (cfg.q_dim, cfg.hidden),
            "gate": (cfg.hidden, cfg.ffn),
            "up": (cfg.hidden, cfg.ffn),
            "down": (cfg.ffn, cfg.hidden),
        }
        self.projections = nn.ModuleDict()
        for block in blocks:
            if block < 0 or block >= cfg.n_layers:
                raise ValueError(f"LoRA block {block} outside model")
            for target in targets:
                if target not in dims:
                    raise ValueError(f"unsupported backbone LoRA target {target}")
                in_dim, out_dim = dims[target]
                key = self.key(block, LAYER_IDS[target])
                self.projections[key] = LoRAProjection(
                    block, LAYER_IDS[target], in_dim, out_dim,
                    rank, alpha, device=device)

    @staticmethod
    def key(block_index: int, layer_id: int) -> str:
        return f"b{block_index}_l{layer_id}"

    def delta(self, block_index: int, layer_id: int, hidden):
        key = self.key(block_index, layer_id)
        if key not in self.projections:
            return None
        return self.projections[key](hidden)

    def tensors(self):
        return list(self.projections.values())


def save_lora_bundle(path: str, tensors) -> None:
    tensors = list(tensors)
    data = bytearray()
    data += b"BNLORA1\x00"
    data += struct.pack("<II", 1, len(tensors))
    data += struct.pack("<fI", 1.0, 0)
    for tensor in tensors:
        data += struct.pack(
            "<IIIIIf",
            tensor.block_index,
            tensor.layer_id,
            tensor.in_dim,
            tensor.out_dim,
            tensor.rank,
            tensor.scale,
        )
        data += tensor.a.detach().cpu().numpy().astype("<f4").tobytes()
        data += tensor.b.detach().cpu().numpy().astype("<f4").tobytes()
    Path(path).write_bytes(data)


def load_lora_bundle(path: str, cfg, device: str = "cuda"):
    tensors = []
    with open(path, "rb") as handle:
        if handle.read(8) != b"BNLORA1\x00":
            raise ValueError("file is not BNLORA1")
        version, tensor_count = struct.unpack("<II", handle.read(8))
        file_scale, persona_len = struct.unpack("<fI", handle.read(8))
        if version != 1 or persona_len != 0:
            raise ValueError("unsupported BNLORA layout")
        for _ in range(tensor_count):
            block, layer, in_dim, out_dim, rank, tensor_scale = struct.unpack(
                "<IIIIIf", handle.read(24))
            module = LoRAProjection(
                block, layer, in_dim, out_dim, rank,
                alpha=tensor_scale * rank * file_scale,
                device=device)
            a = np.fromfile(handle, dtype="<f4", count=rank * in_dim)
            b = np.fromfile(handle, dtype="<f4", count=out_dim * rank)
            if a.size != rank * in_dim or b.size != out_dim * rank:
                raise ValueError("truncated BNLORA tensor")
            with torch.no_grad():
                module.a.copy_(
                    torch.from_numpy(a.reshape(rank, in_dim)).to(device))
                module.b.copy_(
                    torch.from_numpy(b.reshape(out_dim, rank)).to(device))
            tensors.append(module)
        if handle.read(1):
            raise ValueError("trailing BNLORA data")

    bundle = BackboneLoRA.__new__(BackboneLoRA)
    nn.Module.__init__(bundle)
    bundle.projections = nn.ModuleDict()
    output = None
    for tensor in tensors:
        if (
            tensor.block_index == OUTPUT_BLOCK_INDEX
            and tensor.layer_id == LAYER_IDS["output"]
        ):
            output = tensor
            continue
        if tensor.block_index >= cfg.n_layers:
            raise ValueError("BNLORA block outside model")
        bundle.projections[
            bundle.key(tensor.block_index, tensor.layer_id)
        ] = tensor
    return bundle, output
