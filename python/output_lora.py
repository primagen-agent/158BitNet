#!/usr/bin/env python3
"""Trainable output projection LoRA and BNLORA1 serialization."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


OUTPUT_BLOCK_INDEX = 0xFFFFFFFF
OUTPUT_LAYER_ID = 7


class OutputLoRA(nn.Module):
    def __init__(
        self, hidden: int, vocab: int, rank: int, alpha: float,
        device: str = "cuda"
    ):
        super().__init__()
        if rank < 1 or rank > 256:
            raise ValueError("output LoRA rank must be in [1, 256]")
        self.hidden = hidden
        self.vocab = vocab
        self.block_index = OUTPUT_BLOCK_INDEX
        self.layer_id = OUTPUT_LAYER_ID
        self.in_dim = hidden
        self.out_dim = vocab
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.a = nn.Parameter(torch.empty(
            rank, hidden, device=device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(
            vocab, rank, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, hidden):
        low = F.linear(hidden.float(), self.a)
        return F.linear(low, self.b) * self.scale


def save_output_lora(path: str, module: OutputLoRA) -> None:
    data = bytearray()
    data += b"BNLORA1\x00"
    data += struct.pack("<II", 1, 1)
    data += struct.pack("<fI", 1.0, 0)
    data += struct.pack(
        "<IIIIIf",
        OUTPUT_BLOCK_INDEX,
        OUTPUT_LAYER_ID,
        module.hidden,
        module.vocab,
        module.rank,
        module.scale,
    )
    data += module.a.detach().cpu().numpy().astype("<f4").tobytes()
    data += module.b.detach().cpu().numpy().astype("<f4").tobytes()
    Path(path).write_bytes(data)


def load_output_lora(
    path: str, hidden: int, vocab: int, device: str = "cuda"
) -> OutputLoRA:
    with open(path, "rb") as handle:
        if handle.read(8) != b"BNLORA1\x00":
            raise ValueError("file is not BNLORA1")
        version, tensor_count = struct.unpack("<II", handle.read(8))
        file_scale, persona_len = struct.unpack("<fI", handle.read(8))
        if version != 1 or tensor_count != 1 or persona_len != 0:
            raise ValueError("unsupported output BNLORA layout")
        block, layer, in_dim, out_dim, rank, tensor_scale = struct.unpack(
            "<IIIIIf", handle.read(24))
        if (
            block != OUTPUT_BLOCK_INDEX
            or layer != OUTPUT_LAYER_ID
            or in_dim != hidden
            or out_dim != vocab
        ):
            raise ValueError("output BNLORA geometry mismatch")
        a = np.fromfile(handle, dtype="<f4", count=rank * hidden)
        b = np.fromfile(handle, dtype="<f4", count=vocab * rank)
        if a.size != rank * hidden or b.size != vocab * rank or handle.read(1):
            raise ValueError("truncated or trailing output BNLORA data")
    module = OutputLoRA(
        hidden, vocab, rank, alpha=tensor_scale * rank * file_scale,
        device=device)
    with torch.no_grad():
        module.a.copy_(torch.from_numpy(a.reshape(rank, hidden)).to(device))
        module.b.copy_(torch.from_numpy(b.reshape(vocab, rank)).to(device))
    return module
