"""Trainable memory-answer decoder.

This is a standalone nonlinear residual decoder, not a LoRA update.  It
transforms the final normalized hidden state before the frozen vocabulary
projection and is saved independently from backbone weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryAnswerDecoder(nn.Module):
    MAGIC = "BNANSWER1"

    def __init__(self, hidden: int, width: int, device="cuda",
                 memory_aware: bool = False,
                 structured_memory: bool = False):
        super().__init__()
        if hidden < 1 or width < 1:
            raise ValueError("answer decoder dimensions must be positive")
        self.hidden = int(hidden)
        self.width = int(width)
        self.structured_memory = bool(structured_memory)
        self.memory_aware = bool(
            memory_aware or self.structured_memory)
        self.norm = nn.Parameter(torch.ones(hidden, device=device))
        self.gate = nn.Linear(hidden, width, bias=False, device=device)
        self.value = nn.Linear(hidden, width, bias=False, device=device)
        if self.memory_aware:
            self.memory_norm = nn.Parameter(
                torch.ones(hidden, device=device))
            if self.structured_memory:
                self.memory_query = nn.Linear(
                    hidden, width, bias=False, device=device)
                self.memory_key = nn.Linear(
                    hidden, width, bias=False, device=device)
                self.memory_value = nn.Linear(
                    hidden, width, bias=False, device=device)
            else:
                self.memory_gate = nn.Linear(
                    hidden, width, bias=False, device=device)
                self.memory_value = nn.Linear(
                    hidden, width, bias=False, device=device)
        self.out = nn.Linear(width, hidden, bias=False, device=device)
        self.scale = nn.Parameter(torch.zeros((), device=device))
        nn.init.normal_(self.gate.weight, std=hidden ** -0.5)
        nn.init.normal_(self.value.weight, std=hidden ** -0.5)
        if self.structured_memory:
            nn.init.normal_(
                self.memory_query.weight, std=hidden ** -0.5)
            nn.init.normal_(
                self.memory_key.weight, std=hidden ** -0.5)
            nn.init.normal_(
                self.memory_value.weight, std=hidden ** -0.5)
        elif self.memory_aware:
            nn.init.normal_(
                self.memory_gate.weight, std=hidden ** -0.5)
            nn.init.normal_(
                self.memory_value.weight, std=hidden ** -0.5)
        nn.init.normal_(self.out.weight, std=width ** -0.5)

    def forward(self, hidden, memory_summary=None):
        work = hidden.float()
        inv = torch.rsqrt(work.pow(2).mean(-1, keepdim=True) + 1e-6)
        normalized = work * inv * self.norm
        gate = self.gate(normalized)
        value = self.value(normalized)
        if self.memory_aware:
            if memory_summary is None:
                memory_summary = work.new_zeros(
                    (1,) + tuple(work.shape))
            memory = memory_summary.float()
            memory_inv = torch.rsqrt(
                memory.pow(2).mean(-1, keepdim=True) + 1e-6)
            memory_normalized = (
                memory * memory_inv * self.memory_norm)
            if self.structured_memory:
                if memory_normalized.ndim == work.ndim:
                    memory_normalized = memory_normalized.unsqueeze(0)
                query = self.memory_query(normalized)
                keys = self.memory_key(memory_normalized)
                memory_values = self.memory_value(memory_normalized)
                scores = torch.einsum(
                    "tw,ltw->tl", query, keys)
                scores = scores / self.width ** 0.5
                weights = F.softmax(scores, dim=-1)
                context = torch.einsum(
                    "tl,ltw->tw", weights, memory_values)
                gate = gate + context
                value = value + context
            else:
                if memory_normalized.ndim > work.ndim:
                    memory_normalized = memory_normalized.mean(dim=0)
                gate = gate + self.memory_gate(memory_normalized)
                value = value + self.memory_value(memory_normalized)
        update = self.out(F.silu(gate) * value)
        return (work + torch.tanh(self.scale) * update).to(hidden.dtype)

    def save(self, path: str) -> None:
        payload = {
            "magic": self.MAGIC,
            "version": (
                3 if self.structured_memory
                else 2 if self.memory_aware else 1),
            "hidden": self.hidden,
            "width": self.width,
            "memory_aware": self.memory_aware,
            "structured_memory": self.structured_memory,
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in self.state_dict().items()
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, expected_hidden: int, device="cuda"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("magic") != cls.MAGIC
            or payload.get("version") not in (1, 2, 3)
        ):
            raise ValueError("unsupported memory answer decoder")
        hidden = int(payload["hidden"])
        if hidden != int(expected_hidden):
            raise ValueError(
                f"answer decoder hidden {hidden} != backbone {expected_hidden}")
        memory_aware = bool(
            payload.get("memory_aware", payload.get("version") == 2))
        structured_memory = bool(
            payload.get(
                "structured_memory", payload.get("version") == 3))
        module = cls(
            hidden, int(payload["width"]), device=device,
            memory_aware=memory_aware,
            structured_memory=structured_memory)
        module.load_state_dict(payload["state_dict"], strict=True)
        return module
