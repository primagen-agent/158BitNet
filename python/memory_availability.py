"""Neural task/evidence control, separate from the memory content read.

Research prototype: accepts hidden/token features, never semantic gold fields.
Insufficient evidence includes conflict, which is not separately certified here.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from memory_fusion import GatedMemoryFusion


STATES = ("no_memory_needed", "supported", "insufficient")


@dataclass(frozen=True)
class AvailabilityRead:
    residual: torch.Tensor
    content_residual: torch.Tensor
    uncertainty_residual: torch.Tensor
    state_logits: torch.Tensor | None
    state_probabilities: torch.Tensor | None


class AvailabilityMemoryFusion(nn.Module):
    def __init__(self, hidden, layers, heads=8):
        super().__init__()
        self.content = GatedMemoryFusion(hidden, layers, heads)
        self.hidden, self.layers, self.heads = hidden, layers, heads
        self.state_encoder = nn.Linear(4 * hidden + 2, hidden)
        self.state_head = nn.Linear(hidden, len(STATES))
        self.uncertainty_output = nn.Linear(hidden, hidden, bias=False)
        nn.init.zeros_(self.uncertainty_output.weight)

    def prepare(self, features):
        return self.content.prepare(features)

    def inspect(self, layer, hidden, prepared, *, enabled=True):
        if type(enabled) is not bool or type(layer) is not int or not 0 <= layer < self.layers:
            raise ValueError("explicit feature switch and valid layer required")
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 2 or not len(hidden) or
                hidden.shape[1] != self.hidden or hidden.dtype != torch.float32 or not torch.isfinite(hidden).all()):
            raise ValueError("finite FP32 native hidden matrix required")
        zero = torch.zeros_like(hidden)
        if not enabled:
            return AvailabilityRead(zero, zero, zero, None, None)
        x = F.layer_norm(hidden, (self.hidden,))
        if prepared is None:
            read = zero
            null_mass = torch.ones(len(hidden), 1, device=hidden.device, dtype=hidden.dtype)
            present = torch.zeros_like(null_mass)
        else:
            if type(prepared) is not tuple or len(prepared) != 2:
                raise ValueError("prepared neural key/value pair required")
            key, value = prepared
            if any(not isinstance(t, torch.Tensor) or t.ndim != 3 or t.dtype != hidden.dtype or
                   t.device != hidden.device or not torch.isfinite(t).all() for t in prepared):
                raise ValueError("invalid memory feature domain")
            if key.shape != value.shape or key.shape[0] != self.heads or key.shape[1] < 2 or key.shape[2] != self.hidden // self.heads:
                raise ValueError("invalid neural memory geometry")
            query = self.content.query(x).view(-1, self.heads, self.hidden // self.heads).transpose(0, 1)
            attention = (query @ key.transpose(-1, -2) / math.sqrt(self.hidden // self.heads)).softmax(-1)
            read = (attention @ value).transpose(0, 1).contiguous().view(-1, self.hidden)
            null_mass = attention[:, :, 0].mean(0)[:, None]
            present = torch.ones_like(null_mass)
        evidence = F.layer_norm(read, (self.hidden,))
        state_hidden = self.state_encoder(torch.cat((x, evidence, x * evidence, (x - evidence).abs(), null_mass, present), -1)).tanh()
        logits = self.state_head(state_hidden)
        if prepared is None:
            # Structural absence is observable, not a gold relevance label.
            logits = logits.masked_fill(torch.tensor([False, True, False], device=hidden.device), -torch.inf)
        probabilities = logits.softmax(-1)
        content = probabilities[:, 1:2] * self.content.residual(layer, hidden, prepared)
        uncertainty = probabilities[:, 2:3] * self.uncertainty_output(state_hidden)
        return AvailabilityRead(content + uncertainty, content, uncertainty, logits, probabilities)

    def residual(self, layer, hidden, prepared, observations=None):
        result = self.inspect(layer, hidden, prepared)
        if observations is not None:
            observations.append(result.state_logits)
        return result.residual
