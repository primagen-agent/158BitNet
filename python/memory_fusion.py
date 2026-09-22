"""Full-rank neural memory conditioning for a frozen decoder.

Research gate one: supplied episode boundaries, no automatic writer or C
deployment claim. Persistent inputs are token features, never chat KV caches.
The source text is not appended to the generation prompt.
"""
import torch
from torch import nn
from torch.nn import functional as F

FORMAT = "BITNET_MEMORY_FUSION_RESEARCH_V1"


class GatedMemoryFusion(nn.Module):
    def __init__(self, hidden, layers, heads=8, evidence_gate=False):
        super().__init__()
        if hidden <= 0 or layers <= 0 or heads <= 0 or hidden % heads:
            raise ValueError("invalid fusion geometry")
        self.hidden, self.layers, self.heads = hidden, layers, heads
        self.evidence_gate = evidence_gate
        self.encoder = nn.Linear(2 * hidden, hidden, bias=False)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)
        self.null_key = nn.Parameter(torch.zeros(heads, 1, hidden // heads))
        self.layer_gain = nn.Parameter(torch.zeros(layers))
        self.use_gate = nn.Linear((4 if evidence_gate else 1) * hidden, 1)
        nn.init.zeros_(self.use_gate.weight)
        nn.init.zeros_(self.use_gate.bias)

    def prepare(self, features):
        if features.ndim != 2 or features.shape[-1] != 2 * self.hidden:
            raise ValueError("memory feature geometry mismatch")
        if not torch.isfinite(features).all():
            raise ValueError("nonfinite memory features")
        if not len(features):
            return None
        memory = F.layer_norm(self.encoder(features.float()), (self.hidden,))
        key = self.key(memory).view(-1, self.heads, self.hidden // self.heads).transpose(0, 1)
        value = self.value(memory).view(-1, self.heads, self.hidden // self.heads).transpose(0, 1)
        key = torch.cat((self.null_key, key), dim=1)
        value = torch.cat((torch.zeros_like(self.null_key), value), dim=1)
        return key, value

    def residual(self, layer, hidden, prepared, observations=None):
        """Return the learned increment directly, avoiding add/sub cancellation."""
        if not 0 <= layer < self.layers:
            raise ValueError("fusion layer outside backbone")
        if prepared is None:
            return torch.zeros_like(hidden)
        x = F.layer_norm(hidden.float(), (self.hidden,))
        query = self.query(x).view(-1, self.heads, self.hidden // self.heads).transpose(0, 1)
        read = F.scaled_dot_product_attention(query, *prepared)
        read = read.transpose(0, 1).contiguous().view(-1, self.hidden)
        delta = self.output(read)
        if self.evidence_gate:
            evidence = F.layer_norm(delta, (self.hidden,))
            gate_input = torch.cat((x, evidence, x * evidence, (x - evidence).abs()), dim=-1)
        else:
            gate_input = x
        gate_logits = self.use_gate(gate_input)
        if observations is not None:
            observations.append(gate_logits)
        delta = delta * gate_logits.sigmoid()
        return (self.layer_gain[layer].tanh() * delta).to(hidden.dtype)

    def forward(self, layer, hidden, prepared, observations=None):
        if not 0 <= layer < self.layers:
            raise ValueError("fusion layer outside backbone")
        if prepared is None:
            return hidden  # Preserve the original empty-memory bypass exactly.
        return hidden + self.residual(layer, hidden, prepared, observations)

    def bind(self, features, observations=None):
        prepared = self.prepare(features)
        return lambda layer, hidden: self(layer, hidden, prepared, observations)
