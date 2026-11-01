"""Learned read-decision head over frozen lexical token alignment.

The frozen backbone exposes two feature channels per token: a contextual hidden
half and a context-free lexical half. The lexical half is exact for identical
tokens, so a decision head over token alignment can compare an unseen question
wording with a stored fact without a keyword rule: every input is a model
embedding, every weight is learned from fit data, and the only fixed constants
are feature standardization statistics recorded in the artifact.

Pair features (per query, fact):
    s1  query-token coverage of fact tokens (mean over query tokens of the best
        cosine to any fact token)
    s2  fact-token coverage by query tokens
    s3  single best token-pair cosine
    r1, r2, m1, peak   within-query normalization (row maximum / mean), which
        keeps the decision independent of how many facts the bank holds
    score * r1, score * r2, s1 * s2   interactions with the reader's own logit

The head replaces the reader logit at activation: it sees the reader logit as
one input and decides support. Write, reference and update decisions are not
affected and keep using the reader logit directly.
"""
import hashlib
import struct
from pathlib import Path

import torch
from torch.nn import functional as F

FORMAT = "neural-read-head-v1"
MAGIC = b"NSRDHEAD"
VERSION = 1
FEATURE_NAMES = ("score", "s1", "s2", "s3", "r1", "r2", "m1", "peak",
                 "score_r1", "score_r2", "s1_s2")
HIDDEN = 32


def padded_lexical(features, texts):
    """Stack variable-length lexical token matrices from the frozen feature cache."""
    matrices = [F.normalize(features[text]["x"][:, features[text]["x"].shape[-1] // 2:], dim=-1)
                for text in texts]
    width = max(len(m) for m in matrices)
    values = torch.zeros(len(matrices), width, matrices[0].shape[-1])
    mask = torch.zeros(len(matrices), width, dtype=torch.bool)
    for i, matrix in enumerate(matrices):
        values[i, : len(matrix)] = matrix
        mask[i, : len(matrix)] = True
    return values, mask


def pair_stats(query_values, query_mask, fact_values, fact_mask, chunk=16):
    """Lexical alignment statistics for every query/fact pair."""
    s1, s2, s3 = [], [], []
    for start in range(0, len(query_values), chunk):
        q = query_values[start : start + chunk]
        qm = query_mask[start : start + chunk]
        sim = torch.einsum("qad,fbd->qfab", q, fact_values)
        sim = sim.masked_fill(~fact_mask[None, :, None, :], -2.0)
        sim = sim.masked_fill(~qm[:, None, :, None], -2.0)
        row, col = sim.max(-1).values, sim.max(-2).values
        # Padded positions contribute nothing: C sums only real tokens.
        s1.append((row * qm[:, None, :]).sum(-1)
                  / qm.sum(-1)[:, None].clamp_min(1))
        s2.append((col * fact_mask[None]).sum(-1)
                  / fact_mask.sum(-1)[None].clamp_min(1))
        s3.append(sim.amax((2, 3)))
    return torch.cat(s1), torch.cat(s2), torch.cat(s3)


def features(score, s1, s2, s3):
    r1 = s1 - s1.max(-1).values[:, None]
    r2 = s2 - s2.max(-1).values[:, None]
    m1 = s1 - s1.mean(-1)[:, None]
    peak = s3 - s3.mean(-1)[:, None]
    return torch.stack((score, s1, s2, s3, r1, r2, m1, peak,
                        score * r1, score * r2, s1 * s2), -1)


def view_features(features_cache, score, queries, facts):
    """Compose the head input exactly as serving composes it."""
    query_values, query_mask = padded_lexical(features_cache, queries)
    fact_values, fact_mask = padded_lexical(features_cache, facts)
    return features(score, *pair_stats(query_values, query_mask, fact_values, fact_mask))


class ReadHead(torch.nn.Module):
    def __init__(self, width=len(FEATURE_NAMES), hidden=HIDDEN):
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("std", torch.ones(width))
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(width, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1)])

    def normalize(self, values):
        return (values - self.mean) / self.std

    def forward(self, values):
        out = self.normalize(values)
        for layer in self.layers:
            out = layer(out)
        return out.squeeze(-1)


def fit_head(views, *, seeds, steps=1200, learning_rate=3e-3, weight_decay=1e-3):
    """Train on fit views only; return the seed with the lowest fit loss."""
    stacked = torch.cat([values.reshape(-1, values.shape[-1]) for values, _ in views])
    mean, std = stacked.mean(0), stacked.std(0).clamp_min(1e-6)
    best = None
    for seed in seeds:
        torch.manual_seed(seed)
        head = ReadHead()
        head.mean.copy_(mean)
        head.std.copy_(std)
        optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate,
                                     weight_decay=weight_decay)
        loss = None
        for _ in range(steps):
            optimizer.zero_grad()
            loss = sum(F.binary_cross_entropy_with_logits(head(values), gold.float())
                       for values, gold in views) / len(views)
            loss.backward()
            optimizer.step()
        if best is None or float(loss) < best[0]:
            best = (float(loss), head.eval())
    return best[1]


def write_binary(handle, head, metadata):
    """Explicit little-endian artifact; no pickle enters the C runtime."""
    layers = [layer for layer in head.layers if hasattr(layer, "weight")]
    handle.write(MAGIC)
    handle.write(struct.pack("<II", VERSION, len(FEATURE_NAMES)))
    handle.write(struct.pack("<I", len(layers)))
    for layer in layers:
        out_features, in_features = layer.weight.shape
        handle.write(struct.pack("<II", out_features, in_features))
        handle.write(layer.weight.detach().cpu().contiguous().numpy().tobytes())
        handle.write(layer.bias.detach().cpu().contiguous().numpy().tobytes())
    handle.write(head.mean.detach().cpu().contiguous().numpy().tobytes())
    handle.write(head.std.detach().cpu().contiguous().numpy().tobytes())
    handle.write(bytes.fromhex(metadata["reader_sha256"]))
    handle.write(bytes.fromhex(metadata["backbone_sha256"]))


def save(path, head, metadata):
    torch.save(dict(format=FORMAT, metadata=metadata,
                    state_dict={k: v.detach().cpu() for k, v in head.state_dict().items()}),
               path)


def load(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError("unsupported read head format")
    head = ReadHead().eval()
    head.load_state_dict(payload["state_dict"])
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head, payload["metadata"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
