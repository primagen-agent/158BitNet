"""Label-free boundary for reader component diagnostics (supplied episodes).

Only natural messages enter the encoder; metadata IDs never enter neural forward.
This adapter is not an automatic writer or a generation prompt builder.
"""
from dataclasses import dataclass
import json

import torch


def validate_input(record):
    if not isinstance(record, dict) or set(record) != {"id", "context", "episodes"}:
        raise ValueError("runtime input must contain only id, context and episodes")
    if not isinstance(record["id"], str) or not record["id"]:
        raise ValueError("case id required")
    if not isinstance(record["context"], list) or not record["context"]:
        raise ValueError("nonempty context required")
    if not isinstance(record["episodes"], list):
        raise ValueError("episode boundaries required")
    for message in record["context"] + record["episodes"]:
        if not isinstance(message, dict) or set(message) != {"role", "speaker", "text"}:
            raise ValueError("message contains non-runtime fields")
        if message["role"] not in ("user", "assistant", "tool", "system"):
            raise ValueError("invalid role")
        if any(not isinstance(message[key], str) or not message[key].strip() for key in message):
            raise ValueError("nonempty natural message required")
    if record["context"][-1]["role"] != "user":
        raise ValueError("context must end at current user message")


def encoder_texts(record):
    validate_input(record)
    # Sources stay in independent encoder calls, never in the query/context text.
    render = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return render(record["context"]), tuple(render(message) for message in record["episodes"])


@dataclass(frozen=True)
class ReaderFeatures:
    query: torch.Tensor
    episodes: tuple[torch.Tensor, ...]

    def __post_init__(self):
        if type(self.episodes) is not tuple:
            raise ValueError("episode tensor boundaries must be immutable")
        for tokens in (self.query,) + self.episodes:
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or not len(tokens):
                raise ValueError("nonempty token features required")
            if tokens.shape[1] == 0 or tokens.shape[1] % 2 or tokens.shape[1] != self.query.shape[1]:
                raise ValueError("contextual/lexical feature geometry mismatch")
            if not tokens.is_floating_point() or not torch.isfinite(tokens).all():
                raise ValueError("finite floating-point features required")
            if tokens.device != self.query.device or tokens.dtype != self.query.dtype:
                raise ValueError("mixed feature domain")


def encode_reader_input(record, encoder):
    query, episodes = encoder_texts(record)
    # Own storage: downstream mutation must not modify a shared source feature bank.
    return ReaderFeatures(encoder(query).detach().clone(),
                          tuple(encoder(text).detach().clone() for text in episodes))


def reader_forward(reader, features):
    if type(features) is not ReaderFeatures:
        raise ValueError("forward requires label-free ReaderFeatures")
    return reader(features.query, features.episodes)
