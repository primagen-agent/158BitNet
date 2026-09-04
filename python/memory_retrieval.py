"""Utilities for turning slot attention into ordered memory excerpts."""
from __future__ import annotations

import re

import torch


def append_pointer_token_ids(memory, token_ids, device):
    if memory.state_mode != "slots":
        return
    previous = getattr(memory, "pointer_token_ids", None)
    current = torch.tensor(token_ids, device=device, dtype=torch.long)
    if current.numel() > 0:
        current = current.clone()
        current[0] = -1
    if previous is None:
        previous = current.new_empty((0,))
    memory.pointer_token_ids = torch.cat(
        (previous, current))[-memory.max_memory_slots:]


def _attention(memory, layer_mode):
    weights = getattr(memory, "last_pointer_weights", None)
    layer_weights = [
        item for item in getattr(
            memory, "last_pointer_weights_by_layer", [])
        if item is not None]
    if layer_mode != "last" and layer_weights:
        stacked = torch.stack(layer_weights)
        if layer_mode == "mean":
            weights = stacked.mean(dim=0)
        elif layer_mode == "max":
            weights = stacked.max(dim=0).values
        elif layer_mode == "rrf":
            final_scores = stacked[:, -1, :]
            order = final_scores.argsort(dim=-1, descending=True)
            ranks = order.argsort(dim=-1).float()
            weights = (
                1.0 / (60.0 + ranks + 1.0)
            ).sum(dim=0, keepdim=True)
    return weights


def clean_excerpt(text):
    text = re.sub(r"<\|im_(?:start|end)\|>", " ", text)
    text = re.sub(
        r"Store this conversation in long-term memory\."
        r"\s*Reply OK\.?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:assistant|user|OK)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def select_memory_excerpts(memory, decoder, window_count, window_size,
                           layer_mode="max"):
    weights = _attention(memory, layer_mode)
    token_ids = getattr(memory, "pointer_token_ids", None)
    if (
        weights is None
        or token_ids is None
        or weights.shape[-1] != token_ids.numel()
        or token_ids.numel() == 0
    ):
        return []
    scores = weights[-1].float().clone()
    scores[token_ids < 0] = -float("inf")
    radius = max(window_size // 2, 1)
    windows = []
    blocked = torch.zeros_like(scores, dtype=torch.bool)
    for _ in range(window_count):
        candidate_scores = scores.masked_fill(blocked, -float("inf"))
        center = int(candidate_scores.argmax().item())
        if not torch.isfinite(candidate_scores[center]):
            break
        start = max(0, center - radius)
        end = min(token_ids.numel(), start + window_size)
        start = max(0, end - window_size)
        ids = token_ids[start:end]
        ids = ids[ids >= 0].tolist()
        if ids:
            text = clean_excerpt(decoder.decode(ids))
            if text and text not in windows:
                windows.append(text)
        blocked[max(0, start - radius):min(
            token_ids.numel(), end + radius)] = True
    return windows


def add_retrieved_excerpts(prompt, excerpts):
    if not excerpts:
        return prompt
    marker = "<|im_end|>\n<|im_start|>assistant\n"
    if marker not in prompt:
        raise ValueError("query prompt is missing the assistant marker")
    memory_text = "\nRetrieved memory excerpts:\n" + "\n".join(
        f"- {text}" for text in excerpts)
    return prompt.replace(
        marker, memory_text + "<|im_end|>\n<|im_start|>assistant\n", 1)
