"""Backbone feature encoding shared by addressed-memory training tools."""

from __future__ import annotations

import torch


@torch.inference_mode()
def encode_text(backbone, tokenizer, text, max_tokens, pooling="mean_last"):
    ids = tokenizer.encode(text, add_bos=True)[:max_tokens]
    hidden = backbone(
        torch.tensor(ids, device=backbone.device, dtype=torch.long),
        return_hidden=True)
    if pooling == "last":
        vector = hidden[-1].float()
    elif pooling == "mean_last":
        vector = hidden.float().mean(dim=0) + hidden[-1].float()
    else:
        raise ValueError(f"unsupported pooling mode {pooling}")
    return torch.nn.functional.normalize(vector, dim=0).cpu().numpy()


def query_text(question):
    return (
        "Find the stored conversation facts needed to answer this question: "
        + str(question))
