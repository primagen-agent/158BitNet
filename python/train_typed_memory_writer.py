#!/usr/bin/env python3
"""Train an open-set typed writer for immutable versioned memory."""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ggw import GGUFWeights
from c_tokenizer import CTokenizer
from typed_memory_training import (
    clone_state_dict,
    file_fingerprint,
    token_span_variants,
)


SLOT_NAMES = ("entity", "predicate", "value", "time")


def load_raw_worlds(path):
    worlds = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            world_id = str(metadata.get("world_id", ""))
            semantic_world_id = str(
                metadata.get(
                    "semantic_world_id",
                    world_id))
            layout_view = int(
                metadata.get("layout_view", 0))
            events = metadata.get("typed_events") or []
            if not world_id or not events:
                raise ValueError(
                    f"missing typed event metadata in {path}")
            encoded = json.dumps(
                events, sort_keys=True, separators=(",", ":"))
            previous = worlds.setdefault(
                world_id, (
                    events, encoded,
                    semantic_world_id,
                    layout_view,
                ))
            if previous[1] != encoded:
                raise ValueError(
                    f"typed events changed within {world_id}")
    return {
        world_id: {
            "events": values[0],
            "semantic_world_id": values[2],
            "layout_view": values[3],
        }
        for world_id, values in worlds.items()
    }


def load_writer_examples(
    feature_cache, raw_jsonl, backbone_sha, tokenizer,
    token_embedding_table=None,
):
    payload = torch.load(
        feature_cache, map_location="cpu",
        weights_only=True)
    if payload.get("backbone_sha256") != backbone_sha:
        raise ValueError(
            f"feature cache backbone mismatch: {feature_cache}")
    feature_worlds = {}
    for row in payload.get("rows") or []:
        world_id = str(row.get("world_id", ""))
        if not world_id:
            raise ValueError("feature row has no world_id")
        episode_ids = row.get("episode_ids")
        if episode_ids is None:
            raise ValueError(
                "feature cache has no episode token ids")
        feature_worlds.setdefault(
            world_id,
            (episode_ids, row["episode_hidden"]))
    raw_worlds = load_raw_worlds(raw_jsonl)
    if set(feature_worlds) != set(raw_worlds):
        raise ValueError(
            "raw and encoded typed worlds do not match")
    examples = []
    for world_id in sorted(raw_worlds):
        raw_world = raw_worlds[world_id]
        events = raw_world["events"]
        token_ids, hidden = feature_worlds[world_id]
        if (
            len(events) != len(hidden)
            or len(events) != len(token_ids)
        ):
            raise ValueError(
                f"event/feature count mismatch: {world_id}")
        count = len(events)
        for event, ids, features in zip(
            events, token_ids, hidden,
        ):
            if len(ids) != features.shape[0]:
                raise ValueError(
                    f"token/feature length mismatch: "
                    f"{world_id}/{event['episode']}")
            episode = int(event["episode"])
            example = {
                "world_id": world_id,
                "semantic_world_id":
                    raw_world["semantic_world_id"],
                "layout_view":
                    raw_world["layout_view"],
                "episode": episode,
                "hidden": features,
                "token_ids": [int(token) for token in ids],
                "entity": str(event["entity"]),
                "predicate": str(event["predicate"]),
                "entity_surface": str(
                    event.get(
                        "entity_surface", event["entity"])),
                "predicate_surface": str(
                    event.get(
                        "predicate_surface",
                        event["predicate"])),
                "value": str(event["value"]),
                "time": str(event["time"]),
                "value_surface": str(event.get(
                    "value_surface", event["value"])),
                "time_surface": str(event.get(
                    "time_surface", event["time"])),
                "operation": (
                    0 if event["operation"] == "create" else 1),
                "position": (
                    2.0 * episode / max(count - 1, 1) - 1.0),
                "previous_episode":
                    event.get("previous_episode"),
            }
            if token_embedding_table is not None:
                example["identity_hidden"] = (
                    token_embedding_table[
                        example["token_ids"]].clone())
            for field in SLOT_NAMES:
                span = token_span_variants(
                    example["token_ids"],
                    example[f"{field}_surface"],
                    tokenizer)
                if span is None:
                    raise ValueError(
                        f"cannot locate unique {field} token span: "
                        f"{world_id}/{episode} "
                        f"{example[f'{field}_surface']!r}")
                start, end = span
                target = torch.zeros(
                    len(example["token_ids"]),
                    dtype=torch.bool)
                target[start:end + 1] = True
                example[f"{field}_token_target"] = target
            examples.append(example)
    return examples


def load_token_embedding_table(
    gguf, lib, expected_sha, hidden,
):
    if file_fingerprint(gguf) != expected_sha:
        raise ValueError("GGUF backbone identity mismatch")
    weights = GGUFWeights(gguf, lib)
    try:
        if weights.hidden != hidden:
            raise ValueError(
                "GGUF hidden size does not match checkpoint")
        matrix = weights.get_f32(
            "token_embd.weight",
            (weights.vocab, weights.hidden))
        return torch.from_numpy(
            matrix).to(torch.float16)
    finally:
        weights.close()


def collate_examples(examples, selected, device):
    maximum = max(
        examples[index]["hidden"].shape[0]
        for index in selected)
    feature_shape = tuple(
        examples[selected[0]]["hidden"].shape[1:])
    hidden = torch.zeros(
        len(selected), maximum, *feature_shape,
        device=device, dtype=torch.float32)
    identity_hidden = None
    if "identity_hidden" in examples[selected[0]]:
        identity_hidden = torch.zeros(
            len(selected), maximum,
            examples[selected[0]][
                "identity_hidden"].shape[-1],
            device=device, dtype=torch.float32)
    mask = torch.zeros(
        len(selected), maximum,
        device=device, dtype=torch.bool)
    entity_token_target = torch.zeros_like(mask)
    predicate_token_target = torch.zeros_like(mask)
    value_token_target = torch.zeros_like(mask)
    time_token_target = torch.zeros_like(mask)
    operation = []
    position = []
    entity_span_start = []
    entity_span_end = []
    predicate_span_start = []
    predicate_span_end = []
    value_span_start = []
    value_span_end = []
    time_span_start = []
    time_span_end = []
    rows = []
    for batch_index, index in enumerate(selected):
        example = examples[index]
        value = example["hidden"]
        count = value.shape[0]
        hidden[batch_index, :count] = value.to(
            device=device, dtype=torch.float32)
        if identity_hidden is not None:
            identity_hidden[batch_index, :count] = (
                example["identity_hidden"].to(
                    device=device,
                    dtype=torch.float32))
        mask[batch_index, :count] = True
        entity_token_target[batch_index, :count] = (
            example["entity_token_target"].to(device))
        predicate_token_target[batch_index, :count] = (
            example["predicate_token_target"].to(device))
        value_token_target[batch_index, :count] = (
            example["value_token_target"].to(device))
        time_token_target[batch_index, :count] = (
            example["time_token_target"].to(device))
        for name, starts, ends in (
            (
                "entity", entity_span_start,
                entity_span_end,
            ),
            (
                "predicate", predicate_span_start,
                predicate_span_end,
            ),
            (
                "value", value_span_start,
                value_span_end,
            ),
            (
                "time", time_span_start,
                time_span_end,
            ),
        ):
            span = example[
                f"{name}_token_target"].nonzero(
                    as_tuple=False).flatten()
            starts.append(int(span[0]))
            ends.append(int(span[-1]))
        operation.append(example["operation"])
        position.append(example["position"])
        rows.append(example)
    return {
        "hidden": hidden,
        "identity_hidden": identity_hidden,
        "mask": mask,
        "entity_token_target": entity_token_target,
        "predicate_token_target": predicate_token_target,
        "value_token_target": value_token_target,
        "time_token_target": time_token_target,
        "entity_span_start": torch.tensor(
            entity_span_start, device=device,
            dtype=torch.long),
        "entity_span_end": torch.tensor(
            entity_span_end, device=device,
            dtype=torch.long),
        "predicate_span_start": torch.tensor(
            predicate_span_start, device=device,
            dtype=torch.long),
        "predicate_span_end": torch.tensor(
            predicate_span_end, device=device,
            dtype=torch.long),
        "value_span_start": torch.tensor(
            value_span_start, device=device,
            dtype=torch.long),
        "value_span_end": torch.tensor(
            value_span_end, device=device,
            dtype=torch.long),
        "time_span_start": torch.tensor(
            time_span_start, device=device,
            dtype=torch.long),
        "time_span_end": torch.tensor(
            time_span_end, device=device,
            dtype=torch.long),
        "operation": torch.tensor(
            operation, device=device, dtype=torch.long),
        "position": torch.tensor(
            position, device=device, dtype=torch.float32),
        "rows": rows,
    }


class TypedMemoryWriter(nn.Module):
    """Four typed attention slots with open-set address logits."""

    def __init__(
        self, hidden, rank, bands,
        separate_address_localizer=False,
        predict_span_boundaries=False,
        use_token_embeddings=False,
        hard_address_pooling=False,
        contiguous_span_pooling=False,
        max_field_span=8,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.rank = int(rank)
        self.band_count = int(bands)
        self.separate_address_localizer = bool(
            separate_address_localizer)
        self.predict_span_boundaries = bool(
            predict_span_boundaries)
        self.use_token_embeddings = bool(
            use_token_embeddings)
        self.hard_address_pooling = bool(
            hard_address_pooling)
        self.contiguous_span_pooling = bool(
            contiguous_span_pooling)
        self.max_field_span = int(max_field_span)
        if self.max_field_span < 1:
            raise ValueError(
                "maximum field span must be positive")
        self.band_logits = nn.Parameter(
            torch.zeros(len(SLOT_NAMES), bands))
        self.projections = nn.ModuleDict({
            name: nn.Linear(hidden, rank, bias=False)
            for name in SLOT_NAMES
        })
        self.token_keys = nn.Parameter(
            torch.randn(len(SLOT_NAMES), rank)
            / math.sqrt(rank))
        if self.separate_address_localizer:
            self.localizer_band_logits = nn.Parameter(
                torch.zeros(2, bands))
            self.localizer_projections = nn.ModuleDict({
                name: nn.Linear(
                    hidden, rank, bias=False)
                for name in ("entity", "predicate")
            })
        else:
            self.register_parameter(
                "localizer_band_logits", None)
            self.localizer_projections = nn.ModuleDict()
        if self.predict_span_boundaries:
            self.span_boundary_keys = nn.Parameter(
                torch.randn(len(SLOT_NAMES), 2, rank)
                / math.sqrt(rank))
        else:
            self.register_parameter(
                "span_boundary_keys", None)
        if self.contiguous_span_pooling:
            self.segment_length_logits = nn.Parameter(
                torch.zeros(
                    len(SLOT_NAMES),
                    self.max_field_span))
            self.segment_log_scale = nn.Parameter(
                torch.full(
                    (len(SLOT_NAMES),),
                    math.log(10.0)))
            self.segment_heads = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(self.rank, self.rank),
                    nn.GELU(),
                    nn.Linear(self.rank, 1),
                )
                for name in SLOT_NAMES
            })
        else:
            self.register_parameter(
                "segment_length_logits", None)
            self.register_parameter(
                "segment_log_scale", None)
            self.segment_heads = nn.ModuleDict()
        self.operation_head = nn.Sequential(
            nn.Linear(rank * len(SLOT_NAMES), rank),
            nn.GELU(),
            nn.Linear(rank, 2),
        )
        self.time_head = nn.Sequential(
            nn.Linear(rank, rank),
            nn.GELU(),
            nn.Linear(rank, 1),
            nn.Tanh(),
        )
        self.entity_log_scale = nn.Parameter(
            torch.tensor(math.log(10.0)))
        self.entity_bias = nn.Parameter(torch.tensor(-5.0))
        self.predicate_log_scale = nn.Parameter(
            torch.tensor(math.log(10.0)))
        self.predicate_bias = nn.Parameter(torch.tensor(-5.0))

    def slot(
        self, hidden, mask, slot_index, name,
        identity_hidden=None,
    ):
        if (
            self.use_token_embeddings
            and name in ("entity", "predicate")
        ):
            if identity_hidden is None:
                raise ValueError(
                    "token embedding address input is missing")
            address_mixed = identity_hidden
        else:
            address_weight = F.softmax(
                self.band_logits[slot_index], dim=0)
            address_mixed = (
                hidden
                * address_weight.view(
                    1, 1, self.band_count, 1)
            ).sum(dim=2)
        address_token = F.normalize(
            self.projections[name](address_mixed),
            dim=-1, eps=1e-12)
        if (
            self.separate_address_localizer
            and name in ("entity", "predicate")
        ):
            localizer_index = (
                0 if name == "entity" else 1)
            localizer_weight = F.softmax(
                self.localizer_band_logits[
                    localizer_index],
                dim=0)
            localizer_mixed = (
                hidden
                * localizer_weight.view(
                    1, 1, self.band_count, 1)
            ).sum(dim=2)
            score_token = F.normalize(
                self.localizer_projections[name](
                    localizer_mixed),
                dim=-1, eps=1e-12)
        else:
            score_token = address_token
        score = torch.einsum(
            "btr,r->bt", score_token,
            self.token_keys[slot_index])
        score = score.masked_fill(~mask, -1e9)
        attention = F.softmax(score, dim=-1)
        pooling_attention = attention
        segment_logits = None
        if (
            self.contiguous_span_pooling
        ):
            field_index = SLOT_NAMES.index(name)
            (
                segment_logits,
                pooling_attention,
            ) = contiguous_span_distribution(
                self.segment_heads[name](
                    score_token).squeeze(-1)
                * self.segment_log_scale[
                    field_index].exp().clamp(
                        1.0, 100.0),
                mask,
                self.segment_length_logits[
                    field_index],
                self.max_field_span)
        elif (
            self.hard_address_pooling
            and name in ("entity", "predicate")
        ):
            pooling_attention = torch.zeros_like(
                attention).scatter_(
                    1,
                    attention.argmax(
                        dim=-1, keepdim=True),
                    1.0)
        address_attention = (
            pooling_attention.detach()
            if (
                self.separate_address_localizer
                and name in ("entity", "predicate")
            )
            else pooling_attention
        )
        state = F.normalize(
            torch.einsum(
                "bt,btr->br",
                address_attention, address_token),
            dim=-1, eps=1e-12)
        return (
            state, attention, score_token,
            address_token,
            segment_logits,
        )

    def forward(
        self, hidden, mask, identity_hidden=None,
    ):
        states = {}
        attentions = {}
        score_tokens = {}
        address_tokens = {}
        segment_logits = {}
        for index, name in enumerate(SLOT_NAMES):
            (
                states[name],
                attentions[name],
                score_tokens[name],
                address_tokens[name],
                segment,
            ) = self.slot(
                hidden, mask, index, name,
                identity_hidden)
            if segment is not None:
                segment_logits[name] = segment
        combined = torch.cat(
            [states[name] for name in SLOT_NAMES],
            dim=-1)
        result = {
            **states,
            "attentions": attentions,
            "score_tokens": score_tokens,
            "address_tokens": address_tokens,
            "operation_logits":
                self.operation_head(combined),
            "time_position":
                self.time_head(states["time"]).squeeze(-1),
        }
        if self.predict_span_boundaries:
            span_logits = {}
            for index, name in enumerate(SLOT_NAMES):
                logits = torch.einsum(
                    "btr,kr->bkt",
                    score_tokens[name],
                    self.span_boundary_keys[index])
                logits = logits.masked_fill(
                    ~mask[:, None, :], -1e9)
                span_logits[name] = {
                    "start": logits[:, 0],
                    "end": logits[:, 1],
                }
            result["span_logits"] = span_logits
        if self.contiguous_span_pooling:
            result["segment_logits"] = segment_logits
        return result

    def address_logits(self, left, right, kind):
        cosine = left @ right.T
        if kind == "entity":
            scale = self.entity_log_scale.exp().clamp(
                1.0, 100.0)
            return scale * cosine + self.entity_bias
        if kind == "predicate":
            scale = self.predicate_log_scale.exp().clamp(
                1.0, 100.0)
            return scale * cosine + self.predicate_bias
        raise ValueError(f"unknown address kind: {kind}")


def equality_matrix(rows, field, device):
    return torch.tensor([
        [
            float(left[field] == right[field])
            for right in rows
        ]
        for left in rows
    ], device=device, dtype=torch.float32)


def balanced_pair_loss(logits, targets):
    diagonal = torch.eye(
        logits.shape[0], device=logits.device,
        dtype=torch.bool)
    positive = (targets > 0.5) & ~diagonal
    negative = (targets <= 0.5) & ~diagonal
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError(
            "pair batch needs positive and negative pairs")
    return (
        F.softplus(-logits[positive]).mean()
        + F.softplus(logits[negative]).mean()
    ) * 0.5


def hard_negative_pair_loss(
    logits, targets, count=8, margin=0.5,
):
    diagonal = torch.eye(
        logits.shape[0], device=logits.device,
        dtype=torch.bool)
    negative = (targets <= 0.5) & ~diagonal
    masked = logits.masked_fill(
        ~negative, -float("inf"))
    available = negative.sum(dim=-1)
    if not bool((available > 0).all()):
        raise ValueError(
            "hard-negative batch has an empty negative row")
    selected = masked.topk(
        min(count, logits.shape[1] - 1),
        dim=-1).values
    finite = torch.isfinite(selected)
    return F.softplus(
        margin + selected[finite]).mean()


def attention_span_loss(attention, target):
    if attention.shape != target.shape:
        raise ValueError("attention target shape mismatch")
    if not bool(target.any(dim=-1).all()):
        raise ValueError("attention target has an empty row")
    mass = (
        attention * target.to(attention.dtype)
    ).sum(dim=-1)
    return -mass.clamp_min(1e-9).log().mean()


def contiguous_span_distribution(
    token_score, mask, length_logits,
    max_span,
):
    batch, tokens = token_score.shape
    maximum = min(int(max_span), tokens)
    logits = token_score.new_full(
        (batch, tokens, int(max_span)), -1e9)
    for length in range(1, maximum + 1):
        count = tokens - length + 1
        window_score = token_score.unfold(
            1, length, 1).mean(dim=-1)
        window_valid = mask.unfold(
            1, length, 1).all(dim=-1)
        logits[:, :count, length - 1] = (
            window_score
            + length_logits[length - 1]
        ).masked_fill(~window_valid, -1e9)
    probability = F.softmax(
        logits.flatten(1), dim=-1
    ).reshape_as(logits)
    inclusion = token_score.new_zeros(
        batch, tokens)
    for length in range(1, maximum + 1):
        count = tokens - length + 1
        span_probability = probability[
            :, :count, length - 1]
        for offset in range(length):
            inclusion[:, offset:offset + count] += (
                span_probability)
    inclusion = (
        inclusion * mask.to(inclusion.dtype))
    inclusion = (
        inclusion
        / inclusion.sum(
            dim=-1, keepdim=True).clamp_min(1e-9))
    return logits, inclusion


def segment_span_loss(output, batch, max_span):
    losses = {}
    for name in SLOT_NAMES:
        start = batch[f"{name}_span_start"]
        end = batch[f"{name}_span_end"]
        length = end - start + 1
        if bool((length > max_span).any()):
            raise ValueError(
                f"{name} span exceeds maximum length")
        target = start * max_span + length - 1
        losses[name] = F.cross_entropy(
            output["segment_logits"][name].flatten(1),
            target)
    return (
        sum(losses.values()) * (2.0 / len(SLOT_NAMES)),
        losses,
    )


def span_boundary_loss(output, batch):
    losses = {}
    for name in SLOT_NAMES:
        logits = output["span_logits"][name]
        losses[name] = 0.5 * (
            F.cross_entropy(
                logits["start"],
                batch[f"{name}_span_start"])
            + F.cross_entropy(
                logits["end"],
                batch[f"{name}_span_end"]))
    return (
        sum(losses.values()) * (2.0 / len(SLOT_NAMES)),
        losses,
    )


def writer_loss(
    model, output, batch, operation_weight,
    hard_negative_weight=1.0,
    span_attention_weight=0.0,
    span_boundary_weight=0.0,
    span_segment_weight=0.0,
):
    entity_targets = equality_matrix(
        batch["rows"], "entity",
        output["entity"].device)
    predicate_targets = equality_matrix(
        batch["rows"], "predicate",
        output["predicate"].device)
    joint_targets = (
        entity_targets * predicate_targets)
    entity_logits = model.address_logits(
        output["entity"], output["entity"], "entity")
    predicate_logits = model.address_logits(
        output["predicate"], output["predicate"],
        "predicate")
    joint_logits = torch.minimum(
        entity_logits, predicate_logits)
    entity_loss = balanced_pair_loss(
        entity_logits, entity_targets)
    predicate_loss = balanced_pair_loss(
        predicate_logits, predicate_targets)
    joint_loss = balanced_pair_loss(
        joint_logits, joint_targets)
    entity_hard = hard_negative_pair_loss(
        entity_logits, entity_targets)
    predicate_hard = hard_negative_pair_loss(
        predicate_logits, predicate_targets)
    joint_hard = hard_negative_pair_loss(
        joint_logits, joint_targets)
    operation_loss = F.cross_entropy(
        output["operation_logits"],
        batch["operation"],
        weight=operation_weight)
    time_loss = F.smooth_l1_loss(
        output["time_position"],
        batch["position"])
    if span_attention_weight > 0.0:
        attention_parts = {
            name: attention_span_loss(
                output["attentions"][name],
                batch[f"{name}_token_target"])
            for name in SLOT_NAMES
        }
        attention_loss = (
            sum(attention_parts.values())
            * (2.0 / len(SLOT_NAMES))
        )
    else:
        attention_loss = output["entity"].new_zeros(())
        attention_parts = {
            name: attention_loss for name in SLOT_NAMES
        }
    if span_boundary_weight > 0.0:
        boundary_loss, boundary_parts = (
            span_boundary_loss(output, batch))
    else:
        boundary_loss = output["entity"].new_zeros(())
        boundary_parts = {
            name: boundary_loss
            for name in SLOT_NAMES
        }
    if span_segment_weight > 0.0:
        segment_loss, segment_parts = (
            segment_span_loss(
                output, batch,
                model.max_field_span))
    else:
        segment_loss = output["entity"].new_zeros(())
        segment_parts = {
            name: segment_loss
            for name in SLOT_NAMES
        }
    total = (
        entity_loss
        + predicate_loss
        + 0.5 * joint_loss
        + hard_negative_weight * (
            0.25 * entity_hard
            + 0.25 * predicate_hard
            + 0.5 * joint_hard)
        + 0.5 * operation_loss
        + 0.2 * time_loss
        + span_attention_weight * attention_loss
        + span_boundary_weight * boundary_loss
        + span_segment_weight * segment_loss
    )
    return total, {
        "entity": entity_loss.detach(),
        "predicate": predicate_loss.detach(),
        "joint": joint_loss.detach(),
        "entity_hard": entity_hard.detach(),
        "predicate_hard": predicate_hard.detach(),
        "joint_hard": joint_hard.detach(),
        "operation": operation_loss.detach(),
        "time": time_loss.detach(),
        **{
            f"{name}_attention":
                attention_parts[name].detach()
            for name in SLOT_NAMES
        },
        **{
            f"{name}_boundary":
                boundary_parts[name].detach()
            for name in SLOT_NAMES
        },
        "entity_segment":
            segment_parts["entity"].detach(),
        "predicate_segment":
            segment_parts["predicate"].detach(),
        "value_segment":
            segment_parts["value"].detach(),
        "time_segment":
            segment_parts["time"].detach(),
    }


def pair_balanced_accuracy(logits, targets):
    diagonal = torch.eye(
        logits.shape[0], device=logits.device,
        dtype=torch.bool)
    positive = (targets > 0.5) & ~diagonal
    negative = (targets <= 0.5) & ~diagonal
    positive_accuracy = (
        (logits[positive] > 0).float().mean().item())
    negative_accuracy = (
        (logits[negative] <= 0).float().mean().item())
    return (
        0.5 * (positive_accuracy + negative_accuracy),
        positive_accuracy,
        negative_accuracy,
    )


@torch.inference_mode()
def encode_all(model, examples, device, batch_size):
    outputs = {
        "entity": [],
        "predicate": [],
        "operation_logits": [],
        "time_position": [],
    }
    for name in SLOT_NAMES:
        outputs[f"{name}_attention_mass"] = []
        outputs[f"{name}_attention_hit"] = []
        outputs[f"{name}_attention_peak"] = []
        outputs[f"{name}_attention_top4"] = []
    if (
        model.predict_span_boundaries
        or model.contiguous_span_pooling
    ):
        span_names = (
            SLOT_NAMES
            if (
                model.predict_span_boundaries
                or model.contiguous_span_pooling
            )
            else ("entity", "predicate")
        )
        for name in span_names:
            outputs[f"{name}_span_start"] = []
            outputs[f"{name}_span_end"] = []
    model.eval()
    for offset in range(0, len(examples), batch_size):
        selected = list(range(
            offset, min(offset + batch_size, len(examples))))
        batch = collate_examples(
            examples, selected, device)
        result = model(
            batch["hidden"], batch["mask"],
            batch["identity_hidden"])
        for key in (
            "entity", "predicate",
            "operation_logits", "time_position",
        ):
            outputs[key].append(result[key].cpu())
        for name in SLOT_NAMES:
            attention = result["attentions"][name]
            target = batch[f"{name}_token_target"]
            outputs[f"{name}_attention_mass"].append(
                (attention * target.to(attention.dtype))
                .sum(dim=-1).cpu())
            peak = attention.argmax(dim=-1, keepdim=True)
            outputs[f"{name}_attention_peak"].append(
                peak.squeeze(1).cpu())
            if not bool((batch["mask"].sum(
                dim=-1) >= 4).all()):
                raise ValueError(
                    "typed episode has fewer than four tokens")
            outputs[f"{name}_attention_top4"].append(
                attention.topk(
                    4, dim=-1).indices.cpu())
            outputs[f"{name}_attention_hit"].append(
                target.gather(1, peak)
                .squeeze(1).float().cpu())
            if model.predict_span_boundaries:
                span_logits = result[
                    "span_logits"][name]
                outputs[f"{name}_span_start"].append(
                    span_logits["start"].argmax(
                        dim=-1).cpu())
                outputs[f"{name}_span_end"].append(
                    span_logits["end"].argmax(
                        dim=-1).cpu())
            elif model.contiguous_span_pooling:
                flat = result[
                    "segment_logits"][name].flatten(1)
                selected_span = flat.argmax(dim=-1)
                start = (
                    selected_span
                    // model.max_field_span)
                length = (
                    selected_span
                    % model.max_field_span + 1)
                outputs[f"{name}_span_start"].append(
                    start.cpu())
                outputs[f"{name}_span_end"].append(
                    (start + length - 1).cpu())
    return {
        key: torch.cat(values, dim=0)
        for key, values in outputs.items()
    }


def nearest_accuracy(logits, labels):
    logits = logits.clone()
    logits.fill_diagonal_(-float("inf"))
    nearest = logits.argmax(dim=-1).tolist()
    return sum(
        labels[index] == labels[target]
        for index, target in enumerate(nearest)
    ) / max(len(labels), 1)


def version_link_intervals(
    examples, operation_logits, joint_logits,
):
    by_world = collections.defaultdict(list)
    for index, example in enumerate(examples):
        by_world[example["world_id"]].append(index)
    create_correct = 0
    total = 0
    lower_bounds = []
    upper_bounds = []
    for indices in by_world.values():
        indices.sort(
            key=lambda index: examples[index]["episode"])
        episode_to_position = {
            examples[index]["episode"]: position
            for position, index in enumerate(indices)}
        for position, index in enumerate(indices):
            example = examples[index]
            predicted_operation = int(
                operation_logits[index].argmax())
            if example["operation"] == 0:
                create_correct += int(
                    predicted_operation == 0)
                total += 1
                continue
            previous_position = episode_to_position.get(
                example["previous_episode"])
            if (
                predicted_operation != 1
                or previous_position is None
                or previous_position >= position
            ):
                lower_bounds.append(float("inf"))
                upper_bounds.append(-float("inf"))
                total += 1
                continue
            true_index = indices[previous_position]
            later_false = indices[
                previous_position + 1:position]
            lower_bounds.append(
                max(
                    (
                        float(joint_logits[index, candidate])
                        for candidate in later_false
                    ),
                    default=-float("inf"),
                ))
            upper_bounds.append(
                float(joint_logits[index, true_index]))
            total += 1
    return (
        create_correct,
        total,
        torch.tensor(lower_bounds),
        torch.tensor(upper_bounds),
    )


def version_link_accuracy(
    examples, operation_logits, joint_logits,
    threshold=0.0,
):
    create_correct, total, lower, upper = (
        version_link_intervals(
            examples, operation_logits, joint_logits))
    update_correct = int(
        ((float(threshold) >= lower)
         & (float(threshold) < upper)).sum())
    return (
        create_correct + update_correct
    ) / max(total, 1)


def compiled_token_version_accuracy(
    examples, operation_logits, output,
):
    by_world = collections.defaultdict(list)
    for index, example in enumerate(examples):
        by_world[example["world_id"]].append(index)
    correct = 0
    valid_key = 0
    for indices in by_world.values():
        indices.sort(
            key=lambda index: examples[index]["episode"])
        active = {}
        for index in indices:
            example = examples[index]
            key_parts = []
            valid = True
            for name in ("entity", "predicate"):
                start = int(output[
                    f"{name}_span_start"][index])
                end = int(output[
                    f"{name}_span_end"][index])
                token_ids = example["token_ids"]
                if not 0 <= start <= end < len(token_ids):
                    valid = False
                    break
                key_parts.append(tuple(
                    token_ids[start:end + 1]))
            predicted_operation = int(
                operation_logits[index].argmax())
            if not valid:
                continue
            valid_key += 1
            key = tuple(key_parts)
            if example["operation"] == 0:
                correct += int(predicted_operation == 0)
            else:
                previous = active.get(key)
                correct += int(
                    predicted_operation == 1
                    and previous is not None
                    and examples[previous]["episode"]
                    == example["previous_episode"])
            active[key] = index
    return (
        correct / max(len(examples), 1),
        valid_key / max(len(examples), 1),
    )


def compiled_attention_version_accuracy(
    examples, operation_logits, output,
):
    boundary_output = {}
    for name in ("entity", "predicate"):
        peak = output[f"{name}_attention_peak"]
        boundary_output[f"{name}_span_start"] = peak
        boundary_output[f"{name}_span_end"] = peak
    return compiled_token_version_accuracy(
        examples, operation_logits,
        boundary_output)


def compiled_attention_topk_version_accuracy(
    examples, operation_logits, output,
    entity_k, predicate_k,
):
    by_world = collections.defaultdict(list)
    for index, example in enumerate(examples):
        by_world[example["world_id"]].append(index)
    correct = 0
    for indices in by_world.values():
        indices.sort(
            key=lambda index: examples[index]["episode"])
        active = {}
        for index in indices:
            example = examples[index]
            key_parts = []
            for name, count in (
                ("entity", entity_k),
                ("predicate", predicate_k),
            ):
                positions = output[
                    f"{name}_attention_top4"
                ][index, :count].tolist()
                key_parts.append(tuple(sorted(
                    example["token_ids"][position]
                    for position in positions)))
            key = tuple(key_parts)
            predicted_operation = int(
                operation_logits[index].argmax())
            if example["operation"] == 0:
                correct += int(predicted_operation == 0)
            else:
                previous = active.get(key)
                correct += int(
                    predicted_operation == 1
                    and previous is not None
                    and examples[previous]["episode"]
                    == example["previous_episode"])
            active[key] = index
    return correct / max(len(examples), 1)


@torch.inference_mode()
def evaluate(
    model, examples, device, batch_size,
    version_threshold=0.0,
):
    output = encode_all(
        model, examples, device, batch_size)
    entity_logits = model.address_logits(
        output["entity"].to(device),
        output["entity"].to(device),
        "entity").cpu()
    predicate_logits = model.address_logits(
        output["predicate"].to(device),
        output["predicate"].to(device),
        "predicate").cpu()
    entity_targets = equality_matrix(
        examples, "entity", "cpu")
    predicate_targets = equality_matrix(
        examples, "predicate", "cpu")
    joint_targets = (
        entity_targets * predicate_targets)
    joint_logits = torch.minimum(
        entity_logits, predicate_logits)
    entity_balanced, entity_positive, entity_negative = (
        pair_balanced_accuracy(
            entity_logits, entity_targets))
    predicate_balanced, predicate_positive, predicate_negative = (
        pair_balanced_accuracy(
            predicate_logits, predicate_targets))
    joint_balanced, joint_positive, joint_negative = (
        pair_balanced_accuracy(
            joint_logits, joint_targets))
    entities = [
        example["entity"] for example in examples]
    predicates = [
        example["predicate"] for example in examples]
    keys = list(zip(entities, predicates))
    operation_prediction = output[
        "operation_logits"].argmax(dim=-1)
    operation_target = torch.tensor([
        example["operation"] for example in examples])
    operation_scores = []
    for operation in (0, 1):
        selected = operation_target == operation
        operation_scores.append(float(
            (operation_prediction[selected]
             == operation_target[selected]).float().mean()))
    absolute_version_accuracy = version_link_accuracy(
        examples, output["operation_logits"],
        joint_logits, version_threshold)
    (
        attention_compiled_accuracy,
        attention_compiled_valid,
    ) = compiled_attention_version_accuracy(
        examples, output["operation_logits"],
        output)
    metrics = {
        "examples": len(examples),
        "entity_pair_balanced": entity_balanced,
        "entity_pair_positive": entity_positive,
        "entity_pair_negative": entity_negative,
        "predicate_pair_balanced": predicate_balanced,
        "predicate_pair_positive": predicate_positive,
        "predicate_pair_negative": predicate_negative,
        "joint_pair_balanced": joint_balanced,
        "joint_pair_positive": joint_positive,
        "joint_pair_negative": joint_negative,
        "entity_nearest": nearest_accuracy(
            entity_logits, entities),
        "predicate_nearest": nearest_accuracy(
            predicate_logits, predicates),
        "entity_attention_mass": float(
            output["entity_attention_mass"].mean()),
        "entity_attention_hit_accuracy": float(
            output["entity_attention_hit"].mean()),
        "predicate_attention_mass": float(
            output["predicate_attention_mass"].mean()),
        "predicate_attention_hit_accuracy": float(
            output["predicate_attention_hit"].mean()),
        "joint_key_nearest": nearest_accuracy(
            joint_logits, keys),
        "operation_macro_accuracy":
            sum(operation_scores) / len(operation_scores),
        "create_accuracy": operation_scores[0],
        "update_accuracy": operation_scores[1],
        "time_position_mae": float(
            (output["time_position"]
             - torch.tensor([
                 example["position"]
                 for example in examples]))
            .abs().mean()),
        "version_link_accuracy":
            absolute_version_accuracy,
        "absolute_version_link_accuracy":
            absolute_version_accuracy,
        "version_threshold": float(version_threshold),
        "compiled_attention_version_accuracy":
            attention_compiled_accuracy,
        "compiled_attention_valid_key_rate":
            attention_compiled_valid,
    }
    for name in SLOT_NAMES:
        metrics[f"{name}_attention_mass"] = float(
            output[f"{name}_attention_mass"].mean())
        metrics[
            f"{name}_attention_hit_accuracy"
        ] = float(
            output[f"{name}_attention_hit"].mean())
    for entity_k in (1, 2):
        for predicate_k in (1, 2, 3, 4):
            metrics[
                f"compiled_attention_top"
                f"{entity_k}_{predicate_k}_accuracy"
            ] = compiled_attention_topk_version_accuracy(
                examples, output["operation_logits"],
                output, entity_k, predicate_k)
    if (
        model.predict_span_boundaries
        or model.contiguous_span_pooling
    ):
        all_field_exact = torch.ones(
            len(examples), dtype=torch.bool)
        address_exact = torch.ones(
            len(examples), dtype=torch.bool)
        span_names = (
            SLOT_NAMES
            if (
                model.predict_span_boundaries
                or model.contiguous_span_pooling
            )
            else ("entity", "predicate")
        )
        for name in span_names:
            target_start = torch.tensor([
                int(example[
                    f"{name}_token_target"
                ].nonzero(
                    as_tuple=False).flatten()[0])
                for example in examples
            ])
            target_end = torch.tensor([
                int(example[
                    f"{name}_token_target"
                ].nonzero(
                    as_tuple=False).flatten()[-1])
                for example in examples
            ])
            exact = (
                output[f"{name}_span_start"]
                == target_start
            ) & (
                output[f"{name}_span_end"]
                == target_end
            )
            metrics[f"{name}_span_exact"] = float(
                exact.float().mean())
            all_field_exact &= exact
            if name in ("entity", "predicate"):
                address_exact &= exact
        metrics["address_span_exact"] = float(
            address_exact.float().mean())
        metrics["joint_span_exact"] = float(
            all_field_exact.float().mean())
        metrics["all_field_span_exact"] = float(
            all_field_exact.float().mean())
        compiled_accuracy, valid_key_rate = (
            compiled_token_version_accuracy(
                examples,
                output["operation_logits"],
                output))
        metrics["compiled_token_version_accuracy"] = (
            compiled_accuracy)
        metrics["compiled_token_valid_key_rate"] = (
            valid_key_rate)
    return metrics


@torch.inference_mode()
def calibrate_version_threshold(
    model, examples, device, batch_size,
):
    output = encode_all(
        model, examples, device, batch_size)
    entity_logits = model.address_logits(
        output["entity"].to(device),
        output["entity"].to(device),
        "entity").cpu()
    predicate_logits = model.address_logits(
        output["predicate"].to(device),
        output["predicate"].to(device),
        "predicate").cpu()
    joint_logits = torch.minimum(
        entity_logits, predicate_logits)
    create_correct, total, lower, upper = (
        version_link_intervals(
            examples, output["operation_logits"],
            joint_logits))
    candidates = torch.linspace(-8.0, 12.0, 161)
    correct = (
        (candidates[:, None] >= lower[None, :])
        & (candidates[:, None] < upper[None, :])
    ).sum(dim=1) + create_correct
    best_accuracy = float(
        correct.max()) / max(total, 1)
    best_indices = (
        correct == correct.max()).nonzero(
            as_tuple=False).flatten()
    best_index = min(
        best_indices.tolist(),
        key=lambda index: abs(float(candidates[index])))
    best_threshold = float(candidates[best_index])
    return best_threshold, best_accuracy


@torch.inference_mode()
def evaluate_with_train_calibration(
    model, train_examples, valid_examples,
    device, batch_size,
):
    version_threshold, train_version_accuracy = (
        calibrate_version_threshold(
            model, train_examples,
            device, batch_size))
    metrics = evaluate(
        model, valid_examples, device,
        batch_size, version_threshold)
    metrics[
        "train_calibrated_version_link_accuracy"
    ] = train_version_accuracy
    return metrics


def metric_key(metrics):
    optional = []
    if "joint_span_exact" in metrics:
        optional.extend((
            metrics["joint_span_exact"],
            metrics["compiled_token_version_accuracy"],
        ))
    return (
        min(
            metrics["entity_pair_balanced"],
            metrics["predicate_pair_balanced"],
            metrics["entity_attention_hit_accuracy"],
            metrics["predicate_attention_hit_accuracy"],
            metrics["joint_key_nearest"],
            metrics["operation_macro_accuracy"],
            metrics["version_link_accuracy"],
            *optional,
        ),
        metrics["version_link_accuracy"],
        -metrics["time_position_mae"],
    )


def train(
    model, train_examples, valid_examples,
    device, steps, batch_size,
    learning_rate, weight_decay,
    eval_every, patience, seed,
    hard_negative_weight,
    same_world_fraction,
    span_attention_weight,
    span_boundary_weight,
    span_segment_weight,
):
    operation_counts = collections.Counter(
        example["operation"]
        for example in train_examples)
    operation_weight = torch.tensor([
        len(train_examples)
        / max(2 * operation_counts[index], 1)
        for index in (0, 1)
    ], device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate,
        weight_decay=weight_decay)
    best = evaluate_with_train_calibration(
        model, train_examples, valid_examples,
        device, batch_size)
    best_state = clone_state_dict(model)
    best_step = 0
    stale = 0
    rng = random.Random(seed)
    by_world = collections.defaultdict(list)
    for index, example in enumerate(train_examples):
        by_world[example["world_id"]].append(index)
    worlds = list(by_world)
    print(json.dumps({
        "phase": "typed_writer_baseline",
        **best,
    }, separators=(",", ":")), flush=True)
    for step in range(1, steps + 1):
        same_world_count = min(
            batch_size,
            int(round(
                batch_size * same_world_fraction)))
        world = worlds[rng.randrange(len(worlds))]
        selected = [
            by_world[world][
                rng.randrange(len(by_world[world]))]
            for _ in range(same_world_count)]
        selected.extend(
            rng.randrange(len(train_examples))
            for _ in range(
                batch_size - same_world_count))
        rng.shuffle(selected)
        batch = collate_examples(
            train_examples, selected, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["hidden"], batch["mask"],
            batch["identity_hidden"])
        loss, parts = writer_loss(
            model, output, batch, operation_weight,
            hard_negative_weight,
            span_attention_weight,
            span_boundary_weight,
            span_segment_weight)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0)
        optimizer.step()
        if step % eval_every and step != steps:
            continue
        metrics = evaluate_with_train_calibration(
            model, train_examples, valid_examples,
            device, batch_size)
        print(json.dumps({
            "phase": "typed_writer_valid",
            "step": step,
            "loss": float(loss.detach()),
            "gradient_norm": float(gradient_norm),
            "loss_parts": {
                key: float(value)
                for key, value in parts.items()
            },
            **metrics,
        }, separators=(",", ":")), flush=True)
        if metric_key(metrics) > metric_key(best):
            best = metrics
            best_state = clone_state_dict(model)
            best_step = step
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return best, best_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_checkpoint")
    parser.add_argument("train_features")
    parser.add_argument("train_jsonl")
    parser.add_argument("valid_features")
    parser.add_argument("valid_jsonl")
    parser.add_argument("output")
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--lib")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--hard-negative-weight",
        type=float, default=1.0)
    parser.add_argument(
        "--same-world-fraction",
        type=float, default=0.5)
    parser.add_argument(
        "--span-attention-weight",
        type=float, default=1.0)
    parser.add_argument(
        "--separate-address-localizer",
        action="store_true")
    parser.add_argument(
        "--predict-span-boundaries",
        action="store_true")
    parser.add_argument(
        "--span-boundary-weight",
        type=float, default=0.0)
    parser.add_argument(
        "--use-token-embeddings",
        action="store_true")
    parser.add_argument(
        "--hard-address-pooling",
        action="store_true")
    parser.add_argument(
        "--contiguous-span-pooling",
        action="store_true")
    parser.add_argument(
        "--max-field-span",
        type=int, default=8)
    parser.add_argument(
        "--span-segment-weight",
        type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    if args.hard_negative_weight < 0.0:
        parser.error("--hard-negative-weight must be non-negative")
    if not 0.0 <= args.same_world_fraction <= 1.0:
        parser.error("--same-world-fraction must be in [0, 1]")
    if args.span_attention_weight < 0.0:
        parser.error(
            "--span-attention-weight must be non-negative")
    if args.span_boundary_weight < 0.0:
        parser.error(
            "--span-boundary-weight must be non-negative")
    if args.span_segment_weight < 0.0:
        parser.error(
            "--span-segment-weight must be non-negative")
    if (
        args.span_boundary_weight > 0.0
        and not args.predict_span_boundaries
    ):
        parser.error(
            "--span-boundary-weight requires "
            "--predict-span-boundaries")
    if (
        args.span_segment_weight > 0.0
        and not args.contiguous_span_pooling
    ):
        parser.error(
            "--span-segment-weight requires "
            "--contiguous-span-pooling")
    if (
        args.predict_span_boundaries
        and args.contiguous_span_pooling
    ):
        parser.error(
            "boundary and contiguous span heads "
            "are mutually exclusive")
    if (
        args.hard_address_pooling
        and args.contiguous_span_pooling
    ):
        parser.error(
            "hard and contiguous pooling "
            "are mutually exclusive")
    if args.max_field_span < 1:
        parser.error("--max-field-span must be positive")

    checkpoint = torch.load(
        args.base_checkpoint,
        map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "MULTISLOT_ACTIVATOR_V1":
        raise ValueError("bad base activator checkpoint")
    if (
        args.use_token_embeddings
        and not args.lib
    ):
        parser.error(
            "--use-token-embeddings requires --lib")
    if args.use_token_embeddings:
        token_embedding_table = (
            load_token_embedding_table(
                args.gguf, args.lib,
                checkpoint["backbone_sha256"],
                checkpoint["hidden"]))
    else:
        if (
            file_fingerprint(args.gguf)
            != checkpoint["backbone_sha256"]
        ):
            raise ValueError(
                "GGUF backbone identity mismatch")
        token_embedding_table = None
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    train_examples = load_writer_examples(
        args.train_features, args.train_jsonl,
        checkpoint["backbone_sha256"], tokenizer,
        token_embedding_table)
    valid_examples = load_writer_examples(
        args.valid_features, args.valid_jsonl,
        checkpoint["backbone_sha256"], tokenizer,
        token_embedding_table)
    model = TypedMemoryWriter(
        checkpoint["hidden"], args.rank,
        len(checkpoint["hidden_layer_bands"]),
        args.separate_address_localizer,
        args.predict_span_boundaries,
        args.use_token_embeddings,
        args.hard_address_pooling,
        args.contiguous_span_pooling,
        args.max_field_span).to(args.device)
    if args.init_checkpoint:
        initial = torch.load(
            args.init_checkpoint,
            map_location="cpu", weights_only=True)
        expected = {
            "format": "TYPED_MEMORY_WRITER_V1",
            "backbone_sha256":
                checkpoint["backbone_sha256"],
            "hidden": checkpoint["hidden"],
            "rank": args.rank,
            "hidden_layer_bands":
                checkpoint["hidden_layer_bands"],
        }
        for key, value in expected.items():
            if initial.get(key) != value:
                raise ValueError(
                    f"typed writer init mismatch: {key}")
        if bool(initial.get(
            "separate_address_localizer", False
        )) != args.separate_address_localizer:
            raise ValueError(
                "typed writer init mismatch: "
                "separate_address_localizer")
        if bool(initial.get(
            "predict_span_boundaries", False
        )) != args.predict_span_boundaries:
            raise ValueError(
                "typed writer init mismatch: "
                "predict_span_boundaries")
        if bool(initial.get(
            "use_token_embeddings", False
        )) != args.use_token_embeddings:
            raise ValueError(
                "typed writer init mismatch: "
                "use_token_embeddings")
        if bool(initial.get(
            "hard_address_pooling", False
        )) != args.hard_address_pooling:
            raise ValueError(
                "typed writer init mismatch: "
                "hard_address_pooling")
        if bool(initial.get(
            "contiguous_span_pooling", False
        )) != args.contiguous_span_pooling:
            raise ValueError(
                "typed writer init mismatch: "
                "contiguous_span_pooling")
        if (
            int(initial.get(
                "max_field_span",
                args.max_field_span))
            != args.max_field_span
        ):
            raise ValueError(
                "typed writer init mismatch: "
                "max_field_span")
        initial_state = dict(initial["state_dict"])
        old_boundary = initial_state.get(
            "span_boundary_keys")
        if (
            args.predict_span_boundaries
            and old_boundary is not None
            and tuple(old_boundary.shape)
            == (2, 2, args.rank)
            and tuple(model.span_boundary_keys.shape)
            == (len(SLOT_NAMES), 2, args.rank)
        ):
            initial_state.pop("span_boundary_keys")
            missing, unexpected = model.load_state_dict(
                initial_state, strict=False)
            if (
                set(missing) != {"span_boundary_keys"}
                or unexpected
            ):
                raise ValueError(
                    "typed writer legacy boundary "
                    f"init mismatch: {missing} {unexpected}"
                )
            with torch.no_grad():
                model.span_boundary_keys[:2].copy_(
                    old_boundary)
            print(json.dumps({
                "phase":
                    "typed_writer_expand_boundary_slots",
                "copied_fields": ["entity", "predicate"],
                "initialized_fields": ["value", "time"],
            }, separators=(",", ":")), flush=True)
        else:
            model.load_state_dict(
                initial_state, strict=True)
        print(json.dumps({
            "phase": "typed_writer_init",
            "checkpoint": args.init_checkpoint,
            "previous_step":
                initial.get("selected_step"),
            "previous_metrics":
                initial.get("valid_metrics", {}),
        }, separators=(",", ":")), flush=True)
    _, selected_step = train(
        model, train_examples, valid_examples,
        args.device, args.steps, args.batch,
        args.learning_rate, args.weight_decay,
        args.eval_every, args.patience,
        args.seed, args.hard_negative_weight,
        args.same_world_fraction,
        args.span_attention_weight,
        args.span_boundary_weight,
        args.span_segment_weight)
    version_threshold, train_version_accuracy = (
        calibrate_version_threshold(
            model, train_examples,
            args.device, args.batch))
    uncalibrated_metrics = evaluate(
        model, valid_examples, args.device,
        args.batch)
    calibrated_metrics = evaluate(
        model, valid_examples, args.device,
        args.batch, version_threshold)
    calibrated_metrics[
        "uncalibrated_version_link_accuracy"
    ] = uncalibrated_metrics[
        "version_link_accuracy"]
    calibrated_metrics[
        "train_calibrated_version_link_accuracy"
    ] = train_version_accuracy
    metrics = calibrated_metrics
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "TYPED_MEMORY_WRITER_V1",
        "backbone_sha256":
            checkpoint["backbone_sha256"],
        "hidden": checkpoint["hidden"],
        "rank": args.rank,
        "hidden_layer_bands":
            checkpoint["hidden_layer_bands"],
        "state_dict": clone_state_dict(model),
        "selected_step": selected_step,
        "hard_negative_weight":
            args.hard_negative_weight,
        "same_world_fraction":
            args.same_world_fraction,
        "span_attention_weight":
            args.span_attention_weight,
        "token_span_supervision": True,
        "separate_address_localizer":
            args.separate_address_localizer,
        "predict_span_boundaries":
            args.predict_span_boundaries,
        "span_boundary_weight":
            args.span_boundary_weight,
        "use_token_embeddings":
            args.use_token_embeddings,
        "hard_address_pooling":
            args.hard_address_pooling,
        "contiguous_span_pooling":
            args.contiguous_span_pooling,
        "max_field_span":
            args.max_field_span,
        "span_segment_weight":
            args.span_segment_weight,
        "valid_metrics": metrics,
        "version_threshold": version_threshold,
        "address_mode":
            "open_set_entity_predicate_pair_threshold",
        "version_mode":
            "immutable_events_latest_matching_previous_link",
        "runtime_status": "python_proof_only",
        "kv_cache_used": False,
        "lora_used": False,
        "rag_used": False,
        "locomo_used": False,
    }, output)
    print(json.dumps({
        "phase": "typed_writer_done",
        "output": str(output),
        "selected_step": selected_step,
        **metrics,
        "runtime_status": "python_proof_only",
        "kv_cache_used": False,
        "lora_used": False,
        "rag_used": False,
        "locomo_used": False,
    }, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
