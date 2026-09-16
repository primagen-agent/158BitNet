#!/usr/bin/env python3
"""Train token-level pair verification for typed memory addresses."""
from __future__ import annotations

import argparse
import collections
import json
import random
import struct
import zlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from c_tokenizer import CTokenizer
from typed_memory_training import clone_state_dict
from train_typed_memory_writer import (
    SLOT_NAMES,
    TypedMemoryWriter,
    encode_all,
    load_token_embedding_table,
    load_writer_examples,
)

SET_LINK_HEAD_VERSION = 2

TYPED_LINK_TENSORS = (
    "joint_head.0.weight",
    "joint_head.0.bias",
    "joint_head.2.weight",
    "joint_head.2.bias",
    "predecessor_exists_head.0.weight",
    "predecessor_exists_head.0.bias",
    "predecessor_exists_head.2.weight",
    "predecessor_exists_head.2.bias",
)


def export_typed_link_binary(
    checkpoint_path, output_path,
):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu",
        weights_only=True)
    if (
        checkpoint.get("format")
        != "TYPED_PAIR_VERIFIER_V1"
        or not checkpoint.get("set_link_head", False)
        or int(checkpoint.get(
            "set_link_head_version", 0
        )) not in (2, 3)
    ):
        raise ValueError(
            "typed-link export requires a v2 or v3 set-link checkpoint")
    state = checkpoint["verifier_state_dict"]
    version = int(checkpoint["set_link_head_version"])
    exists_features = 7 if version == 3 else 5
    payloads = []
    tensors = {}
    for name in TYPED_LINK_TENSORS:
        if name not in state:
            raise ValueError(
                f"typed-link tensor missing: {name}")
        tensor = state[name].detach().float().contiguous()
        tensors[name] = tensor
        payloads.append(
            tensor.numpy().astype(
                "<f4", copy=False).tobytes())
    rank = int(checkpoint["rank"])
    joint_hidden = rank * 2
    joint_input = int(
        tensors["joint_head.0.weight"].shape[1])
    if (
        tensors["joint_head.0.weight"].shape
        != (joint_hidden, joint_input)
        or joint_input % 2
        or tensors["joint_head.0.bias"].shape
        != (joint_hidden,)
        or tensors["joint_head.2.weight"].shape
        != (1, joint_hidden)
        or tensors["joint_head.2.bias"].shape != (1,)
        or tensors[
            "predecessor_exists_head.0.weight"
        ].shape != (rank, exists_features)
        or tensors[
            "predecessor_exists_head.0.bias"
        ].shape != (rank,)
        or tensors[
            "predecessor_exists_head.2.weight"
        ].shape != (1, rank)
        or tensors[
            "predecessor_exists_head.2.bias"
        ].shape != (1,)
    ):
        raise ValueError(
            "typed-link checkpoint tensor geometry mismatch")
    header = bytearray(b"BNTLINK1")
    header += struct.pack(
        "<IIIIIII",
        1,
        version,
        rank,
        joint_input // 2,
        joint_hidden,
        exists_features,
        len(payloads),
    )
    header += bytes.fromhex(
        checkpoint["backbone_sha256"])
    for payload in payloads:
        header += struct.pack(
            "<II", len(payload),
            zlib.crc32(payload) & 0xFFFFFFFF)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(header + b"".join(payloads))
    return output


def build_active_pairs(examples):
    by_world = collections.defaultdict(list)
    for index, example in enumerate(examples):
        by_world[example["world_id"]].append(index)
    pairs = []
    for indices in by_world.values():
        indices.sort(
            key=lambda index: examples[index]["episode"])
        active = {}
        for index in indices:
            current = examples[index]
            for candidate in active.values():
                previous = examples[candidate]
                entity_same = (
                    current["entity"] == previous["entity"])
                predicate_same = (
                    current["predicate"]
                    == previous["predicate"])
                pairs.append({
                    "left": index,
                    "right": candidate,
                    "entity_same": entity_same,
                    "predicate_same": predicate_same,
                    "joint_same":
                        entity_same and predicate_same,
                })
            active[(
                current["entity"],
                current["predicate"],
            )] = index
    if not pairs:
        raise ValueError("typed pair dataset is empty")
    return pairs


def select_calibration_pairs(pairs, limit, seed):
    if limit <= 0 or len(pairs) <= limit:
        return pairs
    buckets = collections.defaultdict(list)
    for index, pair in enumerate(pairs):
        buckets[(
            bool(pair["entity_same"]),
            bool(pair["predicate_same"]),
        )].append(index)
    keys = (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    )
    rng = random.Random(seed)
    selected = []
    leftovers = []
    base, extra = divmod(limit, len(keys))
    for offset, key in enumerate(keys):
        indices = buckets[key]
        rng.shuffle(indices)
        quota = base + int(offset < extra)
        selected.extend(indices[:quota])
        leftovers.extend(indices[quota:])
    if len(selected) < limit:
        rng.shuffle(leftovers)
        selected.extend(
            leftovers[:limit - len(selected)])
    return [pairs[index] for index in selected]


def build_predecessor_groups(examples, pairs):
    by_left = collections.defaultdict(list)
    for pair_index, pair in enumerate(pairs):
        by_left[pair["left"]].append(pair_index)
    groups = []
    for left, pair_indices in by_left.items():
        example = examples[left]
        if example["operation"] != 1:
            continue
        targets = [
            position
            for position, pair_index in enumerate(pair_indices)
            if examples[pairs[pair_index]["right"]]["episode"]
            == example["previous_episode"]
        ]
        if len(targets) != 1:
            raise ValueError(
                "update has no unique active predecessor")
        if len(pair_indices) > 1:
            groups.append({
                "pairs": pair_indices,
                "target": targets[0],
            })
    if not groups:
        raise ValueError("predecessor ranking dataset is empty")
    return groups


def build_layout_consistency_groups(examples):
    grouped = collections.defaultdict(list)
    for index, example in enumerate(examples):
        key = (
            example["semantic_world_id"],
            example["episode"],
        )
        grouped[key].append(index)
    groups = []
    for indices in grouped.values():
        by_view = {
            examples[index]["layout_view"]: index
            for index in indices
        }
        if len(by_view) >= 2:
            groups.append([
                by_view[view]
                for view in sorted(by_view)[:2]
            ])
    return groups


def collate_pairs(examples, pairs, selected, device):
    left_max = max(
        examples[pairs[index]["left"]]["hidden"].shape[0]
        for index in selected)
    right_max = max(
        examples[pairs[index]["right"]]["hidden"].shape[0]
        for index in selected)
    feature_shape = tuple(
        examples[pairs[selected[0]]["left"]][
            "hidden"].shape[1:])
    left = torch.zeros(
        len(selected), left_max, *feature_shape,
        device=device, dtype=torch.float32)
    right = torch.zeros(
        len(selected), right_max, *feature_shape,
        device=device, dtype=torch.float32)
    left_identity = None
    right_identity = None
    if "identity_hidden" in examples[
        pairs[selected[0]]["left"]
    ]:
        identity_size = examples[
            pairs[selected[0]]["left"]
        ]["identity_hidden"].shape[-1]
        left_identity = torch.zeros(
            len(selected), left_max, identity_size,
            device=device, dtype=torch.float32)
        right_identity = torch.zeros(
            len(selected), right_max, identity_size,
            device=device, dtype=torch.float32)
    left_mask = torch.zeros(
        len(selected), left_max,
        device=device, dtype=torch.bool)
    right_mask = torch.zeros(
        len(selected), right_max,
        device=device, dtype=torch.bool)
    entity = []
    predicate = []
    joint = []
    rows = []
    for batch_index, pair_index in enumerate(selected):
        pair = pairs[pair_index]
        left_value = examples[pair["left"]]["hidden"]
        right_value = examples[pair["right"]]["hidden"]
        left_count = left_value.shape[0]
        right_count = right_value.shape[0]
        left[batch_index, :left_count] = left_value.to(
            device=device, dtype=torch.float32)
        right[batch_index, :right_count] = right_value.to(
            device=device, dtype=torch.float32)
        if left_identity is not None:
            left_identity[
                batch_index, :left_count
            ] = examples[pair["left"]][
                "identity_hidden"].to(
                    device=device,
                    dtype=torch.float32)
            right_identity[
                batch_index, :right_count
            ] = examples[pair["right"]][
                "identity_hidden"].to(
                    device=device,
                    dtype=torch.float32)
        left_mask[batch_index, :left_count] = True
        right_mask[batch_index, :right_count] = True
        entity.append(float(pair["entity_same"]))
        predicate.append(float(pair["predicate_same"]))
        joint.append(float(pair["joint_same"]))
        rows.append(pair)
    return {
        "left": left,
        "right": right,
        "left_identity": left_identity,
        "right_identity": right_identity,
        "left_mask": left_mask,
        "right_mask": right_mask,
        "entity_target": torch.tensor(
            entity, device=device),
        "predicate_target": torch.tensor(
            predicate, device=device),
        "joint_target": torch.tensor(
            joint, device=device),
        "rows": rows,
    }


def collate_event_states(examples, selected, device):
    maximum = max(
        examples[index]["hidden"].shape[0]
        for index in selected)
    feature_shape = tuple(
        examples[selected[0]]["hidden"].shape[1:])
    hidden = torch.zeros(
        len(selected), maximum, *feature_shape,
        device=device, dtype=torch.float32)
    identity = torch.zeros(
        len(selected), maximum,
        examples[selected[0]][
            "identity_hidden"].shape[-1],
        device=device, dtype=torch.float32)
    mask = torch.zeros(
        len(selected), maximum,
        device=device, dtype=torch.bool)
    for batch_index, index in enumerate(selected):
        example = examples[index]
        count = example["hidden"].shape[0]
        hidden[batch_index, :count] = (
            example["hidden"].to(
                device=device,
                dtype=torch.float32))
        identity[batch_index, :count] = (
            example["identity_hidden"].to(
                device=device,
                dtype=torch.float32))
        mask[batch_index, :count] = True
    return {
        "hidden": hidden,
        "identity_hidden": identity,
        "mask": mask,
    }


class TypedPairVerifier(nn.Module):
    """Frozen typed writer plus token-level pair interaction."""

    def __init__(
        self, writer, rank,
        attention_weighted_alignment=False,
        dual_path=False,
        trainable_context_localizer=False,
        set_link_head=False,
        exists_feature_count=5,
    ):
        super().__init__()
        self.writer = writer
        self.writer.requires_grad_(False)
        self.writer.eval()
        self.rank = int(rank)
        self.attention_weighted_alignment = bool(
            attention_weighted_alignment)
        self.dual_path = bool(dual_path)
        self.trainable_context_localizer = bool(
            trainable_context_localizer)
        self.set_link_head = bool(set_link_head)
        if exists_feature_count not in (5, 7):
            raise ValueError("unsupported predecessor feature geometry")
        self.exists_feature_count = exists_feature_count
        if self.dual_path and not writer.use_token_embeddings:
            raise ValueError(
                "dual path requires tied token embeddings")
        if (
            self.trainable_context_localizer
            and not self.dual_path
        ):
            raise ValueError(
                "trainable context localizer requires dual path")
        self.band_logits = nn.Parameter(torch.stack([
            writer.band_logits[
                SLOT_NAMES.index("entity")].detach().clone(),
            writer.band_logits[
                SLOT_NAMES.index("predicate")].detach().clone(),
        ]))
        self.projections = nn.ModuleDict({
            name: nn.Linear(
                writer.hidden, rank, bias=False)
            for name in ("entity", "predicate")
        })
        with torch.no_grad():
            for name in self.projections:
                self.projections[name].weight.copy_(
                    writer.projections[name].weight)
        if self.dual_path:
            self.identity_projections = nn.ModuleDict({
                name: nn.Linear(
                    writer.hidden, rank, bias=False)
                for name in ("entity", "predicate")
            })
            with torch.no_grad():
                for name in self.identity_projections:
                    self.identity_projections[
                        name].weight.copy_(
                            writer.projections[name].weight)
        else:
            self.identity_projections = nn.ModuleDict()
        if writer.separate_address_localizer:
            self.register_parameter("token_keys", None)
        else:
            self.token_keys = nn.Parameter(torch.stack([
                writer.token_keys[
                    SLOT_NAMES.index("entity")].detach().clone(),
                writer.token_keys[
                    SLOT_NAMES.index("predicate")].detach().clone(),
            ]))
        if self.trainable_context_localizer:
            self.context_localizer_band_logits = nn.Parameter(
                writer.localizer_band_logits.detach().clone())
            self.context_localizer_projections = nn.ModuleDict({
                name: nn.Linear(
                    writer.hidden, rank, bias=False)
                for name in ("entity", "predicate")
            })
            with torch.no_grad():
                for name in self.context_localizer_projections:
                    self.context_localizer_projections[
                        name].weight.copy_(
                            writer.localizer_projections[
                                name].weight)
            self.context_token_keys = nn.Parameter(
                torch.stack([
                    writer.token_keys[
                        SLOT_NAMES.index("entity")
                    ].detach().clone(),
                    writer.token_keys[
                        SLOT_NAMES.index("predicate")
                    ].detach().clone(),
                ]))
        else:
            self.register_parameter(
                "context_localizer_band_logits", None)
            self.context_localizer_projections = nn.ModuleDict()
            self.register_parameter(
                "context_token_keys", None)
        pair_width = rank * 2 + 4
        if self.dual_path:
            self.fusion_gates = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(pair_width * 2, rank),
                    nn.GELU(),
                    nn.Linear(rank, 1),
                )
                for name in ("entity", "predicate")
            })
            head_width = pair_width + 1
        else:
            self.fusion_gates = nn.ModuleDict()
            head_width = pair_width
        self.entity_head = nn.Sequential(
            nn.Linear(head_width, rank * 2),
            nn.GELU(),
            nn.Linear(rank * 2, 1),
        )
        self.predicate_head = nn.Sequential(
            nn.Linear(head_width, rank * 2),
            nn.GELU(),
            nn.Linear(rank * 2, 1),
        )
        if self.set_link_head:
            self.joint_head = nn.Sequential(
                nn.Linear(head_width * 2, rank * 2),
                nn.GELU(),
                nn.Linear(rank * 2, 1),
            )
            nn.init.zeros_(
                self.joint_head[-1].weight)
            nn.init.zeros_(
                self.joint_head[-1].bias)
            self.predecessor_exists_head = nn.Sequential(
                nn.Linear(exists_feature_count, rank),
                nn.GELU(),
                nn.Linear(rank, 1),
            )
        else:
            self.joint_head = None
            self.predecessor_exists_head = None

    def typed_tokens(
        self, hidden, identity_hidden,
        slot_index, name, path,
    ):
        if path == "identity":
            if identity_hidden is None:
                raise ValueError(
                    "pair token embedding input is missing")
            mixed = identity_hidden
            projection = (
                self.identity_projections[name]
                if self.dual_path
                else self.projections[name]
            )
        elif path == "contextual":
            weight = F.softmax(
                self.band_logits[slot_index], dim=0)
            mixed = (
                hidden
                * weight.view(
                    1, 1,
                    self.writer.band_count, 1)
            ).sum(dim=2)
            projection = self.projections[name]
        else:
            raise ValueError(f"unknown pair path: {path}")
        return F.normalize(
            projection(mixed),
            dim=-1, eps=1e-12)

    def pool_tokens(
        self, tokens, hidden, mask,
        slot_index, name, path,
    ):
        if (
            path == "contextual"
            and self.trainable_context_localizer
        ):
            localizer_index = (
                0 if name == "entity" else 1)
            weight = F.softmax(
                self.context_localizer_band_logits[
                    localizer_index],
                dim=0)
            mixed = (
                hidden
                * weight.view(
                    1, 1,
                    self.writer.band_count, 1)
            ).sum(dim=2)
            score_tokens = F.normalize(
                self.context_localizer_projections[
                    name](mixed),
                dim=-1, eps=1e-12)
            token_key = self.context_token_keys[
                localizer_index]
        elif self.writer.separate_address_localizer:
            localizer_index = (
                0 if name == "entity" else 1)
            weight = F.softmax(
                self.writer.localizer_band_logits[
                    localizer_index],
                dim=0)
            mixed = (
                hidden
                * weight.view(
                    1, 1,
                    self.writer.band_count, 1)
            ).sum(dim=2)
            score_tokens = F.normalize(
                self.writer.localizer_projections[
                    name](mixed),
                dim=-1, eps=1e-12)
            token_key = self.writer.token_keys[
                slot_index]
        else:
            score_tokens = tokens
            token_key = self.token_keys[slot_index]
        score = torch.einsum(
            "btr,r->bt", score_tokens, token_key)
        score = score.masked_fill(~mask, -1e9)
        attention = F.softmax(score, dim=-1)
        state = F.normalize(
            torch.einsum(
                "bt,btr->br", attention, tokens),
            dim=-1, eps=1e-12)
        return state, attention

    @staticmethod
    def alignment_features(
        left, right, left_mask, right_mask,
        left_weight=None, right_weight=None,
    ):
        similarity = torch.einsum(
            "btr,bsr->bts", left, right)
        valid = (
            left_mask[:, :, None]
            & right_mask[:, None, :])
        similarity = similarity.masked_fill(
            ~valid, -1e9)
        left_best = similarity.max(dim=2).values
        right_best = similarity.max(dim=1).values
        left_mean = (
            (left_best * left_mask).sum(dim=1)
            / left_mask.sum(dim=1).clamp_min(1))
        right_mean = (
            (right_best * right_mask).sum(dim=1)
            / right_mask.sum(dim=1).clamp_min(1))
        left_max = left_best.masked_fill(
            ~left_mask, -1e9).max(dim=1).values
        right_max = right_best.masked_fill(
            ~right_mask, -1e9).max(dim=1).values
        if left_weight is not None:
            if right_weight is None:
                raise ValueError(
                    "both alignment weights are required")
            left_mean = (
                left_best * left_weight
            ).sum(dim=1)
            right_mean = (
                right_best * right_weight
            ).sum(dim=1)
            weighted_mean = torch.einsum(
                "bt,bts,bs->b",
                left_weight, similarity,
                right_weight)
            weighted_score = (
                similarity
                + left_weight.clamp_min(
                    1e-9).log()[:, :, None]
                + right_weight.clamp_min(
                    1e-9).log()[:, None, :]
            ).masked_fill(~valid, -1e9)
            weighted_max = weighted_score.flatten(
                1).max(dim=1).values
            return torch.stack((
                left_mean, right_mean,
                weighted_mean, weighted_max,
            ), dim=-1)
        return torch.stack((
            left_mean, right_mean,
            left_max, right_max,
        ), dim=-1)

    def path_features(
        self, left_hidden, right_hidden,
        left_identity, right_identity,
        left_mask, right_mask,
        slot_index, name, path,
    ):
        left_tokens = self.typed_tokens(
            left_hidden, left_identity,
            slot_index, name, path)
        right_tokens = self.typed_tokens(
            right_hidden, right_identity,
            slot_index, name, path)
        left_state, left_attention = self.pool_tokens(
            left_tokens, left_hidden,
            left_mask, slot_index, name, path)
        right_state, right_attention = self.pool_tokens(
            right_tokens, right_hidden,
            right_mask, slot_index, name, path)
        alignment = self.alignment_features(
            left_tokens, right_tokens,
            left_mask, right_mask,
            (
                left_attention
                if self.attention_weighted_alignment
                else None
            ),
            (
                right_attention
                if self.attention_weighted_alignment
                else None
            ))
        return torch.cat((
            left_state * right_state,
            (left_state - right_state).abs(),
            alignment,
        ), dim=-1)

    def event_state(
        self, hidden, identity_hidden,
        mask, slot_index, name, path,
    ):
        tokens = self.typed_tokens(
            hidden, identity_hidden,
            slot_index, name, path)
        state, attention = self.pool_tokens(
            tokens, hidden, mask,
            slot_index, name, path)
        return state, attention

    def pair_features(
        self, left_hidden, right_hidden,
        left_identity, right_identity,
        left_mask, right_mask,
        slot_index, name,
    ):
        if not self.dual_path:
            feature = self.path_features(
                left_hidden, right_hidden,
                left_identity, right_identity,
                left_mask, right_mask,
                slot_index, name,
                (
                    "identity"
                    if self.writer.use_token_embeddings
                    else "contextual"
                ))
            return feature, {"fused": feature}
        contextual = self.path_features(
            left_hidden, right_hidden,
            left_identity, right_identity,
            left_mask, right_mask,
            slot_index, name, "contextual")
        identity = self.path_features(
            left_hidden, right_hidden,
            left_identity, right_identity,
            left_mask, right_mask,
            slot_index, name, "identity")
        gate = torch.sigmoid(
            self.fusion_gates[name](
                torch.cat(
                    (contextual, identity),
                    dim=-1)))
        fused = (
            gate * contextual
            + (1.0 - gate) * identity)
        return torch.cat((fused, gate), dim=-1), {
            "contextual": contextual,
            "identity": identity,
            "gate": gate,
            "fused": fused,
        }

    def forward(
        self, left, right, left_mask, right_mask,
        left_identity=None, right_identity=None,
        return_internal=False,
    ):
        entity, entity_internal = self.pair_features(
            left, right,
            left_identity, right_identity,
            left_mask, right_mask,
            SLOT_NAMES.index("entity"), "entity")
        predicate, predicate_internal = self.pair_features(
            left, right,
            left_identity, right_identity,
            left_mask, right_mask,
            SLOT_NAMES.index("predicate"), "predicate")
        entity_logit = self.entity_head(
            entity).squeeze(-1)
        predicate_logit = self.predicate_head(
            predicate).squeeze(-1)
        result = {
            "entity_logit": entity_logit,
            "predicate_logit": predicate_logit,
        }
        if self.set_link_head:
            residual = self.joint_head(
                torch.cat((entity, predicate), dim=-1)
            ).squeeze(-1)
            result["joint_logit"] = (
                torch.minimum(
                    entity_logit, predicate_logit)
                + residual)
        if return_internal:
            result["internals"] = {
                "entity": entity_internal,
                "predicate": predicate_internal,
                "pair_features": {
                    "entity": entity,
                    "predicate": predicate,
                },
            }
        return result

    @staticmethod
    def set_link_features(scores, feature_count=5):
        if scores.numel() < 1:
            raise ValueError(
                "set-link features need at least one candidate")
        top = torch.topk(
            scores, min(2, scores.numel())).values
        top1 = top[0]
        top2 = top[1] if top.numel() > 1 else top1
        mean = scores.mean()
        std = scores.std(
            unbiased=False).clamp_min(1e-4)
        normalized = (scores - mean) / std
        probability = F.softmax(
            normalized, dim=0)
        entropy = -(
            probability
            * probability.clamp_min(1e-9).log()
        ).sum()
        if scores.numel() > 1:
            entropy = entropy / scores.new_tensor(
                float(scores.numel())).log()
        else:
            entropy = entropy.new_zeros(())
        features = torch.stack((
            (top1 - mean) / std,
            (top1 - top2) / std,
            probability.max(),
            1.0 - entropy,
            scores.new_tensor(
                float(scores.numel())).log1p(),
        ))
        if feature_count == 7:
            return torch.cat((features, torch.stack((top1, top2))))
        if feature_count != 5:
            raise ValueError("unsupported predecessor feature geometry")
        return features

    def predecessor_exists_logits(self, score_groups):
        if not self.set_link_head:
            raise ValueError(
                "predecessor existence needs set-link head")
        features = torch.stack([
            self.set_link_features(scores, self.exists_feature_count)
            for scores in score_groups
        ])
        return self.predecessor_exists_head(
            features).squeeze(-1)


def joint_pair_scores(output):
    if "joint_logit" in output:
        return output["joint_logit"]
    return torch.minimum(
        output["entity_logit"],
        output["predicate_logit"])


def balanced_binary_loss(
    logits, targets, positive_weight=0.5,
):
    positive = targets > 0.5
    negative = ~positive
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError(
            "pair batch needs both positive and negative labels")
    return (
        positive_weight
        * F.softplus(-logits[positive]).mean()
        + (1.0 - positive_weight)
        * F.softplus(logits[negative]).mean())


def verifier_loss(
    output, batch, positive_weight=0.5,
):
    entity = balanced_binary_loss(
        output["entity_logit"],
        batch["entity_target"],
        positive_weight)
    predicate = balanced_binary_loss(
        output["predicate_logit"],
        batch["predicate_target"],
        positive_weight)
    joint_logit = joint_pair_scores(output)
    joint = balanced_binary_loss(
        joint_logit, batch["joint_target"],
        positive_weight)
    return entity + predicate + joint, {
        "entity": entity.detach(),
        "predicate": predicate.detach(),
        "joint": joint.detach(),
    }


def grouped_ranking_loss(scores, lengths, targets):
    if len(lengths) != len(targets):
        raise ValueError("ranking group target mismatch")
    losses = []
    offset = 0
    for length, target in zip(lengths, targets):
        if length < 2:
            raise ValueError(
                "ranking group needs at least two candidates")
        if not 0 <= target < length:
            raise ValueError("ranking target is out of range")
        group_scores = scores[offset:offset + length]
        if group_scores.numel() != length:
            raise ValueError("ranking score count mismatch")
        losses.append(F.cross_entropy(
            group_scores.unsqueeze(0),
            torch.tensor(
                [target], device=scores.device)))
        offset += length
    if offset != scores.numel():
        raise ValueError("unused ranking scores")
    return torch.stack(losses).mean()


def predecessor_ranking_loss(
    model, examples, pairs, groups,
    selected, device,
):
    chosen = [groups[index] for index in selected]
    pair_indices = [
        pair_index
        for group in chosen
        for pair_index in group["pairs"]
    ]
    batch = collate_pairs(
        examples, pairs, pair_indices, device)
    output = model(
        batch["left"], batch["right"],
        batch["left_mask"], batch["right_mask"],
        batch["left_identity"],
        batch["right_identity"])
    scores = joint_pair_scores(output)
    return grouped_ranking_loss(
        scores,
        [len(group["pairs"]) for group in chosen],
        [group["target"] for group in chosen])


def predecessor_set_losses(
    model, examples, pairs, groups,
    selected, device,
):
    if not model.set_link_head:
        raise ValueError(
            "predecessor set loss needs set-link head")
    chosen = [groups[index] for index in selected]
    pair_indices = [
        pair_index
        for group in chosen
        for pair_index in group["pairs"]
    ]
    batch = collate_pairs(
        examples, pairs, pair_indices, device)
    output = model(
        batch["left"], batch["right"],
        batch["left_mask"], batch["right_mask"],
        batch["left_identity"],
        batch["right_identity"])
    scores = joint_pair_scores(output)
    lengths = [
        len(group["pairs"]) for group in chosen]
    targets = [group["target"] for group in chosen]
    ranking = grouped_ranking_loss(
        scores, lengths, targets)
    positive_groups = []
    negative_groups = []
    offset = 0
    for length, target in zip(lengths, targets):
        group_scores = scores[offset:offset + length]
        positive_groups.append(group_scores)
        keep = torch.ones(
            length, device=scores.device,
            dtype=torch.bool)
        keep[target] = False
        if bool(keep.any()):
            negative_groups.append(group_scores[keep])
        offset += length
    positive_logits = model.predecessor_exists_logits(
        positive_groups)
    negative_logits = model.predecessor_exists_logits(
        negative_groups)
    existence = 0.5 * (
        F.softplus(-positive_logits).mean()
        + F.softplus(negative_logits).mean())
    return ranking, existence


def layout_consistency_loss(
    model, examples, groups,
    selected, device,
):
    example_indices = [
        example_index
        for group_index in selected
        for example_index in groups[group_index]
    ]
    batch = collate_event_states(
        examples, example_indices, device)
    losses = []
    for slot_index, name in enumerate(
        ("entity", "predicate")
    ):
        contextual, _ = model.event_state(
            batch["hidden"],
            batch["identity_hidden"],
            batch["mask"],
            slot_index, name, "contextual")
        contextual = contextual.reshape(
            len(selected), 2, -1)
        losses.append(
            (
                1.0 - F.cosine_similarity(
                    contextual[:, 0],
                    contextual[:, 1],
                    dim=-1)
            ).mean())
    return torch.stack(losses).mean()


@torch.inference_mode()
def score_pairs(
    model, examples, pairs, device, batch_size,
):
    entity = []
    predicate = []
    joint = []
    model.eval()
    for offset in range(0, len(pairs), batch_size):
        selected = list(range(
            offset, min(offset + batch_size, len(pairs))))
        batch = collate_pairs(
            examples, pairs, selected, device)
        output = model(
            batch["left"], batch["right"],
            batch["left_mask"], batch["right_mask"],
            batch["left_identity"],
            batch["right_identity"])
        entity.append(output["entity_logit"].cpu())
        predicate.append(
            output["predicate_logit"].cpu())
        if "joint_logit" in output:
            joint.append(output["joint_logit"].cpu())
    return (
        torch.cat(entity),
        torch.cat(predicate),
        torch.cat(joint) if joint else None,
    )


def balanced_accuracy(
    logits, targets, threshold=0.0,
):
    positive = targets > 0.5
    negative = ~positive
    positive_accuracy = float(
        (logits[positive] > threshold).float().mean())
    negative_accuracy = float(
        (logits[negative] <= threshold).float().mean())
    return (
        0.5 * (positive_accuracy + negative_accuracy),
        positive_accuracy,
        negative_accuracy,
    )


def pair_targets(pairs, field):
    return torch.tensor([
        float(pair[field]) for pair in pairs])


def operation_logits(
    writer, examples, device, batch_size,
):
    return encode_all(
        writer, examples, device,
        batch_size)["operation_logits"]


@torch.inference_mode()
def score_predecessor_existence(
    model, examples, pairs, joint_logits, device,
):
    if not model.set_link_head:
        return None, None
    by_left = collections.defaultdict(list)
    for pair_index, pair in enumerate(pairs):
        by_left[pair["left"]].append(pair_index)
    positive_left = []
    positive_groups = []
    negative_left = []
    negative_groups = []
    missing = {}
    for left, pair_indices in by_left.items():
        example = examples[left]
        if example["operation"] != 1:
            continue
        targets = [
            position
            for position, pair_index
            in enumerate(pair_indices)
            if examples[
                pairs[pair_index]["right"]
            ]["episode"] == example["previous_episode"]
        ]
        if len(targets) != 1:
            raise ValueError(
                "update has no unique active predecessor")
        scores = joint_logits[pair_indices].to(device)
        positive_left.append(left)
        positive_groups.append(scores)
        keep = torch.ones(
            scores.numel(), device=device,
            dtype=torch.bool)
        keep[targets[0]] = False
        if bool(keep.any()):
            negative_left.append(left)
            negative_groups.append(scores[keep])
        else:
            missing[left] = -float("inf")
    positive_values = model.predecessor_exists_logits(
        positive_groups).detach().cpu().tolist()
    positive = dict(zip(
        positive_left, positive_values))
    if negative_groups:
        negative_values = (
            model.predecessor_exists_logits(
                negative_groups
            ).detach().cpu().tolist())
        missing.update(zip(
            negative_left, negative_values))
    return positive, missing


def version_link_diagnostics(
    examples, pairs, entity_logits,
    predicate_logits, operation,
    entity_threshold=0.0,
    predicate_threshold=0.0,
    joint_logits=None,
    predecessor_exists=None,
    missing_predecessor_exists=None,
    predecessor_exists_threshold=0.0,
):
    by_left = collections.defaultdict(list)
    for pair_index, pair in enumerate(pairs):
        by_left[pair["left"]].append(pair_index)
    correct = 0
    create_total = 0
    create_correct = 0
    update_total = 0
    update_operation_correct = 0
    update_selected_correct = 0
    update_no_candidate = 0
    update_wrong_candidate = 0
    predecessor_rank_correct = 0
    predecessor_accepted = 0
    missing_predecessor_rejected = 0
    missing_predecessor_total = 0
    margins = []
    for index, example in enumerate(examples):
        predicted_operation = int(
            operation[index].argmax())
        if example["operation"] == 0:
            create_total += 1
            is_correct = predicted_operation == 0
            create_correct += int(is_correct)
            correct += int(is_correct)
            continue
        update_total += 1
        update_operation_correct += int(
            predicted_operation == 1)
        scored = []
        true_pair = None
        for pair_index in by_left[index]:
            pair = pairs[pair_index]
            if joint_logits is not None:
                score = float(joint_logits[pair_index])
            else:
                score = min(
                    float(entity_logits[pair_index])
                    - entity_threshold,
                    float(predicate_logits[pair_index])
                    - predicate_threshold)
            scored.append((score, pair["right"]))
            if (
                examples[pair["right"]]["episode"]
                == example["previous_episode"]
            ):
                if true_pair is not None:
                    raise ValueError(
                        "multiple true predecessors")
                true_pair = (score, pair["right"])
        if true_pair is None:
            raise ValueError("true predecessor is missing")
        false_scores = [
            score for score, right in scored
            if right != true_pair[1]
        ]
        false_max = max(
            false_scores, default=-float("inf"))
        margins.append(true_pair[0] - false_max)
        predecessor_rank_correct += int(
            true_pair[0] > false_max)
        if predecessor_exists is not None:
            accepted = (
                predecessor_exists[index]
                > predecessor_exists_threshold)
            if missing_predecessor_exists is not None:
                missing_predecessor_total += 1
                missing_predecessor_rejected += int(
                    missing_predecessor_exists[index]
                    <= predecessor_exists_threshold)
        else:
            accepted = true_pair[0] > 0.0
        predecessor_accepted += int(accepted)
        if predicted_operation != 1:
            continue
        candidates = (
            scored if predecessor_exists is not None
            else [
                (score, right)
                for score, right in scored
                if score > 0.0
            ]
        ) if accepted else []
        selected = (
            max(candidates)[1] if candidates else None)
        if selected is None:
            update_no_candidate += 1
            continue
        if (
            examples[selected]["episode"]
            != example["previous_episode"]
        ):
            update_wrong_candidate += 1
            continue
        update_selected_correct += 1
        correct += 1
    metrics = {
        "version_link_accuracy":
            correct / max(len(examples), 1),
        "create_link_accuracy":
            create_correct / max(create_total, 1),
        "update_operation_accuracy":
            update_operation_correct
            / max(update_total, 1),
        "predecessor_rank_accuracy":
            predecessor_rank_correct
            / max(update_total, 1),
        "predecessor_accept_accuracy":
            predecessor_accepted
            / max(update_total, 1),
        "update_link_accuracy":
            update_selected_correct
            / max(update_total, 1),
        "update_no_candidate_rate":
            update_no_candidate
            / max(update_total, 1),
        "update_wrong_candidate_rate":
            update_wrong_candidate
            / max(update_total, 1),
        "predecessor_margin_mean":
            sum(margins) / max(len(margins), 1),
    }
    if missing_predecessor_total:
        rejection = (
            missing_predecessor_rejected
            / missing_predecessor_total)
        metrics[
            "counterfactual_missing_rejection_accuracy"
        ] = rejection
        metrics[
            "predecessor_exists_balanced_accuracy"
        ] = 0.5 * (
            metrics["predecessor_accept_accuracy"]
            + rejection)
    return metrics


def version_link_accuracy(
    examples, pairs, entity_logits,
    predicate_logits, operation,
    entity_threshold=0.0,
    predicate_threshold=0.0,
):
    return version_link_diagnostics(
        examples, pairs, entity_logits,
        predicate_logits, operation,
        entity_threshold,
        predicate_threshold)["version_link_accuracy"]


@torch.inference_mode()
def evaluate(
    model, writer, examples, pairs,
    device, batch_size,
    entity_threshold=0.0,
    predicate_threshold=0.0,
):
    (
        entity_logits,
        predicate_logits,
        learned_joint_logits,
    ) = score_pairs(
        model, examples, pairs, device, batch_size)
    entity_target = pair_targets(
        pairs, "entity_same")
    predicate_target = pair_targets(
        pairs, "predicate_same")
    joint_target = pair_targets(
        pairs, "joint_same")
    joint_logits = (
        learned_joint_logits
        if learned_joint_logits is not None
        else torch.minimum(
            entity_logits - entity_threshold,
            predicate_logits - predicate_threshold)
    )
    entity = balanced_accuracy(
        entity_logits, entity_target,
        entity_threshold)
    predicate = balanced_accuracy(
        predicate_logits, predicate_target,
        predicate_threshold)
    joint = balanced_accuracy(
        joint_logits, joint_target)
    operation = operation_logits(
        writer, examples, device, batch_size)
    (
        predecessor_exists,
        missing_predecessor_exists,
    ) = score_predecessor_existence(
        model, examples, pairs,
        learned_joint_logits, device)
    link_metrics = version_link_diagnostics(
        examples, pairs, entity_logits,
        predicate_logits, operation,
        entity_threshold,
        predicate_threshold,
        learned_joint_logits,
        predecessor_exists,
        missing_predecessor_exists)
    if predecessor_exists is not None:
        positive_values = list(
            predecessor_exists.values())
        negative_values = list(
            missing_predecessor_exists.values())
        existence_logits = torch.tensor(
            positive_values + negative_values)
        existence_targets = torch.tensor(
            [1.0] * len(positive_values)
            + [0.0] * len(negative_values))
        (
            oracle_threshold,
            oracle_balanced,
        ) = calibrate_binary_threshold(
            existence_logits, existence_targets)
        oracle_metrics = version_link_diagnostics(
            examples, pairs, entity_logits,
            predicate_logits, operation,
            entity_threshold,
            predicate_threshold,
            learned_joint_logits,
            predecessor_exists,
            missing_predecessor_exists,
            oracle_threshold)
        link_metrics[
            "oracle_predecessor_exists_threshold"
        ] = oracle_threshold
        link_metrics[
            "oracle_predecessor_exists_balanced_accuracy"
        ] = oracle_balanced
        link_metrics[
            "oracle_exists_version_link_accuracy"
        ] = oracle_metrics["version_link_accuracy"]
    return {
        "examples": len(examples),
        "pairs": len(pairs),
        "entity_pair_balanced": entity[0],
        "entity_pair_positive": entity[1],
        "entity_pair_negative": entity[2],
        "predicate_pair_balanced": predicate[0],
        "predicate_pair_positive": predicate[1],
        "predicate_pair_negative": predicate[2],
        "joint_pair_balanced": joint[0],
        "joint_pair_positive": joint[1],
        "joint_pair_negative": joint[2],
        **link_metrics,
        "entity_threshold":
            float(entity_threshold),
        "predicate_threshold":
            float(predicate_threshold),
    }


def calibrate_binary_threshold(logits, targets):
    positive = targets > 0.5
    negative = ~positive
    lower = float(logits.min()) - 1.0
    upper = float(logits.max()) + 1.0
    candidates = torch.linspace(
        lower, upper, 257)
    positive_accuracy = (
        logits[positive][None, :]
        > candidates[:, None]
    ).float().mean(dim=1)
    negative_accuracy = (
        logits[negative][None, :]
        <= candidates[:, None]
    ).float().mean(dim=1)
    score = 0.5 * (
        positive_accuracy + negative_accuracy)
    best = (
        score == score.max()).nonzero(
            as_tuple=False).flatten()
    index = min(
        best.tolist(),
        key=lambda item: abs(float(candidates[item])))
    return float(candidates[index]), float(score[index])


@torch.inference_mode()
def calibrate_pair_thresholds(
    model, examples, pairs, device, batch_size,
):
    entity_logits, predicate_logits, _ = score_pairs(
        model, examples, pairs, device, batch_size)
    entity_threshold, entity_accuracy = (
        calibrate_binary_threshold(
            entity_logits,
            pair_targets(pairs, "entity_same")))
    predicate_threshold, predicate_accuracy = (
        calibrate_binary_threshold(
            predicate_logits,
            pair_targets(pairs, "predicate_same")))
    return (
        entity_threshold,
        predicate_threshold,
        entity_accuracy,
        predicate_accuracy,
    )


@torch.inference_mode()
def evaluate_with_train_calibration(
    model, writer,
    train_examples, train_pairs,
    valid_examples, valid_pairs,
    device, batch_size,
):
    (
        entity_threshold,
        predicate_threshold,
        train_entity_accuracy,
        train_predicate_accuracy,
    ) = calibrate_pair_thresholds(
        model, train_examples, train_pairs,
        device, batch_size)
    metrics = evaluate(
        model, writer, valid_examples, valid_pairs,
        device, batch_size,
        entity_threshold, predicate_threshold)
    metrics["train_calibrated_entity_accuracy"] = (
        train_entity_accuracy)
    metrics["train_calibrated_predicate_accuracy"] = (
        train_predicate_accuracy)
    metrics["calibration_pairs"] = len(train_pairs)
    return metrics


def metric_key(metrics):
    minimum_metrics = [
        metrics["entity_pair_balanced"],
        metrics["predicate_pair_balanced"],
        metrics["joint_pair_balanced"],
        metrics["version_link_accuracy"],
    ]
    if "predecessor_exists_balanced_accuracy" in metrics:
        minimum_metrics.append(
            metrics[
                "predecessor_exists_balanced_accuracy"])
    return (
        min(minimum_metrics),
        metrics["version_link_accuracy"],
    )


def train(
    model, writer,
    train_examples, train_pairs, calibration_pairs,
    valid_examples, valid_pairs,
    device, steps, batch_size,
    learning_rate, weight_decay,
    eval_every, patience, seed,
    predecessor_rank_weight,
    predecessor_batch,
    predecessor_exists_weight,
    positive_loss_weight,
    layout_consistency_weight,
    layout_consistency_batch,
):
    optimizer = torch.optim.AdamW(
        [
            parameter for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=learning_rate,
        weight_decay=weight_decay)
    best = evaluate_with_train_calibration(
        model, writer,
        train_examples, calibration_pairs,
        valid_examples, valid_pairs,
        device, batch_size)
    best_state = clone_state_dict(model)
    best_step = 0
    stale = 0
    rng = random.Random(seed)
    predecessor_groups = build_predecessor_groups(
        train_examples, train_pairs)
    layout_groups = build_layout_consistency_groups(
        train_examples)
    if (
        layout_consistency_weight > 0.0
        and not layout_groups
    ):
        raise ValueError(
            "layout consistency needs paired views")
    print(json.dumps({
        "phase": "typed_pair_baseline",
        **best,
    }, separators=(",", ":")), flush=True)
    positive = [
        index for index, pair in enumerate(train_pairs)
        if pair["joint_same"]]
    hard = [
        index for index, pair in enumerate(train_pairs)
        if (
            pair["entity_same"]
            or pair["predicate_same"])
        and not pair["joint_same"]]
    easy = [
        index for index, pair in enumerate(train_pairs)
        if (
            not pair["entity_same"]
            and not pair["predicate_same"])]
    if not positive or not hard or not easy:
        raise ValueError(
            "typed pair curriculum lacks a pair class")
    for step in range(1, steps + 1):
        positive_count = batch_size // 4
        hard_count = batch_size // 2
        selected = [
            positive[rng.randrange(len(positive))]
            for _ in range(positive_count)]
        selected.extend(
            hard[rng.randrange(len(hard))]
            for _ in range(hard_count))
        selected.extend(
            easy[rng.randrange(len(easy))]
            for _ in range(
                batch_size - len(selected)))
        rng.shuffle(selected)
        batch = collate_pairs(
            train_examples, train_pairs,
            selected, device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["left"], batch["right"],
            batch["left_mask"], batch["right_mask"],
            batch["left_identity"],
            batch["right_identity"])
        loss, parts = verifier_loss(
            output, batch,
            positive_loss_weight)
        group_selected = None
        if (
            predecessor_rank_weight > 0.0
            or (
                model.set_link_head
                and predecessor_exists_weight > 0.0
            )
        ):
            group_selected = [
                rng.randrange(len(predecessor_groups))
                for _ in range(predecessor_batch)
            ]
        if model.set_link_head and group_selected is not None:
            (
                predecessor_loss,
                existence_loss,
            ) = predecessor_set_losses(
                model, train_examples, train_pairs,
                predecessor_groups, group_selected,
                device)
        elif predecessor_rank_weight > 0.0:
            predecessor_loss = predecessor_ranking_loss(
                model, train_examples, train_pairs,
                predecessor_groups, group_selected,
                device)
            existence_loss = loss.new_zeros(())
        else:
            predecessor_loss = loss.new_zeros(())
            existence_loss = loss.new_zeros(())
        if predecessor_rank_weight > 0.0:
            loss = (
                loss
                + predecessor_rank_weight
                * predecessor_loss)
        parts["predecessor"] = (
            predecessor_loss.detach())
        if (
            model.set_link_head
            and predecessor_exists_weight > 0.0
        ):
            loss = (
                loss
                + predecessor_exists_weight
                * existence_loss)
        parts["predecessor_exists"] = (
            existence_loss.detach())
        if layout_consistency_weight > 0.0:
            layout_selected = [
                rng.randrange(len(layout_groups))
                for _ in range(
                    layout_consistency_batch)
            ]
            consistency_loss = layout_consistency_loss(
                model, train_examples, layout_groups,
                layout_selected,
                device)
            loss = (
                loss
                + layout_consistency_weight
                * consistency_loss)
        else:
            consistency_loss = loss.new_zeros(())
        parts["layout_consistency"] = (
            consistency_loss.detach())
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [
                parameter for parameter in model.parameters()
                if parameter.requires_grad
            ], 1.0)
        optimizer.step()
        if step % eval_every and step != steps:
            continue
        metrics = evaluate_with_train_calibration(
            model, writer,
            train_examples, calibration_pairs,
            valid_examples, valid_pairs,
            device, batch_size)
        print(json.dumps({
            "phase": "typed_pair_valid",
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


def load_writer(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu",
        weights_only=True)
    if checkpoint.get("format") != "TYPED_MEMORY_WRITER_V1":
        raise ValueError("bad typed writer checkpoint")
    writer = TypedMemoryWriter(
        checkpoint["hidden"], checkpoint["rank"],
        len(checkpoint["hidden_layer_bands"]),
        checkpoint.get(
            "separate_address_localizer",
            False),
        checkpoint.get(
            "predict_span_boundaries",
            False),
        checkpoint.get(
            "use_token_embeddings",
            False),
        checkpoint.get(
            "hard_address_pooling",
            False),
        checkpoint.get(
            "contiguous_span_pooling",
            False),
        checkpoint.get(
            "max_field_span",
            8)).to(device)
    writer.load_state_dict(
        checkpoint["state_dict"], strict=True)
    writer.eval()
    for parameter in writer.parameters():
        parameter.requires_grad = False
    return checkpoint, writer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("writer_checkpoint")
    parser.add_argument("train_features")
    parser.add_argument("train_jsonl")
    parser.add_argument("valid_features")
    parser.add_argument("valid_jsonl")
    parser.add_argument("output")
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--tok-probe", required=True)
    parser.add_argument("--lib")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--predecessor-rank-weight",
        type=float, default=1.0)
    parser.add_argument(
        "--predecessor-batch",
        type=int, default=16)
    parser.add_argument(
        "--predecessor-exists-weight",
        type=float, default=1.0)
    parser.add_argument(
        "--attention-weighted-alignment",
        action="store_true")
    parser.add_argument(
        "--dual-path",
        action="store_true")
    parser.add_argument(
        "--trainable-context-localizer",
        action="store_true")
    parser.add_argument(
        "--set-link-head",
        action="store_true")
    parser.add_argument(
        "--set-link-head-only",
        action="store_true")
    parser.add_argument(
        "--positive-loss-weight",
        type=float, default=0.5)
    parser.add_argument(
        "--layout-consistency-weight",
        type=float, default=0.0)
    parser.add_argument(
        "--layout-consistency-batch",
        type=int, default=16)
    parser.add_argument(
        "--calibration-pairs",
        type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    if args.predecessor_rank_weight < 0.0:
        parser.error(
            "--predecessor-rank-weight must be non-negative")
    if args.predecessor_batch < 1:
        parser.error("--predecessor-batch must be positive")
    if args.predecessor_exists_weight < 0.0:
        parser.error(
            "--predecessor-exists-weight must be non-negative")
    if not 0.0 < args.positive_loss_weight < 1.0:
        parser.error(
            "--positive-loss-weight must be in (0, 1)")
    if args.layout_consistency_weight < 0.0:
        parser.error(
            "--layout-consistency-weight must be non-negative")
    if args.layout_consistency_batch < 1:
        parser.error(
            "--layout-consistency-batch must be positive")
    if 0 < args.calibration_pairs < 4:
        parser.error(
            "--calibration-pairs must be zero or at least four")
    if (
        args.layout_consistency_weight > 0.0
        and not args.dual_path
    ):
        parser.error(
            "layout consistency requires --dual-path")
    if (
        args.set_link_head_only
        and not args.set_link_head
    ):
        parser.error(
            "--set-link-head-only requires --set-link-head")

    writer_checkpoint, writer = load_writer(
        args.writer_checkpoint, args.device)
    if writer_checkpoint.get(
        "use_token_embeddings", False
    ):
        if not args.lib:
            parser.error(
                "token-embedding writer requires --lib")
        token_embedding_table = (
            load_token_embedding_table(
                args.gguf, args.lib,
                writer_checkpoint["backbone_sha256"],
                writer_checkpoint["hidden"]))
    else:
        token_embedding_table = None
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    train_examples = load_writer_examples(
        args.train_features, args.train_jsonl,
        writer_checkpoint["backbone_sha256"],
        tokenizer, token_embedding_table)
    valid_examples = load_writer_examples(
        args.valid_features, args.valid_jsonl,
        writer_checkpoint["backbone_sha256"],
        tokenizer, token_embedding_table)
    train_pairs = build_active_pairs(train_examples)
    valid_pairs = build_active_pairs(valid_examples)
    calibration_pairs = select_calibration_pairs(
        train_pairs, args.calibration_pairs,
        args.seed + 97)
    calibration_counts = collections.Counter(
        (
            bool(pair["entity_same"]),
            bool(pair["predicate_same"]),
        )
        for pair in calibration_pairs)
    print(json.dumps({
        "phase": "typed_pair_calibration",
        "total_pairs": len(train_pairs),
        "selected_pairs": len(calibration_pairs),
        "entity_false_predicate_false":
            calibration_counts[(False, False)],
        "entity_false_predicate_true":
            calibration_counts[(False, True)],
        "entity_true_predicate_false":
            calibration_counts[(True, False)],
        "entity_true_predicate_true":
            calibration_counts[(True, True)],
    }, separators=(",", ":")), flush=True)
    model = TypedPairVerifier(
        writer, writer_checkpoint["rank"],
        args.attention_weighted_alignment,
        args.dual_path,
        args.trainable_context_localizer,
        args.set_link_head).to(args.device)
    if args.init_checkpoint:
        initial = torch.load(
            args.init_checkpoint,
            map_location="cpu", weights_only=True)
        initial_set_link_head = bool(
            initial.get("set_link_head", False))
        if (
            initial.get("format")
            != "TYPED_PAIR_VERIFIER_V1"
            or initial.get("backbone_sha256")
            != writer_checkpoint["backbone_sha256"]
            or initial.get("rank")
            != writer_checkpoint["rank"]
            or bool(initial.get(
                "separate_address_localizer",
                False
            )) != bool(writer_checkpoint.get(
                "separate_address_localizer",
                False
            ))
            or bool(initial.get(
                "attention_weighted_alignment",
                False
            )) != args.attention_weighted_alignment
            or bool(initial.get(
                "dual_path", False
            )) != args.dual_path
            or bool(initial.get(
                "trainable_context_localizer",
                False
            )) != args.trainable_context_localizer
            or (
                initial_set_link_head
                and not args.set_link_head
            )
            or (
                initial_set_link_head
                and int(initial.get(
                    "set_link_head_version", 1
                )) != SET_LINK_HEAD_VERSION
            )
        ):
            raise ValueError(
                "typed pair init checkpoint mismatch")
        missing, unexpected = model.load_state_dict(
            initial["verifier_state_dict"],
            strict=False)
        allowed_missing = {
            name for name in model.state_dict()
            if name.startswith("writer.")
        }
        if (
            args.set_link_head
            and not initial_set_link_head
        ):
            allowed_missing.update({
                name for name in model.state_dict()
                if (
                    name.startswith("joint_head.")
                    or name.startswith(
                        "predecessor_exists_head.")
                )
            })
        if set(missing) - allowed_missing or unexpected:
            raise ValueError(
                "typed pair init parameter mismatch: "
                f"{missing} {unexpected}")
        print(json.dumps({
            "phase": "typed_pair_init",
            "checkpoint": args.init_checkpoint,
            "previous_step":
                initial.get("selected_step"),
            "previous_metrics":
                initial.get("valid_metrics", {}),
        }, separators=(",", ":")), flush=True)
    if args.set_link_head_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (
                name.startswith("joint_head.")
                or name.startswith(
                    "predecessor_exists_head."))
    metrics, selected_step = train(
        model, writer,
        train_examples, train_pairs,
        calibration_pairs,
        valid_examples, valid_pairs,
        args.device, args.steps, args.batch,
        args.learning_rate, args.weight_decay,
        args.eval_every, args.patience,
        args.seed,
        args.predecessor_rank_weight,
        args.predecessor_batch,
        args.predecessor_exists_weight,
        args.positive_loss_weight,
        args.layout_consistency_weight,
        args.layout_consistency_batch)
    uncalibrated = (
        None
        if args.set_link_head
        else evaluate(
            model, writer,
            valid_examples, valid_pairs,
            args.device, args.batch)
    )
    metrics["uncalibrated_metrics"] = uncalibrated
    entity_threshold = metrics["entity_threshold"]
    predicate_threshold = metrics["predicate_threshold"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "TYPED_PAIR_VERIFIER_V1",
        "backbone_sha256":
            writer_checkpoint["backbone_sha256"],
        "hidden": writer_checkpoint["hidden"],
        "rank": writer_checkpoint["rank"],
        "hidden_layer_bands":
            writer_checkpoint["hidden_layer_bands"],
        "separate_address_localizer":
            writer_checkpoint.get(
                "separate_address_localizer",
                False),
        "predict_span_boundaries":
            writer_checkpoint.get(
                "predict_span_boundaries",
                False),
        "use_token_embeddings":
            writer_checkpoint.get(
                "use_token_embeddings",
                False),
        "hard_address_pooling":
            writer_checkpoint.get(
                "hard_address_pooling",
                False),
        "contiguous_span_pooling":
            writer_checkpoint.get(
                "contiguous_span_pooling",
                False),
        "max_field_span":
            writer_checkpoint.get(
                "max_field_span", 8),
        "writer_state_dict":
            writer_checkpoint["state_dict"],
        "verifier_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
            if not name.startswith("writer.")
        },
        "selected_step": selected_step,
        "predecessor_rank_weight":
            args.predecessor_rank_weight,
        "predecessor_batch":
            args.predecessor_batch,
        "predecessor_exists_weight":
            args.predecessor_exists_weight,
        "attention_weighted_alignment":
            args.attention_weighted_alignment,
        "dual_path": args.dual_path,
        "trainable_context_localizer":
            args.trainable_context_localizer,
        "set_link_head":
            args.set_link_head,
        "set_link_head_version": (
            SET_LINK_HEAD_VERSION
            if args.set_link_head else 0),
        "set_link_head_only":
            args.set_link_head_only,
        "positive_loss_weight":
            args.positive_loss_weight,
        "layout_consistency_weight":
            args.layout_consistency_weight,
        "layout_consistency_batch":
            args.layout_consistency_batch,
        "calibration_pairs":
            len(calibration_pairs),
        "valid_metrics": metrics,
        "entity_threshold": entity_threshold,
        "predicate_threshold":
            predicate_threshold,
        "runtime_status": "python_proof_only",
        "kv_cache_used": False,
        "lora_used": False,
        "rag_used": False,
        "locomo_used": False,
    }, output)
    print(json.dumps({
        "phase": "typed_pair_done",
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
