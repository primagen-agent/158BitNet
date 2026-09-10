"""Train a backbone-bound, per-layer Metis memory model.

The accuracy-first path follows the public MemTensor/Metis implementation:
learned full-rank query/K/V projections, normalized GQA memory reads,
StraightThrough AlphaTopP selection, gated-delta writes, five-task dynamic
sampling, and a complete differentiable graph across dialogue chunks. The
backbone stays frozen and KV cache is not used.

Positive query/K/V ranks and NAS gates remain available only for later
compression experiments. BNMEM export stores model parameters; dynamic
memory contents are exported separately as BNSTATE data.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ggw import GGUFWeights
from torch_backbone import TorchBackbone
from train_data import (
    CTokenDecoder, CTokenizer, load_dataset, dataset_order, render_chunk,
)
from bnmem_export import load_bnmem_v1, save_bnmem_v3
from answer_decoder import MemoryAnswerDecoder
from memory_retrieval import (
    add_retrieved_excerpts,
    append_pointer_token_ids,
    select_memory_excerpts,
)
from output_lora import OutputLoRA, load_output_lora
from backbone_lora import (
    BackboneLoRA,
    load_lora_bundle,
    save_lora_bundle,
)

# Reference gate-bias init (GatedDeltaRuleMixin._logit_clamped): p=1.0
# clamped to 1-1e-4, then log(p/(1-p)) = log(9999) ~ +9.2102 (sigmoid -> ~1).
# (An earlier sign-flipped version produced -9.21 -> beta ~ 0 -> empty M/S.)
LOGIT_CLAMPED_1 = math.log((1.0 - 1e-4) / 1e-4)        # ~ +9.2102


TASK_WEIGHT_DEFAULTS = {
    0: (0.25, 0.10),
    1: (0.35, 0.25),
    2: (0.20, 0.30),
    3: (0.10, 0.20),
    4: (0.10, 0.15),
}


def parse_task_filter(value):
    """Parse ``all`` or a comma-separated subset of paper task ids."""
    if value == "all":
        return None
    result = {
        int(item.strip())
        for item in str(value).split(",")
        if item.strip()
    }
    if not result or any(task not in TASK_WEIGHT_DEFAULTS for task in result):
        raise ValueError("task filter must be all or ids from 0 through 4")
    return result


def memory_task_id(stratum_name, sample):
    """Map local memory data to the official five-task training schedule."""
    metadata = sample.get("metadata", {})
    v2_task = str(metadata.get("v2_task", ""))
    for task in range(5):
        if v2_task.startswith(f"task{task}"):
            return task
    if stratum_name.startswith("task3"):
        return 3
    if stratum_name.startswith("task4"):
        return 4
    if stratum_name == "multi_fact":
        return 3
    if stratum_name == "memory_pollution":
        return 4
    if stratum_name == "multi_entity":
        return 3
    if stratum_name == "post_memory":
        return 4
    if "distract" in stratum_name:
        return 2
    if stratum_name == "reconstruction" or stratum_name.startswith(
        "remember_"
    ):
        return 0
    if stratum_name.startswith(("update_", "forget_", "reflect_")):
        return 1
    operation = str(metadata.get("type", ""))
    style = str(metadata.get("style", ""))
    if "distract" in style:
        return 2
    if operation == "reconstruction":
        return 0
    if operation in {"update", "forget", "reflection"}:
        return 1
    return 0


def generation_query_evidence_map(sample, query_indices):
    """Normalize current and legacy per-query evidence annotations."""
    current = sample.get("query_evidence_message_indices")
    if isinstance(current, dict):
        return current
    legacy = sample.get("evidence_message_indices")
    if len(query_indices) == 1 and isinstance(legacy, list):
        return {str(query_indices[0]): legacy}
    if isinstance(legacy, list) and len(legacy) == len(query_indices):
        return {
            str(query_index): evidence_index
            for query_index, evidence_index
            in zip(query_indices, legacy)
        }
    if len(query_indices) == 1 and isinstance(legacy, int):
        return {str(query_indices[0]): legacy}
    return {}


def target_ids_by_evidence(
    query_indices, query_evidence, target_ids_by_query
):
    """Associate each query target with only its annotated source messages."""
    result = {}
    for query_index in query_indices:
        target_ids = target_ids_by_query.get(int(query_index))
        if target_ids is None:
            continue
        evidence = query_evidence.get(
            str(query_index), query_evidence.get(int(query_index)))
        if evidence is None:
            continue
        evidence_indices = evidence if isinstance(evidence, list) else [
            evidence]
        for evidence_index in evidence_indices:
            bucket = result.setdefault(int(evidence_index), [])
            normalized = tuple(int(token) for token in target_ids)
            if all(tuple(existing) != normalized for existing in bucket):
                bucket.append(list(normalized))
    return result


def scheduled_task_weights(progress, available_tasks, starts, ends):
    """Linearly anneal and normalize official task sampling weights."""
    progress = min(max(float(progress), 0.0), 1.0)
    raw = {
        task: max(
            0.0,
            starts[task] + (ends[task] - starts[task]) * progress,
        )
        for task in available_tasks
    }
    total = sum(raw.values())
    if total <= 0.0:
        return {task: 1.0 / len(raw) for task in raw}
    return {task: value / total for task, value in raw.items()}


def straight_through_alpha_top_p(
    p, rho, k_min, max_tokens=0, max_fraction=0.0
):
    """Sparse AlphaTopP forward weights with dense softmax gradients."""
    L = p.shape[-1]
    sorted_p, sorted_idx = torch.sort(
        p, descending=True, dim=-1, stable=True)
    cum = torch.cumsum(sorted_p, dim=-1)
    exceed = cum > rho
    if bool(exceed.any()):
        k = int(torch.nonzero(
            exceed, as_tuple=False)[0, -1].item()) + 1
    else:
        k = L
    k = min(max(k, k_min), L)
    k_max = L
    if max_fraction > 0.0:
        k_max = min(k_max, max(1, math.ceil(L * max_fraction)))
    if max_tokens > 0:
        k_max = min(k_max, max_tokens)
    k_max = max(k_min, k_max)
    k = min(k, k_max)
    selected = torch.zeros_like(p, dtype=torch.bool)
    selected[sorted_idx[:k]] = True
    mass = p[selected].sum().clamp_min(1e-6)
    hard = torch.where(selected, p / mass, torch.zeros_like(p))
    return hard.detach() - p.detach() + p


def gated_delta_update(M, S, K, V, beta_weight, alpha):
    """Reference parallel GDU update for value and key-normalizer states."""
    km = K @ M
    ks = K @ S
    new_M = (alpha * M
             + (K.t() * beta_weight) @ (V - alpha * km))
    new_S = (alpha * S
             + (K * (beta_weight * (1.0 - alpha * ks))
                .unsqueeze(-1)).sum(0))
    return new_M, new_S


def normalized_memory_read(keys, memory, normalizer, denom_mode):
    """Read a delta-memory state at explicit latent probe keys."""
    numerator = keys @ memory
    denominator = keys @ normalizer
    if denom_mode == "abs_plus_one":
        denominator = denominator.abs() + 1.0
    else:
        denominator = denominator + 1.0
    return numerator / denominator.unsqueeze(-1)


def memory_read_preservation_loss(
    old_memory, old_normalizer, new_memory, new_normalizer,
    probe_keys, denom_mode,
):
    """Penalize a new write when it changes established latent readouts."""
    if probe_keys.numel() == 0:
        return new_memory.new_zeros(())
    old_read = normalized_memory_read(
        probe_keys, old_memory, old_normalizer, denom_mode).detach()
    new_read = normalized_memory_read(
        probe_keys, new_memory, new_normalizer, denom_mode)
    scale = old_read.float().pow(2).mean(
        dim=-1, keepdim=True).sqrt().clamp_min(1e-3)
    return (((new_read.float() - old_read.float()) / scale)
            .square().mean())


def orthogonalize_write_keys(
    unit_keys, address_weights, existing_basis,
    strength, update_similarity_threshold,
):
    """Remove interference with distinct old addresses, preserving updates."""
    if existing_basis.numel() == 0 or strength <= 0.0:
        return unit_keys
    address = F.normalize(
        (unit_keys * address_weights.unsqueeze(-1)).sum(dim=0),
        dim=-1, eps=1e-12)
    basis = F.normalize(existing_basis, dim=-1, eps=1e-12)
    similarities = torch.abs(basis @ address)
    protected = basis[similarities < update_similarity_threshold]
    if protected.shape[0] == 0:
        return unit_keys
    projected = (unit_keys @ protected.t()) @ protected
    return F.normalize(
        unit_keys - float(strength) * projected,
        dim=-1, eps=1e-12)


def metis_rms_norm(x, weight, eps):
    """Backbone-style RMSNorm: normalize x first, then apply weight."""
    inv = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return x.float() * inv * weight.float()


def balanced_truncated_svd(weight, rank):
    """Return A/B with weight ~= A @ B using the leading singular triplets.

    Splitting sqrt(S) across both factors keeps their initial scales balanced,
    which is friendlier to Adam than placing the complete singular value in
    either factor.
    """
    if weight.ndim != 2:
        raise ValueError("SVD weight must be a matrix")
    if rank < 1 or rank > min(weight.shape):
        raise ValueError(
            f"SVD rank {rank} outside [1, {min(weight.shape)}]")
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    root = s[:rank].clamp_min(0.0).sqrt()
    return u[:, :rank] * root, root.unsqueeze(1) * vh[:rank, :]


def straight_through_binary_gate(logits, temperature=1.0,
                                 threshold=0.5):
    """Binary structural gate in forward, sigmoid surrogate in backward."""
    soft = torch.sigmoid(logits / temperature)
    hard = (soft >= threshold).to(soft.dtype)
    return hard.detach() - soft.detach() + soft


def straight_through_topk(probabilities, k=1):
    """Select sparse latent banks with a soft router gradient."""
    if probabilities.ndim < 1:
        raise ValueError("top-k bank routing expects bank logits")
    if k < 1 or k > probabilities.shape[-1]:
        raise ValueError("bank routing k is outside the bank count")
    hard = torch.zeros_like(probabilities)
    hard.scatter_(
        -1, probabilities.topk(k, dim=-1).indices, 1.0)
    soft = probabilities * float(k)
    return hard.detach() - soft.detach() + soft


def straight_through_top1(probabilities):
    """Backward-compatible one-bank router."""
    return straight_through_topk(probabilities, 1)


def stale_answer_margin_loss(positive_nll, negative_nlls, margin):
    """Require the correct sequence NLL to beat stale/deleted alternatives."""
    if not negative_nlls:
        return positive_nll.new_zeros(())
    negatives = torch.stack(negative_nlls)
    return F.relu(margin + positive_nll - negatives).mean()


def autoregressive_supervision_length(generated_prefix, target_ids):
    """Supervise through the first rollout error, not beyond misalignment.

    Once a generated token differs from the gold token, later gold positions
    no longer describe the same prefix. Training those positions creates
    contradictory gradients. If the whole generated prefix is correct, also
    supervise the final target token (normally EOS).
    """
    comparable = min(len(generated_prefix), max(len(target_ids) - 1, 0))
    for index in range(comparable):
        if int(generated_prefix[index]) != int(target_ids[index]):
            return index + 1
    return len(target_ids)


def matching_prefix_length(prediction_ids, target_ids):
    """Count consecutive correct tokens from the start of a generation."""
    count = 0
    for prediction, target in zip(prediction_ids, target_ids):
        if int(prediction) != int(target):
            break
        count += 1
    return count


def ordered_episode_copy_tail(
    payload_token_ids, payload_positions, generated_ids
):
    """Return an unambiguous remaining run from an activated episode payload.

    Payload tokens remain ordered by their original source positions. Gaps
    split the payload into independent runs, so the pointer cannot jump across
    unselected source text. The longest suffix of the generated answer is
    matched first, allowing the backbone to emit a prefix that the selector
    omitted while preventing a shorter, common digit from overriding a longer
    and more specific match.
    """
    generated = [int(token) for token in generated_ids]
    if not generated:
        return ()
    pairs = sorted(
        (
            (int(position), int(token))
            for token, position in zip(
                payload_token_ids, payload_positions)
            if int(token) >= 0 and int(position) >= 0
        ),
        key=lambda item: item[0],
    )
    if not pairs:
        return ()

    runs = []
    current = []
    previous_position = None
    for position, token in pairs:
        if (
            previous_position is not None
            and position != previous_position + 1
        ):
            runs.append(current)
            current = []
        current.append(token)
        previous_position = position
    if current:
        runs.append(current)

    for matched_length in range(len(generated), 0, -1):
        suffix = generated[-matched_length:]
        candidates = set()
        for run in runs:
            if len(run) <= matched_length:
                continue
            for start in range(len(run) - matched_length):
                if run[start:start + matched_length] == suffix:
                    candidates.add(tuple(run[start + matched_length:]))
        if candidates:
            return candidates.pop() if len(candidates) == 1 else ()
    return ()


def ordered_episode_continuation(
    payload_token_ids, payload_positions, generated_ids
):
    """Return the first token of an unambiguous episode-local copy tail."""
    tail = ordered_episode_copy_tail(
        payload_token_ids, payload_positions, generated_ids)
    return tail[0] if tail else None


def retention_floor_loss(alpha, target):
    """Penalize global state decay below a preferred retention floor."""
    return F.relu(float(target) - alpha).square()


def memory_address_logits(query_groups, keys, temperature):
    """Score write addresses from the query representation used by M/S.

    Average evidence across query tokens and groups instead of using a
    log-sum-exp. A single generic token or one lucky group must not be able
    to satisfy the address objective for the whole query.
    """
    if keys.shape[0] < 2:
        return query_groups.new_zeros((keys.shape[0],))
    query = F.normalize(query_groups.float(), dim=-1, eps=1e-12)
    normalized_keys = F.normalize(keys.float(), dim=-1, eps=1e-12)
    scores = torch.einsum(
        "tgk,nk->tgn", query, normalized_keys) / float(temperature)
    return scores.mean(dim=(0, 1))


def memory_address_contrastive_loss(
    query_groups, keys, target_index, temperature
):
    """Align a neural query with its write address without retrieving text."""
    if keys.shape[0] < 2:
        return query_groups.new_zeros(())
    logits = memory_address_logits(query_groups, keys, temperature)
    target = torch.tensor(
        [int(target_index)], device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits.unsqueeze(0), target)


def memory_key_diversity_loss(new_keys, previous_keys, margin):
    """Penalize correlated write addresses across distinct memories."""
    if previous_keys.numel() == 0:
        return new_keys.new_zeros(())
    new_normalized = F.normalize(new_keys.float(), dim=-1, eps=1e-12)
    previous_normalized = F.normalize(
        previous_keys.float(), dim=-1, eps=1e-12)
    similarities = torch.einsum(
        "lk,nlk->nl", new_normalized, previous_normalized).abs()
    return F.relu(similarities - float(margin)).square().mean()


def memory_address_transform(keys, beta_weight, alpha):
    """Return the exact linear transform applied to prior M/S addresses.

    The gated-delta update can be written as ``new_M = A @ M + write`` and
    ``new_S = A @ S + write``. Historical address vectors must therefore be
    carried forward by the same A after every later commit.
    """
    eye = torch.eye(
        keys.shape[-1], device=keys.device, dtype=keys.dtype)
    gram = (keys.t() * beta_weight) @ keys
    return alpha * (eye - gram)


def evidence_attention_loss(weights, labels):
    """Penalize query attention that misses labelled evidence token slots."""
    positive = labels > 0
    negative = labels == 0
    if not bool(positive.any()) or not bool(negative.any()):
        return weights.new_zeros(())
    positive_mass = weights[..., positive].sum(dim=-1)
    return -positive_mass.clamp_min(1e-8).log().mean()


def balanced_token_selection_loss(logits, labels):
    """Teach a selector to cover every labelled token, not just one of them.

    Attention-mass supervision is sufficient for locating one evidence span,
    but a write selector must retain all payload tokens needed to reconstruct
    a sequence. Balanced binary supervision prevents the many negative source
    tokens from overwhelming the smaller positive address/value spans.
    """
    known = labels >= 0
    if not bool(known.any()):
        return logits.new_zeros(())
    selected_logits = logits[known]
    targets = (labels[known] > 0).to(selected_logits.dtype)
    positives = targets.sum()
    negatives = targets.numel() - positives
    if float(positives) == 0.0 or float(negatives) == 0.0:
        return F.binary_cross_entropy_with_logits(
            selected_logits, targets)
    positive_weight = (negatives / positives).detach()
    return F.binary_cross_entropy_with_logits(
        selected_logits, targets, pos_weight=positive_weight)


def answer_token_slot_labels(source_ids, target_ids, eos_id):
    """Label the longest exact target-token span in a source chunk.

    Positive slots are the best contiguous source/answer token match,
    negatives are the remaining tokens in the same answer-bearing chunk,
    and all slots stay unknown when there is no meaningful match.
    """
    target = [int(token) for token in target_ids if int(token) != eos_id]
    source = [int(token) for token in source_ids]
    unknown = [-1] * len(source)
    if not source or not target:
        return unknown

    previous = [0] * (len(target) + 1)
    best_length = 0
    best_source_ends = []
    for source_index, source_token in enumerate(source):
        current = [0] * (len(target) + 1)
        for target_index, target_token in enumerate(target, start=1):
            if source_token != target_token:
                continue
            current[target_index] = previous[target_index - 1] + 1
            match_length = current[target_index]
            if match_length > best_length:
                best_length = match_length
                best_source_ends = [source_index]
            elif match_length == best_length:
                best_source_ends.append(source_index)
        previous = current

    minimum_length = 1 if len(target) == 1 else 2
    if best_length < minimum_length:
        return unknown
    labels = [0] * len(source)
    for source_end in best_source_ends:
        source_start = source_end - best_length + 1
        for index in range(source_start, source_end + 1):
            labels[index] = 1
    return labels


def memory_target_slot_labels(
    source_ids, target_id_sequences, eos_id
):
    """Mark payload tokens for the memory written by one source message.

    Each write is supervised from the value contained in that same message,
    never from a future query.  Multiple payloads are merged when a message
    intentionally carries more than one memory.
    """
    merged = [0] * len(source_ids)
    matched = False
    for target_ids in target_id_sequences:
        labels = answer_token_slot_labels(source_ids, target_ids, eos_id)
        if any(label > 0 for label in labels):
            matched = True
            merged = [
                max(left, right)
                for left, right in zip(merged, labels)
            ]
    return merged if matched else [-1] * len(source_ids)


class MetisMemory(nn.Module):
    """32-layer memory module; frozen projections borrowed from the
    backbone (q_proj pre-RoPE read side, o_proj fuse side)."""

    def __init__(self, backbone: TorchBackbone, layer_ids, gamma=0.9,
                 tau=1.0, rho=0.9, k_min=1, beta_scale=0.9,
                 alpha_max_tokens=0, alpha_max_fraction=0.0,
                 query_rank=128, query_gate_lambda=0.0,
                 query_gate_temperature=1.0, query_gate_threshold=0.5,
                 query_gate_min_rank=1, query_mode="backbone_delta",
                 kv_rank=128, kv_gate_lambda=0.0,
                 kv_gate_temperature=1.0, kv_gate_threshold=0.5,
                 kv_gate_min_rank=1,
                 layer_gate_lambda=0.0, layer_gate_temperature=1.0,
                 layer_gate_threshold=0.5, layer_gate_min_layers=1,
                 fusion_mode="fixed",
                 gdu_alpha_init=1.0, gdu_beta_init=1.0,
                 retention_target=0.99,
                 address_temperature=0.1,
                 key_diversity_margin=0.1,
                 denom_mode="signed_plus_one",
                 state_mode="delta", max_memory_slots=4096,
                 slot_temperature=0.07, memory_banks=1,
                 max_memory_episodes=64, episode_temperature=0.1,
                 episode_read_mode="soft",
                 episode_address_rank=64,
                 episode_address_mode="storage",
                 episode_address_granularity="pooled",
                 episode_address_tokens=16,
                 episode_payload_tokens=32,
                 episode_payload_min_layer_votes=1,
                 episode_address_source="contextual",
                 episode_identity_uniqueness_floor=0.05,
                 episode_identity_match_scale=0.0,
                 episode_identity_ngram=1,
                 episode_identity_route_mode="maxsim",
                 episode_identity_selection_mode="topk",
                 episode_router_mode="cosine",
                 episode_router_rank=32,
                 bank_temperature=0.1, bank_write_top_k=1,
                 bank_read_mode="soft", bank_router_mode="learned",
                 write_mode="paired_tokens",
                 value_alpha_max_tokens=-1,
                 value_alpha_max_fraction=-1.0,
                 write_orthogonalization=0.0,
                 update_similarity_threshold=0.85,
                 device="cuda",
                 dtype=torch.float32, seed=42):
        super().__init__()
        cfg = backbone.cfg
        self.backbone = backbone
        self.layer_ids = list(layer_ids)
        self.n_layers = len(self.layer_ids)
        self.d_model = cfg.hidden
        self.kv_dim = cfg.kv_dim
        self.q_dim = cfg.q_dim
        self.head_dim = cfg.head_dim
        self.groups = cfg.q_dim // cfg.kv_dim
        self.heads_per_group = cfg.n_heads // self.groups
        self.gamma, self.tau, self.rho, self.k_min = gamma, tau, rho, k_min
        if fusion_mode not in {"fixed", "residual_gate"}:
            raise ValueError(f"unsupported fusion mode {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.alpha_max_tokens = int(alpha_max_tokens)
        self.alpha_max_fraction = float(alpha_max_fraction)
        if self.alpha_max_tokens < 0:
            raise ValueError("alpha_max_tokens must be non-negative")
        if not 0.0 <= self.alpha_max_fraction <= 1.0:
            raise ValueError("alpha_max_fraction must be in [0, 1]")
        self.value_alpha_max_tokens = (
            self.alpha_max_tokens
            if value_alpha_max_tokens < 0
            else int(value_alpha_max_tokens))
        self.value_alpha_max_fraction = (
            self.alpha_max_fraction
            if value_alpha_max_fraction < 0.0
            else float(value_alpha_max_fraction))
        if self.value_alpha_max_tokens < 0:
            raise ValueError(
                "value alpha max tokens must be non-negative")
        if not 0.0 <= self.value_alpha_max_fraction <= 1.0:
            raise ValueError(
                "value alpha max fraction must be in [0, 1]")
        if not 0.0 <= write_orthogonalization <= 1.0:
            raise ValueError(
                "write orthogonalization must be in [0, 1]")
        if not 0.0 <= update_similarity_threshold <= 1.0:
            raise ValueError(
                "update similarity threshold must be in [0, 1]")
        self.write_orthogonalization = float(write_orthogonalization)
        self.update_similarity_threshold = float(
            update_similarity_threshold)
        self.beta_scale = beta_scale
        if not 0.0 < retention_target <= 1.0:
            raise ValueError("retention target must be in (0, 1]")
        self.retention_target = float(retention_target)
        if address_temperature <= 0.0:
            raise ValueError("address temperature must be positive")
        if not 0.0 <= key_diversity_margin < 1.0:
            raise ValueError("key diversity margin must be in [0, 1)")
        self.address_temperature = float(address_temperature)
        self.key_diversity_margin = float(key_diversity_margin)
        # Paper/reference initialization: both gates start at sigmoid≈1.
        # They remain trainable and can learn forgetting/update behavior.
        NL = self.n_layers
        def gate_logit(value):
            probability = min(max(float(value), 1e-4), 1.0 - 1e-4)
            return math.log(probability / (1.0 - probability))
        self.gdu_ab = nn.Parameter(torch.full(
            (NL,), gate_logit(gdu_alpha_init), device=device, dtype=dtype))
        self.gdu_bb = nn.Parameter(torch.full(
            (NL,), gate_logit(gdu_beta_init), device=device, dtype=dtype))
        self.device, self.dtype = device, dtype

        g = torch.Generator(device="cpu").manual_seed(seed)
        def zeros(*shape):
            return torch.zeros(*shape, device=device, dtype=dtype)
        # A zero rank selects the accuracy-first full-rank path. Positive
        # ranks keep the legacy factorized path for later compression work.
        if kv_rank < 0 or kv_rank > min(self.kv_dim, self.d_model):
            raise ValueError(
                f"kv_rank {kv_rank} outside [0, "
                f"{min(self.kv_dim, self.d_model)}]")
        self.kv_rank = kv_rank
        if kv_rank == 0:
            self.wk = nn.Parameter(zeros(NL, self.kv_dim, self.d_model))
            self.wv = nn.Parameter(zeros(NL, self.kv_dim, self.d_model))
            self.wk_a = self.wk_b = None
            self.wv_a = self.wv_b = None
        else:
            self.wk = self.wv = None
            self.wk_a = nn.Parameter(zeros(NL, self.kv_dim, kv_rank))
            self.wk_b = nn.Parameter(zeros(NL, kv_rank, self.d_model))
            self.wv_a = nn.Parameter(zeros(NL, self.kv_dim, kv_rank))
            self.wv_b = nn.Parameter(zeros(NL, kv_rank, self.d_model))
        self.w_agg = nn.Parameter(zeros(NL, self.d_model))
        self.episode_identity_selector_w = (
            nn.Parameter(zeros(self.d_model))
            if (
                state_mode == "episodes"
                and episode_address_source == "embedding"
            ) else None)
        if write_mode not in (
            "paired_tokens", "factorized", "dual_tokens"
        ):
            raise ValueError(
                "write mode must be paired_tokens, factorized, or "
                "dual_tokens")
        self.write_mode = write_mode
        self.w_value_agg = (
            nn.Parameter(zeros(NL, self.d_model))
            if write_mode in ("factorized", "dual_tokens") else None)
        self.gdu_aw = nn.Parameter(zeros(NL, self.d_model))
        self.gdu_bw = nn.Parameter(zeros(NL, self.d_model))
        self.mem_norm = nn.Parameter(torch.ones(NL, self.q_dim,
                                                device=device, dtype=dtype))
        if self.fusion_mode == "residual_gate":
            self.fusion_gate_w = nn.Parameter(zeros(NL, self.d_model))
            self.fusion_gate_b = nn.Parameter(torch.full(
                (NL,), math.log(0.1 / 0.9),
                device=device, dtype=dtype))
        else:
            self.fusion_gate_w = None
            self.fusion_gate_b = None
        # Zero selects an independent full-rank query projection. Positive
        # ranks use the factorized form.  ``backbone_delta`` remains only for
        # loading historical experiments; accuracy-first training uses the
        # reference independent query because it has one unambiguous input
        # domain in both Python and C.
        if query_rank < 0 or query_rank > min(self.q_dim, self.d_model):
            raise ValueError(
                f"query_rank {query_rank} outside [0, "
                f"{min(self.q_dim, self.d_model)}]")
        self.q_rank = query_rank
        if query_rank == 0:
            self.query_proj = nn.Parameter(
                zeros(NL, self.q_dim, self.d_model))
            self.query_a = self.query_b = None
            if query_mode == "independent":
                with torch.no_grad():
                    for layer in range(NL):
                        query_init = torch.randn(
                            self.q_dim, self.d_model, generator=g,
                            dtype=torch.float32) / math.sqrt(self.d_model)
                        self.query_proj[layer].copy_(query_init.to(
                            device=device, dtype=dtype))
        else:
            self.query_proj = None
            self.query_a = nn.Parameter(
                zeros(NL, self.q_dim, query_rank))
            self.query_b = nn.Parameter(
                zeros(NL, query_rank, self.d_model))
            with torch.no_grad():
                for layer in range(NL):
                    init_b = torch.randn(
                        query_rank, self.d_model, generator=g,
                        dtype=torch.float32) / math.sqrt(self.d_model)
                    self.query_b[layer].copy_(
                        init_b.to(device=device, dtype=dtype))
        self.query_gate_lambda = query_gate_lambda
        self.query_gate_temperature = query_gate_temperature
        self.query_gate_threshold = query_gate_threshold
        self.query_gate_min_rank = query_gate_min_rank
        if query_mode not in {"backbone_delta", "independent"}:
            raise ValueError(f"unsupported query mode {query_mode}")
        self.query_mode = query_mode
        self.denom_mode = denom_mode
        if state_mode not in {"delta", "slots", "banked", "episodes"}:
            raise ValueError(f"unsupported memory state mode {state_mode}")
        if max_memory_slots < 1:
            raise ValueError("max_memory_slots must be positive")
        if slot_temperature <= 0.0:
            raise ValueError("slot_temperature must be positive")
        self.state_mode = state_mode
        self.max_memory_slots = max_memory_slots
        self.slot_temperature = slot_temperature
        if max_memory_episodes < 1:
            raise ValueError("max memory episodes must be positive")
        if episode_temperature <= 0.0:
            raise ValueError("episode temperature must be positive")
        if episode_read_mode not in ("soft", "hard"):
            raise ValueError(
                "episode read mode must be soft or hard")
        if episode_address_mode not in ("storage", "shared_query"):
            raise ValueError(
                "episode address mode must be storage or shared_query")
        if episode_address_granularity not in ("pooled", "tokens"):
            raise ValueError(
                "episode address granularity must be pooled or tokens")
        if episode_address_tokens < 1:
            raise ValueError("episode address tokens must be positive")
        if episode_payload_tokens < 1:
            raise ValueError("episode payload tokens must be positive")
        if (
            episode_payload_min_layer_votes < 1
            or episode_payload_min_layer_votes > self.n_layers
        ):
            raise ValueError(
                "episode payload minimum layer votes must fit layer count")
        if episode_address_source not in ("contextual", "embedding"):
            raise ValueError(
                "episode address source must be contextual or embedding")
        if (
            episode_address_source == "embedding"
            and episode_address_granularity != "tokens"
        ):
            raise ValueError(
                "embedding episode addresses require token granularity")
        if (
            episode_address_source == "embedding"
            and episode_router_mode != "cosine"
        ):
            raise ValueError(
                "embedding episode addresses currently use cosine routing")
        if not 0.0 <= episode_identity_uniqueness_floor <= 1.0:
            raise ValueError(
                "episode identity uniqueness floor must be in [0, 1]")
        if episode_identity_match_scale < 0.0:
            raise ValueError(
                "episode identity match scale must be non-negative")
        if (
            episode_identity_ngram < 1
            or episode_identity_ngram > episode_address_tokens
        ):
            raise ValueError(
                "episode identity ngram must fit the address token count")
        if episode_identity_route_mode not in ("maxsim", "span"):
            raise ValueError(
                "episode identity route mode must be maxsim or span")
        if episode_identity_selection_mode not in ("topk", "window"):
            raise ValueError(
                "episode identity selection mode must be topk or window")
        if episode_router_mode not in ("cosine", "token_metric"):
            raise ValueError(
                "episode router mode must be cosine or token_metric")
        if episode_router_rank < 1:
            raise ValueError("episode router rank must be positive")
        if (
            state_mode == "episodes"
            and episode_router_mode == "token_metric"
            and episode_router_rank > self.kv_dim
        ):
            raise ValueError("episode router rank is outside model geometry")
        if (
            episode_router_mode == "token_metric"
            and episode_address_granularity != "tokens"
        ):
            raise ValueError(
                "token_metric router requires token address granularity")
        self.max_memory_episodes = int(max_memory_episodes)
        self.episode_temperature = float(episode_temperature)
        self.episode_read_mode = episode_read_mode
        self.episode_address_mode = episode_address_mode
        self.episode_address_granularity = (
            episode_address_granularity)
        self.episode_address_tokens = int(episode_address_tokens)
        self.episode_payload_tokens = int(episode_payload_tokens)
        self.episode_payload_min_layer_votes = int(
            episode_payload_min_layer_votes)
        self.episode_address_source = episode_address_source
        self.episode_identity_uniqueness_floor = float(
            episode_identity_uniqueness_floor)
        self.episode_identity_match_scale = float(
            episode_identity_match_scale)
        self.episode_identity_ngram = int(episode_identity_ngram)
        self.episode_identity_route_mode = episode_identity_route_mode
        self.episode_identity_selection_mode = (
            episode_identity_selection_mode)
        self.episode_router_mode = episode_router_mode
        self.episode_router_rank = int(episode_router_rank)
        if (
            state_mode == "episodes"
            and episode_address_source == "contextual"
            and episode_address_mode == "storage"
            and (
                episode_address_rank < 0
                or episode_address_rank > min(
                    self.kv_dim, self.d_model)
            )
        ):
            raise ValueError("episode address rank is outside model geometry")
        self.episode_address_rank = (
            int(episode_address_rank)
            if (
                state_mode == "episodes"
                and episode_address_source == "contextual"
                and episode_address_mode == "storage"
            ) else 0)
        if (
            state_mode == "episodes"
            and episode_address_source == "contextual"
            and episode_address_mode == "storage"
            and self.episode_address_rank > 0
        ):
            self.episode_address_a = nn.Parameter(zeros(
                NL, self.kv_dim, self.episode_address_rank))
            address_b = torch.randn(
                NL, self.episode_address_rank, self.d_model,
                generator=g, dtype=torch.float32
            ) / math.sqrt(self.d_model)
            self.episode_address_b = nn.Parameter(
                address_b.to(device=device, dtype=dtype))
        else:
            self.episode_address_a = None
            self.episode_address_b = None
        if (
            state_mode == "episodes"
            and episode_router_mode == "token_metric"
        ):
            metric_scale = 1.0 / math.sqrt(self.kv_dim)
            self.episode_match_q_down = nn.Parameter(
                torch.randn(
                    NL, self.episode_router_rank, self.kv_dim,
                    generator=g, dtype=torch.float32
                ).mul_(metric_scale).to(device=device, dtype=dtype))
            self.episode_match_q_up = nn.Parameter(zeros(
                NL, self.kv_dim, self.episode_router_rank))
            self.episode_match_k_down = nn.Parameter(
                torch.randn(
                    NL, self.episode_router_rank, self.kv_dim,
                    generator=g, dtype=torch.float32
                ).mul_(metric_scale).to(device=device, dtype=dtype))
            self.episode_match_k_up = nn.Parameter(zeros(
                NL, self.kv_dim, self.episode_router_rank))
            self.episode_layer_logits = nn.Parameter(zeros(NL))
        else:
            self.episode_match_q_down = None
            self.episode_match_q_up = None
            self.episode_match_k_down = None
            self.episode_match_k_up = None
            self.episode_layer_logits = None
        if memory_banks < 1:
            raise ValueError("memory banks must be positive")
        if state_mode == "banked" and memory_banks < 2:
            raise ValueError("banked memory requires at least two banks")
        if state_mode != "banked" and memory_banks != 1:
            raise ValueError(
                "multiple memory banks require state mode banked")
        if bank_temperature <= 0.0:
            raise ValueError("bank temperature must be positive")
        if bank_write_top_k < 1 or bank_write_top_k > memory_banks:
            raise ValueError("bank write top-k is outside the bank count")
        self.memory_banks = int(memory_banks)
        self.bank_temperature = float(bank_temperature)
        self.bank_write_top_k = int(bank_write_top_k)
        if bank_read_mode not in ("soft", "hard"):
            raise ValueError("bank read mode must be soft or hard")
        self.bank_read_mode = bank_read_mode
        if bank_router_mode not in ("learned", "sequential"):
            raise ValueError(
                "bank router mode must be learned or sequential")
        if (
            bank_router_mode == "sequential"
            and bank_write_top_k != 1
        ):
            raise ValueError(
                "sequential bank allocation requires write top-k one")
        self.bank_router_mode = bank_router_mode
        if state_mode == "banked" and bank_router_mode == "learned":
            router = torch.randn(
                NL, self.memory_banks, self.d_model,
                generator=g, dtype=torch.float32
            ) / math.sqrt(self.d_model)
            self.bank_router_w = nn.Parameter(
                router.to(device=device, dtype=dtype))
        else:
            self.bank_router_w = None
        self.query_gate_logits = None
        if query_gate_lambda > 0.0 and query_rank == 0:
            raise ValueError("query rank gates require a factorized query")
        if query_gate_lambda > 0.0:
            # Global gates keep the selected rank components consistent
            # across all memory layers, allowing a compact common-rank
            # runtime model after thresholding.
            self.query_gate_logits = nn.Parameter(
                torch.full((query_rank,), 4.0, device=device, dtype=dtype))
        self.kv_gate_lambda = kv_gate_lambda
        self.kv_gate_temperature = kv_gate_temperature
        self.kv_gate_threshold = kv_gate_threshold
        self.kv_gate_min_rank = kv_gate_min_rank
        self.kv_gate_logits = None
        if kv_gate_lambda > 0.0 and kv_rank == 0:
            raise ValueError("K/V rank gates require factorized projections")
        if kv_gate_lambda > 0.0:
            # One shared bottleneck architecture keeps K and V at the same
            # searched rank and makes the deployed projection pair compact.
            self.kv_gate_logits = nn.Parameter(
                torch.full((kv_rank,), 4.0, device=device, dtype=dtype))
        self.layer_gate_lambda = layer_gate_lambda
        self.layer_gate_temperature = layer_gate_temperature
        self.layer_gate_threshold = layer_gate_threshold
        self.layer_gate_min_layers = layer_gate_min_layers
        self.layer_gate_logits = None
        self.layer_gate_override = None
        if layer_gate_lambda > 0.0:
            self.layer_gate_logits = nn.Parameter(
                torch.full((NL,), 4.0, device=device, dtype=dtype))
        self.query_norm = nn.Parameter(
            torch.ones(NL, self.head_dim, device=device, dtype=dtype))
        if self.query_mode == "independent" and self.q_rank > 0:
            with torch.no_grad():
                self.query_a.normal_(
                    mean=0.0, std=1.0 / math.sqrt(query_rank))
        # frozen backbone tensors we borrow every forward (registered as
        # buffers-less plain attrs: they live on the backbone, requires_grad
        # False; keeping them out of state_dict avoids double storage)
        self.attn_norm_w = torch.stack([
            backbone.layers[i]["attn_norm"] for i in self.layer_ids
        ]).to(dtype)                                            # [NL, d] fp32-ish
        self.rms_eps = cfg.rms_eps

        # runtime state (fresh tensors per sample)
        self.M = None   # [NL, kv, kv] or [NL, banks, kv, kv]
        self.S = None   # [NL, kv] or [NL, banks, kv]
        self.slot_K = None  # [NL, token slots, kv]
        self.slot_V = None  # [NL, token slots, kv]
        self.slot_labels = None  # [token slots], 1=evidence, 0=distractor
        self.episode_M = None
        self.episode_S = None
        self.episode_address = None
        self.episode_token_address = None
        self.episode_token_weight = None
        self.episode_identity_address = None
        self.episode_identity_weight = None
        self.episode_identity_token_ids = None
        self.episode_identity_positions = None
        self.episode_payload_token_ids = None
        self.episode_payload_positions = None
        self.episode_payload_votes = None
        self.episode_ids = []
        self.episode_route_override = None
        self.episode_activation_by_layer = [None] * self.n_layers
        self.last_episode_route_index = None
        self.last_episode_route_logits = None
        self.last_commit_episode_identity = None
        self.last_commit_episode_identity_weight = None
        self.last_commit_episode_token_ids = None
        self.last_commit_episode_positions = None
        self.last_commit_episode_payload_token_ids = None
        self.last_commit_episode_payload_positions = None
        self.last_commit_episode_payload_votes = None
        self.last_commit_episode_payload_source_ids = None
        self.pending_slot_label = -1
        self.pending_slot_labels = None
        self.pending_key_labels = None
        self.pending_value_labels = None
        self.pending_key_labels = None
        self.pending_value_labels = None
        self.collect_evidence_loss = False
        self.evidence_supervision_tail = None
        self.evidence_supervision_range = None
        self.evidence_losses = []
        self.collect_write_selection_loss = False
        self.write_selection_losses = []
        self.collect_retention_loss = False
        self.retention_losses = []
        self.collect_preservation_loss = False
        self.preservation_losses = []
        self.preservation_probe_keys = []
        self.write_address_basis = []
        self.collect_address_training = False
        self.address_losses = []
        self.key_diversity_losses = []
        self.address_correct = 0
        self.address_total = 0
        self.address_target_id = None
        self.address_token_range = None
        self.pending_memory_id = None
        self.address_ids = []
        self.address_keys = []
        self.last_commit_address_key_by_layer = [None] * self.n_layers
        self.last_commit_episode_tokens_by_layer = [
            None] * self.n_layers
        self.last_commit_episode_weights_by_layer = [
            None] * self.n_layers
        self.last_commit_address_transform_by_layer = [
            None] * self.n_layers
        self.last_commit_bank_route_by_layer = [None] * self.n_layers
        self.next_bank_index = 0
        self.current_write_bank = None
        self.last_commit_bank_route_by_layer = [None] * self.n_layers
        self.next_bank_index = 0
        self.current_write_bank = None
        self.collect_fusion_gate_loss = False
        self.fusion_gate_target = 0.0
        self.fusion_gate_losses = []
        self.last_pointer_weights = None
        self.last_pointer_weights_by_layer = [None] * self.n_layers
        self.last_memory_fused_by_layer = [None] * self.n_layers
        self.last_commit_stats_by_layer = [None] * self.n_layers
        self.last_read_stats_by_layer = [None] * self.n_layers
        self.query_read_override_by_layer = [None] * self.n_layers
        self.query_read_context_by_layer = [None] * self.n_layers
        self.collect_runtime_diagnostics = False
        self.pointer_token_ids = None
        # Reference/runtime semantics have no empty-state bypass: even before
        # the first commit the memory branch reads zero and the attention
        # branch is scaled by gamma.
        self.active = True
        self._captured = None    # list of per-layer [T, d] tensors
        self._captured_token_identity = None
        self._captured_token_ids = None

    def init_from_backbone(self):
        """Initialize memory projections from the matching backbone.

        The accuracy-first learned query follows the reference implementation:
        an independent query_proj(hidden_states), initialized from the
        backbone q_proj weights.  It must not reuse the backbone's live
        q_proj output because that output consumes input_layernorm(hidden),
        while the learned memory query consumes the raw block input.

        Full-rank K/V projections are exact copies. Factorized projections
        retain the legacy truncated-SVD initialization for compressed
        experiments.
        """
        with torch.no_grad():
            for s, blk in enumerate(self.layer_ids):
                lw = self.backbone.layers[blk]
                if self.query_mode == "independent":
                    if self.q_rank == 0:
                        self.query_proj[s].copy_(lw["q"].float())
                    else:
                        query_a, query_b = balanced_truncated_svd(
                            lw["q"].float(), self.q_rank)
                        self.query_a[s].copy_(query_a.to(
                            device=self.query_a.device,
                            dtype=self.query_a.dtype))
                        self.query_b[s].copy_(query_b.to(
                            device=self.query_b.device,
                            dtype=self.query_b.dtype))
                if self.kv_rank == 0:
                    self.wk[s].copy_(lw["k"].float())
                    self.wv[s].copy_(lw["v"].float())
                else:
                    for source, factor_a, factor_b in (
                        (lw["k"], self.wk_a, self.wk_b),
                        (lw["v"], self.wv_a, self.wv_b),
                    ):
                        a, b = balanced_truncated_svd(
                            source, self.kv_rank)
                        factor_a[s].copy_(a.to(
                            device=factor_a.device, dtype=factor_a.dtype))
                        factor_b[s].copy_(b.to(
                            device=factor_b.device, dtype=factor_b.dtype))

    def load_checkpoint(self, path, allow_query_rank_expand=False,
                        allow_selection_mismatch=False):
        checkpoint = load_bnmem_v1(path)
        expected_sha256 = checkpoint["backbone_sha256"]
        actual_sha256 = self.backbone.model_sha256()
        if expected_sha256 is None:
            raise ValueError(
                "checkpoint is not bound to a backbone model")
        if expected_sha256 != actual_sha256:
            raise ValueError(
                "checkpoint was trained for a different backbone model")
        expected_geometry = (
            self.d_model, self.kv_dim, self.q_dim, self.head_dim)
        actual_geometry = (
            checkpoint["d_model"], checkpoint["kv_dim"],
            checkpoint["q_dim"], checkpoint["head_dim"])
        if actual_geometry != expected_geometry:
            raise ValueError(
                f"checkpoint geometry {actual_geometry} != "
                f"trainer geometry {expected_geometry}")
        if checkpoint["layer_ids"] != self.layer_ids:
            raise ValueError(
                f"checkpoint layers {checkpoint['layer_ids']} != "
                f"trainer layers {self.layer_ids}")
        checkpoint_rank = checkpoint["query_rank"]
        rank_expanded = (
            allow_query_rank_expand and checkpoint_rank < self.q_rank)
        if checkpoint_rank != self.q_rank and not rank_expanded:
            raise ValueError(
                f"checkpoint query rank {checkpoint_rank} != "
                f"trainer rank {self.q_rank}")
        expected_denom = 2 if self.denom_mode == "abs_plus_one" else 1
        scalar_checks = {
            "gamma": self.gamma,
            "tau": self.tau,
            "beta_scale": self.beta_scale,
        }
        if not allow_selection_mismatch:
            scalar_checks["rho"] = self.rho
        for name, expected in scalar_checks.items():
            actual = checkpoint[name]
            if not math.isclose(
                actual, expected, rel_tol=1e-5, abs_tol=1e-6
            ):
                raise ValueError(
                    f"checkpoint {name} {actual} != trainer {expected}")
        if checkpoint["k_min"] != self.k_min:
            raise ValueError("checkpoint k_min does not match trainer")
        for name, expected in (
            ("alpha_max_tokens", self.alpha_max_tokens),
            ("alpha_max_fraction", self.alpha_max_fraction),
        ):
            actual = checkpoint.get(name, 0)
            if not math.isclose(
                float(actual), float(expected),
                rel_tol=1e-5, abs_tol=1e-6
            ):
                raise ValueError(
                    f"checkpoint {name} {actual} != trainer {expected}")
        if allow_selection_mismatch and not math.isclose(
            checkpoint["rho"], self.rho, rel_tol=1e-5, abs_tol=1e-6
        ):
            print(json.dumps({
                "phase": "memory_init",
                "selection_policy_changed": True,
                "checkpoint_rho": checkpoint["rho"],
                "training_rho": self.rho,
            }, separators=(",", ":")), flush=True)
        if checkpoint["denom_mode"] != expected_denom:
            raise ValueError("checkpoint denominator mode does not match")
        expected_add_backbone = self.query_mode == "backbone_delta"
        if checkpoint["query_add_backbone"] != expected_add_backbone:
            checkpoint_mode = (
                "backbone_delta"
                if checkpoint["query_add_backbone"]
                else "independent"
            )
            raise ValueError(
                f"checkpoint query mode {checkpoint_mode} != "
                f"trainer query mode {self.query_mode}")
        checkpoint_fusion = checkpoint.get("fusion_mode", "fixed")
        if checkpoint_fusion != self.fusion_mode:
            raise ValueError(
                f"checkpoint fusion mode {checkpoint_fusion} != "
                f"trainer fusion mode {self.fusion_mode}")
        checkpoint_tensors = dict(checkpoint["tensors"])
        if checkpoint["kv_rank"] > 0:
            checkpoint_tensors["wk"] = torch.matmul(
                checkpoint_tensors.pop("wk_a"),
                checkpoint_tensors.pop("wk_b"))
            checkpoint_tensors["wv"] = torch.matmul(
                checkpoint_tensors.pop("wv_a"),
                checkpoint_tensors.pop("wv_b"))
        with torch.no_grad():
            for name, value in checkpoint_tensors.items():
                if name in {"wk", "wv"}:
                    if self.kv_rank == 0:
                        target = getattr(self, name)
                        target.copy_(
                            value.to(
                                device=target.device, dtype=target.dtype))
                    else:
                        factor_a = getattr(self, name + "_a")
                        factor_b = getattr(self, name + "_b")
                        for layer in range(self.n_layers):
                            a, b = balanced_truncated_svd(
                                value[layer].to(device=self.device),
                                self.kv_rank)
                            factor_a[layer].copy_(
                                a.to(device=factor_a.device,
                                     dtype=factor_a.dtype))
                            factor_b[layer].copy_(
                                b.to(device=factor_b.device,
                                     dtype=factor_b.dtype))
                    continue
                target = getattr(self, name)
                if rank_expanded and name == "query_a":
                    target.zero_()
                    target[:, :, :checkpoint_rank].copy_(
                        value.to(device=target.device, dtype=target.dtype))
                    continue
                if rank_expanded and name == "query_b":
                    target[:, :checkpoint_rank, :].copy_(
                        value.to(device=target.device, dtype=target.dtype))
                    continue
                if tuple(target.shape) != tuple(value.shape):
                    raise ValueError(
                        f"checkpoint tensor {name} shape {tuple(value.shape)} "
                        f"!= trainer shape {tuple(target.shape)}")
                target.copy_(
                    value.to(device=target.device, dtype=target.dtype))
            if self.episode_identity_selector_w is not None:
                self.episode_identity_selector_w.copy_(self.w_agg[0])
            if (
                self.state_mode == "banked"
                and self.bank_router_mode == "learned"
            ):
                router_path = path + ".bankrouter.pt"
                if os.path.isfile(router_path):
                    payload = torch.load(
                        router_path, map_location="cpu",
                        weights_only=True)
                    if int(payload.get("memory_banks", 0)) != (
                        self.memory_banks
                    ):
                        raise ValueError(
                            "bank router count does not match trainer")
                    if not math.isclose(
                        float(payload.get("bank_temperature", -1.0)),
                        self.bank_temperature,
                        rel_tol=1e-6, abs_tol=1e-8,
                    ):
                        raise ValueError(
                            "bank router temperature does not match trainer")
                    if int(payload.get("bank_write_top_k", 1)) != (
                        self.bank_write_top_k
                    ):
                        raise ValueError(
                            "bank router write top-k does not match trainer")
                    router = payload["bank_router_w"]
                    if tuple(router.shape) != tuple(
                        self.bank_router_w.shape
                    ):
                        raise ValueError(
                            "bank router tensor shape does not match trainer")
                    self.bank_router_w.copy_(
                        router.to(
                            device=self.bank_router_w.device,
                            dtype=self.bank_router_w.dtype))
                    print(json.dumps({
                        "phase": "memory_init",
                        "bank_router": router_path,
                        "memory_banks": self.memory_banks,
                    }, separators=(",", ":")), flush=True)
            if self.write_mode in ("factorized", "dual_tokens"):
                selector_path = path + ".write_selector.pt"
                if os.path.isfile(selector_path):
                    payload = torch.load(
                        selector_path, map_location="cpu",
                        weights_only=True)
                    selector = payload["w_value_agg"]
                    if tuple(selector.shape) != tuple(
                        self.w_value_agg.shape
                    ):
                        raise ValueError(
                            "value selector shape does not match trainer")
                    if int(payload.get(
                        "value_alpha_max_tokens",
                        self.value_alpha_max_tokens,
                    )) != self.value_alpha_max_tokens:
                        raise ValueError(
                            "value selector token cap does not match trainer")
                    if not math.isclose(
                        float(payload.get(
                            "value_alpha_max_fraction",
                            self.value_alpha_max_fraction,
                        )),
                        self.value_alpha_max_fraction,
                        rel_tol=1e-6, abs_tol=1e-8,
                    ):
                        raise ValueError(
                            "value selector fraction cap does not match "
                            "trainer")
                    if not math.isclose(
                        float(payload.get(
                            "write_orthogonalization",
                            self.write_orthogonalization,
                        )),
                        self.write_orthogonalization,
                        rel_tol=1e-6, abs_tol=1e-8,
                    ):
                        raise ValueError(
                            "write orthogonalization does not match trainer")
                    if not math.isclose(
                        float(payload.get(
                            "update_similarity_threshold",
                            self.update_similarity_threshold,
                        )),
                        self.update_similarity_threshold,
                        rel_tol=1e-6, abs_tol=1e-8,
                    ):
                        raise ValueError(
                            "update similarity threshold does not match "
                            "trainer")
                    self.w_value_agg.copy_(
                        selector.to(
                            device=self.w_value_agg.device,
                            dtype=self.w_value_agg.dtype))
                    print(json.dumps({
                        "phase": "memory_init",
                        "write_selector": selector_path,
                    }, separators=(",", ":")), flush=True)
                else:
                    # Start a two-selector experiment from a legacy checkpoint
                    # without changing its initial write-token distribution.
                    self.w_value_agg.copy_(self.w_agg)
            if self.state_mode == "episodes":
                router_path = path + ".episode_router.pt"
                if os.path.isfile(router_path):
                    payload = torch.load(
                        router_path, map_location="cpu",
                        weights_only=True)
                    if (
                        self.episode_address_a is not None
                        and "episode_address_a" in payload
                    ):
                        if int(payload.get(
                            "episode_address_rank", -1
                        )) != self.episode_address_rank:
                            raise ValueError(
                                "episode address rank does not match trainer")
                        for name in (
                            "episode_address_a", "episode_address_b"
                        ):
                            source = payload[name]
                            target = getattr(self, name)
                            if tuple(source.shape) != tuple(target.shape):
                                raise ValueError(
                                    f"{name} shape does not match trainer")
                            target.copy_(source.to(
                                device=target.device, dtype=target.dtype))
                    if self.episode_router_mode == "token_metric":
                        if payload.get(
                            "episode_router_mode"
                        ) != self.episode_router_mode:
                            raise ValueError(
                                "episode router mode does not match trainer")
                        if int(payload.get(
                            "episode_router_rank", -1
                        )) != self.episode_router_rank:
                            raise ValueError(
                                "episode router rank does not match trainer")
                        for name in (
                            "episode_match_q_down",
                            "episode_match_q_up",
                            "episode_match_k_down",
                            "episode_match_k_up",
                            "episode_layer_logits",
                        ):
                            source = payload[name]
                            target = getattr(self, name)
                            if tuple(source.shape) != tuple(target.shape):
                                raise ValueError(
                                    f"{name} shape does not match trainer")
                            target.copy_(source.to(
                                device=target.device, dtype=target.dtype))
                    if (
                        self.episode_identity_selector_w is not None
                        and "episode_identity_selector_w" in payload
                    ):
                        source = payload[
                            "episode_identity_selector_w"]
                        target = self.episode_identity_selector_w
                        if tuple(source.shape) != tuple(target.shape):
                            raise ValueError(
                                "episode identity selector shape does not "
                                "match trainer")
                        target.copy_(source.to(
                            device=target.device, dtype=target.dtype))
                    print(json.dumps({
                        "phase": "memory_init",
                        "episode_router": router_path,
                    }, separators=(",", ":")), flush=True)

    # ---- state control ----
    def reset_state(self):
        # drop the old tensors FIRST (frees any autograd graph still attached
        # to M/S before allocating the fresh zeros -- on a fragmented GPU the
        # zeros allocation itself can OOM if the old graph is still resident)
        self.M = None
        self.S = None
        if self.state_mode == "banked":
            self.M = torch.zeros(
                self.n_layers, self.memory_banks,
                self.kv_dim, self.kv_dim,
                device=self.device, dtype=self.dtype)
            self.S = torch.zeros(
                self.n_layers, self.memory_banks, self.kv_dim,
                device=self.device, dtype=self.dtype)
        else:
            self.M = torch.zeros(
                self.n_layers, self.kv_dim, self.kv_dim,
                device=self.device, dtype=self.dtype)
            self.S = torch.zeros(
                self.n_layers, self.kv_dim,
                device=self.device, dtype=self.dtype)
        self.slot_K = torch.empty(
            self.n_layers, 0, self.kv_dim,
            device=self.device, dtype=self.dtype)
        self.slot_V = torch.empty(
            self.n_layers, 0, self.kv_dim,
            device=self.device, dtype=self.dtype)
        self.slot_labels = torch.empty(
            0, device=self.device, dtype=torch.int8)
        self.episode_M = torch.empty(
            self.n_layers, 0, self.kv_dim, self.kv_dim,
            device=self.device, dtype=self.dtype)
        self.episode_S = torch.empty(
            self.n_layers, 0, self.kv_dim,
            device=self.device, dtype=self.dtype)
        self.episode_address = torch.empty(
            self.n_layers, 0, self.kv_dim,
            device=self.device, dtype=self.dtype)
        self.episode_token_address = torch.empty(
            self.n_layers, 0, self.episode_address_tokens, self.kv_dim,
            device=self.device, dtype=self.dtype)
        self.episode_token_weight = torch.empty(
            self.n_layers, 0, self.episode_address_tokens,
            device=self.device, dtype=self.dtype)
        self.episode_identity_address = torch.empty(
            0, self.episode_address_tokens, self.d_model,
            device=self.device, dtype=self.dtype)
        self.episode_identity_weight = torch.empty(
            0, self.episode_address_tokens,
            device=self.device, dtype=self.dtype)
        self.episode_identity_token_ids = torch.empty(
            0, self.episode_address_tokens,
            device=self.device, dtype=torch.long)
        self.episode_identity_positions = torch.empty(
            0, self.episode_address_tokens,
            device=self.device, dtype=torch.long)
        self.episode_payload_token_ids = torch.empty(
            0, self.episode_payload_tokens,
            device=self.device, dtype=torch.long)
        self.episode_payload_positions = torch.empty(
            0, self.episode_payload_tokens,
            device=self.device, dtype=torch.long)
        self.episode_payload_votes = torch.empty(
            0, self.episode_payload_tokens,
            device=self.device, dtype=torch.int16)
        self.episode_ids = []
        self.episode_route_override = None
        self.episode_activation_by_layer = [None] * self.n_layers
        self.last_episode_route_index = None
        self.last_episode_route_logits = None
        self.pending_slot_label = -1
        self.pending_slot_labels = None
        self.collect_evidence_loss = False
        self.evidence_supervision_tail = None
        self.evidence_supervision_range = None
        self.evidence_losses = []
        self.collect_write_selection_loss = False
        self.write_selection_losses = []
        self.collect_retention_loss = False
        self.retention_losses = []
        self.collect_preservation_loss = False
        self.preservation_losses = []
        self.preservation_probe_keys = []
        self.write_address_basis = []
        self.collect_address_training = False
        self.address_losses = []
        self.key_diversity_losses = []
        self.address_correct = 0
        self.address_total = 0
        self.address_target_id = None
        self.address_token_range = None
        self.pending_memory_id = None
        self.address_ids = []
        self.address_keys = []
        self.last_commit_address_key_by_layer = [None] * self.n_layers
        self.last_commit_episode_tokens_by_layer = [
            None] * self.n_layers
        self.last_commit_episode_weights_by_layer = [
            None] * self.n_layers
        self.last_commit_episode_identity = None
        self.last_commit_episode_identity_weight = None
        self.last_commit_episode_token_ids = None
        self.last_commit_episode_positions = None
        self.last_commit_episode_payload_token_ids = None
        self.last_commit_episode_payload_positions = None
        self.last_commit_episode_payload_votes = None
        self.last_commit_episode_payload_source_ids = None
        self.last_commit_address_transform_by_layer = [
            None] * self.n_layers
        self.last_commit_bank_route_by_layer = [None] * self.n_layers
        self.collect_fusion_gate_loss = False
        self.fusion_gate_target = 0.0
        self.fusion_gate_losses = []
        self.last_pointer_weights = None
        self.last_pointer_weights_by_layer = [None] * self.n_layers
        self.last_memory_fused_by_layer = [None] * self.n_layers
        self.last_commit_stats_by_layer = [None] * self.n_layers
        self.last_read_stats_by_layer = [None] * self.n_layers
        self.query_read_override_by_layer = [None] * self.n_layers
        self.query_read_context_by_layer = [None] * self.n_layers
        self.pointer_token_ids = None
        # Keep reset behavior identical to the C runtime and official Metis:
        # zero the state but continue applying the reweighted memory block.
        self.active = True
        self._captured = None
        self._captured_token_identity = None
        self._captured_token_ids = None

    def clone_runtime_state(self):
        """Clone dynamic memory data for repeatable no-KV evaluation."""
        return {
            "M": self.M.clone(),
            "S": self.S.clone(),
            "slot_K": self.slot_K.clone(),
            "slot_V": self.slot_V.clone(),
            "slot_labels": self.slot_labels.clone(),
            "episode_M": self.episode_M.clone(),
            "episode_S": self.episode_S.clone(),
            "episode_address": self.episode_address.clone(),
            "episode_token_address": (
                self.episode_token_address.clone()),
            "episode_token_weight": (
                self.episode_token_weight.clone()),
            "episode_identity_address": (
                self.episode_identity_address.clone()),
            "episode_identity_weight": (
                self.episode_identity_weight.clone()),
            "episode_identity_token_ids": (
                self.episode_identity_token_ids.clone()),
            "episode_identity_positions": (
                self.episode_identity_positions.clone()),
            "episode_payload_token_ids": (
                self.episode_payload_token_ids.clone()),
            "episode_payload_positions": (
                self.episode_payload_positions.clone()),
            "episode_payload_votes": (
                self.episode_payload_votes.clone()),
            "episode_ids": list(self.episode_ids),
            "next_bank_index": self.next_bank_index,
            "active": self.active,
        }

    def restore_runtime_state(self, state):
        self.M = state["M"].clone()
        self.S = state["S"].clone()
        self.slot_K = state["slot_K"].clone()
        self.slot_V = state["slot_V"].clone()
        self.slot_labels = state["slot_labels"].clone()
        self.episode_M = state["episode_M"].clone()
        self.episode_S = state["episode_S"].clone()
        self.episode_address = state["episode_address"].clone()
        self.episode_token_address = (
            state["episode_token_address"].clone())
        self.episode_token_weight = (
            state["episode_token_weight"].clone())
        self.episode_identity_address = (
            state["episode_identity_address"].clone())
        self.episode_identity_weight = (
            state["episode_identity_weight"].clone())
        self.episode_identity_token_ids = (
            state["episode_identity_token_ids"].clone())
        self.episode_identity_positions = (
            state["episode_identity_positions"].clone())
        self.episode_payload_token_ids = (
            state["episode_payload_token_ids"].clone())
        self.episode_payload_positions = (
            state["episode_payload_positions"].clone())
        self.episode_payload_votes = (
            state["episode_payload_votes"].clone())
        self.episode_ids = list(state.get("episode_ids", []))
        self.episode_route_override = None
        self.episode_activation_by_layer = [None] * self.n_layers
        self.last_episode_route_index = None
        self.last_episode_route_logits = None
        self.next_bank_index = int(state.get("next_bank_index", 0))
        self.current_write_bank = None
        self.active = state["active"]
        self.discard_captured()

    def set_episode_route_override(self, episode_index):
        if episode_index is None:
            self.episode_route_override = None
            return
        episode_index = int(episode_index)
        if (
            episode_index < 0
            or episode_index >= self.episode_M.shape[1]
        ):
            raise ValueError("episode route override is out of range")
        self.episode_route_override = episode_index

    def set_pending_slot_label(self, label):
        self.pending_slot_label = int(label)
        self.pending_slot_labels = None

    def set_pending_slot_labels(self, labels):
        labels = torch.as_tensor(
            labels, device=self.device, dtype=torch.int8).reshape(-1)
        if bool(torch.any((labels < -1) | (labels > 1))):
            raise ValueError("slot labels must be -1, 0, or 1")
        self.pending_slot_labels = labels
        self.pending_slot_label = -1

    def set_pending_factorized_labels(self, key_labels, value_labels):
        self.pending_key_labels = torch.as_tensor(
            key_labels, device=self.device, dtype=torch.int8).reshape(-1)
        self.pending_value_labels = torch.as_tensor(
            value_labels, device=self.device, dtype=torch.int8).reshape(-1)
        if self.pending_key_labels.shape != self.pending_value_labels.shape:
            raise ValueError(
                "factorized key/value labels must have the same shape")

    def set_pending_memory_id(self, memory_id):
        self.pending_memory_id = (
            None if memory_id is None else int(memory_id))

    def begin_address_training(self):
        self.collect_address_training = True
        self.address_losses = []
        self.key_diversity_losses = []
        self.address_correct = 0
        self.address_total = 0

    def set_address_target(self, memory_id, token_range):
        self.address_target_id = (
            None if memory_id is None else int(memory_id))
        self.address_token_range = (
            None if token_range is None
            else (int(token_range[0]), int(token_range[1])))

    def clear_address_target(self):
        self.address_target_id = None
        self.address_token_range = None

    def end_address_training(self):
        self.collect_address_training = False
        self.clear_address_target()
        address_loss = (
            torch.stack(self.address_losses).mean()
            if self.address_losses
            else torch.zeros((), device=self.device))
        diversity_loss = (
            torch.stack(self.key_diversity_losses).mean()
            if self.key_diversity_losses
            else torch.zeros((), device=self.device))
        self.address_losses = []
        self.key_diversity_losses = []
        correct = self.address_correct
        total = self.address_total
        self.address_correct = 0
        self.address_total = 0
        return address_loss, diversity_loss, correct, total

    def begin_evidence_supervision(self, tail_tokens=None,
                                   token_range=None):
        if tail_tokens is not None and token_range is not None:
            raise ValueError(
                "evidence supervision accepts tail_tokens or token_range")
        self.evidence_losses = []
        self.evidence_supervision_tail = (
            None if tail_tokens is None else int(tail_tokens))
        self.evidence_supervision_range = (
            None if token_range is None
            else (int(token_range[0]), int(token_range[1])))
        self.collect_evidence_loss = True

    def end_evidence_supervision(self):
        self.collect_evidence_loss = False
        self.evidence_supervision_tail = None
        self.evidence_supervision_range = None
        if not self.evidence_losses:
            return torch.zeros((), device=self.device)
        result = torch.stack(self.evidence_losses).mean()
        self.evidence_losses = []
        return result

    def begin_write_selection_supervision(self):
        self.write_selection_losses = []
        self.collect_write_selection_loss = True

    def end_write_selection_supervision(self):
        self.collect_write_selection_loss = False
        if not self.write_selection_losses:
            return torch.zeros((), device=self.device)
        result = torch.stack(self.write_selection_losses).mean()
        self.write_selection_losses = []
        return result

    def begin_retention_supervision(self):
        self.retention_losses = []
        self.collect_retention_loss = True

    def end_retention_supervision(self):
        self.collect_retention_loss = False
        if not self.retention_losses:
            return torch.zeros((), device=self.device)
        result = torch.stack(self.retention_losses).mean()
        self.retention_losses = []
        return result

    def begin_preservation_supervision(self):
        self.preservation_losses = []
        self.collect_preservation_loss = True

    def end_preservation_supervision(self):
        self.collect_preservation_loss = False
        if not self.preservation_losses:
            return torch.zeros((), device=self.device)
        result = torch.stack(self.preservation_losses).mean()
        self.preservation_losses = []
        return result

    def begin_fusion_gate_supervision(self, target):
        self.fusion_gate_losses = []
        self.fusion_gate_target = float(target)
        self.collect_fusion_gate_loss = True

    def end_fusion_gate_supervision(self):
        self.collect_fusion_gate_loss = False
        if not self.fusion_gate_losses:
            return torch.zeros((), device=self.device)
        result = torch.stack(self.fusion_gate_losses).mean()
        self.fusion_gate_losses = []
        return result

    def slot_of(self, blk):
        """Backbone block index -> memory slot (None when not a memory
        layer; all 32 are memory layers but keep the general shape)."""
        if not hasattr(self, "_slot_map"):
            self._slot_map = {b: s for s, b in enumerate(self.layer_ids)}
        return self._slot_map.get(blk)

    def capture(self, per_layer_hn):
        """per_layer_hn: list of NL tensors-or-None ([T, d] attn-normed rows
        of the current chunk at each memory layer; the backbone fills every
        slot on each call). Accumulates per layer until a commit consumes it
        (C1 semantics). Rows keep their autograd graph so write params get
        credit through the state they leave."""
        if self._captured is None:
            self._captured = list(per_layer_hn)
        else:
            self._captured = [
                None if (a is None and b is None)
                else (a if b is None else (b if a is None
                                           else torch.cat([a, b], dim=0)))
                for a, b in zip(self._captured, per_layer_hn)
            ]

    def capture_token_identity(self, token_rows):
        """Capture input-embedding rows aligned with the current chunk."""
        if self._captured_token_identity is None:
            self._captured_token_identity = token_rows
        else:
            self._captured_token_identity = torch.cat(
                (self._captured_token_identity, token_rows), dim=0)

    def capture_token_ids(self, token_ids):
        """Capture token ids only for address-selection diagnostics."""
        if self._captured_token_ids is None:
            self._captured_token_ids = token_ids
        else:
            self._captured_token_ids = torch.cat(
                (self._captured_token_ids, token_ids), dim=0)

    def take_captured(self):
        return self._captured

    def discard_captured(self):
        self._captured = None
        self._captured_token_identity = None
        self._captured_token_ids = None

    def _finalize_address_commit(self):
        memory_id = self.pending_memory_id
        self.pending_memory_id = None
        if any(
            key is None
            for key in self.last_commit_address_key_by_layer
        ):
            return
        keys = torch.stack(self.last_commit_address_key_by_layer)
        # Runtime interference protection is part of every committed write;
        # it must not depend on training-only memory ids or annotations.
        if self.write_orthogonalization > 0.0:
            self.write_address_basis.append(keys.detach())
        if memory_id is None:
            return
        # These immutable probes represent the read locations that existed
        # before later writes. They are deliberately not transformed with
        # the delta update: the deployment query representation does not
        # change merely because another memory was committed.
        self.preservation_probe_keys.append(keys.detach())
        if any(
            transform is None
            for transform in self.last_commit_address_transform_by_layer
        ):
            return
        transforms = torch.stack(
            self.last_commit_address_transform_by_layer)
        # New writes transform every contribution already present in M/S.
        # Keep the training-only address trace in the same effective space.
        self.address_keys = [
            F.normalize(
                torch.einsum("lij,lj->li", transforms, previous),
                dim=-1, eps=1e-12)
            for previous in self.address_keys
        ]
        if self.collect_address_training and self.address_keys:
            previous = torch.stack(self.address_keys)
            self.key_diversity_losses.append(
                memory_key_diversity_loss(
                    keys, previous, self.key_diversity_margin))
        self.address_ids.append(memory_id)
        self.address_keys.append(keys)

    def _finalize_bank_commit(self):
        """Record the neural bank chosen for one committed memory."""
        memory_id = self.pending_memory_id
        self.pending_memory_id = None
        if memory_id is None:
            return
        if any(
            route is None
            for route in self.last_commit_bank_route_by_layer
        ):
            return
        routes = torch.stack(self.last_commit_bank_route_by_layer)
        if self.collect_address_training and self.address_keys:
            previous = torch.stack(self.address_keys)
            self.key_diversity_losses.append(
                memory_key_diversity_loss(
                    routes, previous, self.key_diversity_margin))
        self.address_ids.append(memory_id)
        self.address_keys.append(routes)

    def clear_query_read_override(self):
        self.query_read_override_by_layer = [None] * self.n_layers
        self.query_read_context_by_layer = [None] * self.n_layers
        self.episode_activation_by_layer = [None] * self.n_layers

    def prepare_query_read_override(self, tail_tokens):
        """Build one frozen memory read per layer from a query-only prefill.

        The first pass runs with fusion disabled and leaves raw per-layer
        query activations in ``_captured``. Each layer reads committed
        memory once from the final query-token window; the pooled result is
        reused throughout the second prefill/decode pass. No dynamic memory
        state is written or changed here.
        """
        caps = self._captured
        if caps is None:
            raise RuntimeError("two-pass query read requires captured rows")
        if tail_tokens < 1:
            raise ValueError("two-pass query tail must be positive")
        overrides = []
        for slot, rows in enumerate(caps):
            if rows is None or rows.shape[0] == 0:
                overrides.append(None)
                continue
            query_rows = rows[-min(tail_tokens, rows.shape[0]):]
            overrides.append(
                self.read(slot, query_rows).mean(dim=0, keepdim=True))
        self.query_read_override_by_layer = overrides
        self.discard_captured()

    def prepare_query_read_context(self, tail_tokens):
        """Pool query context for dynamic token-by-token M/S activation."""
        caps = self._captured
        if caps is None:
            raise RuntimeError(
                "contextual query read requires captured rows")
        if tail_tokens < 1:
            raise ValueError("contextual query tail must be positive")
        contexts = []
        for rows in caps:
            if rows is None or rows.shape[0] == 0:
                contexts.append(None)
                continue
            query_rows = rows[-min(tail_tokens, rows.shape[0]):]
            contexts.append(
                query_rows.float().mean(dim=0, keepdim=True))
        self.query_read_context_by_layer = contexts
        self.discard_captured()

    def prepare_episode_activation(self, tail_tokens):
        """Activate latent episodes from query prefill, then freeze routing."""
        if self.state_mode != "episodes":
            raise RuntimeError(
                "episode activation requires episode state")
        caps = self._captured
        if caps is None:
            raise RuntimeError(
                "episode activation requires captured query rows")
        if tail_tokens < 1:
            raise ValueError("episode activation tail must be positive")
        layer_logits = []
        valid_slots = []
        global_logits = None
        self.last_episode_route_index = None
        self.last_episode_route_logits = None
        episode_count = self.episode_M.shape[1]
        if (
            self.episode_address_source == "embedding"
            and episode_count > 0
        ):
            identity_rows = self._captured_token_identity
            if identity_rows is None or identity_rows.shape[0] == 0:
                raise RuntimeError(
                    "embedding episode activation requires token identities")
            query_identity = identity_rows[
                -min(tail_tokens, identity_rows.shape[0]):]
            global_logits = self._episode_identity_route_logits(
                query_identity)
            valid_slots = list(range(self.n_layers))
        else:
            for slot, rows in enumerate(caps):
                if (
                    rows is None
                    or rows.shape[0] == 0
                    or episode_count == 0
                ):
                    continue
                query_rows = rows[-min(tail_tokens, rows.shape[0]):]
                query_groups = self._query_groups(slot, query_rows)
                logits = self._episode_route_logits(slot, query_groups)
                layer_logits.append(logits)
                valid_slots.append(slot)
        activations = [None] * self.n_layers
        if layer_logits and global_logits is None:
            stacked_logits = torch.stack(layer_logits)
            if self.episode_layer_logits is not None:
                layer_indices = torch.tensor(
                    valid_slots, device=stacked_logits.device,
                    dtype=torch.long)
                layer_weights = F.softmax(
                    self.episode_layer_logits[layer_indices].float(),
                    dim=0)
                global_logits = torch.einsum(
                    "l,le->e", layer_weights, stacked_logits)
            else:
                global_logits = stacked_logits.mean(dim=0)
        if global_logits is not None:
            probabilities = F.softmax(global_logits, dim=-1)
            self.last_episode_route_index = int(
                global_logits.detach().argmax().item())
            self.last_episode_route_logits = (
                global_logits.detach().float().cpu())
            activation = (
                straight_through_top1(probabilities)
                if self.episode_read_mode == "hard"
                else probabilities)
            for slot in valid_slots:
                activations[slot] = activation
            if (
                self.collect_address_training
                and self.address_target_id is not None
                and self.address_target_id in self.episode_ids
            ):
                target_index = max(
                    index for index, memory_id
                    in enumerate(self.episode_ids)
                    if memory_id == self.address_target_id)
                target = torch.tensor(
                    [target_index], device=global_logits.device,
                    dtype=torch.long)
                self.address_losses.append(
                    F.cross_entropy(
                        global_logits.unsqueeze(0), target))
                with torch.no_grad():
                    self.address_correct += int(
                        global_logits.argmax().item() == target_index)
                    self.address_total += 1
        self.episode_activation_by_layer = activations
        self.discard_captured()

    def _append_episode_state(self, memory, normalizer):
        if any(
            key is None
            for key in self.last_commit_address_key_by_layer
        ):
            raise RuntimeError(
                "episode commit did not produce every layer address")
        address = torch.stack(
            self.last_commit_address_key_by_layer, dim=0)
        if any(
            value is None
            for value in self.last_commit_episode_tokens_by_layer
        ) or any(
            value is None
            for value in self.last_commit_episode_weights_by_layer
        ):
            raise RuntimeError(
                "episode commit did not produce token addresses")
        token_address = torch.stack(
            self.last_commit_episode_tokens_by_layer, dim=0)
        token_weight = torch.stack(
            self.last_commit_episode_weights_by_layer, dim=0)
        if self.episode_address_source == "embedding":
            if (
                self.last_commit_episode_identity is None
                or self.last_commit_episode_identity_weight is None
                or self.last_commit_episode_token_ids is None
                or self.last_commit_episode_positions is None
            ):
                raise RuntimeError(
                    "episode commit did not produce identity addresses")
            identity_address = self.last_commit_episode_identity
            identity_weight = self.last_commit_episode_identity_weight
            identity_token_ids = self.last_commit_episode_token_ids
            identity_positions = self.last_commit_episode_positions
        else:
            identity_address = torch.zeros(
                self.episode_address_tokens, self.d_model,
                device=memory.device, dtype=memory.dtype)
            identity_weight = torch.zeros(
                self.episode_address_tokens,
                device=memory.device, dtype=memory.dtype)
            identity_token_ids = torch.full(
                (self.episode_address_tokens,), -1,
                device=memory.device, dtype=torch.long)
            identity_positions = torch.full(
                (self.episode_address_tokens,), -1,
                device=memory.device, dtype=torch.long)
        if (
            self.last_commit_episode_payload_votes is not None
            and self.last_commit_episode_payload_source_ids is not None
        ):
            positive_indices = torch.nonzero(
                self.last_commit_episode_payload_votes
                >= self.episode_payload_min_layer_votes,
                as_tuple=False,
            ).flatten()
            if positive_indices.numel() > self.episode_payload_tokens:
                vote_order = torch.argsort(
                    self.last_commit_episode_payload_votes[
                        positive_indices],
                    descending=True,
                    stable=True,
                )[:self.episode_payload_tokens]
                positive_indices = positive_indices[vote_order]
            positive_indices = torch.sort(positive_indices).values
            payload_token_ids = (
                self.last_commit_episode_payload_source_ids[
                    positive_indices])
            payload_positions = positive_indices
            payload_votes = self.last_commit_episode_payload_votes[
                positive_indices]
            if positive_indices.numel() < self.episode_payload_tokens:
                pad_count = (
                    self.episode_payload_tokens
                    - positive_indices.numel())
                payload_token_ids = torch.cat((
                    payload_token_ids,
                    torch.full(
                        (pad_count,), -1,
                        device=memory.device, dtype=torch.long),
                ))
                payload_positions = torch.cat((
                    payload_positions,
                    torch.full(
                        (pad_count,), -1,
                        device=memory.device, dtype=torch.long),
                ))
                payload_votes = torch.cat((
                    payload_votes,
                    torch.zeros(
                        pad_count,
                        device=memory.device, dtype=torch.int16),
                ))
            self.last_commit_episode_payload_token_ids = payload_token_ids
            self.last_commit_episode_payload_positions = payload_positions
        elif (
            self.last_commit_episode_payload_token_ids is None
            or self.last_commit_episode_payload_positions is None
        ):
            payload_token_ids = torch.full(
                (self.episode_payload_tokens,), -1,
                device=memory.device, dtype=torch.long)
            payload_positions = torch.full(
                (self.episode_payload_tokens,), -1,
                device=memory.device, dtype=torch.long)
            payload_votes = torch.zeros(
                self.episode_payload_tokens,
                device=memory.device, dtype=torch.int16)
        else:
            payload_token_ids = (
                self.last_commit_episode_payload_token_ids)
            payload_positions = (
                self.last_commit_episode_payload_positions)
            payload_votes = torch.zeros(
                self.episode_payload_tokens,
                device=memory.device, dtype=torch.int16)
        self.episode_M = torch.cat(
            (self.episode_M, memory.unsqueeze(1)), dim=1)
        self.episode_S = torch.cat(
            (self.episode_S, normalizer.unsqueeze(1)), dim=1)
        self.episode_address = torch.cat(
            (self.episode_address, address.unsqueeze(1)), dim=1)
        self.episode_token_address = torch.cat((
            self.episode_token_address,
            token_address.unsqueeze(1),
        ), dim=1)
        self.episode_token_weight = torch.cat((
            self.episode_token_weight,
            token_weight.unsqueeze(1),
        ), dim=1)
        self.episode_identity_address = torch.cat((
            self.episode_identity_address,
            identity_address.unsqueeze(0),
        ), dim=0)
        self.episode_identity_weight = torch.cat((
            self.episode_identity_weight,
            identity_weight.unsqueeze(0),
        ), dim=0)
        self.episode_identity_token_ids = torch.cat((
            self.episode_identity_token_ids,
            identity_token_ids.unsqueeze(0),
        ), dim=0)
        self.episode_identity_positions = torch.cat((
            self.episode_identity_positions,
            identity_positions.unsqueeze(0),
        ), dim=0)
        self.episode_payload_token_ids = torch.cat((
            self.episode_payload_token_ids,
            payload_token_ids.unsqueeze(0),
        ), dim=0)
        self.episode_payload_positions = torch.cat((
            self.episode_payload_positions,
            payload_positions.unsqueeze(0),
        ), dim=0)
        self.episode_payload_votes = torch.cat((
            self.episode_payload_votes,
            payload_votes.unsqueeze(0),
        ), dim=0)
        self.episode_ids.append(self.pending_memory_id)
        if self.episode_M.shape[1] > self.max_memory_episodes:
            self.episode_M = self.episode_M[
                :, -self.max_memory_episodes:]
            self.episode_S = self.episode_S[
                :, -self.max_memory_episodes:]
            self.episode_address = self.episode_address[
                :, -self.max_memory_episodes:]
            self.episode_token_address = self.episode_token_address[
                :, -self.max_memory_episodes:]
            self.episode_token_weight = self.episode_token_weight[
                :, -self.max_memory_episodes:]
            self.episode_identity_address = (
                self.episode_identity_address[
                    -self.max_memory_episodes:])
            self.episode_identity_weight = (
                self.episode_identity_weight[
                    -self.max_memory_episodes:])
            self.episode_identity_token_ids = (
                self.episode_identity_token_ids[
                    -self.max_memory_episodes:])
            self.episode_identity_positions = (
                self.episode_identity_positions[
                    -self.max_memory_episodes:])
            self.episode_payload_token_ids = (
                self.episode_payload_token_ids[
                    -self.max_memory_episodes:])
            self.episode_payload_positions = (
                self.episode_payload_positions[
                    -self.max_memory_episodes:])
            self.episode_payload_votes = (
                self.episode_payload_votes[
                    -self.max_memory_episodes:])
            self.episode_ids = self.episode_ids[
                -self.max_memory_episodes:]
        self.pending_memory_id = None

    def commit_all(self):
        """Commit every layer's captured rows (no-grad path, e.g. valid).
        Batched like commit_all_grad_enabled: one stack, not 32."""
        caps = self._captured
        eps = self.rms_eps
        ran = False
        advance_bank = (
            self.state_mode == "banked"
            and self.pending_memory_id is not None)
        if self.state_mode == "banked":
            self.current_write_bank = self.next_bank_index
        self.last_commit_address_key_by_layer = [None] * self.n_layers
        self.last_commit_episode_tokens_by_layer = [
            None] * self.n_layers
        self.last_commit_episode_weights_by_layer = [
            None] * self.n_layers
        self.last_commit_episode_identity = None
        self.last_commit_episode_identity_weight = None
        self.last_commit_episode_token_ids = None
        self.last_commit_episode_positions = None
        self.last_commit_episode_payload_token_ids = None
        self.last_commit_episode_payload_positions = None
        self.last_commit_episode_payload_votes = None
        self.last_commit_episode_payload_source_ids = None
        self.last_commit_address_transform_by_layer = [
            None] * self.n_layers
        if self.state_mode == "slots":
            keys, values = [], []
            if caps is not None:
                with torch.no_grad():
                    length = next(
                        (rows.shape[0] for rows in caps
                         if rows is not None and rows.shape[0] > 0), 0)
                    for slot, rows in enumerate(caps):
                        if rows is None or rows.shape[0] == 0:
                            keys.append(torch.zeros(
                                length, self.kv_dim, device=self.device,
                                dtype=self.dtype))
                            values.append(torch.zeros(
                                length, self.kv_dim, device=self.device,
                                dtype=self.dtype))
                        else:
                            key, value = self._slot_commit_math(
                                slot, rows.float(), eps)
                            keys.append(key)
                            values.append(value)
                            ran = True
                    if ran:
                        self._append_memory_slot(
                            torch.stack(keys), torch.stack(values))
            self.discard_captured()
            if ran:
                self.active = True
            return ran
        if self.state_mode == "episodes":
            new_M_parts = []
            new_S_parts = []
            if caps is not None:
                with torch.no_grad():
                    for slot, rows in enumerate(caps):
                        if rows is None or rows.shape[0] == 0:
                            continue
                        base_M = torch.zeros_like(self.M[slot])
                        base_S = torch.zeros_like(self.S[slot])
                        nm, ns = self._commit_math(
                            slot, rows.float(), eps,
                            base_memory=base_M,
                            base_normalizer=base_S)
                        new_M_parts.append(nm)
                        new_S_parts.append(ns)
                        ran = True
                    if ran:
                        self._append_episode_state(
                            torch.stack(new_M_parts),
                            torch.stack(new_S_parts))
            self.discard_captured()
            if ran:
                self.active = True
            return ran
        if caps is not None:
            M_parts = list(self.M)
            S_parts = list(self.S)
            with torch.no_grad():
                for slot, rows in enumerate(caps):
                    if rows is not None and rows.shape[0] > 0:
                        nm, ns = self._commit_math(slot, rows.float(), eps)
                        M_parts[slot] = nm
                        S_parts[slot] = ns
                        ran = True
                if ran:
                    self.M = torch.stack(M_parts, dim=0)
                    self.S = torch.stack(S_parts, dim=0)
                    if self.state_mode == "banked":
                        self._finalize_bank_commit()
                    else:
                        self._finalize_address_commit()
        self.discard_captured()
        if ran:
            self.active = True
            if advance_bank:
                self.next_bank_index = (
                    self.next_bank_index + 1
                ) % self.memory_banks
        self.current_write_bank = None
        return ran

    def commit_all_grad_enabled(self):
        """Commit ALL slots from captured rows in ONE batched update.

        Memory-critical design (12GB rule): the naive loop called commit()
        per slot, and each commit() stacked the full [32,kv,kv] M — 32
        stacks per chunk each retaining every slot = 537MB/chunk of graph.
        The batched form computes every slot's new_M/new_S first (graphs on
        per-slot math only), then does ONE stack. Retained per chunk:
        the per-slot math tensors + one 16.7MB stack output.
        """
        caps = self._captured
        eps = self.rms_eps
        ran = False
        advance_bank = (
            self.state_mode == "banked"
            and self.pending_memory_id is not None)
        if self.state_mode == "banked":
            self.current_write_bank = self.next_bank_index
        self.last_commit_address_key_by_layer = [None] * self.n_layers
        self.last_commit_episode_tokens_by_layer = [
            None] * self.n_layers
        self.last_commit_episode_weights_by_layer = [
            None] * self.n_layers
        self.last_commit_episode_identity = None
        self.last_commit_episode_identity_weight = None
        self.last_commit_episode_token_ids = None
        self.last_commit_episode_positions = None
        self.last_commit_episode_payload_token_ids = None
        self.last_commit_episode_payload_positions = None
        self.last_commit_episode_payload_votes = None
        self.last_commit_episode_payload_source_ids = None
        self.last_commit_address_transform_by_layer = [
            None] * self.n_layers
        self.last_commit_bank_route_by_layer = [None] * self.n_layers
        if self.state_mode == "slots":
            keys, values = [], []
            if caps is not None:
                length = next(
                    (rows.shape[0] for rows in caps
                     if rows is not None and rows.shape[0] > 0), 0)
                for slot, rows in enumerate(caps):
                    if rows is None or rows.shape[0] == 0:
                        keys.append(torch.zeros(
                            length, self.kv_dim, device=self.device,
                            dtype=self.dtype))
                        values.append(torch.zeros(
                            length, self.kv_dim, device=self.device,
                            dtype=self.dtype))
                    else:
                        with torch.enable_grad():
                            key, value = self._slot_commit_math(
                                slot, rows.float(), eps)
                        keys.append(key)
                        values.append(value)
                        ran = True
                if ran:
                    with torch.enable_grad():
                        self._append_memory_slot(
                            torch.stack(keys), torch.stack(values))
            self.discard_captured()
            if ran:
                self.active = True
            return ran
        if self.state_mode == "episodes":
            new_M_parts = []
            new_S_parts = []
            if caps is not None:
                for slot, rows in enumerate(caps):
                    if rows is None or rows.shape[0] == 0:
                        continue
                    base_M = torch.zeros_like(self.M[slot])
                    base_S = torch.zeros_like(self.S[slot])
                    with torch.enable_grad():
                        nm, ns = self._commit_math(
                            slot, rows.float(), eps,
                            base_memory=base_M,
                            base_normalizer=base_S)
                    new_M_parts.append(nm)
                    new_S_parts.append(ns)
                    ran = True
                if ran:
                    with torch.enable_grad():
                        self._append_episode_state(
                            torch.stack(new_M_parts),
                            torch.stack(new_S_parts))
            self.discard_captured()
            if ran:
                self.active = True
            return ran
        if caps is not None:
            new_M_parts = []
            new_S_parts = []
            for slot, rows in enumerate(caps):
                if rows is not None and rows.shape[0] > 0:
                    with torch.enable_grad():
                        nm, ns = self._commit_math(
                            slot, rows.float(), eps)
                    new_M_parts.append((slot, nm))
                    new_S_parts.append((slot, ns))
                    ran = True
            if ran:
                M_parts = list(self.M)
                S_parts = list(self.S)
                for slot, nm in new_M_parts:
                    M_parts[slot] = nm
                for slot, ns in new_S_parts:
                    S_parts[slot] = ns
                with torch.enable_grad():
                    self.M = torch.stack(M_parts, dim=0)
                    self.S = torch.stack(S_parts, dim=0)
                if self.state_mode == "banked":
                    self._finalize_bank_commit()
                else:
                    self._finalize_address_commit()
        self.discard_captured()
        if ran:
            self.active = True
            if advance_bank:
                self.next_bank_index = (
                    self.next_bank_index + 1
                ) % self.memory_banks
        self.current_write_bank = None
        return ran

    # ---- forward math ----
    def query_gates(self):
        if self.query_gate_logits is None:
            return None
        return torch.sigmoid(
            self.query_gate_logits / self.query_gate_temperature)

    def query_gate_penalty(self):
        gates = self.query_gates()
        if gates is None:
            return torch.zeros((), device=self.device)
        return gates.mean()

    def kv_gates(self):
        if self.kv_gate_logits is None:
            return None
        return torch.sigmoid(
            self.kv_gate_logits / self.kv_gate_temperature)

    def layer_gates(self, hard=False):
        gates = None
        if self.layer_gate_logits is not None:
            if hard:
                gates = straight_through_binary_gate(
                    self.layer_gate_logits,
                    self.layer_gate_temperature,
                    self.layer_gate_threshold)
            else:
                gates = torch.sigmoid(
                    self.layer_gate_logits / self.layer_gate_temperature)
        if self.layer_gate_override is not None:
            override = self.layer_gate_override.to(
                device=self.device, dtype=self.dtype)
            gates = override if gates is None else gates * override
        return gates

    def architecture_parameters(self):
        return [
            parameter
            for parameter in (
                self.query_gate_logits,
                self.kv_gate_logits,
                self.layer_gate_logits,
            )
            if parameter is not None
        ]

    def architecture_penalty(self):
        penalty = torch.zeros((), device=self.device)
        query_gates = self.query_gates()
        if query_gates is not None:
            penalty = penalty + self.query_gate_lambda * query_gates.mean()
        kv_gates = self.kv_gates()
        if kv_gates is not None:
            penalty = penalty + self.kv_gate_lambda * kv_gates.mean()
        layer_gates = self.layer_gates()
        if layer_gates is not None:
            penalty = (
                penalty + self.layer_gate_lambda * layer_gates.mean())
        return penalty

    def architecture_stats(self):
        stats = {}
        for prefix, gates, threshold in (
            ("query", self.query_gates(), self.query_gate_threshold),
            ("kv", self.kv_gates(), self.kv_gate_threshold),
            ("layer", self.layer_gates(), self.layer_gate_threshold),
        ):
            if gates is not None:
                values = gates.detach()
                stats[prefix + "_gate_mean"] = float(values.mean())
                stats[prefix + "_active"] = int(
                    (values >= threshold).sum().item())
        return stats

    @staticmethod
    def _selected_gate_indices(gates, threshold, minimum):
        active = torch.nonzero(
            gates >= threshold, as_tuple=False).reshape(-1)
        if active.numel() < minimum:
            active = torch.topk(
                gates, k=min(minimum, gates.numel())
            ).indices.sort().values
        return active

    def _query_groups(self, slot, h_raw):
        """Project raw residual rows into normalized memory query groups."""
        context = self.query_read_context_by_layer[slot]
        if context is not None:
            h_raw = h_raw.float() + context.expand(h_raw.shape[0], -1)
        T = h_raw.shape[0]
        hd = self.head_dim
        blk = self.layer_ids[slot]
        base_q_w = self.backbone.layers[blk]["q"]
        if self.q_rank == 0:
            q = F.linear(
                h_raw.float(), self.query_proj[slot].float())
            if self.query_mode == "backbone_delta":
                q = q + F.linear(
                    h_raw.to(base_q_w.dtype), base_q_w).float()
        else:
            query_hidden = F.linear(
                h_raw.float(), self.query_b[slot].float())
            gates = self.query_gates()
            if gates is not None:
                query_hidden = query_hidden * gates
            q = F.linear(query_hidden, self.query_a[slot].float())
            if self.query_mode == "backbone_delta":
                q = q + F.linear(
                    h_raw.to(base_q_w.dtype), base_q_w).float()
        q = q.view(T, -1, hd)
        q = metis_rms_norm(q, self.query_norm[slot], 1e-6)
        q = F.normalize(q, dim=-1, eps=1e-12)
        n_heads = q.shape[1]
        groups = self.q_dim // self.kv_dim
        heads_per_group = n_heads // groups
        return q.reshape(T, groups, heads_per_group * hd)

    def _episode_route_logits(self, slot, query_groups):
        """Score every latent episode from pooled or token-level addresses."""
        if self.episode_address_granularity == "tokens":
            if self.episode_address_mode == "shared_query":
                query_vectors = F.normalize(
                    query_groups.mean(dim=1),
                    dim=-1, eps=1e-12)
            else:
                query_vectors = F.normalize(
                    query_groups.reshape(-1, self.kv_dim),
                    dim=-1, eps=1e-12)
            addresses = F.normalize(
                self.episode_token_address[slot].float(),
                dim=-1, eps=1e-12)
            if self.episode_router_mode == "token_metric":
                query_vectors = query_vectors + F.linear(
                    F.silu(F.linear(
                        query_vectors,
                        self.episode_match_q_down[slot].float())),
                    self.episode_match_q_up[slot].float())
                addresses = addresses + F.linear(
                    F.silu(F.linear(
                        addresses,
                        self.episode_match_k_down[slot].float())),
                    self.episode_match_k_up[slot].float())
                query_vectors = F.normalize(
                    query_vectors, dim=-1, eps=1e-12)
                addresses = F.normalize(
                    addresses, dim=-1, eps=1e-12)
            similarities = torch.einsum(
                "qk,eak->qea", query_vectors, addresses)
            best_match = similarities.max(dim=0).values
            weights = self.episode_token_weight[slot].float()
            return (
                (best_match * weights).sum(dim=-1)
                / weights.sum(dim=-1).clamp_min(1e-6)
                / self.episode_temperature)
        addresses = F.normalize(
            self.episode_address[slot].float(),
            dim=-1, eps=1e-12)
        if self.episode_address_mode == "shared_query":
            query = F.normalize(
                query_groups.mean(dim=1),
                dim=-1, eps=1e-12)
            return (
                torch.einsum("tk,ek->te", query, addresses)
                .mean(dim=0) / self.episode_temperature)
        query = F.normalize(query_groups, dim=-1, eps=1e-12)
        return (
            torch.einsum("tgk,ek->tge", query, addresses)
            .mean(dim=(0, 1)) / self.episode_temperature)

    def _episode_identity_route_logits(self, query_identity):
        """Score latent episodes from tied backbone token embeddings."""
        if self.episode_identity_route_mode == "span":
            return self._episode_identity_span_logits(query_identity)
        query = query_identity.float()
        addresses = self.episode_identity_address.float()
        present = (self.episode_identity_weight > 0)
        gram = min(
            self.episode_identity_ngram,
            query.shape[0], addresses.shape[1])
        if gram > 1:
            query_windows = query.shape[0] - gram + 1
            address_windows = addresses.shape[1] - gram + 1
            query = torch.stack([
                query[index:index + gram].reshape(-1)
                for index in range(query_windows)
            ])
            addresses = torch.stack([
                addresses[:, index:index + gram].reshape(
                    addresses.shape[0], -1)
                for index in range(address_windows)
            ], dim=1)
            positions = self.episode_identity_positions
            window_present = torch.stack([
                present[:, index:index + gram].all(dim=-1)
                for index in range(address_windows)
            ], dim=1)
            consecutive = torch.stack([
                (
                    positions[:, index + 1:index + gram]
                    - positions[:, index:index + gram - 1]
                    == 1
                ).all(dim=-1)
                for index in range(address_windows)
            ], dim=1)
            present = window_present & consecutive
        query = F.normalize(query, dim=-1, eps=1e-12)
        addresses = F.normalize(addresses, dim=-1, eps=1e-12)
        similarities = torch.einsum(
            "qd,ead->qea", query, addresses)
        best_match = similarities.max(dim=0).values
        if self.episode_identity_match_scale > 0.0:
            best_match = torch.exp(
                (
                    best_match.clamp(max=1.0) - 1.0
                ) / self.episode_identity_match_scale)
        present = present.float()
        episode_count = addresses.shape[0]
        if episode_count > 1:
            cross_similarity = torch.einsum(
                "ead,fbd->eafb", addresses, addresses)
            best_cross_episode = cross_similarity.max(dim=-1).values
            self_mask = torch.eye(
                episode_count, device=addresses.device,
                dtype=torch.bool).unsqueeze(1)
            best_cross_episode = best_cross_episode.masked_fill(
                self_mask, -1.0)
            max_other_similarity = best_cross_episode.max(dim=-1).values
            uniqueness = (
                1.0 - max_other_similarity).clamp(min=0.0, max=1.0)
        else:
            uniqueness = torch.ones_like(present)
        weights = present * (
            self.episode_identity_uniqueness_floor + uniqueness)
        return (
            (best_match * weights).sum(dim=-1)
            / weights.sum(dim=-1).clamp_min(1e-6)
            / self.episode_temperature)

    def _episode_identity_span_logits(self, query_identity):
        """Align each episode's longest latent address span to the query."""
        query = F.normalize(
            query_identity.float(), dim=-1, eps=1e-12)
        addresses = F.normalize(
            self.episode_identity_address.float(),
            dim=-1, eps=1e-12)
        logits = []
        for episode_index in range(addresses.shape[0]):
            positions = self.episode_identity_positions[
                episode_index].detach().cpu().tolist()
            present = [
                index for index, position in enumerate(positions)
                if int(position) >= 0
            ]
            runs = []
            if present:
                run_start = 0
                for offset in range(1, len(present)):
                    previous = positions[present[offset - 1]]
                    current = positions[present[offset]]
                    if int(current) != int(previous) + 1:
                        runs.append(present[run_start:offset])
                        run_start = offset
                runs.append(present[run_start:])
            run = max(runs, key=len) if runs else []
            if not run or query.shape[0] < len(run):
                logits.append(query.new_tensor(-1.0))
                continue
            address_span = addresses[episode_index, run]
            window_count = query.shape[0] - len(run) + 1
            window_scores = torch.stack([
                (
                    query[start:start + len(run)]
                    * address_span
                ).sum(dim=-1).mean()
                for start in range(window_count)
            ])
            score = window_scores.max()
            if self.episode_identity_match_scale > 0.0:
                score = torch.exp(
                    (
                        score.clamp(max=1.0) - 1.0
                    ) / self.episode_identity_match_scale)
            logits.append(score / self.episode_temperature)
        return torch.stack(logits)

    def read(self, slot, h_raw):
        """v4 reference read law (matches metis_v6_read_v4 in C):
        q = query_proj(h_raw) [T, q_dim]; per head: RMSNorm(head_dim,
        query_norm, eps 1e-6) then L2 (eps 1e-12); group reads with
        (denom + 1). h_raw is the layer INPUT residual (not normed)."""
        T = h_raw.shape[0]
        groups = self.q_dim // self.kv_dim
        qg = self._query_groups(slot, h_raw)
        if (
            self.collect_address_training
            and self.state_mode not in ("banked", "episodes")
            and self.address_target_id is not None
            and self.address_target_id in self.address_ids
            and self.address_keys
        ):
            start, end = (
                self.address_token_range
                if self.address_token_range is not None
                else (0, T))
            start = max(0, min(start, T))
            end = max(start + 1, min(end, T))
            target_index = self.address_ids.index(
                self.address_target_id)
            keys = torch.stack(
                [entry[slot] for entry in self.address_keys])
            supervised_query = qg[start:end]
            self.address_losses.append(memory_address_contrastive_loss(
                supervised_query, keys, target_index,
                self.address_temperature))
            with torch.no_grad():
                logits = memory_address_logits(
                    supervised_query, keys, self.address_temperature)
                self.address_correct += int(
                    logits.argmax().item() == target_index)
                self.address_total += 1
        if self.state_mode == "episodes":
            episode_count = self.episode_M.shape[1]
            if episode_count == 0:
                return torch.zeros(
                    T, self.q_dim, device=h_raw.device,
                    dtype=torch.float32)
            global_logits = self._episode_route_logits(slot, qg)
            route_logits = global_logits.view(
                1, 1, -1).expand(T, groups, -1)
            activation = self.episode_activation_by_layer[slot]
            if (
                self.collect_address_training
                and activation is None
                and self.address_target_id is not None
                and self.address_target_id in self.episode_ids
            ):
                start, end = (
                    self.address_token_range
                    if self.address_token_range is not None
                    else (0, T))
                start = max(0, min(start, T))
                end = max(start + 1, min(end, T))
                logits = route_logits[start:end].mean(dim=(0, 1))
                target_index = max(
                    index for index, memory_id
                    in enumerate(self.episode_ids)
                    if memory_id == self.address_target_id)
                target = torch.tensor(
                    [target_index], device=logits.device,
                    dtype=torch.long)
                self.address_losses.append(
                    F.cross_entropy(logits.unsqueeze(0), target))
                with torch.no_grad():
                    self.address_correct += int(
                        logits.argmax().item() == target_index)
                    self.address_total += 1
            if self.episode_route_override is not None:
                route_weights = torch.zeros_like(route_logits)
                route_weights[..., self.episode_route_override] = 1.0
            elif activation is not None:
                route_weights = activation.view(
                    1, 1, -1).expand(T, groups, -1)
            else:
                probabilities = F.softmax(route_logits, dim=-1)
                route_weights = (
                    straight_through_top1(probabilities)
                    if self.episode_read_mode == "hard"
                    else probabilities)
            memory = self.episode_M[slot].float()
            normalizer = self.episode_S[slot].float()
            numerator = torch.einsum(
                "tgk,ekc->tgec", qg, memory)
            denominator = torch.einsum(
                "tgk,ek->tge", qg, normalizer)
            if self.denom_mode == "abs_plus_one":
                denominator = denominator.abs() + 1.0
            else:
                denominator = denominator + 1.0
            episode_outputs = numerator / denominator.unsqueeze(-1)
            out = torch.einsum(
                "tge,tgek->tgk",
                route_weights, episode_outputs)
            return out.reshape(T, self.q_dim)
        if self.state_mode == "banked":
            memory = self.M[slot].float()
            normalizer = self.S[slot].float()
            if self.bank_router_mode == "sequential":
                router_query = F.normalize(
                    qg, dim=-1, eps=1e-12)
                bank_keys = F.normalize(
                    normalizer, dim=-1, eps=1e-12)
                route_logits = torch.einsum(
                    "tgk,bk->tgb", router_query, bank_keys
                ) / self.bank_temperature
                occupied = normalizer.norm(
                    dim=-1) > 1e-8
                route_logits = route_logits.masked_fill(
                    ~occupied.view(1, 1, -1), -1e4)
            else:
                router_hidden = metis_rms_norm(
                    h_raw, self.attn_norm_w[slot], self.rms_eps)
                route_logits = F.linear(
                    router_hidden,
                    self.bank_router_w[slot].float()
                ) / self.bank_temperature
            route_probabilities = F.softmax(route_logits, dim=-1)
            route_weights = (
                straight_through_topk(route_probabilities, 1)
                if self.bank_read_mode == "hard"
                else route_probabilities
            )
            if route_weights.ndim == 2:
                route_weights = route_weights.unsqueeze(1).expand(
                    -1, groups, -1)
            if (
                self.collect_address_training
                and self.address_target_id is not None
                and self.address_target_id in self.address_ids
                and self.address_keys
            ):
                start, end = (
                    self.address_token_range
                    if self.address_token_range is not None
                    else (0, T))
                start = max(0, min(start, T))
                end = max(start + 1, min(end, T))
                target_index = self.address_ids.index(
                    self.address_target_id)
                target_route = self.address_keys[target_index][slot]
                target_distribution = (
                    target_route
                    / target_route.sum().clamp_min(1e-6)
                ).detach()
                target_bank = int(target_route.argmax().item())
                supervised_logits = route_logits[start:end].mean(
                    dim=tuple(range(route_logits.ndim - 1)))
                self.address_losses.append(-(
                    target_distribution
                    * F.log_softmax(supervised_logits, dim=-1)
                ).sum())
                with torch.no_grad():
                    self.address_correct += int(
                        supervised_logits.argmax().item() == target_bank)
                    self.address_total += 1
            numerator = torch.einsum(
                "tgk,bkc->tgbc", qg, memory)
            denominator = torch.einsum(
                "tgk,bk->tgb", qg, normalizer)
            if self.denom_mode == "abs_plus_one":
                denominator = denominator.abs() + 1.0
            else:
                denominator = denominator + 1.0
            bank_outputs = numerator / denominator.unsqueeze(-1)
            out = torch.einsum(
                "tgb,tgbk->tgk",
                route_weights, bank_outputs)
            return out.reshape(T, self.q_dim)
        if self.state_mode == "slots":
            if self.slot_K.shape[1] == 0:
                return torch.zeros(
                    T, self.q_dim, device=h_raw.device,
                    dtype=torch.float32)
            query = F.normalize(qg, dim=-1, eps=1e-12)
            keys = F.normalize(
                self.slot_K[slot], dim=-1, eps=1e-12)
            scores = torch.einsum(
                "tgk,nk->tgn", query, keys) / self.slot_temperature
            weights = F.softmax(scores, dim=-1)
            self.last_pointer_weights_by_layer[slot] = weights.mean(dim=1)
            if slot == self.n_layers - 1:
                self.last_pointer_weights = weights.mean(dim=1)
            if self.collect_evidence_loss:
                supervised_weights = weights
                if self.evidence_supervision_range is not None:
                    start, end = self.evidence_supervision_range
                    supervised_weights = weights[start:end]
                elif self.evidence_supervision_tail is not None:
                    supervised_weights = weights[
                        -self.evidence_supervision_tail:]
                self.evidence_losses.append(
                    evidence_attention_loss(
                        supervised_weights, self.slot_labels))
            out = torch.einsum(
                "tgn,nk->tgk", weights, self.slot_V[slot])
            return out.reshape(T, self.q_dim)
        M = self.M[slot]                             # [kv, kv]
        S = self.S[slot]                             # [kv]
        num = torch.einsum("tgk,kc->tgc", qg, M)
        denom = torch.einsum("tgk,k->tg", qg, S)
        if self.denom_mode == "abs_plus_one":
            denom = denom.abs() + 1.0
        else:
            denom = denom + 1.0
        out = num / denom.unsqueeze(-1)
        return out.reshape(T, self.q_dim)

    def fuse(self, slot, attn_branch, h_raw):
        """Apply a structurally gated memory layer.

        With gate=1 this is the original gamma blend.  With gate=0 the layer
        is an exact identity on the attention branch, so thresholded layer
        pruning has the same forward semantics as NAS training.
        """
        blk = self.layer_ids[slot]
        o_w = self.backbone.layers[blk]["o"]                 # [d, q_dim] bf16
        read_override = self.query_read_override_by_layer[slot]
        if read_override is None:
            mem = self.read(slot, h_raw)
        else:
            mem = read_override.expand(h_raw.shape[0], -1)
        memn = metis_rms_norm(mem, self.mem_norm[slot], 1e-6)
        fused = F.linear(memn.to(o_w.dtype), o_w)            # [T, d]
        self.last_memory_fused_by_layer[slot] = fused
        if self.collect_runtime_diagnostics:
            with torch.no_grad():
                memory_norm = fused.detach().float().norm(dim=-1).mean()
                attention_norm = (
                    attn_branch.detach().float().norm(dim=-1).mean())
                self.last_read_stats_by_layer[slot] = {
                    "memory_norm": float(memory_norm),
                    "attention_norm": float(attention_norm),
                    "memory_attention_ratio": float(
                        memory_norm / attention_norm.clamp_min(1e-12)),
                }
        if self.fusion_mode == "residual_gate":
            gate_logit = (
                h_raw.float() @ self.fusion_gate_w[slot].float() +
                self.fusion_gate_b[slot].float())
            gate = torch.sigmoid(gate_logit)
            if self.collect_fusion_gate_loss:
                target = torch.full_like(
                    gate_logit, self.fusion_gate_target)
                self.fusion_gate_losses.append(
                    F.binary_cross_entropy_with_logits(
                        gate_logit, target))
            return (
                attn_branch.float() +
                gate.unsqueeze(-1) * fused.float()
            ).to(attn_branch.dtype)
        layer_gates = self.layer_gates(hard=True)
        gate = (
            torch.ones((), device=attn_branch.device)
            if layer_gates is None else layer_gates[slot])
        strength = (1.0 - self.gamma) * gate.float()
        return ((1.0 - strength) * attn_branch.float()
                + strength * fused.float()).to(attn_branch.dtype)

    def answer_memory_summary(self):
        values = [
            value for value in self.last_memory_fused_by_layer
            if value is not None]
        if not values:
            return None
        return torch.stack(values, dim=0).mean(dim=0)

    def answer_memory_layers(self):
        values = [
            value for value in self.last_memory_fused_by_layer
            if value is not None]
        if not values:
            return None
        return torch.stack(values, dim=0)

    def commit(self, slot, raw_rows, eps):
        """raw_rows [L, d] captured layer-input residuals for ONE layer.
        Single-slot commit: computes new_M/new_S math and stacks once."""
        with torch.enable_grad():
            nm, ns = self._commit_math(slot, raw_rows.float(), eps)
        M_parts = list(self.M)
        S_parts = list(self.S)
        M_parts[slot] = nm
        S_parts[slot] = ns
        self.M = torch.stack(M_parts, dim=0)
        self.S = torch.stack(S_parts, dim=0)
        self.active = True

    def _commit_math(
        self, slot, raw_rows, eps,
        base_memory=None, base_normalizer=None,
    ):
        """Pure math: returns (new_M[slot], new_S[slot]) tensors with graph.
        Caller owns the stacking (batched single-stack for commit_all)."""
        if raw_rows.ndim == 1:
            raw_rows = raw_rows.unsqueeze(0)
        L = raw_rows.shape[0]
        norm_w = self.attn_norm_w[slot]                      # [d]
        h = metis_rms_norm(raw_rows, norm_w, eps)             # [L, d]
        scores = h @ self.w_agg[slot]                        # [L]
        p = F.softmax(scores / self.tau, dim=-1)
        w_sel = straight_through_alpha_top_p(
            p, self.rho, self.k_min,
            self.alpha_max_tokens, self.alpha_max_fraction)
        identity_scores = None
        identity_sel = None
        if (
            self.state_mode == "episodes"
            and self.episode_address_source == "embedding"
            and slot == 0
        ):
            identity_scores = h @ self.episode_identity_selector_w
            identity_p = F.softmax(
                identity_scores / self.tau, dim=-1)
            identity_sel = straight_through_alpha_top_p(
                identity_p, self.rho, self.k_min,
                self.alpha_max_tokens, self.alpha_max_fraction)
        if self.write_mode in ("factorized", "dual_tokens"):
            value_scores = h @ self.w_value_agg[slot]
            value_p = F.softmax(value_scores / self.tau, dim=-1)
            value_sel = straight_through_alpha_top_p(
                value_p, self.rho, self.k_min,
                self.value_alpha_max_tokens,
                self.value_alpha_max_fraction)
        else:
            value_p = None
            value_sel = None

        if self.kv_rank == 0:
            Kall = F.linear(h, self.wk[slot])                 # [L, kv]
            Vall = F.linear(h, self.wv[slot])                # [L, kv]
        else:
            kv_gates = self.kv_gates()
            kh = F.linear(h, self.wk_b[slot])
            vh = F.linear(h, self.wv_b[slot])
            if kv_gates is not None:
                kh = kh * kv_gates
                vh = vh * kv_gates
            Kall = F.linear(kh, self.wk_a[slot])              # [L, kv]
            Vall = F.linear(vh, self.wv_a[slot])             # [L, kv]
        unit_keys = F.normalize(Kall, dim=-1, eps=1e-12)
        address_unit_keys = unit_keys
        if (
            self.state_mode == "episodes"
            and self.episode_address_mode == "shared_query"
        ):
            address_unit_keys = F.normalize(
                self._query_groups(
                    slot, raw_rows.float()).mean(dim=1),
                dim=-1, eps=1e-12)
        elif (
            self.state_mode == "episodes"
            and self.episode_address_rank > 0
        ):
            address_hidden = F.linear(
                h, self.episode_address_b[slot])
            address_delta = F.linear(
                address_hidden, self.episode_address_a[slot])
            address_unit_keys = F.normalize(
                Kall.detach() + address_delta,
                dim=-1, eps=1e-12)
        if (
            self.state_mode != "episodes"
            and self.write_orthogonalization > 0.0
            and self.write_address_basis
        ):
            existing_basis = torch.stack([
                entry[slot] for entry in self.write_address_basis
            ])
            unit_keys = orthogonalize_write_keys(
                unit_keys,
                w_sel / w_sel.sum().clamp_min(1e-6),
                existing_basis,
                self.write_orthogonalization,
                self.update_similarity_threshold)
        Kn = unit_keys / math.sqrt(self.kv_dim)

        a_pre = h @ self.gdu_aw[slot] + self.gdu_ab[slot]
        b_pre = h @ self.gdu_bw[slot] + self.gdu_bb[slot]
        if (
            self.collect_write_selection_loss
            and self.pending_slot_labels is not None
            and self.pending_slot_labels.numel() == L
        ):
            labels = self.pending_slot_labels
            known = labels >= 0
            if bool(known.any()):
                targets = (labels[known] > 0).to(b_pre.dtype)
                write_loss = F.binary_cross_entropy_with_logits(
                    b_pre[known], targets)
                if (
                    self.write_mode == "paired_tokens"
                    and bool((labels > 0).any())
                    and bool((labels == 0).any())
                ):
                    write_loss = (
                        write_loss + evidence_attention_loss(p, labels))
                if (
                    self.write_mode in ("factorized", "dual_tokens")
                    and self.pending_key_labels is not None
                    and self.pending_value_labels is not None
                    and self.pending_key_labels.numel() == L
                    and self.pending_value_labels.numel() == L
                ):
                    write_loss = (
                        write_loss
                        + balanced_token_selection_loss(
                            scores, self.pending_key_labels)
                        + balanced_token_selection_loss(
                            value_scores, self.pending_value_labels)
                    )
                if (
                    identity_scores is not None
                    and self.pending_key_labels is not None
                    and self.pending_key_labels.numel() == L
                ):
                    write_loss = (
                        write_loss
                        + balanced_token_selection_loss(
                            identity_scores, self.pending_key_labels)
                    )
                self.write_selection_losses.append(write_loss)
        beta_i = self.beta_scale * torch.sigmoid(b_pre)      # [L]
        alpha_selection = (
            0.5 * (w_sel + value_sel)
            if self.write_mode == "dual_tokens"
            else w_sel)
        alpha_scalar = (
            (torch.sigmoid(a_pre) * alpha_selection).sum()
            / alpha_selection.sum().clamp_min(1e-6))
        if self.collect_retention_loss:
            self.retention_losses.append(
                retention_floor_loss(
                    alpha_scalar, self.retention_target))
        if self.collect_runtime_diagnostics:
            with torch.no_grad():
                selected_mask = w_sel.detach() > 0
                if self.write_mode == "dual_tokens":
                    selected_mask = (
                        selected_mask | (value_sel.detach() > 0))
                selected = int(selected_mask.sum().item())
                beta_detached = beta_i.detach()
                self.last_commit_stats_by_layer[slot] = {
                    "selected_tokens": selected,
                    "total_tokens": L,
                    "selected_ratio": selected / max(L, 1),
                    "alpha": float(alpha_scalar.detach()),
                    "beta_mean": float(beta_detached.mean()),
                    "beta_min": float(beta_detached.min()),
                    "beta_max": float(beta_detached.max()),
                }

        if self.write_mode == "factorized":
            key_vector = F.normalize(
                (
                    unit_keys
                    * w_sel.unsqueeze(-1)
                ).sum(dim=0),
                dim=-1, eps=1e-12,
            ) / math.sqrt(self.kv_dim)
            value_vector = (
                Vall * value_sel.unsqueeze(-1)
            ).sum(dim=0)
            beta_scalar = (
                beta_i * value_sel
            ).sum() / value_sel.sum().clamp_min(1e-6)
            Kn = key_vector.unsqueeze(0)
            Vall = value_vector.unsqueeze(0)
            b_eff = beta_scalar.unsqueeze(0)
            trace_address_weight = torch.ones_like(b_eff)
        elif self.write_mode == "dual_tokens":
            # Keep each selected token as its own K/V association. Address
            # and payload selectors contribute equal total write mass, which
            # preserves the legacy update exactly while they are identical.
            b_eff = beta_i * (0.5 * (w_sel + value_sel))
            trace_address_weight = (
                w_sel / w_sel.sum().clamp_min(1e-6))
        else:
            b_eff = beta_i * w_sel                           # [L]
            trace_address_weight = (
                b_eff / b_eff.sum().clamp_min(1e-6))
        address_weight = b_eff / b_eff.sum().clamp_min(1e-6)
        if self.state_mode == "banked":
            if self.bank_router_mode == "sequential":
                if self.current_write_bank is None:
                    raise RuntimeError(
                        "sequential bank write requires commit context")
                route = torch.zeros(
                    self.memory_banks, device=h.device,
                    dtype=h.dtype)
                route[self.current_write_bank] = 1.0
                route_probabilities = route
            else:
                token_route_logits = F.linear(
                    h, self.bank_router_w[slot].float())
                route_logits = (
                    token_route_logits
                    * address_weight.unsqueeze(-1)
                ).sum(dim=0) / self.bank_temperature
                route_probabilities = F.softmax(route_logits, dim=-1)
                route = straight_through_topk(
                    route_probabilities, self.bank_write_top_k)
            self.last_commit_bank_route_by_layer[slot] = route
            if self.collect_runtime_diagnostics:
                route_entropy = -(
                    route_probabilities
                    * route_probabilities.clamp_min(1e-12).log()
                ).sum()
                self.last_commit_stats_by_layer[slot].update({
                    "bank_index": int(route.argmax().detach()),
                    "bank_entropy": float(route_entropy.detach()),
                })
            bank_M = self.M[slot]
            bank_S = self.S[slot]
            new_M_parts = []
            new_S_parts = []
            for bank_index in range(self.memory_banks):
                candidate_M, candidate_S = gated_delta_update(
                    bank_M[bank_index], bank_S[bank_index],
                    Kn, Vall, b_eff, alpha_scalar)
                weight = route[bank_index]
                new_M_parts.append(
                    bank_M[bank_index]
                    + weight * (candidate_M - bank_M[bank_index]))
                new_S_parts.append(
                    bank_S[bank_index]
                    + weight * (candidate_S - bank_S[bank_index]))
            return (
                torch.stack(new_M_parts),
                torch.stack(new_S_parts),
            )
        if self.state_mode == "episodes":
            address_count = min(self.episode_address_tokens, L)
            address_indices = torch.topk(
                w_sel.detach(), k=address_count).indices
            address_tokens = address_unit_keys[address_indices]
            address_weights = w_sel[address_indices]
            if address_count < self.episode_address_tokens:
                pad_count = self.episode_address_tokens - address_count
                address_tokens = torch.cat((
                    address_tokens,
                    torch.zeros(
                        pad_count, self.kv_dim,
                        device=address_tokens.device,
                        dtype=address_tokens.dtype),
                ), dim=0)
                address_weights = torch.cat((
                    address_weights,
                    torch.zeros(
                        pad_count,
                        device=address_weights.device,
                        dtype=address_weights.dtype),
                ), dim=0)
            self.last_commit_episode_tokens_by_layer[slot] = (
                address_tokens)
            self.last_commit_episode_weights_by_layer[slot] = (
                address_weights)
            if (
                slot == 0
                and self.episode_address_source == "embedding"
            ):
                if self.episode_identity_selection_mode == "window":
                    window_scores = identity_scores.detach().unfold(
                        0, address_count, 1).mean(dim=-1)
                    window_start = int(
                        window_scores.argmax().item())
                    identity_address_indices = torch.arange(
                        window_start, window_start + address_count,
                        device=h.device, dtype=torch.long)
                    identity_address_weights = torch.ones(
                        address_count, device=h.device, dtype=h.dtype)
                else:
                    identity_address_indices = torch.topk(
                        identity_sel.detach(), k=address_count).indices
                    identity_address_indices = torch.sort(
                        identity_address_indices).values
                    identity_address_weights = identity_sel[
                        identity_address_indices]
                identity_rows = self._captured_token_identity
                token_ids = self._captured_token_ids
                if (
                    identity_rows is None
                    or identity_rows.shape[0] != L
                    or token_ids is None
                    or token_ids.shape[0] != L
                ):
                    raise RuntimeError(
                        "episode token identities do not align with write")
                identity_tokens = F.normalize(
                    identity_rows.float()[identity_address_indices],
                    dim=-1, eps=1e-12)
                identity_weights = identity_address_weights
                identity_token_ids = token_ids[
                    identity_address_indices]
                identity_positions = identity_address_indices
                if address_count < self.episode_address_tokens:
                    pad_count = (
                        self.episode_address_tokens - address_count)
                    identity_tokens = torch.cat((
                        identity_tokens,
                        torch.zeros(
                            pad_count, self.d_model,
                            device=identity_tokens.device,
                            dtype=identity_tokens.dtype),
                    ), dim=0)
                    identity_weights = torch.cat((
                        identity_weights,
                        torch.zeros(
                            pad_count,
                            device=identity_weights.device,
                            dtype=identity_weights.dtype),
                        ), dim=0)
                    identity_token_ids = torch.cat((
                        identity_token_ids,
                        torch.full(
                            (pad_count,), -1,
                            device=identity_token_ids.device,
                            dtype=torch.long),
                        ), dim=0)
                    identity_positions = torch.cat((
                        identity_positions,
                        torch.full(
                            (pad_count,), -1,
                            device=identity_positions.device,
                            dtype=torch.long),
                    ), dim=0)
                self.last_commit_episode_identity = identity_tokens
                self.last_commit_episode_identity_weight = identity_weights
                self.last_commit_episode_token_ids = identity_token_ids
                self.last_commit_episode_positions = identity_positions
            if value_sel is not None:
                token_ids = self._captured_token_ids
                if token_ids is None or token_ids.shape[0] != L:
                    raise RuntimeError(
                        "episode payload token ids do not align with write")
                selected = (
                    value_sel.detach() > 0).to(torch.int16)
                if self.last_commit_episode_payload_votes is None:
                    self.last_commit_episode_payload_votes = selected
                    self.last_commit_episode_payload_source_ids = (
                        token_ids.detach().clone())
                else:
                    if (
                        self.last_commit_episode_payload_votes.shape[0] != L
                        or not torch.equal(
                            self.last_commit_episode_payload_source_ids,
                            token_ids,
                        )
                    ):
                        raise RuntimeError(
                            "episode payload layers do not align")
                    self.last_commit_episode_payload_votes = (
                        self.last_commit_episode_payload_votes + selected)
        self.last_commit_address_key_by_layer[slot] = F.normalize(
            (
                (
                    address_unit_keys / math.sqrt(self.kv_dim)
                    if self.state_mode == "episodes"
                    else Kn
                )
                * trace_address_weight.unsqueeze(-1)
            ).sum(dim=0),
            dim=-1, eps=1e-12)
        self.last_commit_address_transform_by_layer[slot] = (
            memory_address_transform(Kn, b_eff, alpha_scalar))
        old_M = self.M[slot] if base_memory is None else base_memory
        old_S = self.S[slot] if base_normalizer is None else base_normalizer
        new_M, new_S = gated_delta_update(
            old_M, old_S, Kn, Vall,
            b_eff, alpha_scalar)
        if (
            self.collect_preservation_loss
            and self.preservation_probe_keys
        ):
            probe_keys = torch.stack([
                entry[slot]
                for entry in self.preservation_probe_keys
            ])
            self.preservation_losses.append(
                memory_read_preservation_loss(
                    old_M, old_S,
                    new_M, new_S, probe_keys,
                    self.denom_mode))
        # caller (commit / commit_all) owns the stacking so a 32-slot
        # batched commit performs ONE stack instead of 32.
        return new_M, new_S

    def runtime_diagnostics(self):
        """Compact actual gate, selection, and branch statistics."""
        commits = [
            value for value in self.last_commit_stats_by_layer
            if value is not None]
        reads = [
            value for value in self.last_read_stats_by_layer
            if value is not None]

        def mean(rows, key):
            return (
                sum(float(row[key]) for row in rows) / len(rows)
                if rows else 0.0)

        return {
            "layers_with_commit": len(commits),
            "layers_with_read": len(reads),
            "selected_ratio": mean(commits, "selected_ratio"),
            "alpha": mean(commits, "alpha"),
            "beta_mean": mean(commits, "beta_mean"),
            "beta_min": (
                min((row["beta_min"] for row in commits), default=0.0)),
            "beta_max": (
                max((row["beta_max"] for row in commits), default=0.0)),
            "memory_attention_ratio": mean(
                reads, "memory_attention_ratio"),
        }

    def _slot_commit_math(self, slot, raw_rows, eps):
        """Encode every source token as an independent K/V memory slot."""
        if raw_rows.ndim == 1:
            raw_rows = raw_rows.unsqueeze(0)
        norm_w = self.attn_norm_w[slot]
        h = metis_rms_norm(raw_rows, norm_w, eps)

        if self.kv_rank == 0:
            keys = F.linear(h, self.wk[slot])
            values = F.linear(h, self.wv[slot])
        else:
            kv_gates = self.kv_gates()
            key_hidden = F.linear(h, self.wk_b[slot])
            value_hidden = F.linear(h, self.wv_b[slot])
            if kv_gates is not None:
                key_hidden = key_hidden * kv_gates
                value_hidden = value_hidden * kv_gates
            keys = F.linear(key_hidden, self.wk_a[slot])
            values = F.linear(value_hidden, self.wv_a[slot])
        return F.normalize(keys, dim=-1, eps=1e-12), values

    def _append_memory_slot(self, keys, values):
        """Append [layers, tokens, kv] data without token aggregation."""
        length = keys.shape[1]
        self.slot_K = torch.cat(
            (self.slot_K, keys), dim=1)
        self.slot_V = torch.cat(
            (self.slot_V, values), dim=1)
        if self.pending_slot_labels is not None:
            if self.pending_slot_labels.numel() != length:
                raise ValueError(
                    "pending slot label count does not match captured rows")
            labels = self.pending_slot_labels
        else:
            labels = torch.full(
                (length,), self.pending_slot_label,
                device=self.device, dtype=torch.int8)
        self.slot_labels = torch.cat((self.slot_labels, labels), dim=0)
        self.pending_slot_label = -1
        self.pending_slot_labels = None
        if self.slot_K.shape[1] > self.max_memory_slots:
            self.slot_K = self.slot_K[:, -self.max_memory_slots:]
            self.slot_V = self.slot_V[:, -self.max_memory_slots:]
            self.slot_labels = self.slot_labels[-self.max_memory_slots:]

    # ---- export ----
    def export(self, path):
        kv_gates = self.kv_gates()
        wk = self.wk.data if self.kv_rank == 0 else None
        wv = self.wv.data if self.kv_rank == 0 else None
        wk_a = self.wk_a.data if self.kv_rank > 0 else None
        wk_b = self.wk_b.data if self.kv_rank > 0 else None
        wv_a = self.wv_a.data if self.kv_rank > 0 else None
        wv_b = self.wv_b.data if self.kv_rank > 0 else None
        if kv_gates is not None:
            kv_values = kv_gates.detach()
            kv_active = self._selected_gate_indices(
                kv_values, self.kv_gate_threshold,
                self.kv_gate_min_rank)
            wk_a = wk_a[:, :, kv_active] * kv_values[kv_active]
            wk_b = wk_b[:, kv_active, :]
            wv_a = wv_a[:, :, kv_active] * kv_values[kv_active]
            wv_b = wv_b[:, kv_active, :]
            print(f'{{"kv_rank_searched":{self.kv_rank},'
                  f'"kv_rank_exported":{kv_active.numel()},'
                  f'"kv_gate_mean":{float(kv_values.mean()):.6f}}}',
                  flush=True)
        query_proj = (
            self.query_proj.data if self.q_rank == 0 else None)
        query_a = self.query_a.data if self.q_rank > 0 else None
        query_b = self.query_b.data if self.q_rank > 0 else None
        query_add_backbone = self.query_mode == "backbone_delta"
        if query_add_backbone and self.q_rank == 0:
            # Deploy one unambiguous learned-query operator.  Python training
            # defines the delta path as:
            #   Q_mem(h_raw) = delta(h_raw) + Wq_backbone(h_raw)
            # The C attention Q snapshot is Wq_backbone(RMSNorm(h_raw)), so
            # serializing the delta flag would change the function at runtime.
            # Fold Wq into the full-rank matrix and export the reference form
            # query_proj(h_raw) instead.
            backbone_query = torch.stack([
                self.backbone.layers[layer]["q"].float()
                for layer in self.layer_ids
            ]).to(device=query_proj.device, dtype=query_proj.dtype)
            query_proj = query_proj + backbone_query
            query_add_backbone = False
            print(json.dumps({
                "phase": "memory_export",
                "query_conversion": "backbone_delta_to_independent",
            }, separators=(",", ":")), flush=True)
        elif query_add_backbone:
            raise ValueError(
                "low-rank backbone-delta query cannot be exported exactly; "
                "use --query-rank 0 or --query-mode independent")
        gates = self.query_gates()
        if gates is not None:
            gate_values = gates.detach()
            active = self._selected_gate_indices(
                gate_values, self.query_gate_threshold,
                self.query_gate_min_rank)
            query_a = query_a[:, :, active] * gate_values[active]
            query_b = query_b[:, active, :]
            print(f'{{"query_rank_searched":{self.q_rank},'
                  f'"query_rank_exported":{active.numel()},'
                  f'"gate_mean":{float(gate_values.mean()):.6f}}}',
                  flush=True)

        layer_ids = self.layer_ids
        layer_active = None
        layer_gates = self.layer_gates()
        if layer_gates is not None:
            layer_values = layer_gates.detach()
            layer_active = self._selected_gate_indices(
                layer_values, self.layer_gate_threshold,
                self.layer_gate_min_layers)
            active_slots = layer_active.tolist()
            layer_ids = [self.layer_ids[index] for index in active_slots]
            if self.kv_rank == 0:
                wk = wk[layer_active]
                wv = wv[layer_active]
            else:
                wk_a = wk_a[layer_active]
                wk_b = wk_b[layer_active]
                wv_a = wv_a[layer_active]
                wv_b = wv_b[layer_active]
            if self.q_rank == 0:
                query_proj = query_proj[layer_active]
            else:
                query_a = query_a[layer_active]
                query_b = query_b[layer_active]
            print(f'{{"layers_searched":{self.n_layers},'
                  f'"layers_exported":{layer_active.numel()},'
                  f'"layer_gate_mean":{float(layer_values.mean()):.6f},'
                  f'"layer_ids":{json.dumps(layer_ids)}}}', flush=True)

        def selected(tensor):
            return tensor if layer_active is None else tensor[layer_active]

        save_bnmem_v3(path,
                      layer_ids=layer_ids,
                      d_model=self.d_model, kv_dim=self.kv_dim,
                      q_dim=self.q_dim, head_dim=self.head_dim,
                      gamma=self.gamma, tau=self.tau, rho=self.rho,
                      k_min=self.k_min,
                      alpha_max_tokens=self.alpha_max_tokens,
                      alpha_max_fraction=self.alpha_max_fraction,
                      gdu_ab=selected(self.gdu_ab.data),
                      gdu_bb=selected(self.gdu_bb.data),
                      beta_scale=self.beta_scale,
                      wk=wk, wv=wv,
                      wk_a=wk_a, wk_b=wk_b,
                      wv_a=wv_a, wv_b=wv_b,
                      w_agg=selected(self.w_agg.data),
                      gdu_aw=selected(self.gdu_aw.data),
                      gdu_bw=selected(self.gdu_bw.data),
                      mem_norm=selected(self.mem_norm.data),
                      query_proj=query_proj,
                      query_a=query_a, query_b=query_b,
                      query_norm=selected(self.query_norm.data),
                      denom_mode=(2 if self.denom_mode == "abs_plus_one"
                                  else 1),
                      query_add_backbone=query_add_backbone,
                      fusion_gate_w=(
                          None if self.fusion_gate_w is None
                          else selected(self.fusion_gate_w.data)),
                      fusion_gate_b=(
                          None if self.fusion_gate_b is None
                          else selected(self.fusion_gate_b.data)),
                      backbone_sha256=self.backbone.model_sha256())
        print(f'{{"exported":"{path}"}}', flush=True)
        if (
            self.state_mode == "banked"
            and self.bank_router_mode == "learned"
        ):
            router_path = path + ".bankrouter.pt"
            torch.save({
                "version": 1,
                "memory_banks": self.memory_banks,
                "bank_temperature": self.bank_temperature,
                "bank_write_top_k": self.bank_write_top_k,
                "bank_read_mode": self.bank_read_mode,
                "bank_router_mode": self.bank_router_mode,
                "bank_router_w": (
                    self.bank_router_w.detach().float().cpu()),
            }, router_path)
            print(json.dumps({
                "exported_bank_router": router_path,
                "prototype_only": True,
            }, separators=(",", ":")), flush=True)
        if self.write_mode in ("factorized", "dual_tokens"):
            selector_path = path + ".write_selector.pt"
            torch.save({
                "version": 1,
                "write_mode": self.write_mode,
                "value_alpha_max_tokens": self.value_alpha_max_tokens,
                "value_alpha_max_fraction": (
                    self.value_alpha_max_fraction),
                "write_orthogonalization": (
                    self.write_orthogonalization),
                "update_similarity_threshold": (
                    self.update_similarity_threshold),
                "w_value_agg": (
                    self.w_value_agg.detach().float().cpu()),
            }, selector_path)
            print(json.dumps({
                "exported_write_selector": selector_path,
                "prototype_only": True,
            }, separators=(",", ":")), flush=True)
        if (
            self.state_mode == "episodes"
            and (
                self.episode_address_a is not None
                or self.episode_router_mode == "token_metric"
                or self.episode_identity_selector_w is not None
            )
        ):
            router_path = path + ".episode_router.pt"
            router_payload = {
                "version": 1,
                "episode_address_rank": self.episode_address_rank,
                "episode_router_mode": self.episode_router_mode,
                "episode_router_rank": self.episode_router_rank,
                "episode_address_granularity": (
                    self.episode_address_granularity),
                "episode_address_tokens": self.episode_address_tokens,
            }
            if self.episode_address_a is not None:
                router_payload.update({
                    "episode_address_a": (
                        self.episode_address_a.detach().float().cpu()),
                    "episode_address_b": (
                        self.episode_address_b.detach().float().cpu()),
                })
            if self.episode_router_mode == "token_metric":
                for name in (
                    "episode_match_q_down",
                    "episode_match_q_up",
                    "episode_match_k_down",
                    "episode_match_k_up",
                    "episode_layer_logits",
                ):
                    router_payload[name] = (
                        getattr(self, name).detach().float().cpu())
            if self.episode_identity_selector_w is not None:
                router_payload["episode_identity_selector_w"] = (
                    self.episode_identity_selector_w
                    .detach().float().cpu())
            torch.save(router_payload, router_path)
            print(json.dumps({
                "exported_episode_router": router_path,
                "prototype_only": True,
            }, separators=(",", ":")), flush=True)


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------

class MemoryTrainer:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)
        if args.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = args.device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")

        print(f'{{"phase":"load_backbone","device":"{self.device}"}}',
              flush=True)
        t0 = time.time()
        gw = GGUFWeights(args.gguf, args.lib)
        self.gw = gw
        self.backbone = TorchBackbone(gw, device=self.device,
                                      dtype=torch.bfloat16)
        self.answer_decoder = None
        if args.init_answer_decoder:
            self.answer_decoder = MemoryAnswerDecoder.load(
                args.init_answer_decoder, self.backbone.cfg.hidden,
                device=self.device)
        elif args.answer_decoder_width > 0:
            self.answer_decoder = MemoryAnswerDecoder(
                self.backbone.cfg.hidden, args.answer_decoder_width,
                device=self.device,
                memory_aware=args.memory_aware_answer_decoder,
                structured_memory=(
                    args.structured_memory_answer_decoder))
        if self.answer_decoder is not None:
            self.backbone.answer_decoder = self.answer_decoder
        self.backbone_lora = None
        if args.init_backbone_lora:
            self.backbone_lora, bundled_output = load_lora_bundle(
                args.init_backbone_lora, self.backbone.cfg,
                device=self.device)
            self.backbone.backbone_lora = self.backbone_lora
        elif args.backbone_lora_rank > 0:
            if args.backbone_lora_blocks == "all":
                lora_blocks = list(range(self.backbone.cfg.n_layers))
            elif args.backbone_lora_blocks.startswith("last:"):
                count = int(args.backbone_lora_blocks.split(":", 1)[1])
                if count < 1 or count > self.backbone.cfg.n_layers:
                    raise ValueError("invalid --backbone-lora-blocks count")
                lora_blocks = list(range(
                    self.backbone.cfg.n_layers - count,
                    self.backbone.cfg.n_layers))
            else:
                lora_blocks = [
                    int(value)
                    for value in args.backbone_lora_blocks.split(",")
                    if value.strip()
                ]
            targets = tuple(
                value.strip()
                for value in args.backbone_lora_targets.split(",")
                if value.strip()
            )
            self.backbone_lora = BackboneLoRA(
                self.backbone.cfg,
                blocks=lora_blocks,
                targets=targets,
                rank=args.backbone_lora_rank,
                alpha=args.backbone_lora_alpha,
                device=self.device)
            self.backbone.backbone_lora = self.backbone_lora
            bundled_output = None
        else:
            bundled_output = None
        self.output_lora = None
        if args.init_output_lora:
            self.output_lora = load_output_lora(
                args.init_output_lora,
                self.backbone.cfg.hidden,
                self.backbone.cfg.vocab)
        elif args.output_lora_rank > 0:
            self.output_lora = OutputLoRA(
                self.backbone.cfg.hidden,
                self.backbone.cfg.vocab,
                args.output_lora_rank,
                args.output_lora_alpha)
        elif bundled_output is not None:
            self.output_lora = bundled_output
        if self.output_lora is not None:
            self.backbone.output_lora = self.output_lora
        if args.freeze_lora:
            if self.backbone_lora is not None:
                for parameter in self.backbone_lora.parameters():
                    parameter.requires_grad_(False)
            if self.output_lora is not None:
                for parameter in self.output_lora.parameters():
                    parameter.requires_grad_(False)
        print(f'{{"phase":"backbone_loaded","sec":{time.time()-t0:.1f},'
              f'"layers":{self.backbone.cfg.n_layers}}}', flush=True)

        n_blk = self.backbone.cfg.n_layers
        if args.layers == "all":
            layer_ids = list(range(n_blk))
        else:
            layer_ids = [int(x) for x in args.layers.split(",")]
        self.mem = MetisMemory(self.backbone, layer_ids,
                                gamma=args.gamma, tau=args.tau,
                                rho=args.rho, k_min=1,
                                alpha_max_tokens=args.alpha_max_tokens,
                                alpha_max_fraction=args.alpha_max_fraction,
                                beta_scale=args.beta_scale,
                                query_rank=args.query_rank,
                                query_gate_lambda=args.query_gate_lambda,
                                query_gate_temperature=args.query_gate_temperature,
                                query_gate_threshold=args.query_gate_threshold,
                                query_gate_min_rank=args.query_gate_min_rank,
                                query_mode=args.query_mode,
                                kv_rank=args.kv_rank,
                                kv_gate_lambda=args.kv_gate_lambda,
                                kv_gate_temperature=args.kv_gate_temperature,
                                kv_gate_threshold=args.kv_gate_threshold,
                                kv_gate_min_rank=args.kv_gate_min_rank,
                                layer_gate_lambda=args.layer_gate_lambda,
                                layer_gate_temperature=(
                                    args.layer_gate_temperature),
                                layer_gate_threshold=(
                                    args.layer_gate_threshold),
                                layer_gate_min_layers=(
                                    args.layer_gate_min_layers),
                                fusion_mode=args.fusion_mode,
                                gdu_alpha_init=args.gdu_alpha_init,
                                gdu_beta_init=args.gdu_beta_init,
                                retention_target=args.retention_target,
                                address_temperature=(
                                    args.address_temperature),
                                key_diversity_margin=(
                                    args.key_diversity_margin),
                                denom_mode=args.denom_mode,
                                state_mode=args.state_mode,
                                max_memory_slots=args.max_memory_slots,
                                slot_temperature=args.slot_temperature,
                                memory_banks=args.memory_banks,
                                max_memory_episodes=(
                                    args.max_memory_episodes),
                                episode_temperature=(
                                    args.episode_temperature),
                                episode_read_mode=(
                                    args.episode_read_mode),
                                episode_address_rank=(
                                    args.episode_address_rank),
                                episode_address_mode=(
                                    args.episode_address_mode),
                                episode_address_granularity=(
                                    args.episode_address_granularity),
                                episode_address_tokens=(
                                    args.episode_address_tokens),
                                episode_payload_tokens=(
                                    args.episode_payload_tokens),
                                episode_payload_min_layer_votes=(
                                    args.episode_payload_min_layer_votes),
                                episode_address_source=(
                                    args.episode_address_source),
                                episode_identity_uniqueness_floor=(
                                    args.episode_identity_uniqueness_floor),
                                episode_identity_match_scale=(
                                    args.episode_identity_match_scale),
                                episode_identity_ngram=(
                                    args.episode_identity_ngram),
                                episode_identity_route_mode=(
                                    args.episode_identity_route_mode),
                                episode_identity_selection_mode=(
                                    args.episode_identity_selection_mode),
                                episode_router_mode=(
                                    args.episode_router_mode),
                                episode_router_rank=(
                                    args.episode_router_rank),
                                bank_temperature=args.bank_temperature,
                                bank_write_top_k=args.bank_write_top_k,
                                bank_read_mode=args.bank_read_mode,
                                bank_router_mode=args.bank_router_mode,
                                write_mode=args.write_mode,
                                value_alpha_max_tokens=(
                                    args.value_alpha_max_tokens),
                                value_alpha_max_fraction=(
                                    args.value_alpha_max_fraction),
                                write_orthogonalization=(
                                    args.write_orthogonalization),
                                update_similarity_threshold=(
                                    args.update_similarity_threshold),
                                device=self.device, dtype=torch.float32,
                                seed=args.seed)
        if args.bb_init:
            self.mem.init_from_backbone()
            init_kind = (
                "backbone_exact_full_rank"
                if self.mem.kv_rank == 0 and self.mem.q_rank == 0
                else "backbone_factorized")
            print(f'{{"phase":"memory_init","source":"{init_kind}",'
                  f'"kv_rank":{self.mem.kv_rank}}}', flush=True)
        if args.init_memory:
            self.mem.load_checkpoint(
                args.init_memory,
                allow_query_rank_expand=args.allow_query_rank_expand,
                allow_selection_mismatch=(
                    args.allow_init_selection_mismatch))
            print(f'{{"phase":"memory_init","source":"checkpoint",'
                  f'"path":"{args.init_memory}",'
                  f'"query_rank":{self.mem.q_rank},'
                  f'"query_mode":"{self.mem.query_mode}"}}', flush=True)

        self.tok = CTokenizer(args.tok_probe, args.gguf)
        self.decoder = (
            CTokenDecoder(args.tok_probe, args.gguf)
            if (
                args.retrieval_windows > 0
                or args.generation_valid_samples_per_task > 0
            ) else None)
        self.eos = self.tok.eos()

        train_strata = load_dataset(args.data)
        ov = None
        if args.oversample_distract > 1:
            ov = {"distract": args.oversample_distract,
                  "multi_entity": args.oversample_distract}
        train_order = dataset_order(train_strata, args.seed, oversample=ov)
        if args.valid_data:
            valid_strata = load_dataset(args.valid_data)
            valid_order = dataset_order(valid_strata, args.seed + 1)
            deduped_valid_order = []
            seen_valid = set()
            for stratum_index, line_index in valid_order:
                line = valid_strata[stratum_index][1][line_index]
                sample = json.loads(line)
                locomo_id = str(sample.get("locomo_id", ""))
                if locomo_id:
                    key = ":".join(locomo_id.split(":")[:2])
                else:
                    key = (stratum_index, line_index)
                if key in seen_valid:
                    continue
                seen_valid.add(key)
                deduped_valid_order.append((stratum_index, line_index))
            valid_order = deduped_valid_order
            assert len(train_order) >= args.samples, \
                f"train data too small: {len(train_order)} < {args.samples}"
            assert len(valid_order) >= args.valid, \
                f"valid data too small: {len(valid_order)} < {args.valid}"
            self.train_strata = train_strata
            self.valid_strata = valid_strata
            self.train_idx = train_order[:args.samples]
            self.valid_idx = valid_order[:args.valid]
            print(f'{{"phase":"data_split","mode":"independent",'
                  f'"train":{len(self.train_idx)},'
                  f'"valid":{len(self.valid_idx)},'
                  f'"valid_unique":true}}', flush=True)
        else:
            assert len(train_order) >= args.samples + args.valid, \
                (f"data too small: {len(train_order)} < "
                 f"{args.samples}+{args.valid}")
            self.train_strata = train_strata
            self.valid_strata = train_strata
            self.train_idx = train_order[:args.samples]
            self.valid_idx = train_order[
                args.samples:args.samples + args.valid]
            print('{"warning":"validation is split from the training data; '
                  'use --valid-data for leakage-safe model selection"}',
                  flush=True)

        train_tasks = parse_task_filter(args.train_tasks)
        valid_tasks = parse_task_filter(args.valid_tasks)
        if train_tasks is not None:
            self.train_idx = [
                index for index in self.train_idx
                if memory_task_id(
                    self.train_strata[index[0]][0],
                    json.loads(self.train_strata[index[0]][1][index[1]]))
                in train_tasks
            ]
        if valid_tasks is not None:
            self.valid_idx = [
                index for index in self.valid_idx
                if memory_task_id(
                    self.valid_strata[index[0]][0],
                    json.loads(self.valid_strata[index[0]][1][index[1]]))
                in valid_tasks
            ]
        if not self.train_idx:
            raise ValueError("training task filter selected no samples")
        if not self.valid_idx:
            raise ValueError("validation task filter selected no samples")
        print(json.dumps({
            "phase": "task_filter",
            "train_tasks": (
                "all" if train_tasks is None else sorted(train_tasks)),
            "valid_tasks": (
                "all" if valid_tasks is None else sorted(valid_tasks)),
            "train_samples": len(self.train_idx),
            "valid_samples": len(self.valid_idx),
        }, separators=(",", ":")), flush=True)

        self.train_by_task = {}
        for stratum_index, line_index in self.train_idx:
            stratum_name, lines = self.train_strata[stratum_index]
            sample = json.loads(lines[line_index])
            task = memory_task_id(stratum_name, sample)
            self.train_by_task.setdefault(task, []).append(
                (stratum_index, line_index))
        self.train_task_positions = {
            task: 0 for task in self.train_by_task}
        self.valid_by_task = {}
        for stratum_index, line_index in self.valid_idx:
            stratum_name, lines = self.valid_strata[stratum_index]
            sample = json.loads(lines[line_index])
            task = memory_task_id(stratum_name, sample)
            self.valid_by_task.setdefault(task, []).append(
                (stratum_index, line_index))
        self.task_weight_starts = {
            task: getattr(args, f"task{task}_weight_start")
            for task in TASK_WEIGHT_DEFAULTS}
        self.task_weight_ends = {
            task: getattr(args, f"task{task}_weight_end")
            for task in TASK_WEIGHT_DEFAULTS}
        task_counts = {
            str(task): len(indices)
            for task, indices in sorted(self.train_by_task.items())}
        print(json.dumps({
            "phase": "task_schedule",
            "task_counts": task_counts,
            "weight_start": self.task_weight_starts,
            "weight_end": self.task_weight_ends,
        }, separators=(",", ":")), flush=True)

        memory_parameters = [
            parameter
            for name, parameter in self.mem.named_parameters()
            if not name.startswith("backbone.")
        ]
        if args.train_retrieval_only:
            if args.train_episode_selector_only:
                retrieval_prefixes = (
                    "episode_identity_selector_w",
                )
            elif args.train_episode_router_only:
                retrieval_prefixes = (
                    "episode_match_q_down", "episode_match_q_up",
                    "episode_match_k_down", "episode_match_k_up",
                    "episode_layer_logits",
                )
            elif args.train_episode_address_only:
                retrieval_prefixes = (
                    "episode_address_a", "episode_address_b",
                )
            elif self.mem.state_mode == "episodes":
                retrieval_prefixes = (
                    "query_proj", "query_a", "query_b", "query_norm",
                    "episode_address_a", "episode_address_b",
                    "query_gate_logits",
                )
            else:
                retrieval_prefixes = (
                    "query_proj", "query_a", "query_b", "query_norm",
                    "wk", "wk_a", "wk_b",
                    "query_gate_logits", "kv_gate_logits",
                )
            for name, parameter in self.mem.named_parameters():
                if (
                    not name.startswith("backbone.")
                    and not name.startswith(retrieval_prefixes)
                ):
                    parameter.requires_grad_(False)
        if args.freeze_memory:
            for parameter in memory_parameters:
                parameter.requires_grad_(False)
        architecture_parameters = (
            []
            if args.freeze_memory or args.train_retrieval_only
            else self.mem.architecture_parameters())
        architecture_ids = {id(parameter)
                            for parameter in architecture_parameters}
        optimizer_parameters = [
            parameter for parameter in memory_parameters
            if parameter.requires_grad
            and id(parameter) not in architecture_ids]
        if self.output_lora is not None and not args.freeze_lora:
            optimizer_parameters.extend(self.output_lora.parameters())
        if self.backbone_lora is not None and not args.freeze_lora:
            optimizer_parameters.extend(self.backbone_lora.parameters())
        if self.answer_decoder is not None:
            optimizer_parameters.extend(self.answer_decoder.parameters())
        deduped_parameters = []
        seen_parameters = set()
        for parameter in optimizer_parameters:
            if id(parameter) not in seen_parameters:
                seen_parameters.add(id(parameter))
                deduped_parameters.append(parameter)
        optimizer_parameters = deduped_parameters
        if not optimizer_parameters and not architecture_parameters:
            raise ValueError("no trainable parameters selected")
        self.weight_parameters = optimizer_parameters
        self.architecture_parameters = architecture_parameters
        self.trainable_parameters = (
            optimizer_parameters + architecture_parameters)
        router_parameter = (
            self.mem.bank_router_w
            if self.mem.state_mode == "banked" else None)
        write_selector_parameter = self.mem.w_value_agg
        if optimizer_parameters:
            router_ids = (
                {id(router_parameter)}
                if router_parameter is not None else set())
            write_selector_ids = (
                {id(write_selector_parameter)}
                if write_selector_parameter is not None else set())
            specialized_ids = router_ids | write_selector_ids
            base_parameters = [
                parameter for parameter in optimizer_parameters
                if id(parameter) not in specialized_ids]
            router_parameters = [
                parameter for parameter in optimizer_parameters
                if id(parameter) in router_ids]
            write_selector_parameters = [
                parameter for parameter in optimizer_parameters
                if id(parameter) in write_selector_ids]
            optimizer_groups = []
            if base_parameters:
                optimizer_groups.append({
                    "params": base_parameters,
                    "lr": args.lr,
                    "lr_scale": 1.0,
                })
            if router_parameters:
                optimizer_groups.append({
                    "params": router_parameters,
                    "lr": args.lr * args.bank_router_lr_scale,
                    "lr_scale": args.bank_router_lr_scale,
                })
            if write_selector_parameters:
                optimizer_groups.append({
                    "params": write_selector_parameters,
                    "lr": args.lr * args.write_selector_lr_scale,
                    "lr_scale": args.write_selector_lr_scale,
                })
            self.opt = torch.optim.AdamW(
                optimizer_groups, lr=args.lr, weight_decay=args.wd)
        else:
            self.opt = None
        self.arch_opt = (
            torch.optim.Adam(
                architecture_parameters, lr=args.nas_lr,
                weight_decay=0.0)
            if architecture_parameters else None)
        print(f'{{"phase":"optimizer_init",'
              f'"trainable_parameters":'
              f'{sum(parameter.numel() for parameter in self.trainable_parameters)},'
              f'"weight_parameters":'
              f'{sum(parameter.numel() for parameter in optimizer_parameters)},'
              f'"architecture_parameters":'
              f'{sum(parameter.numel() for parameter in architecture_parameters)},'
              f'"memory_frozen":{str(args.freeze_memory).lower()},'
              f'"retrieval_only":'
              f'{str(args.train_retrieval_only).lower()},'
              f'"answer_decoder":'
              f'{str(self.answer_decoder is not None).lower()},'
              f'"lora_frozen":{str(args.freeze_lora).lower()},'
              f'"output_lora":{str(self.output_lora is not None).lower()},'
              f'"backbone_lora":'
              f'{str(self.backbone_lora is not None).lower()},'
              f'"bank_router_lr_scale":'
              f'{args.bank_router_lr_scale},'
              f'"write_selector_lr_scale":'
              f'{args.write_selector_lr_scale}}}',
              flush=True)
        self.base_lr = args.lr
        self.warmup = args.warmup
        self.best_valid = float("inf")
        self.best_generation_recall = float("-inf")
        self.checkpoint_saved = False
        self.patience = 0
        self.schedule_rng = random.Random(args.seed + 0x5E1F)
        self.self_prefix_used = 0
        self.contrastive_pairs_used = 0
        self.write_selection_loss_sum = 0.0
        self.write_selection_loss_count = 0
        self.retention_loss_sum = 0.0
        self.retention_loss_count = 0
        self.preservation_loss_sum = 0.0
        self.preservation_loss_count = 0
        self.interference_distill_loss_sum = 0.0
        self.interference_distill_loss_count = 0
        self.interference_teacher_used = 0
        self.address_loss_sum = 0.0
        self.address_loss_count = 0
        self.address_correct = 0
        self.address_total = 0
        self.key_diversity_loss_sum = 0.0
        self.key_diversity_loss_count = 0
        self.arch_cursor = 0

    def warmup_lr(self, step):
        if self.warmup <= 0 or step >= self.warmup:
            return self.base_lr
        return self.base_lr * (step + 1) / self.warmup

    def next_training_batch(self, step, batch_size):
        available = sorted(self.train_by_task)
        progress = step / max(self.args.steps - 1, 1)
        weights = scheduled_task_weights(
            progress, available,
            self.task_weight_starts, self.task_weight_ends)
        task = self.schedule_rng.choices(
            available, weights=[weights[value] for value in available],
            k=1)[0]
        indices = self.train_by_task[task]
        position = self.train_task_positions[task]
        batch = [
            indices[(position + offset) % len(indices)]
            for offset in range(batch_size)
        ]
        self.train_task_positions[task] = (
            position + batch_size) % len(indices)
        return task, batch, weights

    def run_architecture_step(self, tok_cache):
        """One validation-gradient step for differentiable NAS gates."""
        if self.arch_opt is None:
            return None
        s_idx = self.valid_idx[self.arch_cursor % len(self.valid_idx)]
        self.arch_cursor += 1
        line = self.valid_strata[s_idx[0]][1][s_idx[1]]
        sample = json.loads(line)
        self.arch_opt.zero_grad(set_to_none=True)
        for parameter in self.weight_parameters:
            parameter.requires_grad_(False)
        for parameter in self.architecture_parameters:
            parameter.requires_grad_(True)
        try:
            ls, nt, _nc, _ne = self.run_sample(
                sample, True, tok_cache, allow_self_prefix=False)
            if ls is None or nt == 0:
                return None
            task_loss = ls / nt
            penalty = self.mem.architecture_penalty()
            objective = task_loss + penalty
            objective.backward()
            torch.nn.utils.clip_grad_norm_(
                self.architecture_parameters, self.args.clip)
            self.arch_opt.step()
            return (float(task_loss.detach()),
                    float(penalty.detach()))
        finally:
            for parameter in self.weight_parameters:
                parameter.requires_grad_(True)

    def greedy_prefix(self, prompt_ids, length):
        """Roll out the model's own prefix without KV cache or state writes."""
        generated = []
        with torch.no_grad():
            for _ in range(length):
                tokens = torch.tensor(
                    prompt_ids + generated, device=self.device)
                logits = self._query_forward(
                    tokens, query_token_count=len(prompt_ids))
                self.mem.discard_captured()
                generated.append(int(logits.argmax().item()))
        return generated

    def _query_forward(
        self, tokens, logits_all=False, query_token_count=None
    ):
        """Run one query with inline, frozen, or contextual memory reads."""
        mem = self.mem
        # Query rows are never committed before their answer is scored.
        # Starting from a clean capture window also prevents a previous
        # teacher-forced or greedy replay from contaminating the two-pass
        # query summary.
        mem.discard_captured()
        mem.clear_query_read_override()
        if (
            self.args.query_read_mode in (
                "two_pass", "contextual", "episode_activation")
            and mem.active
        ):
            if query_token_count is None:
                query_token_count = int(tokens.shape[0])
            if query_token_count < 1 or query_token_count > tokens.shape[0]:
                raise ValueError("invalid two-pass query token count")
            query_tokens = tokens[:query_token_count]
            saved_active = mem.active
            mem.active = False
            try:
                with torch.no_grad():
                    self.backbone(
                        query_tokens, memory_v6=mem, fuse_start=0)
            finally:
                mem.active = saved_active
            if self.args.query_read_mode == "two_pass":
                mem.prepare_query_read_override(
                    self.args.two_pass_query_tokens)
            elif self.args.query_read_mode == "contextual":
                mem.prepare_query_read_context(
                    self.args.two_pass_query_tokens)
            else:
                mem.prepare_episode_activation(
                    self.args.two_pass_query_tokens)
        try:
            return self.backbone(
                tokens, memory_v6=mem, logits_all=logits_all,
                fuse_start=0)
        finally:
            mem.clear_query_read_override()

    def _query_forward_from_snapshot(
        self, tokens, snapshot, logits_all=False, query_token_count=None
    ):
        """Run a training-only counterfactual query from detached M/S state."""
        mem = self.mem
        saved_memory = mem.M
        saved_normalizer = mem.S
        saved_active = mem.active
        saved_collect_address = mem.collect_address_training
        mem.M = snapshot["M"]
        mem.S = snapshot["S"]
        mem.active = snapshot["active"]
        mem.collect_address_training = False
        mem.discard_captured()
        try:
            with torch.no_grad():
                return self._query_forward(
                    tokens, logits_all=logits_all,
                    query_token_count=query_token_count).detach()
        finally:
            mem.M = saved_memory
            mem.S = saved_normalizer
            mem.active = saved_active
            mem.collect_address_training = saved_collect_address
            mem.discard_captured()

    def run_sample(self, sample, want_grads, tok_cache,
                   query_memory_enabled=True, allow_self_prefix=True):
        """Forward one sample: full-prefix replays for exchanges (fusion
        gated to the current chunk), fresh-context query + teacher-forced
        target scoring. Returns (loss_sum, n_label)."""
        chunks = sample["messages"]
        query_value = sample.get("query_turn_id", len(chunks) - 1)
        if isinstance(query_value, list):
            query_indices = [int(index) for index in query_value]
        else:
            query_indices = [int(query_value)]
        if not query_indices:
            query_indices = [len(chunks) - 1]
        query_index_set = set(query_indices)
        if any(index < 0 or index >= len(chunks)
               for index in query_indices):
            raise ValueError("query_turn_id contains an invalid chunk index")
        first_query = min(query_indices)
        gradient_tail_commits = int(
            sample.get("metadata", {}).get("gradient_tail_commits", 0))
        if gradient_tail_commits < 0:
            raise ValueError("gradient_tail_commits must be non-negative")
        gradient_start = (
            max(0, first_query - gradient_tail_commits)
            if want_grads and gradient_tail_commits else 0)
        evidence_indices = {
            int(index)
            for index in sample.get("evidence_message_indices", [])}
        distractor_indices = {
            int(index)
            for index in sample.get("distractor_message_indices", [])}
        query_evidence = generation_query_evidence_map(
            sample, query_indices)
        interference_lambda = float(getattr(
            self.args, "interference_distill_lambda", 0.0))
        snapshot_evidence_indices = set()
        if want_grads and interference_lambda > 0.0:
            for value in query_evidence.values():
                if isinstance(value, list):
                    snapshot_evidence_indices.update(
                        int(index) for index in value)
                elif value is not None:
                    snapshot_evidence_indices.add(int(value))
        interference_snapshots = {}
        addressable_indices = {
            int(index)
            for index in sample.get(
                "memory_targets_by_message", {}).keys()}
        for value in query_evidence.values():
            if isinstance(value, list):
                addressable_indices.update(
                    int(index) for index in value)
            else:
                addressable_indices.add(int(value))
        self.mem.reset_state()
        mem, bb = self.mem, self.backbone
        eps = bb.cfg.rms_eps

        prompts = []
        for c in range(len(chunks)):
            is_q = c in query_index_set
            text, target = render_chunk(chunks[c], is_q)
            # Every chunk is evaluated in a fresh no-KV context, so every
            # chunk needs the same BOS treatment as a standalone request.
            key = ("b", text)
            ids = tok_cache.get(key)
            if ids is None:
                ids = self.tok.encode(text, add_bos=True)
                tok_cache[key] = ids
            tgt_ids = None
            if is_q:
                tkey = ("t", target)
                tgt_ids = tok_cache.get(tkey)
                if tgt_ids is None:
                    tgt_ids = self.tok.encode(target, add_bos=False)
                    tgt_ids = tgt_ids + [self.eos]
                    tok_cache[tkey] = tgt_ids
            prompts.append((ids, is_q, target, text, tgt_ids))
        evidence_target_ids = next(
            (target_ids for _ids, is_q, _target, _text, target_ids in prompts
             if is_q and target_ids is not None),
            None)
        if evidence_target_ids is None:
            self.last_sample_position_stats = {
                "first_correct": 0,
                "first_total": 0,
                "prefix_correct": 0,
                "prefix_total": 0,
            }
            return None, 0
        evidence_target_ids_by_message = target_ids_by_evidence(
            query_indices,
            query_evidence,
            {
                index: prompts[index][4]
                for index in query_indices
                if prompts[index][4] is not None
            },
        )

        ctx = torch.enable_grad() if want_grads else torch.no_grad()
        loss_sum = torch.zeros((), device=self.device)
        n_label = 0
        n_correct = 0
        sequence_exact = 0
        first_correct = 0
        first_total = 0
        prefix_correct = 0
        prefix_total = 0
        sample_write_losses = []
        sample_retention_losses = []
        sample_preservation_losses = []
        collect_address = (
            want_grads
            and (
                self.args.address_lambda > 0.0
                or self.args.key_diversity_lambda > 0.0
            )
            and mem.state_mode in ("delta", "banked", "episodes"))
        if collect_address:
            mem.begin_address_training()
        with ctx:
            for chunk_index, (
                ids, is_q, target, text, tgt_ids
            ) in enumerate(prompts):
                if not is_q:
                    truncated_prefix = (
                        want_grads and chunk_index < gradient_start)
                    # Official Metis keeps the complete multi-chunk graph:
                    # later query loss differentiates through every earlier
                    # memory read and write. The backbone weights remain
                    # frozen, but its activations must not be detached.
                    # Very long local trajectories may explicitly request a
                    # bounded gradient suffix. Prefix chunks still execute
                    # and update the memory state, but are detached so a
                    # 32/64-commit sample remains trainable on 16 GB MPS.
                    toks = torch.tensor(ids, device=self.device)
                    if truncated_prefix:
                        with torch.no_grad():
                            _ = bb(toks, memory_v6=mem, fuse_start=0)
                    else:
                        _ = bb(toks, memory_v6=mem, fuse_start=0)
                    memory_targets = sample.get(
                        "memory_targets_by_message", {})
                    raw_targets = memory_targets.get(
                        str(chunk_index),
                        memory_targets.get(chunk_index))
                    target_id_sequences = None
                    if raw_targets is not None:
                        if not isinstance(raw_targets, list):
                            raw_targets = [raw_targets]
                        target_id_sequences = []
                        for memory_target in raw_targets:
                            target_key = (
                                "memory-target", str(memory_target))
                            memory_target_ids = tok_cache.get(target_key)
                            if memory_target_ids is None:
                                memory_target_ids = self.tok.encode(
                                    str(memory_target), add_bos=False)
                                memory_target_ids = (
                                    memory_target_ids + [self.eos])
                                tok_cache[target_key] = memory_target_ids
                            target_id_sequences.append(memory_target_ids)
                        slot_labels = memory_target_slot_labels(
                            ids, target_id_sequences, self.eos)
                    elif chunk_index in evidence_target_ids_by_message:
                        target_id_sequences = (
                            evidence_target_ids_by_message[chunk_index])
                        slot_labels = memory_target_slot_labels(
                            ids, target_id_sequences, self.eos)
                    elif chunk_index in distractor_indices:
                        slot_labels = [0] * len(ids)
                    elif chunk_index in evidence_indices:
                        if self.args.evidence_label_mode == "chunk":
                            slot_labels = [1] * len(ids)
                        else:
                            slot_labels = answer_token_slot_labels(
                                ids, evidence_target_ids, self.eos)
                    else:
                        slot_labels = [-1] * len(ids)
                    mem.set_pending_slot_labels(slot_labels)
                    if (
                        getattr(mem, "write_mode", "paired_tokens")
                        in ("factorized", "dual_tokens")
                        and target_id_sequences
                    ):
                        key_sequences = (
                            target_id_sequences[:-1]
                            if len(target_id_sequences) > 1
                            else target_id_sequences)
                        value_sequences = target_id_sequences[-1:]
                        mem.set_pending_factorized_labels(
                            memory_target_slot_labels(
                                ids, key_sequences, self.eos),
                            memory_target_slot_labels(
                                ids, value_sequences, self.eos),
                        )
                    else:
                        mem.pending_key_labels = None
                        mem.pending_value_labels = None
                    mem.set_pending_memory_id(
                        chunk_index
                        if chunk_index in addressable_indices
                        else None)
                    supervise_write = (
                        want_grads
                        and self.args.write_selection_lambda > 0.0
                        and mem.state_mode in (
                            "delta", "banked", "episodes"))
                    supervise_retention = (
                        want_grads
                        and self.args.retention_lambda > 0.0
                        and mem.state_mode in (
                            "delta", "banked", "episodes"))
                    supervise_preservation = (
                        want_grads
                        and self.args.preservation_lambda > 0.0
                        and mem.state_mode == "delta")
                    if supervise_write:
                        mem.begin_write_selection_supervision()
                    if supervise_retention:
                        mem.begin_retention_supervision()
                    if supervise_preservation:
                        mem.begin_preservation_supervision()
                    if truncated_prefix:
                        mem.commit_all()
                    else:
                        mem.commit_all_grad_enabled()
                    if supervise_write:
                        write_loss = mem.end_write_selection_supervision()
                        sample_write_losses.append(write_loss)
                    if supervise_retention:
                        retention_loss = mem.end_retention_supervision()
                        sample_retention_losses.append(retention_loss)
                    if supervise_preservation:
                        preservation_loss = (
                            mem.end_preservation_supervision())
                        sample_preservation_losses.append(
                            preservation_loss)
                    if mem.state_mode in (
                        "delta", "banked", "episodes"
                    ):
                        mem.pending_slot_labels = None
                        mem.pending_slot_label = -1
                        mem.pending_key_labels = None
                        mem.pending_value_labels = None
                    append_pointer_token_ids(mem, ids, bb.device)
                    if (
                        chunk_index in snapshot_evidence_indices
                        and mem.state_mode == "delta"
                    ):
                        interference_snapshots[chunk_index] = {
                            "M": mem.M.detach().clone(),
                            "S": mem.S.detach().clone(),
                            "active": mem.active,
                        }
                    continue
                # query: FRESH context (memory is the only fact source).
                # M4(b) ruling: the query chunk is NEVER committed before
                # scoring — deploy commits the query turn AFTER generation
                # (it only affects the NEXT turn). Training with a
                # pre-scoring query commit taught the reads to expect the
                # question tokens inside M/S; at eval they are absent ->
                # out-of-distribution S -> degenerate decode (torch probe
                # proved the collapse is model-side, not C-i8).
                # Inline mode reads committed state during the scoring
                # prefill. Two-pass mode first captures a backbone-only
                # query representation, performs one frozen memory read per
                # layer, then replays the scoring prefill with those reads.
                # Neither mode commits the query before scoring.
                saved_active = mem.active
                if not query_memory_enabled:
                    mem.active = False
                if (
                    query_memory_enabled
                    and self.args.retrieval_windows > 0
                ):
                    with torch.no_grad():
                        toks = torch.tensor(ids, device=self.device)
                        _ = self._query_forward(
                            toks, query_token_count=len(ids))
                    mem.discard_captured()
                    excerpts = select_memory_excerpts(
                        mem, self.decoder,
                        self.args.retrieval_windows,
                        self.args.retrieval_window_size,
                        self.args.retrieval_layer_mode)
                    if excerpts:
                        text = add_retrieved_excerpts(text, excerpts)
                        key = ("retrieval", text)
                        ids = tok_cache.get(key)
                        if ids is None:
                            ids = self.tok.encode(text, add_bos=True)
                            tok_cache[key] = ids
                # Scheduled-prefix training closes the exposure gap between
                # teacher forcing and deployment generation. Score every gold
                # next-token target after the model's own greedy prefix
                # instead of the gold prefix. The
                # rollout is no-grad and never commits, so memory remains the
                # only cross-chunk carrier and the differentiable scoring
                # forward has the same size as ordinary teacher forcing.
                prefix = tgt_ids[:-1]
                rollout_supervision = len(tgt_ids)
                if (
                    want_grads
                    and allow_self_prefix
                    and self.args.self_prefix_prob > 0.0
                    and self.schedule_rng.random()
                    < self.args.self_prefix_prob
                ):
                    prefix = self.greedy_prefix(ids, len(tgt_ids) - 1)
                    rollout_supervision = (
                        autoregressive_supervision_length(
                            prefix, tgt_ids))
                    self.self_prefix_used += 1
                # score targets on the same fresh context with fusion active
                full = ids + prefix
                toks = torch.tensor(full, device=self.device)
                address_target = query_evidence.get(
                    str(chunk_index),
                    query_evidence.get(chunk_index))
                if isinstance(address_target, list):
                    address_target = (
                        address_target[-1]
                        if address_target else None)
                oracle_logits_all = None
                if (
                    want_grads
                    and interference_lambda > 0.0
                    and address_target is not None
                    and int(address_target) in interference_snapshots
                ):
                    oracle_logits_all = self._query_forward_from_snapshot(
                        toks,
                        interference_snapshots[int(address_target)],
                        logits_all=True,
                        query_token_count=len(ids))
                supervise_fusion = (
                    want_grads
                    and query_memory_enabled
                    and mem.fusion_mode == "residual_gate"
                    and self.args.fusion_gate_lambda > 0.0)
                if supervise_fusion:
                    memory_relevant = (
                        sample.get("metadata", {}).get("type") != "normal")
                    mem.begin_fusion_gate_supervision(
                        1.0 if memory_relevant else 0.0)
                supervise_evidence = (
                    want_grads
                    and query_memory_enabled
                    and self.args.evidence_lambda > 0.0)
                if supervise_evidence:
                    if self.args.evidence_supervision_position == "query":
                        mem.begin_evidence_supervision(token_range=(
                            max(0, len(ids) - 64), len(ids)))
                    else:
                        mem.begin_evidence_supervision(len(tgt_ids))
                if collect_address and address_target is not None:
                    mem.set_address_target(
                        address_target,
                        # The final query token produces the first answer
                        # token. Supervise that deployment-critical read,
                        # not generic wording elsewhere in the question.
                        (max(0, len(ids) - 1), len(ids)))
                logits_all = self._query_forward(
                    toks, logits_all=True,
                    query_token_count=len(ids))
                if collect_address:
                    mem.clear_address_target()
                if supervise_fusion:
                    fusion_loss = mem.end_fusion_gate_supervision()
                    loss_sum = (
                        loss_sum + self.args.fusion_gate_lambda *
                        fusion_loss * rollout_supervision)
                if supervise_evidence:
                    evidence_loss = mem.end_evidence_supervision()
                    loss_sum = (
                        loss_sum
                        + self.args.evidence_lambda
                        * evidence_loss * rollout_supervision)
                lp = logits_all[len(full) - len(tgt_ids): len(full)]
                tgt_t = torch.tensor(tgt_ids, device=self.device)
                scored_lp = lp[:rollout_supervision]
                scored_target = tgt_t[:rollout_supervision]
                nll = F.cross_entropy(
                    scored_lp.float(), scored_target, reduction="sum")
                loss_sum = loss_sum + nll
                if oracle_logits_all is not None:
                    oracle_lp = oracle_logits_all[
                        len(full) - len(tgt_ids): len(full)
                    ][:rollout_supervision].float()
                    with torch.no_grad():
                        oracle_nll = F.cross_entropy(
                            oracle_lp, scored_target, reduction="mean")
                        current_nll = F.cross_entropy(
                            scored_lp.float().detach(),
                            scored_target, reduction="mean")
                    if bool(oracle_nll < current_nll):
                        temperature = float(
                            self.args.interference_distill_temperature)
                        teacher = F.softmax(
                            oracle_lp / temperature, dim=-1)
                        distill_loss = F.kl_div(
                            F.log_softmax(
                                scored_lp.float() / temperature, dim=-1),
                            teacher,
                            reduction="batchmean",
                        ) * (temperature * temperature)
                        loss_sum = (
                            loss_sum
                            + interference_lambda
                            * distill_loss * rollout_supervision)
                        self.interference_distill_loss_sum += float(
                            distill_loss.detach())
                        self.interference_distill_loss_count += 1
                        self.interference_teacher_used += 1
                if self.args.first_token_lambda > 0.0:
                    first_nll = F.cross_entropy(
                        lp[:1].float(), tgt_t[:1], reduction="mean")
                    loss_sum = (
                        loss_sum
                        + self.args.first_token_lambda
                        * first_nll * rollout_supervision)
                if self.args.prefix_token_lambda > 0.0:
                    prefix_count = min(
                        self.args.prefix_token_count,
                        rollout_supervision)
                    prefix_nll = F.cross_entropy(
                        lp[:prefix_count].float(),
                        tgt_t[:prefix_count],
                        reduction="mean")
                    loss_sum = (
                        loss_sum
                        + self.args.prefix_token_lambda
                        * prefix_nll * rollout_supervision)
                n_label += rollout_supervision
                negatives_by_query = sample.get(
                    "negative_answers_by_query", {})
                negatives = negatives_by_query.get(
                    str(chunk_index),
                    negatives_by_query.get(
                        chunk_index,
                        sample.get("negative_answers", [])))
                if self.args.contrastive_lambda > 0.0 and negatives:
                    ranking_losses = []
                    for negative in negatives[
                        :self.args.contrastive_negatives
                    ]:
                        nkey = ("negative", str(negative))
                        negative_ids = tok_cache.get(nkey)
                        if negative_ids is None:
                            negative_ids = self.tok.encode(
                                str(negative), add_bos=False) + [self.eos]
                            tok_cache[nkey] = negative_ids
                        negative_full = ids + negative_ids[:-1]
                        negative_logits = self._query_forward(
                            torch.tensor(
                                negative_full, device=self.device),
                            logits_all=True,
                            query_token_count=len(ids))
                        mem.discard_captured()
                        negative_selected = negative_logits[
                            len(negative_full) - len(negative_ids):
                            len(negative_full)]
                        negative_target = torch.tensor(
                            negative_ids, device=self.device)
                        negative_nll = F.cross_entropy(
                            negative_selected.float(),
                            negative_target, reduction="mean")
                        ranking_losses.append(negative_nll)
                    if ranking_losses:
                        ranking_loss = stale_answer_margin_loss(
                            nll / rollout_supervision, ranking_losses,
                            self.args.contrastive_margin)
                        # run_sample returns a token-sum objective. Scaling
                        # preserves CE_mean + lambda*ranking_mean after ls/nt.
                        loss_sum = (
                            loss_sum
                            + self.args.contrastive_lambda
                            * ranking_loss * rollout_supervision)
                        self.contrastive_pairs_used += len(ranking_losses)
                pred = lp.argmax(dim=-1)
                correct = int((pred == tgt_t).sum().item())
                n_correct += correct
                sequence_exact += int(correct == len(tgt_ids))
                first_correct += int(pred[0] == tgt_t[0])
                first_total += 1
                prefix_count = min(
                    self.args.prefix_token_count, len(tgt_ids))
                prefix_correct += int(
                    (pred[:prefix_count] == tgt_t[:prefix_count])
                    .sum().item())
                prefix_total += prefix_count
                mem.active = saved_active
                mem.discard_captured()
                # Recall queries are read-only. Neither the question nor its
                # gold/model answer may enter M/S before a later query.
                # Subsequent non-query chunks still commit normally.
        if sample_write_losses and n_label > 0:
            write_loss = torch.stack(sample_write_losses).mean()
            self.write_selection_loss_sum += float(write_loss.detach())
            self.write_selection_loss_count += 1
            loss_sum = (
                loss_sum +
                self.args.write_selection_lambda * write_loss * n_label)
        if sample_retention_losses and n_label > 0:
            retention_loss = torch.stack(
                sample_retention_losses).mean()
            self.retention_loss_sum += float(retention_loss.detach())
            self.retention_loss_count += 1
            loss_sum = (
                loss_sum
                + self.args.retention_lambda
                * retention_loss * n_label)
        if sample_preservation_losses and n_label > 0:
            preservation_loss = torch.stack(
                sample_preservation_losses).mean()
            self.preservation_loss_sum += float(
                preservation_loss.detach())
            self.preservation_loss_count += 1
            loss_sum = (
                loss_sum
                + self.args.preservation_lambda
                * preservation_loss * n_label)
        if collect_address:
            (
                address_loss, diversity_loss,
                address_correct, address_total,
            ) = mem.end_address_training()
            if n_label > 0:
                self.address_loss_sum += float(address_loss.detach())
                self.address_loss_count += 1
                self.address_correct += address_correct
                self.address_total += address_total
                self.key_diversity_loss_sum += float(
                    diversity_loss.detach())
                self.key_diversity_loss_count += 1
                loss_sum = (
                    loss_sum
                    + self.args.address_lambda
                    * address_loss * n_label
                    + self.args.key_diversity_lambda
                    * diversity_loss * n_label)
        self.last_sample_position_stats = {
            "first_correct": first_correct,
            "first_total": first_total,
            "prefix_correct": prefix_correct,
            "prefix_total": prefix_total,
        }
        return loss_sum, n_label, n_correct, sequence_exact

    def run_valid(self, n, tok_cache, query_memory_enabled=True):
        """Evaluate a stable prefix of the held-out split.

        Early stopping requires directly comparable measurements.  Rotating
        through validation samples makes successive losses describe different
        subsets and can select a worse checkpoint merely because its batch was
        easier.
        """
        total, tok, n_ok, correct, exact, query_count = (
            0.0, 0, 0, 0, 0, 0)
        first_correct = 0
        first_total = 0
        prefix_correct = 0
        prefix_total = 0
        for ci in range(min(n, len(self.valid_idx))):
            s_idx = self.valid_idx[ci]
            line = self.valid_strata[s_idx[0]][1][s_idx[1]]
            sample = json.loads(line)
            try:
                with torch.no_grad():
                    ls, nt, nc, ne = self.run_sample(
                        sample, False, tok_cache,
                        query_memory_enabled=query_memory_enabled)
            except Exception:
                continue
            if ls is None or nt == 0:
                continue
            total += ls.item()
            tok += nt
            n_ok += 1
            correct += nc
            exact += ne
            position_stats = self.last_sample_position_stats
            first_correct += position_stats["first_correct"]
            first_total += position_stats["first_total"]
            prefix_correct += position_stats["prefix_correct"]
            prefix_total += position_stats["prefix_total"]
            query_value = sample.get(
                "query_turn_id", len(sample["messages"]) - 1)
            query_count += (
                len(query_value) if isinstance(query_value, list) else 1)
        self.last_valid_position_stats = {
            "first_token_acc": first_correct / max(first_total, 1),
            "prefix_token_acc": (
                prefix_correct / max(prefix_total, 1)),
        }
        return (total / max(tok, 1), tok, n_ok,
                correct / max(tok, 1), exact / max(query_count, 1))

    def _generation_stop_ids(self):
        stop_ids = {self.eos}
        im_end = self.tok.encode("<|im_end|>", add_bos=False)
        if len(im_end) == 1:
            stop_ids.add(im_end[0])
        return stop_ids

    def _greedy_generate(self, prompt_ids, max_tokens, stop_ids):
        if self.decoder is None:
            raise RuntimeError(
                "generation validation requires a token decoder")
        generated = []
        copy_tail = []
        copy_episode = None
        copy_finished = False
        with torch.no_grad():
            for _ in range(max_tokens):
                tokens = torch.tensor(
                    prompt_ids + generated,
                    device=self.device, dtype=torch.long)
                logits = self._query_forward(
                    tokens, query_token_count=len(prompt_ids))
                self.mem.discard_captured()
                token = int(logits.float().argmax().item())
                if (
                    self.args.episode_copy_mode == "continuation"
                    and self.mem.active
                    and self.mem.state_mode == "episodes"
                    and self.mem.last_episode_route_index is not None
                ):
                    episode_index = self.mem.last_episode_route_index
                    if copy_episode is not None:
                        if episode_index != copy_episode:
                            copy_tail = []
                            copy_finished = True
                        elif copy_tail:
                            token = copy_tail.pop(0)
                        else:
                            copy_finished = True
                    elif not copy_finished:
                        remaining = ordered_episode_copy_tail(
                            self.mem.episode_payload_token_ids[
                                episode_index].detach().cpu().tolist(),
                            self.mem.episode_payload_positions[
                                episode_index].detach().cpu().tolist(),
                            generated,
                        )
                        if remaining:
                            copy_episode = episode_index
                            token = remaining[0]
                            copy_tail = list(remaining[1:])
                if token in stop_ids:
                    break
                generated.append(token)
        return self.decoder.decode(generated).strip(), generated

    def run_generation_valid(self, samples_per_task, max_tokens, tok_cache):
        """No-KV greedy validation grouped by the paper's five tasks.

        Tasks 0-3 form the primary recall score. Task 4 is reported as a
        pollution guardrail but cannot improve the checkpoint-selection
        score by copying an answer from the current prompt.
        """
        def empty_stats():
            return {
                "samples": 0,
                "queries": 0,
                "memory_exact": 0,
                "no_memory_exact": 0,
                "oracle_snapshot_exact": 0,
                "first_token_correct": 0,
                "first_token_total": 0,
                "prefix_token_correct": 0,
                "prefix_token_total": 0,
                "recall_queries": 0,
                "recall_exact": 0,
                "no_memory_recall_exact": 0,
                "matching_prefix_tokens": 0,
                "oracle_snapshot_matching_prefix_tokens": 0,
                "target_tokens": 0,
                "prediction_tokens": 0,
                "episode_route_correct": 0,
                "episode_route_total": 0,
                "episode_route_latest": 0,
            }

        stop_ids = self._generation_stop_ids()
        totals = {}
        style_totals = {}
        age_totals = {}
        diagnostic_rows = []
        self.mem.collect_runtime_diagnostics = True
        for task, indices in sorted(self.valid_by_task.items()):
            stats = empty_stats()
            for s_idx in indices[:samples_per_task]:
                line = self.valid_strata[s_idx[0]][1][s_idx[1]]
                sample = json.loads(line)
                style = str(
                    sample.get("metadata", {}).get("style", "unknown"))
                style_stats = style_totals.setdefault(
                    style, empty_stats())
                style_stats["samples"] += 1
                chunks = sample["messages"]
                query_value = sample.get(
                    "query_turn_id", len(chunks) - 1)
                query_indices = (
                    [int(value) for value in query_value]
                    if isinstance(query_value, list)
                    else [int(query_value)])
                query_index_set = set(query_indices)
                query_evidence = generation_query_evidence_map(
                    sample, query_indices)
                self.mem.reset_state()
                memory_snapshots = {}
                episode_by_chunk = {}
                with torch.no_grad():
                    for chunk_index, chunk in enumerate(chunks):
                        is_query = chunk_index in query_index_set
                        text, target = render_chunk(chunk, is_query)
                        cache_key = ("generation", text)
                        ids = tok_cache.get(cache_key)
                        if ids is None:
                            ids = self.tok.encode(text, add_bos=True)
                            tok_cache[cache_key] = ids
                        if not is_query:
                            self.backbone(
                                torch.tensor(
                                    ids, device=self.device,
                                    dtype=torch.long),
                                memory_v6=self.mem, fuse_start=0)
                            self.mem.commit_all()
                            if self.mem.state_mode == "episodes":
                                episode_by_chunk[chunk_index] = (
                                    self.mem.episode_M.shape[1] - 1)
                                if (
                                    self.args.debug_episode_addresses
                                    and stats["samples"] == 0
                                    and self.mem.episode_address_source
                                    == "embedding"
                                ):
                                    selected_ids = [
                                        int(token_id)
                                        for token_id in
                                        self.mem.episode_identity_token_ids[
                                            -1].detach().cpu().tolist()
                                        if int(token_id) >= 0
                                    ]
                                    print(json.dumps({
                                        "phase": "episode_address_debug",
                                        "chunk": chunk_index,
                                        "memory_targets": sample.get(
                                            "memory_targets_by_message",
                                            {}).get(str(chunk_index)),
                                        "token_ids": selected_ids,
                                        "pieces": [
                                            self.decoder.decode(
                                                [token_id])
                                            for token_id in selected_ids
                                        ],
                                    }, separators=(",", ":")),
                                          flush=True)
                                if (
                                    self.args.debug_episode_payload
                                    and stats["samples"] == 0
                                ):
                                    payload_ids = [
                                        int(token_id)
                                        for token_id in
                                        self.mem.episode_payload_token_ids[
                                            -1].detach().cpu().tolist()
                                        if int(token_id) >= 0
                                    ]
                                    payload_positions = [
                                        int(position)
                                        for position in
                                        self.mem.episode_payload_positions[
                                            -1].detach().cpu().tolist()
                                        if int(position) >= 0
                                    ]
                                    payload_votes = [
                                        int(vote)
                                        for vote in
                                        self.mem.episode_payload_votes[
                                            -1].detach().cpu().tolist()
                                        if int(vote) > 0
                                    ]
                                    print(json.dumps({
                                        "phase": "episode_payload_debug",
                                        "chunk": chunk_index,
                                        "memory_targets": sample.get(
                                            "memory_targets_by_message",
                                            {}).get(str(chunk_index)),
                                        "positions": payload_positions,
                                        "votes": payload_votes,
                                        "token_ids": payload_ids,
                                        "pieces": [
                                            self.decoder.decode(
                                                [token_id])
                                            for token_id in payload_ids
                                        ],
                                    }, separators=(",", ":")),
                                          flush=True)
                            if self.args.generation_oracle_snapshot:
                                memory_snapshots[chunk_index] = (
                                    self.mem.clone_runtime_state())
                            continue

                        target = target.strip()
                        evidence_index = query_evidence.get(
                            str(chunk_index),
                            query_evidence.get(chunk_index))
                        if isinstance(evidence_index, list):
                            evidence_index = (
                                evidence_index[-1]
                                if evidence_index else None)
                        if evidence_index is None:
                            age_key = "unknown"
                        else:
                            age = sum(
                                int(index not in query_index_set)
                                for index in range(
                                    int(evidence_index) + 1,
                                    chunk_index))
                            if age == 0:
                                age_key = "0"
                            elif age <= 3:
                                age_key = "1-3"
                            elif age <= 7:
                                age_key = "4-7"
                            else:
                                age_key = "8+"
                        age_stats = age_totals.setdefault(
                            age_key, empty_stats())
                        age_stats["samples"] += 1
                        target_key = ("generation_target", target)
                        target_ids = tok_cache.get(target_key)
                        if target_ids is None:
                            target_ids = self.tok.encode(
                                target, add_bos=False)
                            tok_cache[target_key] = target_ids
                        answer_limit = min(
                            max_tokens, max(8, len(target_ids) + 8))
                        if (
                            self.args.generation_oracle_episode
                            and self.mem.state_mode == "episodes"
                            and evidence_index is not None
                            and int(evidence_index) in episode_by_chunk
                        ):
                            self.mem.set_episode_route_override(
                                episode_by_chunk[int(evidence_index)])
                        saved_state = self.mem.clone_runtime_state()
                        prediction, prediction_ids = self._greedy_generate(
                            ids, answer_limit, stop_ids)
                        selected_episode = (
                            self.mem.last_episode_route_index)
                        selected_route_logits = (
                            None
                            if self.mem.last_episode_route_logits is None
                            else self.mem.last_episode_route_logits.tolist())
                        self.mem.restore_runtime_state(saved_state)
                        self.mem.active = False
                        no_memory_prediction, _ = self._greedy_generate(
                            ids, answer_limit, stop_ids)
                        self.mem.restore_runtime_state(saved_state)
                        oracle_prediction = None
                        oracle_prediction_ids = []
                        if (
                            self.args.generation_oracle_snapshot
                            and evidence_index is not None
                            and int(evidence_index) in memory_snapshots
                        ):
                            self.mem.restore_runtime_state(
                                memory_snapshots[int(evidence_index)])
                            (
                                oracle_prediction,
                                oracle_prediction_ids,
                            ) = self._greedy_generate(
                                ids, answer_limit, stop_ids)
                            self.mem.restore_runtime_state(saved_state)

                        stats["queries"] += 1
                        stats["memory_exact"] += int(
                            prediction == target)
                        stats["no_memory_exact"] += int(
                            no_memory_prediction == target)
                        stats["oracle_snapshot_exact"] += int(
                            oracle_prediction == target)
                        matched_prefix = matching_prefix_length(
                            prediction_ids, target_ids)
                        oracle_matched_prefix = matching_prefix_length(
                            oracle_prediction_ids, target_ids)
                        stats["matching_prefix_tokens"] += matched_prefix
                        stats[
                            "oracle_snapshot_matching_prefix_tokens"
                        ] += oracle_matched_prefix
                        stats["target_tokens"] += len(target_ids)
                        stats["prediction_tokens"] += len(prediction_ids)
                        style_stats["queries"] += 1
                        style_stats["memory_exact"] += int(
                            prediction == target)
                        style_stats["no_memory_exact"] += int(
                            no_memory_prediction == target)
                        style_stats["oracle_snapshot_exact"] += int(
                            oracle_prediction == target)
                        age_stats["queries"] += 1
                        age_stats["memory_exact"] += int(
                            prediction == target)
                        age_stats["no_memory_exact"] += int(
                            no_memory_prediction == target)
                        age_stats["oracle_snapshot_exact"] += int(
                            oracle_prediction == target)
                        expected_episode = (
                            episode_by_chunk.get(int(evidence_index))
                            if (
                                evidence_index is not None
                                and self.mem.state_mode == "episodes"
                            ) else None)
                        if (
                            self.args.debug_episode_addresses
                            and stats["samples"] == 0
                            and self.mem.state_mode == "episodes"
                        ):
                            print(json.dumps({
                                "phase": "episode_route_debug",
                                "chunk": chunk_index,
                                "evidence_chunk": evidence_index,
                                "expected_episode": expected_episode,
                                "selected_episode": selected_episode,
                                "route_logits": selected_route_logits,
                                "query_token_count": len(ids),
                                "query_pieces": [
                                    self.decoder.decode([int(token_id)])
                                    for token_id in ids
                                ],
                            }, separators=(",", ":")), flush=True)
                        if (
                            self.args.debug_generation_errors
                            and prediction != target
                        ):
                            print(json.dumps({
                                "phase": "generation_error_debug",
                                "task": task,
                                "style": style,
                                "chunk": chunk_index,
                                "age": age_key,
                                "target": target,
                                "prediction": prediction,
                                "target_ids": target_ids,
                                "prediction_ids": prediction_ids,
                                "expected_episode": expected_episode,
                                "selected_episode": selected_episode,
                            }, separators=(",", ":")), flush=True)
                        if (
                            selected_episode is not None
                            and expected_episode is not None
                        ):
                            route_correct = int(
                                selected_episode == expected_episode)
                            route_latest = int(
                                selected_episode
                                == self.mem.episode_M.shape[1] - 1)
                            for route_stats in (
                                stats, style_stats, age_stats
                            ):
                                route_stats[
                                    "episode_route_total"] += 1
                                route_stats[
                                    "episode_route_correct"] += route_correct
                                route_stats[
                                    "episode_route_latest"] += route_latest
                        style_stats[
                            "matching_prefix_tokens"] += matched_prefix
                        style_stats[
                            "oracle_snapshot_matching_prefix_tokens"
                        ] += oracle_matched_prefix
                        style_stats["target_tokens"] += len(target_ids)
                        style_stats[
                            "prediction_tokens"] += len(prediction_ids)
                        age_stats[
                            "matching_prefix_tokens"] += matched_prefix
                        age_stats[
                            "oracle_snapshot_matching_prefix_tokens"
                        ] += oracle_matched_prefix
                        age_stats["target_tokens"] += len(target_ids)
                        age_stats[
                            "prediction_tokens"] += len(prediction_ids)
                        if target_ids:
                            stats["first_token_total"] += 1
                            stats["first_token_correct"] += int(
                                bool(prediction_ids)
                                and prediction_ids[0] == target_ids[0])
                            style_stats["first_token_total"] += 1
                            style_stats["first_token_correct"] += int(
                                bool(prediction_ids)
                                and prediction_ids[0] == target_ids[0])
                            age_stats["first_token_total"] += 1
                            age_stats["first_token_correct"] += int(
                                bool(prediction_ids)
                                and prediction_ids[0] == target_ids[0])
                            prefix_count = min(
                                self.args.prefix_token_count,
                                len(target_ids))
                            stats["prefix_token_total"] += prefix_count
                            prefix_correct = sum(
                                int(
                                    index < len(prediction_ids)
                                    and prediction_ids[index]
                                    == target_ids[index])
                                for index in range(prefix_count))
                            stats["prefix_token_correct"] += prefix_correct
                            style_stats[
                                "prefix_token_total"] += prefix_count
                            style_stats[
                                "prefix_token_correct"] += prefix_correct
                            age_stats[
                                "prefix_token_total"] += prefix_count
                            age_stats[
                                "prefix_token_correct"] += prefix_correct
                        if (
                            task < 4
                            and target != "No information available."
                        ):
                            stats["recall_queries"] += 1
                            stats["recall_exact"] += int(
                                prediction == target)
                            stats["no_memory_recall_exact"] += int(
                                no_memory_prediction == target)
                            style_stats["recall_queries"] += 1
                            style_stats["recall_exact"] += int(
                                prediction == target)
                            style_stats[
                                "no_memory_recall_exact"] += int(
                                    no_memory_prediction == target)
                            age_stats["recall_queries"] += 1
                            age_stats["recall_exact"] += int(
                                prediction == target)
                            age_stats[
                                "no_memory_recall_exact"] += int(
                                    no_memory_prediction == target)
                        diagnostic_rows.append(
                            self.mem.runtime_diagnostics())
                        # Query and generated/gold answers are read-only.
                        # Only later non-query chunks may modify M/S.
                stats["samples"] += 1
            queries = max(stats["queries"], 1)
            stats["memory_exact"] /= queries
            stats["no_memory_exact"] /= queries
            stats["oracle_snapshot_exact"] /= queries
            stats["first_token_acc"] = (
                stats["first_token_correct"]
                / max(stats["first_token_total"], 1))
            stats["prefix_token_acc"] = (
                stats["prefix_token_correct"]
                / max(stats["prefix_token_total"], 1))
            recall_queries = max(stats["recall_queries"], 1)
            stats["recall_exact"] /= recall_queries
            stats["no_memory_recall_exact"] /= recall_queries
            stats["mean_matching_prefix_tokens"] = (
                stats["matching_prefix_tokens"] / queries)
            stats["oracle_snapshot_mean_matching_prefix_tokens"] = (
                stats["oracle_snapshot_matching_prefix_tokens"] / queries)
            stats["mean_target_tokens"] = (
                stats["target_tokens"] / queries)
            stats["mean_prediction_tokens"] = (
                stats["prediction_tokens"] / queries)
            stats["episode_route_acc"] = (
                stats["episode_route_correct"]
                / max(stats["episode_route_total"], 1))
            stats["episode_route_latest_ratio"] = (
                stats["episode_route_latest"]
                / max(stats["episode_route_total"], 1))
            totals[task] = stats

        for stats in (
            list(style_totals.values()) + list(age_totals.values())
        ):
            queries = max(stats["queries"], 1)
            stats["memory_exact"] /= queries
            stats["no_memory_exact"] /= queries
            stats["oracle_snapshot_exact"] /= queries
            stats["first_token_acc"] = (
                stats["first_token_correct"]
                / max(stats["first_token_total"], 1))
            stats["prefix_token_acc"] = (
                stats["prefix_token_correct"]
                / max(stats["prefix_token_total"], 1))
            recall_queries = max(stats["recall_queries"], 1)
            stats["recall_exact"] /= recall_queries
            stats["no_memory_recall_exact"] /= recall_queries
            stats["mean_matching_prefix_tokens"] = (
                stats["matching_prefix_tokens"] / queries)
            stats["oracle_snapshot_mean_matching_prefix_tokens"] = (
                stats["oracle_snapshot_matching_prefix_tokens"] / queries)
            stats["mean_target_tokens"] = (
                stats["target_tokens"] / queries)
            stats["mean_prediction_tokens"] = (
                stats["prediction_tokens"] / queries)
            stats["episode_route_acc"] = (
                stats["episode_route_correct"]
                / max(stats["episode_route_total"], 1))
            stats["episode_route_latest_ratio"] = (
                stats["episode_route_latest"]
                / max(stats["episode_route_total"], 1))

        recall_tasks = [
            task for task in (0, 1, 2, 3)
            if task in totals and totals[task]["recall_queries"] > 0]
        selection_tasks = parse_task_filter(
            self.args.generation_selection_tasks)
        if selection_tasks is not None:
            recall_tasks = [
                task for task in recall_tasks
                if task in selection_tasks]
        recall_exact = (
            sum(totals[task]["recall_exact"] for task in recall_tasks)
            / max(len(recall_tasks), 1))
        no_memory_recall_exact = (
            sum(
                totals[task]["no_memory_recall_exact"]
                for task in recall_tasks)
            / max(len(recall_tasks), 1))
        pollution_exact = (
            totals.get(4, {}).get("memory_exact", 0.0))
        diagnostic_keys = (
            "selected_ratio", "alpha", "beta_mean", "beta_min",
            "beta_max", "memory_attention_ratio")
        memory_stats = {
            key: (
                sum(float(row[key]) for row in diagnostic_rows)
                / max(len(diagnostic_rows), 1))
            for key in diagnostic_keys
        }
        self.mem.collect_runtime_diagnostics = False
        return {
            "recall_exact": recall_exact,
            "no_memory_recall_exact": no_memory_recall_exact,
            "pollution_exact": pollution_exact,
            "memory_stats": memory_stats,
            "by_task": totals,
            "by_style": style_totals,
            "by_memory_age": age_totals,
        }

    def export(self, path):
        self.mem.export(path)
        if self.answer_decoder is not None:
            decoder_path = path + ".bnanswer"
            self.answer_decoder.save(decoder_path)
            print(f'{{"exported_answer_decoder":"{decoder_path}",'
                  f'"width":{self.answer_decoder.width}}}', flush=True)
        lora_tensors = []
        if self.backbone_lora is not None:
            lora_tensors.extend(self.backbone_lora.tensors())
        if self.output_lora is not None:
            lora_tensors.append(self.output_lora)
        if lora_tensors:
            lora_path = path + ".bnlora"
            save_lora_bundle(lora_path, lora_tensors)
            print(f'{{"exported_lora":"{lora_path}",'
                  f'"tensors":{len(lora_tensors)}}}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("data")
    ap.add_argument("--valid-data", default=None,
                    help="independent validation JSONL directory")
    ap.add_argument("--output", default="metis_v6.bnmem")
    ap.add_argument(
        "--final-output", default=None,
        help="also export the final parameters, even when validation selected "
             "an earlier best checkpoint")
    ap.add_argument(
        "--init-memory", default=None,
        help="initialize all trainable memory parameters from BNMEM1")
    ap.add_argument(
        "--allow-query-rank-expand", action="store_true",
        help="zero-pad a lower-rank checkpoint into --query-rank")
    ap.add_argument(
        "--allow-init-selection-mismatch", action="store_true",
        help="load checkpoint weights while adopting the current --rho; "
             "all backbone, geometry, layer, and query-mode checks remain")
    ap.add_argument(
        "--answer-decoder-width", type=int, default=0,
        help="width of the nonlinear full-rank memory answer decoder; "
             "zero disables it")
    ap.add_argument(
        "--memory-aware-answer-decoder", action="store_true",
        help="condition the non-LoRA answer decoder on the mean memory "
             "read vector across active memory layers")
    ap.add_argument(
        "--structured-memory-answer-decoder", action="store_true",
        help="cross-attend from each query token to all per-layer memory "
             "read vectors without averaging away layer structure")
    ap.add_argument(
        "--init-answer-decoder", default=None,
        help="initialize the non-LoRA memory answer decoder from BNANSWER1")
    ap.add_argument("--output-lora-rank", type=int, default=0)
    ap.add_argument("--output-lora-alpha", type=float, default=16.0)
    ap.add_argument("--init-output-lora", default=None)
    ap.add_argument("--backbone-lora-rank", type=int, default=0)
    ap.add_argument("--backbone-lora-alpha", type=float, default=16.0)
    ap.add_argument("--backbone-lora-targets", default="q,v,o")
    ap.add_argument("--backbone-lora-blocks", default="all")
    ap.add_argument("--init-backbone-lora", default=None)
    ap.add_argument(
        "--freeze-lora", action="store_true",
        help="load backbone/output LoRA for forward use but do not update it")
    ap.add_argument(
        "--freeze-memory", action="store_true",
        help="keep BNMEM parameters fixed while training another enabled "
             "memory component")
    ap.add_argument(
        "--train-retrieval-only", action="store_true",
        help="update only memory query/key projections; freeze value, "
             "fusion, gate, and write parameters")
    ap.add_argument(
        "--train-episode-address-only", action="store_true",
        help="freeze storage and query projections; train only the "
             "routing-only episode address projection")
    ap.add_argument("--lib", default=None)
    ap.add_argument("--tok-probe", default="tok_probe")
    ap.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
        help="training device; auto prefers CUDA, then Apple MPS")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument(
        "--fusion-mode", choices=("fixed", "residual_gate"),
        default="fixed",
        help="fixed gamma blend or identity-preserving per-token residual gate")
    ap.add_argument("--gdu-alpha-init", type=float, default=1.0)
    ap.add_argument("--gdu-beta-init", type=float, default=1.0)
    ap.add_argument(
        "--retention-target", type=float, default=0.99,
        help="minimum preferred global M/S retention per commit")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--alpha-max-tokens", type=int, default=0)
    ap.add_argument("--alpha-max-fraction", type=float, default=0.0)
    ap.add_argument(
        "--value-alpha-max-tokens", type=int, default=-1,
        help="payload-selector token cap; negative inherits "
             "--alpha-max-tokens")
    ap.add_argument(
        "--value-alpha-max-fraction", type=float, default=-1.0,
        help="payload-selector fraction cap; negative inherits "
             "--alpha-max-fraction")
    ap.add_argument(
        "--write-orthogonalization", type=float, default=0.0,
        help="strength for projecting new write keys out of distinct old "
             "address subspaces")
    ap.add_argument(
        "--update-similarity-threshold", type=float, default=0.85,
        help="address cosine similarity above which a write is treated as "
             "an intentional update rather than protected from overwrite")
    ap.add_argument("--beta-scale", type=float, default=0.9)
    ap.add_argument("--query-rank", type=int, default=128,
                    help="query projection rank; zero selects full rank")
    ap.add_argument(
        "--query-mode",
        choices=("independent", "backbone_delta"),
        default="independent",
        help="paper-aligned independent query or legacy compatibility delta")
    ap.add_argument("--query-gate-lambda", type=float, default=1e-4,
                    help="NAS-NG sparsity weight; zero disables rank gates")
    ap.add_argument("--query-gate-temperature", type=float, default=1.0)
    ap.add_argument("--query-gate-threshold", type=float, default=0.5)
    ap.add_argument("--query-gate-min-rank", type=int, default=1)
    ap.add_argument("--kv-rank", type=int, default=128,
                    help="K/V projection rank; zero selects full rank")
    ap.add_argument("--kv-gate-lambda", type=float, default=1e-4,
                    help="NAS-NG sparsity weight for K/V rank neurons")
    ap.add_argument("--kv-gate-temperature", type=float, default=1.0)
    ap.add_argument("--kv-gate-threshold", type=float, default=0.5)
    ap.add_argument("--kv-gate-min-rank", type=int, default=1)
    ap.add_argument("--layer-gate-lambda", type=float, default=1e-4,
                    help="NAS-NG sparsity weight for memory layers")
    ap.add_argument("--layer-gate-temperature", type=float, default=1.0)
    ap.add_argument("--layer-gate-threshold", type=float, default=0.5)
    ap.add_argument("--layer-gate-min-layers", type=int, default=1)
    ap.add_argument("--nas-lr", type=float, default=1e-2,
                    help="validation-gradient learning rate for NG logits")
    ap.add_argument("--nas-warmup", type=int, default=200,
                    help="weight-only steps before NAS gate updates")
    ap.add_argument("--nas-every", type=int, default=4,
                    help="run one validation NAS step every N weight steps")
    ap.add_argument("--denom-mode",
                    choices=("signed_plus_one", "abs_plus_one"),
                    default="signed_plus_one",
                    help="normalized-memory read denominator")
    ap.add_argument(
        "--state-mode", choices=("delta", "slots", "banked", "episodes"),
        default="delta",
        help="single delta matrix, token slots, routed banks, or immutable "
             "latent episodes")
    ap.add_argument(
        "--max-memory-slots", type=int, default=4096,
        help="maximum source-token K/V slots retained by memory")
    ap.add_argument(
        "--slot-temperature", type=float, default=0.07,
        help="softmax temperature for query-to-slot retrieval")
    ap.add_argument(
        "--max-memory-episodes", type=int, default=64,
        help="maximum immutable latent episodes retained in memory")
    ap.add_argument(
        "--episode-temperature", type=float, default=0.1,
        help="softmax temperature for neural episode activation")
    ap.add_argument(
        "--episode-read-mode",
        choices=("soft", "hard"), default="soft",
        help="mix episode reads softly or activate one episode")
    ap.add_argument(
        "--episode-address-rank", type=int, default=64,
        help="rank of the routing-only address delta projection")
    ap.add_argument(
        "--episode-address-mode",
        choices=("storage", "shared_query"), default="storage",
        help="derive episode addresses from storage keys or the same query "
             "projection used during recall")
    ap.add_argument(
        "--episode-address-granularity",
        choices=("pooled", "tokens"), default="pooled",
        help="route from one pooled address or token-level latent addresses")
    ap.add_argument(
        "--episode-address-tokens", type=int, default=16,
        help="maximum latent address tokens retained per episode and layer")
    ap.add_argument(
        "--episode-payload-tokens", type=int, default=32,
        help="maximum ordered value-selector tokens retained per episode")
    ap.add_argument(
        "--episode-payload-min-layer-votes", type=int, default=1,
        help="minimum number of memory layers selecting a token before it "
             "may enter the ordered episode payload")
    ap.add_argument(
        "--episode-address-source",
        choices=("contextual", "embedding"), default="contextual",
        help="build activation addresses from contextual layer states or "
             "tied backbone token embeddings")
    ap.add_argument(
        "--episode-identity-uniqueness-floor", type=float, default=0.05,
        help="minimum weight retained for address tokens shared across "
             "multiple episodes")
    ap.add_argument(
        "--episode-identity-match-scale", type=float, default=0.0,
        help="positive values apply a sharp near-identity activation kernel "
             "to token-embedding similarity")
    ap.add_argument(
        "--episode-identity-ngram", type=int, default=1,
        help="number of consecutive selected token embeddings combined into "
             "one sequence-aware activation address")
    ap.add_argument(
        "--episode-identity-route-mode",
        choices=("maxsim", "span"), default="maxsim",
        help="activate by unordered token/ngram matches or by aligned "
             "longest-span sequence matching")
    ap.add_argument(
        "--episode-identity-selection-mode",
        choices=("topk", "window"), default="topk",
        help="retain independently selected address tokens or one contiguous "
             "highest-scoring latent token window")
    ap.add_argument(
        "--episode-router-mode",
        choices=("cosine", "token_metric"), default="cosine",
        help="fixed cosine activation or a learned latent token matcher")
    ap.add_argument(
        "--episode-router-rank", type=int, default=32,
        help="hidden rank of the learned token-level episode matcher")
    ap.add_argument(
        "--memory-banks", type=int, default=1,
        help="number of independent latent M/S banks in banked mode")
    ap.add_argument(
        "--bank-temperature", type=float, default=0.1,
        help="softmax temperature for neural latent-bank activation")
    ap.add_argument(
        "--bank-write-top-k", type=int, default=1,
        help="number of latent banks updated by each committed memory")
    ap.add_argument(
        "--bank-read-mode", choices=("soft", "hard"), default="soft",
        help="mix bank reads softly or activate one latent bank per token")
    ap.add_argument(
        "--bank-router-mode",
        choices=("learned", "sequential"), default="learned",
        help="learn a write/read router or allocate writes sequentially "
             "and activate banks from their latent S content")
    ap.add_argument(
        "--write-mode",
        choices=("paired_tokens", "factorized", "dual_tokens"),
        default="paired_tokens",
        help="pair K/V per selected token, aggregate address and payload "
             "into one K-to-V write, or select address and payload token "
             "associations independently")
    ap.add_argument(
        "--bank-router-lr-scale", type=float, default=5.0,
        help="learning-rate multiplier for the shared write/read router")
    ap.add_argument(
        "--write-selector-lr-scale", type=float, default=1.0,
        help="learning-rate multiplier for the payload token selector")
    ap.add_argument(
        "--retrieval-windows", type=int, default=0,
        help="model-selected memory windows inserted into each query")
    ap.add_argument(
        "--retrieval-window-size", type=int, default=32,
        help="tokens per inserted memory window")
    ap.add_argument(
        "--retrieval-layer-mode",
        choices=("last", "mean", "max", "rrf"), default="max",
        help="layer aggregation for retrieved memory windows")
    ap.add_argument(
        "--query-read-mode",
        choices=(
            "inline", "two_pass", "contextual", "episode_activation"),
        default="inline",
        help="read committed memory inside one prefill, or derive one "
             "frozen read/context/episode activation from a backbone-only "
             "query prefill and replay the query")
    ap.add_argument(
        "--two-pass-query-tokens", type=int, default=16,
        help="final query-prefill tokens pooled for each two-pass read")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument(
        "--grad-accum", type=int, default=1,
        help="micro-batches accumulated per optimizer step")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--valid", type=int, default=300)
    ap.add_argument(
        "--train-tasks", default="all",
        help="all or comma-separated paper task ids used for optimization")
    ap.add_argument(
        "--valid-tasks", default="all",
        help="all or comma-separated paper task ids used for validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--valid-every", type=int, default=0)
    ap.add_argument("--valid-subset", type=int, default=32)
    ap.add_argument(
        "--generation-valid-every", type=int, default=0,
        help="run no-KV greedy validation every N optimizer steps; "
             "zero disables it")
    ap.add_argument(
        "--generation-valid-samples-per-task", type=int, default=0,
        help="greedy validation samples per paper task")
    ap.add_argument(
        "--generation-valid-max-tokens", type=int, default=160,
        help="maximum no-KV greedy answer tokens per validation query")
    ap.add_argument(
        "--generation-oracle-snapshot", action="store_true",
        help="diagnose later-write interference by also generating from "
             "the target memory's post-commit M/S snapshot")
    ap.add_argument(
        "--generation-oracle-episode", action="store_true",
        help="diagnose immutable latent episodes by activating the episode "
             "known to contain the target memory")
    ap.add_argument(
        "--generation-selection-tasks", default="all",
        help="paper task ids whose greedy recall selects checkpoints; "
             "all uses recall tasks 0 through 3")
    ap.add_argument(
        "--selection-metric",
        choices=("loss", "generation_recall_exact"), default="loss",
        help="checkpoint and early-stop metric")
    ap.add_argument(
        "--min-steps-before-early-stop", type=int, default=0,
        help="never early-stop before this many optimizer steps")
    ap.add_argument(
        "--early-stop-patience", type=int, default=5,
        help="number of selection evaluations without improvement")
    ap.add_argument(
        "--self-prefix-prob", type=float, default=0.0,
        help="probability of using a no-KV greedy model prefix for "
             "training targets instead of the gold teacher-forcing prefix")
    ap.add_argument(
        "--first-token-lambda", type=float, default=0.0,
        help="extra mean NLL weight on the first generated answer token")
    ap.add_argument(
        "--prefix-token-lambda", type=float, default=0.0,
        help="extra mean NLL weight on the first answer-token prefix")
    ap.add_argument(
        "--prefix-token-count", type=int, default=8,
        help="answer prefix length used for loss and position metrics")
    ap.add_argument(
        "--contrastive-lambda", type=float, default=1.0,
        help="weight for positive-vs-stale answer sequence margin loss")
    ap.add_argument(
        "--contrastive-margin", type=float, default=1.0,
        help="required NLL margin between correct and stale answers")
    ap.add_argument(
        "--contrastive-negatives", type=int, default=1,
        help="maximum stale/deleted answers scored per sample")
    ap.add_argument(
        "--evidence-lambda", type=float, default=0.0,
        help="weight for attention mass supervision on labelled evidence")
    ap.add_argument(
        "--write-selection-lambda", type=float, default=0.0,
        help="direct supervision for AlphaTopP importance and GDU write beta")
    ap.add_argument(
        "--retention-lambda", type=float, default=0.0,
        help="penalty weight when global per-commit retention falls below "
             "--retention-target")
    ap.add_argument(
        "--preservation-lambda", type=float, default=0.0,
        help="penalty weight for changes to existing latent readouts caused "
             "by each later memory write")
    ap.add_argument(
        "--interference-distill-lambda", type=float, default=0.0,
        help="training-only KL weight that matches full-state query logits "
             "to the target memory's pre-interference M/S snapshot")
    ap.add_argument(
        "--interference-distill-temperature", type=float, default=1.0)
    ap.add_argument(
        "--address-lambda", type=float, default=0.0,
        help="training-only query-to-write neural address contrastive loss")
    ap.add_argument(
        "--address-temperature", type=float, default=0.1)
    ap.add_argument(
        "--key-diversity-lambda", type=float, default=0.0,
        help="training-only penalty for correlated memory write addresses")
    ap.add_argument(
        "--key-diversity-margin", type=float, default=0.1)
    ap.add_argument(
        "--fusion-gate-lambda", type=float, default=0.0,
        help="direct memory-relevant versus normal-query fusion supervision")
    ap.add_argument(
        "--evidence-label-mode",
        choices=("answer_span", "chunk"), default="answer_span",
        help="positive slot labels within annotated evidence chunks")
    ap.add_argument(
        "--evidence-supervision-position",
        choices=("answer", "query"), default="answer",
        help="apply retrieval supervision on answer-prefix positions or "
             "the final 64 query-prompt tokens")
    ap.add_argument("--bb-init", action="store_true", default=True)
    ap.add_argument("--no-bb-init", dest="bb_init", action="store_false")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument(
        "--eval-only", action="store_true",
        help="evaluate --init-memory on the selected validation data "
             "without training or exporting")
    ap.add_argument(
        "--debug-episode-addresses", action="store_true",
        help="print selected latent address token pieces for the first "
             "generation-validation sample")
    ap.add_argument(
        "--debug-episode-payload", action="store_true",
        help="print ordered value-selector token pieces for the first "
             "generation-validation sample")
    ap.add_argument(
        "--episode-copy-mode",
        choices=("none", "continuation"), default="none",
        help="optionally continue an answer only inside the neurally activated "
             "episode's ordered payload")
    ap.add_argument(
        "--debug-generation-errors", action="store_true",
        help="print target and prediction details for failed generation "
             "validation queries")
    ap.add_argument(
        "--train-episode-router-only", action="store_true",
        help="freeze memory content projections and train only the latent "
             "episode activation matcher")
    ap.add_argument(
        "--train-episode-selector-only", action="store_true",
        help="freeze memory contents and train only the token selector used "
             "to form latent episode activation addresses")
    ap.add_argument("--oversample-distract", type=int, default=1,
                    help="rounds per pass for distract strata (v10run: 3)")
    for task, (start, end) in TASK_WEIGHT_DEFAULTS.items():
        ap.add_argument(
            f"--task{task}-weight-start", type=float, default=start)
        ap.add_argument(
            f"--task{task}-weight-end", type=float, default=end)
    args = ap.parse_args()
    if (
        args.train_episode_address_only
        or args.train_episode_router_only
        or args.train_episode_selector_only
    ):
        args.train_retrieval_only = True
    if sum(map(int, (
        args.train_episode_address_only,
        args.train_episode_router_only,
        args.train_episode_selector_only,
    ))) > 1:
        ap.error(
            "episode address/router/selector-only modes are mutually "
            "exclusive")
    try:
        parse_task_filter(args.train_tasks)
        parse_task_filter(args.valid_tasks)
        parse_task_filter(args.generation_selection_tasks)
    except ValueError as error:
        ap.error(str(error))
    if args.retrieval_windows < 0:
        ap.error("--retrieval-windows must be non-negative")
    if (
        args.train_episode_address_only
        and args.state_mode != "episodes"
    ):
        ap.error(
            "--train-episode-address-only requires --state-mode episodes")
    if (
        args.train_episode_address_only
        and args.episode_address_mode != "storage"
    ):
        ap.error(
            "--train-episode-address-only requires storage address mode")
    if (
        args.train_episode_router_only
        and (
            args.state_mode != "episodes"
            or args.episode_router_mode != "token_metric"
        )
    ):
        ap.error(
            "--train-episode-router-only requires episodes with "
            "--episode-router-mode token_metric")
    if (
        args.train_episode_selector_only
        and (
            args.state_mode != "episodes"
            or args.episode_address_source != "embedding"
        )
    ):
        ap.error(
            "--train-episode-selector-only requires episodes with "
            "--episode-address-source embedding")
    if args.alpha_max_tokens < 0:
        ap.error("--alpha-max-tokens must be non-negative")
    if not 0.0 <= args.alpha_max_fraction <= 1.0:
        ap.error("--alpha-max-fraction must be in [0, 1]")
    if args.value_alpha_max_tokens < -1:
        ap.error("--value-alpha-max-tokens must be -1 or non-negative")
    if (
        args.value_alpha_max_fraction < -1.0
        or args.value_alpha_max_fraction > 1.0
    ):
        ap.error(
            "--value-alpha-max-fraction must be -1 or in [0, 1]")
    if not 0.0 <= args.write_orthogonalization <= 1.0:
        ap.error("--write-orthogonalization must be in [0, 1]")
    if not 0.0 <= args.update_similarity_threshold <= 1.0:
        ap.error("--update-similarity-threshold must be in [0, 1]")
    if args.retrieval_window_size < 1:
        ap.error("--retrieval-window-size must be positive")
    if args.two_pass_query_tokens < 1:
        ap.error("--two-pass-query-tokens must be positive")
    if not 0.0 <= args.self_prefix_prob <= 1.0:
        ap.error("--self-prefix-prob must be in [0, 1]")
    if args.first_token_lambda < 0.0:
        ap.error("--first-token-lambda must be non-negative")
    if args.prefix_token_lambda < 0.0:
        ap.error("--prefix-token-lambda must be non-negative")
    if args.prefix_token_count < 1:
        ap.error("--prefix-token-count must be positive")
    if (
        args.query_read_mode in (
            "two_pass", "contextual", "episode_activation")
        and args.retrieval_windows > 0
    ):
        ap.error(
            "--query-read-mode two_pass is not compatible with text-window "
            "retrieval")
    if (
        args.query_read_mode in (
            "two_pass", "contextual", "episode_activation")
        and args.evidence_lambda > 0.0
    ):
        ap.error(
            "--query-read-mode two_pass currently requires "
            "--evidence-lambda 0")
    if (
        args.query_read_mode == "episode_activation"
        and args.state_mode != "episodes"
    ):
        ap.error(
            "--query-read-mode episode_activation requires "
            "--state-mode episodes")
    if (
        args.episode_address_source == "embedding"
        and args.query_read_mode != "episode_activation"
    ):
        ap.error(
            "--episode-address-source embedding requires "
            "--query-read-mode episode_activation")
    if args.contrastive_lambda < 0.0:
        ap.error("--contrastive-lambda must be non-negative")
    if args.contrastive_margin < 0.0:
        ap.error("--contrastive-margin must be non-negative")
    if args.contrastive_negatives < 1:
        ap.error("--contrastive-negatives must be positive")
    if args.evidence_lambda < 0.0:
        ap.error("--evidence-lambda must be non-negative")
    if args.write_selection_lambda < 0.0:
        ap.error("--write-selection-lambda must be non-negative")
    if args.fusion_gate_lambda < 0.0:
        ap.error("--fusion-gate-lambda must be non-negative")
    if not 0.0 < args.gdu_alpha_init <= 1.0:
        ap.error("--gdu-alpha-init must be in (0, 1]")
    if not 0.0 < args.gdu_beta_init <= 1.0:
        ap.error("--gdu-beta-init must be in (0, 1]")
    for task in TASK_WEIGHT_DEFAULTS:
        for suffix in ("start", "end"):
            if getattr(args, f"task{task}_weight_{suffix}") < 0.0:
                ap.error(
                    f"--task{task}-weight-{suffix} must be non-negative")
    if args.query_rank == 0 and args.query_gate_lambda > 0.0:
        ap.error("--query-rank 0 requires --query-gate-lambda 0")
    if args.kv_rank == 0 and args.kv_gate_lambda > 0.0:
        ap.error("--kv-rank 0 requires --kv-gate-lambda 0")
    for name in ("query_gate_temperature", "kv_gate_temperature",
                 "layer_gate_temperature"):
        if getattr(args, name) <= 0.0:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("query_gate_threshold", "kv_gate_threshold",
                 "layer_gate_threshold"):
        if not 0.0 < getattr(args, name) < 1.0:
            ap.error(f"--{name.replace('_', '-')} must be in (0, 1)")
    for name in ("query_gate_lambda", "kv_gate_lambda",
                 "layer_gate_lambda"):
        if getattr(args, name) < 0.0:
            ap.error(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("query_gate_min_rank", "kv_gate_min_rank",
                 "layer_gate_min_layers"):
        if getattr(args, name) < 1:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    if args.nas_lr <= 0.0:
        ap.error("--nas-lr must be positive")
    if args.nas_warmup < 0:
        ap.error("--nas-warmup must be non-negative")
    if args.nas_every < 1:
        ap.error("--nas-every must be positive")
    if args.max_memory_slots < 1:
        ap.error("--max-memory-slots must be positive")
    if not 0.0 < args.retention_target <= 1.0:
        ap.error("--retention-target must be in (0, 1]")
    if args.retention_lambda < 0.0:
        ap.error("--retention-lambda must be non-negative")
    if args.preservation_lambda < 0.0:
        ap.error("--preservation-lambda must be non-negative")
    if args.interference_distill_lambda < 0.0:
        ap.error("--interference-distill-lambda must be non-negative")
    if args.interference_distill_temperature <= 0.0:
        ap.error(
            "--interference-distill-temperature must be positive")
    if args.address_lambda < 0.0:
        ap.error("--address-lambda must be non-negative")
    if args.address_temperature <= 0.0:
        ap.error("--address-temperature must be positive")
    if args.key_diversity_lambda < 0.0:
        ap.error("--key-diversity-lambda must be non-negative")
    if not 0.0 <= args.key_diversity_margin < 1.0:
        ap.error("--key-diversity-margin must be in [0, 1)")
    if args.slot_temperature <= 0.0:
        ap.error("--slot-temperature must be positive")
    if args.max_memory_episodes < 1:
        ap.error("--max-memory-episodes must be positive")
    if args.episode_temperature <= 0.0:
        ap.error("--episode-temperature must be positive")
    if args.episode_address_rank < 0:
        ap.error("--episode-address-rank must be non-negative")
    if args.episode_address_tokens < 1:
        ap.error("--episode-address-tokens must be positive")
    if args.episode_payload_tokens < 1:
        ap.error("--episode-payload-tokens must be positive")
    if args.episode_payload_min_layer_votes < 1:
        ap.error("--episode-payload-min-layer-votes must be positive")
    if not 0.0 <= args.episode_identity_uniqueness_floor <= 1.0:
        ap.error(
            "--episode-identity-uniqueness-floor must be in [0, 1]")
    if args.episode_identity_match_scale < 0.0:
        ap.error("--episode-identity-match-scale must be non-negative")
    if (
        args.episode_identity_ngram < 1
        or args.episode_identity_ngram > args.episode_address_tokens
    ):
        ap.error(
            "--episode-identity-ngram must fit the address token count")
    if args.episode_router_rank < 1:
        ap.error("--episode-router-rank must be positive")
    if args.memory_banks < 1:
        ap.error("--memory-banks must be positive")
    if args.state_mode == "banked" and args.memory_banks < 2:
        ap.error("--state-mode banked requires --memory-banks >= 2")
    if args.state_mode != "banked" and args.memory_banks != 1:
        ap.error("--memory-banks > 1 requires --state-mode banked")
    if args.bank_temperature <= 0.0:
        ap.error("--bank-temperature must be positive")
    if (
        args.bank_write_top_k < 1
        or args.bank_write_top_k > args.memory_banks
    ):
        ap.error("--bank-write-top-k must be within --memory-banks")
    if args.bank_router_lr_scale <= 0.0:
        ap.error("--bank-router-lr-scale must be positive")
    if args.write_selector_lr_scale <= 0.0:
        ap.error("--write-selector-lr-scale must be positive")
    if (
        args.bank_router_mode == "sequential"
        and args.bank_write_top_k != 1
    ):
        ap.error(
            "--bank-router-mode sequential requires "
            "--bank-write-top-k 1")
    if (
        args.write_mode != "paired_tokens"
        and args.state_mode not in ("delta", "episodes")
    ):
        ap.error(
            "two-selector write modes require delta or episode state")
    if args.grad_accum < 1:
        ap.error("--grad-accum must be positive")
    if args.generation_valid_every < 0:
        ap.error("--generation-valid-every must be non-negative")
    if args.generation_valid_samples_per_task < 0:
        ap.error(
            "--generation-valid-samples-per-task must be non-negative")
    if args.generation_valid_max_tokens < 1:
        ap.error("--generation-valid-max-tokens must be positive")
    if args.min_steps_before_early_stop < 0:
        ap.error("--min-steps-before-early-stop must be non-negative")
    if args.early_stop_patience < 1:
        ap.error("--early-stop-patience must be positive")
    if (
        args.selection_metric == "generation_recall_exact"
        and (
            args.generation_valid_every < 1
            or args.generation_valid_samples_per_task < 1
        )
    ):
        ap.error(
            "generation_recall_exact selection requires generation "
            "validation to be enabled")

    tr = MemoryTrainer(args)
    tok_cache = {}
    if args.eval_only:
        if not args.init_memory:
            ap.error("--eval-only requires --init-memory")
        v, vt, vn, vacc, vexact = tr.run_valid(
            args.valid_subset, tok_cache)
        position_stats = dict(tr.last_valid_position_stats)
        vbase, _bt, _bn, _ba, _be = tr.run_valid(
            args.valid_subset, tok_cache, query_memory_enabled=False)
        payload = {
            "split": "eval_only",
            "loss": v,
            "tokens": vt,
            "samples": vn,
            "token_acc": vacc,
            "exact_acc": vexact,
            "first_token_acc": position_stats["first_token_acc"],
            "prefix_token_acc": position_stats["prefix_token_acc"],
            "no_memory_loss": vbase,
            "memory_gain": vbase - v,
        }
        if args.generation_valid_samples_per_task > 0:
            payload["generation"] = tr.run_generation_valid(
                args.generation_valid_samples_per_task,
                args.generation_valid_max_tokens,
                tok_cache)
        print(json.dumps(payload, separators=(",", ":")), flush=True)
        return
    if args.export_only:
        tr.mem.reset_state()
        tr.export(args.output)
        return

    t0 = time.time()
    completed_steps = 0
    eval_every = args.valid_every or max(1, args.steps // 20)
    for step in range(args.steps):
        if tr.opt is None:
            break
        for parameter in tr.architecture_parameters:
            parameter.requires_grad_(False)
        tr.opt.zero_grad(set_to_none=True)
        lr = tr.warmup_lr(step)
        for group in tr.opt.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))
        total_loss, total_tok, n_ok = 0.0, 0, 0
        task_sample_counts = {}
        tr.write_selection_loss_sum = 0.0
        tr.write_selection_loss_count = 0
        tr.retention_loss_sum = 0.0
        tr.retention_loss_count = 0
        tr.preservation_loss_sum = 0.0
        tr.preservation_loss_count = 0
        tr.interference_distill_loss_sum = 0.0
        tr.interference_distill_loss_count = 0
        tr.interference_teacher_used = 0
        tr.address_loss_sum = 0.0
        tr.address_loss_count = 0
        tr.address_correct = 0
        tr.address_total = 0
        tr.key_diversity_loss_sum = 0.0
        tr.key_diversity_loss_count = 0
        for _micro_batch in range(args.grad_accum):
            batch_task, batch_indices, _task_weights = (
                tr.next_training_batch(step, args.batch))
            task_sample_counts[batch_task] = (
                task_sample_counts.get(batch_task, 0)
                + len(batch_indices))
            for s_idx in batch_indices:
                line = tr.train_strata[s_idx[0]][1][s_idx[1]]
                sample = json.loads(line)
                try:
                    ls, nt, _nc, _ne = tr.run_sample(
                        sample, True, tok_cache)
                    if ls is not None and nt > 0:
                        # Backward per sample frees its graph immediately.
                        # Gradients are averaged across every successful
                        # sample in all accumulated micro-batches below.
                        (ls / nt).backward()
                        total_loss += ls.item()
                        total_tok += nt
                        n_ok += 1
                except Exception as e:
                    print(
                        f'{{"step":{step},'
                        f'"skip_sample":"{type(e).__name__}:'
                        f'"{str(e)[:120]}"}}', flush=True)
                    traceback.print_exc()
                    if isinstance(e, torch.OutOfMemoryError):
                        tr.mem.M = None
                        tr.mem.S = None
                        tr.mem.discard_captured()
                        if tr.device == "cuda":
                            torch.cuda.empty_cache()
                        elif tr.device == "mps":
                            torch.mps.empty_cache()
                    tr.mem.reset_state()
                    continue
        if total_tok == 0:
            continue
        if n_ok > 1:
            for parameter in tr.weight_parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(n_ok)
        torch.nn.utils.clip_grad_norm_(tr.weight_parameters, args.clip)
        tr.opt.step()
        completed_steps = step + 1
        mean_loss = total_loss / total_tok
        el = time.time() - t0
        sps = (step + 1) / el * 3600 if el > 0 else 0
        train_payload = {
            "step": step,
            "split": "train",
            "loss": round(mean_loss, 6),
            "tokens": total_tok,
            "samples": n_ok,
            "micro_batches": args.grad_accum,
            "task_samples": task_sample_counts,
            "self_prefix_used": tr.self_prefix_used,
            "contrastive_pairs_used": tr.contrastive_pairs_used,
            "write_selection_loss": round(
                tr.write_selection_loss_sum
                / max(tr.write_selection_loss_count, 1), 6),
            "retention_loss": round(
                tr.retention_loss_sum
                / max(tr.retention_loss_count, 1), 6),
            "preservation_loss": round(
                tr.preservation_loss_sum
                / max(tr.preservation_loss_count, 1), 6),
            "interference_distill_loss": round(
                tr.interference_distill_loss_sum
                / max(tr.interference_distill_loss_count, 1), 6),
            "interference_teacher_used": (
                tr.interference_teacher_used),
            "address_loss": round(
                tr.address_loss_sum
                / max(tr.address_loss_count, 1), 6),
            "address_top1": round(
                tr.address_correct / max(tr.address_total, 1), 6),
            "key_diversity_loss": round(
                tr.key_diversity_loss_sum
                / max(tr.key_diversity_loss_count, 1), 6),
            "steps_per_hour": round(sps),
        }
        print(json.dumps(train_payload, separators=(",", ":")),
              flush=True)

        if (
            tr.arch_opt is not None
            and step + 1 >= args.nas_warmup
            and (step + 1) % args.nas_every == 0
        ):
            nas_result = tr.run_architecture_step(tok_cache)
            if nas_result is not None:
                nas_loss, nas_penalty = nas_result
                nas_stats = tr.mem.architecture_stats()
                stats_json = ",".join(
                    f'"{name}":{value}'
                    for name, value in nas_stats.items())
                print(f'{{"step":{step},"split":"nas_valid",'
                      f'"loss":{nas_loss:.6f},'
                      f'"penalty":{nas_penalty:.6f}'
                      f'{"," if stats_json else ""}{stats_json}}}',
                      flush=True)

        should_stop = False
        if (step + 1) % eval_every == 0:
            v, vt, vn, vacc, vexact = tr.run_valid(
                args.valid_subset, tok_cache)
            position_stats = dict(tr.last_valid_position_stats)
            vbase, _bt, _bn, _ba, _be = tr.run_valid(
                args.valid_subset, tok_cache, query_memory_enabled=False)
            print(f'{{"step":{step},"split":"valid","loss":{v:.6f},'
                  f'"tokens":{vt},"samples":{vn},'
                  f'"token_acc":{vacc:.6f},"exact_acc":{vexact:.6f},'
                  f'"first_token_acc":'
                  f'{position_stats["first_token_acc"]:.6f},'
                  f'"prefix_token_acc":'
                  f'{position_stats["prefix_token_acc"]:.6f},'
                  f'"no_memory_loss":{vbase:.6f},'
                  f'"memory_gain":{vbase-v:.6f}}}', flush=True)
            loss_improved = v < tr.best_valid - 1e-6
            if loss_improved:
                tr.best_valid = v
            if args.selection_metric == "loss":
                if loss_improved:
                    tr.patience = 0
                    tr.export(args.output)
                    tr.checkpoint_saved = True
                else:
                    tr.patience += 1
                should_stop = (
                    step + 1 >= args.min_steps_before_early_stop
                    and tr.patience >= args.early_stop_patience)

        if (
            args.generation_valid_every > 0
            and (step + 1) % args.generation_valid_every == 0
        ):
            generation = tr.run_generation_valid(
                args.generation_valid_samples_per_task,
                args.generation_valid_max_tokens,
                tok_cache)
            print(json.dumps({
                "step": step,
                "split": "generation_valid",
                **generation,
            }, separators=(",", ":")), flush=True)
            if args.selection_metric == "generation_recall_exact":
                score = generation["recall_exact"]
                if score > tr.best_generation_recall + 1e-9:
                    tr.best_generation_recall = score
                    tr.patience = 0
                    tr.export(args.output)
                    tr.checkpoint_saved = True
                else:
                    tr.patience += 1
                should_stop = (
                    step + 1 >= args.min_steps_before_early_stop
                    and tr.patience >= args.early_stop_patience)

        if should_stop:
            print(json.dumps({
                "early_stop": True,
                "selection_metric": args.selection_metric,
                "patience": tr.patience,
            }, separators=(",", ":")), flush=True)
            break

    if (
        args.final_output
        and os.path.abspath(args.final_output)
        != os.path.abspath(args.output)
    ):
        tr.export(args.final_output)
        print(json.dumps({
            "final_output": args.final_output,
            "source": "final_parameters",
        }, separators=(",", ":")), flush=True)
    if not tr.checkpoint_saved:
        tr.export(args.output)
        tr.checkpoint_saved = True
    else:
        # Report and hand off the same checkpoint selected by validation,
        # rather than evaluating the potentially overfit early-stop state.
        tr.mem.load_checkpoint(args.output)
    v, vt, vn, vacc, vexact = tr.run_valid(
        min(args.valid_subset, 16), tok_cache)
    print(
        f'{{"result":"done","final_valid_nll":{v:.6f},'
        f'"best_valid_nll":{tr.best_valid:.6f},'
        f'"best_generation_recall":'
        f'{tr.best_generation_recall if math.isfinite(tr.best_generation_recall) else -1.0:.6f},'
        f'"steps":{completed_steps}}}',
        flush=True)


if __name__ == "__main__":
    main()
