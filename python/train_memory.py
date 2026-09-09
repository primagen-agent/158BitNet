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
import shutil
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


def memory_task_id(stratum_name, sample):
    """Map local memory data to the official five-task training schedule."""
    metadata = sample.get("metadata", {})
    v2_task = str(metadata.get("v2_task", ""))
    if v2_task.startswith("task3") or stratum_name.startswith("task3"):
        return 3
    if v2_task.startswith("task4") or stratum_name.startswith("task4"):
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


def stale_answer_margin_loss(positive_nll, negative_nlls, margin):
    """Require the correct sequence NLL to beat stale/deleted alternatives."""
    if not negative_nlls:
        return positive_nll.new_zeros(())
    negatives = torch.stack(negative_nlls)
    return F.relu(margin + positive_nll - negatives).mean()


def evidence_attention_loss(weights, labels):
    """Penalize query attention that misses labelled evidence token slots."""
    positive = labels > 0
    negative = labels == 0
    if not bool(positive.any()) or not bool(negative.any()):
        return weights.new_zeros(())
    positive_mass = weights[..., positive].sum(dim=-1)
    return -positive_mass.clamp_min(1e-8).log().mean()


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
                 denom_mode="signed_plus_one",
                 state_mode="delta", max_memory_slots=4096,
                 slot_temperature=0.07,
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
        self.beta_scale = beta_scale
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
        if state_mode not in {"delta", "slots"}:
            raise ValueError(f"unsupported memory state mode {state_mode}")
        if max_memory_slots < 1:
            raise ValueError("max_memory_slots must be positive")
        if slot_temperature <= 0.0:
            raise ValueError("slot_temperature must be positive")
        self.state_mode = state_mode
        self.max_memory_slots = max_memory_slots
        self.slot_temperature = slot_temperature
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
        self.M = None   # [NL, kv, kv]
        self.S = None   # [NL, kv]
        self.slot_K = None  # [NL, token slots, kv]
        self.slot_V = None  # [NL, token slots, kv]
        self.slot_labels = None  # [token slots], 1=evidence, 0=distractor
        self.pending_slot_label = -1
        self.pending_slot_labels = None
        self.collect_evidence_loss = False
        self.evidence_supervision_tail = None
        self.evidence_supervision_range = None
        self.evidence_losses = []
        self.collect_write_selection_loss = False
        self.write_selection_losses = []
        self.collect_fusion_gate_loss = False
        self.fusion_gate_target = 0.0
        self.fusion_gate_losses = []
        self.last_pointer_weights = None
        self.last_pointer_weights_by_layer = [None] * self.n_layers
        self.last_memory_fused_by_layer = [None] * self.n_layers
        self.pointer_token_ids = None
        # Reference/runtime semantics have no empty-state bypass: even before
        # the first commit the memory branch reads zero and the attention
        # branch is scaled by gamma.
        self.active = True
        self._captured = None    # list of per-layer [T, d] tensors

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

    # ---- state control ----
    def reset_state(self):
        # drop the old tensors FIRST (frees any autograd graph still attached
        # to M/S before allocating the fresh zeros -- on a fragmented GPU the
        # zeros allocation itself can OOM if the old graph is still resident)
        self.M = None
        self.S = None
        self.M = torch.zeros(self.n_layers, self.kv_dim, self.kv_dim,
                             device=self.device, dtype=self.dtype)
        self.S = torch.zeros(self.n_layers, self.kv_dim,
                             device=self.device, dtype=self.dtype)
        self.slot_K = torch.empty(
            self.n_layers, 0, self.kv_dim,
            device=self.device, dtype=self.dtype)
        self.slot_V = torch.empty(
            self.n_layers, 0, self.kv_dim,
            device=self.device, dtype=self.dtype)
        self.slot_labels = torch.empty(
            0, device=self.device, dtype=torch.int8)
        self.pending_slot_label = -1
        self.pending_slot_labels = None
        self.collect_evidence_loss = False
        self.evidence_supervision_tail = None
        self.evidence_supervision_range = None
        self.evidence_losses = []
        self.collect_write_selection_loss = False
        self.write_selection_losses = []
        self.collect_fusion_gate_loss = False
        self.fusion_gate_target = 0.0
        self.fusion_gate_losses = []
        self.last_pointer_weights = None
        self.last_pointer_weights_by_layer = [None] * self.n_layers
        self.last_memory_fused_by_layer = [None] * self.n_layers
        self.pointer_token_ids = None
        # Keep reset behavior identical to the C runtime and official Metis:
        # zero the state but continue applying the reweighted memory block.
        self.active = True
        self._captured = None

    def clone_runtime_state(self):
        """Clone dynamic memory data for repeatable no-KV evaluation."""
        return {
            "M": self.M.clone(),
            "S": self.S.clone(),
            "slot_K": self.slot_K.clone(),
            "slot_V": self.slot_V.clone(),
            "slot_labels": self.slot_labels.clone(),
            "active": self.active,
        }

    def restore_runtime_state(self, state):
        self.M = state["M"].clone()
        self.S = state["S"].clone()
        self.slot_K = state["slot_K"].clone()
        self.slot_V = state["slot_V"].clone()
        self.slot_labels = state["slot_labels"].clone()
        self.active = state["active"]
        self.discard_captured()

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

    def take_captured(self):
        return self._captured

    def discard_captured(self):
        self._captured = None

    def commit_all(self):
        """Commit every layer's captured rows (no-grad path, e.g. valid).
        Batched like commit_all_grad_enabled: one stack, not 32."""
        caps = self._captured
        eps = self.rms_eps
        ran = False
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
        self.discard_captured()
        if ran:
            self.active = True
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
        self.discard_captured()
        if ran:
            self.active = True
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

    def read(self, slot, h_raw):
        """v4 reference read law (matches metis_v6_read_v4 in C):
        q = query_proj(h_raw) [T, q_dim]; per head: RMSNorm(head_dim,
        query_norm, eps 1e-6) then L2 (eps 1e-12); group reads with
        (denom + 1). h_raw is the layer INPUT residual (not normed)."""
        T = h_raw.shape[0]
        hd = self.head_dim
        blk = self.layer_ids[slot]
        base_q_w = self.backbone.layers[blk]["q"]
        # Memory-efficient trainable low-rank query.
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
        q = q.view(T, -1, hd)                       # [T, H, hd]
        q = metis_rms_norm(
            q, self.query_norm[slot], 1e-6)           # per-head RMSNorm
        q = F.normalize(q, dim=-1, eps=1e-12)        # per-head L2
        n_heads = q.shape[1]
        groups = self.q_dim // self.kv_dim
        hpg = n_heads // groups
        qg = q.reshape(T, groups, hpg * hd)          # [T, G, kv]
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
        mem = self.read(slot, h_raw)
        memn = metis_rms_norm(mem, self.mem_norm[slot], 1e-6)
        fused = F.linear(memn.to(o_w.dtype), o_w)            # [T, d]
        self.last_memory_fused_by_layer[slot] = fused
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

    def _commit_math(self, slot, raw_rows, eps):
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
        Kn = F.normalize(Kall, dim=-1, eps=1e-12) / math.sqrt(self.kv_dim)

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
                if bool((labels > 0).any()) and bool((labels == 0).any()):
                    write_loss = (
                        write_loss + evidence_attention_loss(p, labels))
                self.write_selection_losses.append(write_loss)
        beta_i = self.beta_scale * torch.sigmoid(b_pre)      # [L]
        alpha_scalar = ((torch.sigmoid(a_pre) * w_sel).sum()
                        / w_sel.sum().clamp_min(1e-6))

        b_eff = beta_i * w_sel                               # [L]
        new_M, new_S = gated_delta_update(
            self.M[slot], self.S[slot], Kn, Vall,
            b_eff, alpha_scalar)
        # caller (commit / commit_all) owns the stacking so a 32-slot
        # batched commit performs ONE stack instead of 32.
        return new_M, new_S

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
                                denom_mode=args.denom_mode,
                                state_mode=args.state_mode,
                                max_memory_slots=args.max_memory_slots,
                                slot_temperature=args.slot_temperature,
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
            if args.retrieval_windows > 0 else None)
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

        self.train_by_task = {}
        for stratum_index, line_index in self.train_idx:
            stratum_name, lines = self.train_strata[stratum_index]
            sample = json.loads(lines[line_index])
            task = memory_task_id(stratum_name, sample)
            self.train_by_task.setdefault(task, []).append(
                (stratum_index, line_index))
        self.train_task_positions = {
            task: 0 for task in self.train_by_task}
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
        self.opt = (
            torch.optim.AdamW(
                optimizer_parameters, lr=args.lr, weight_decay=args.wd)
            if optimizer_parameters else None)
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
              f'{str(self.backbone_lora is not None).lower()}}}',
              flush=True)
        self.base_lr = args.lr
        self.warmup = args.warmup
        self.best_valid = float("inf")
        self.patience = 0
        self.schedule_rng = random.Random(args.seed + 0x5E1F)
        self.self_prefix_used = 0
        self.contrastive_pairs_used = 0
        self.write_selection_loss_sum = 0.0
        self.write_selection_loss_count = 0
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
                logits = self.backbone(
                    tokens, memory_v6=self.mem, fuse_start=0)
                self.mem.discard_captured()
                generated.append(int(logits.argmax().item()))
        return generated

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
            return None, 0

        ctx = torch.enable_grad() if want_grads else torch.no_grad()
        loss_sum = torch.zeros((), device=self.device)
        n_label = 0
        n_correct = 0
        sequence_exact = 0
        sample_write_losses = []
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
                    if chunk_index in distractor_indices:
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
                    supervise_write = (
                        want_grads
                        and self.args.write_selection_lambda > 0.0
                        and mem.state_mode == "delta")
                    if supervise_write:
                        mem.begin_write_selection_supervision()
                    if truncated_prefix:
                        mem.commit_all()
                    else:
                        mem.commit_all_grad_enabled()
                    if supervise_write:
                        write_loss = mem.end_write_selection_supervision()
                        sample_write_losses.append(write_loss)
                    if mem.state_mode == "delta":
                        mem.pending_slot_labels = None
                        mem.pending_slot_label = -1
                    append_pointer_token_ids(mem, ids, bb.device)
                    continue
                # query: FRESH context (memory is the only fact source).
                # M4(b) ruling: the query chunk is NEVER committed before
                # scoring — deploy commits the query turn AFTER generation
                # (it only affects the NEXT turn). Training with a
                # pre-scoring query commit taught the reads to expect the
                # question tokens inside M/S; at eval they are absent ->
                # out-of-distribution S -> degenerate decode (torch probe
                # proved the collapse is model-side, not C-i8).
                # The query forward still FUSES (reads run on exchange-
                # committed state) but its captures are DISCARDED.
                saved_active = mem.active
                if not query_memory_enabled:
                    mem.active = False
                with torch.no_grad():
                    toks = torch.tensor(ids, device=self.device)
                    _ = bb(toks, memory_v6=mem, fuse_start=0)
                mem.discard_captured()
                if (
                    query_memory_enabled
                    and self.args.retrieval_windows > 0
                ):
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
                if (
                    want_grads
                    and allow_self_prefix
                    and self.args.self_prefix_prob > 0.0
                    and self.schedule_rng.random()
                    < self.args.self_prefix_prob
                ):
                    prefix = self.greedy_prefix(ids, len(tgt_ids) - 1)
                    self.self_prefix_used += 1
                # score targets on the same fresh context with fusion active
                full = ids + prefix
                toks = torch.tensor(full, device=self.device)
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
                logits_all = bb(toks, memory_v6=mem, logits_all=True,
                                fuse_start=0)
                if supervise_fusion:
                    fusion_loss = mem.end_fusion_gate_supervision()
                    loss_sum = (
                        loss_sum + self.args.fusion_gate_lambda *
                        fusion_loss * len(tgt_ids))
                if supervise_evidence:
                    evidence_loss = mem.end_evidence_supervision()
                    loss_sum = (
                        loss_sum
                        + self.args.evidence_lambda
                        * evidence_loss * len(tgt_ids))
                lp = logits_all[len(full) - len(tgt_ids): len(full)]
                tgt_t = torch.tensor(tgt_ids, device=self.device)
                nll = F.cross_entropy(lp.float(), tgt_t, reduction="sum")
                loss_sum = loss_sum + nll
                n_label += len(tgt_ids)
                negatives = sample.get("negative_answers", [])
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
                        negative_logits = bb(
                            torch.tensor(
                                negative_full, device=self.device),
                            memory_v6=mem, logits_all=True,
                            fuse_start=0)
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
                            nll / len(tgt_ids), ranking_losses,
                            self.args.contrastive_margin)
                        # run_sample returns a token-sum objective. Scaling
                        # preserves CE_mean + lambda*ranking_mean after ls/nt.
                        loss_sum = (
                            loss_sum
                            + self.args.contrastive_lambda
                            * ranking_loss * len(tgt_ids))
                        self.contrastive_pairs_used += len(ranking_losses)
                pred = lp.argmax(dim=-1)
                correct = int((pred == tgt_t).sum().item())
                n_correct += correct
                sequence_exact += int(correct == len(tgt_ids))
                mem.active = saved_active
                mem.discard_captured()
                if chunk_index + 1 < len(chunks):
                    # A supervised exchange becomes ordinary conversation
                    # history for later queries, matching deploy semantics:
                    # score first, then commit the completed user+assistant
                    # turn for the next exchange.
                    complete_text, _ = render_chunk(
                        chunks[chunk_index], False)
                    ckey = ("complete", complete_text)
                    complete_ids = tok_cache.get(ckey)
                    if complete_ids is None:
                        complete_ids = self.tok.encode(
                            complete_text, add_bos=True)
                        tok_cache[ckey] = complete_ids
                    _ = bb(
                        torch.tensor(complete_ids, device=self.device),
                        memory_v6=mem, fuse_start=0)
                    mem.set_pending_slot_labels(
                        [-1] * len(complete_ids))
                    if want_grads:
                        mem.commit_all_grad_enabled()
                    else:
                        mem.commit_all()
                    append_pointer_token_ids(
                        mem, complete_ids, bb.device)
        if sample_write_losses and n_label > 0:
            write_loss = torch.stack(sample_write_losses).mean()
            self.write_selection_loss_sum += float(write_loss.detach())
            self.write_selection_loss_count += 1
            loss_sum = (
                loss_sum +
                self.args.write_selection_lambda * write_loss * n_label)
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
            query_value = sample.get(
                "query_turn_id", len(sample["messages"]) - 1)
            query_count += (
                len(query_value) if isinstance(query_value, list) else 1)
        return (total / max(tok, 1), tok, n_ok,
                correct / max(tok, 1), exact / max(query_count, 1))

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
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--alpha-max-tokens", type=int, default=0)
    ap.add_argument("--alpha-max-fraction", type=float, default=0.0)
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
        "--state-mode", choices=("delta", "slots"), default="delta",
        help="dynamic delta matrix or sparse token-preserving memory slots")
    ap.add_argument(
        "--max-memory-slots", type=int, default=4096,
        help="maximum source-token K/V slots retained by memory")
    ap.add_argument(
        "--slot-temperature", type=float, default=0.07,
        help="softmax temperature for query-to-slot retrieval")
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
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--valid", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--valid-every", type=int, default=0)
    ap.add_argument("--valid-subset", type=int, default=32)
    ap.add_argument(
        "--self-prefix-prob", type=float, default=0.0,
        help="probability of using a no-KV greedy model prefix for "
             "training targets instead of the gold teacher-forcing prefix")
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
    ap.add_argument("--oversample-distract", type=int, default=1,
                    help="rounds per pass for distract strata (v10run: 3)")
    for task, (start, end) in TASK_WEIGHT_DEFAULTS.items():
        ap.add_argument(
            f"--task{task}-weight-start", type=float, default=start)
        ap.add_argument(
            f"--task{task}-weight-end", type=float, default=end)
    args = ap.parse_args()
    if args.retrieval_windows < 0:
        ap.error("--retrieval-windows must be non-negative")
    if args.alpha_max_tokens < 0:
        ap.error("--alpha-max-tokens must be non-negative")
    if not 0.0 <= args.alpha_max_fraction <= 1.0:
        ap.error("--alpha-max-fraction must be in [0, 1]")
    if args.retrieval_window_size < 1:
        ap.error("--retrieval-window-size must be positive")
    if not 0.0 <= args.self_prefix_prob <= 1.0:
        ap.error("--self-prefix-prob must be in [0, 1]")
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
    if args.slot_temperature <= 0.0:
        ap.error("--slot-temperature must be positive")

    tr = MemoryTrainer(args)
    tok_cache = {}
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
            group["lr"] = lr
        total_loss, total_tok, n_ok = 0.0, 0, 0
        batch_task, batch_indices, task_weights = tr.next_training_batch(
            step, args.batch)
        tr.write_selection_loss_sum = 0.0
        tr.write_selection_loss_count = 0
        for s_idx in batch_indices:
            line = tr.train_strata[s_idx[0]][1][s_idx[1]]
            sample = json.loads(line)
            try:
                ls, nt, _nc, _ne = tr.run_sample(sample, True, tok_cache)
                if ls is not None and nt > 0:
                    # backward PER SAMPLE: the graph frees immediately,
                    # capping activation memory at one sample (12GB rule).
                    (ls / nt).backward()
                    total_loss += ls.item()
                    total_tok += nt
                    n_ok += 1
            except Exception as e:
                print(f'{{"step":{step},"skip_sample":"{type(e).__name__}:'
                      f'"{str(e)[:120]}"}}', flush=True)
                traceback.print_exc()
                if isinstance(e, torch.OutOfMemoryError):
                    # free the failed sample's graph + cached blocks before
                    # the next allocation; reset_state drops M/S to None
                    # first so the fresh zeros cannot OOM on fragmentation
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
            # Match ordinary batched CE semantics. Each sample is forwarded
            # and backpropagated separately to cap activation memory, so the
            # accumulated gradients must be averaged before the optimizer
            # update rather than implicitly scaling the learning rate.
            for parameter in tr.weight_parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(n_ok)
        torch.nn.utils.clip_grad_norm_(tr.weight_parameters, args.clip)
        tr.opt.step()
        completed_steps = step + 1
        mean_loss = total_loss / total_tok
        el = time.time() - t0
        sps = (step + 1) / el * 3600 if el > 0 else 0
        print(f'{{"step":{step},"split":"train","loss":{mean_loss:.6f},'
              f'"tokens":{total_tok},"samples":{n_ok},'
              f'"task":{batch_task},'
              f'"task_weight":{task_weights[batch_task]:.6f},'
              f'"self_prefix_used":{tr.self_prefix_used},'
              f'"contrastive_pairs_used":{tr.contrastive_pairs_used},'
              f'"write_selection_loss":'
              f'{tr.write_selection_loss_sum / max(tr.write_selection_loss_count, 1):.6f},'
              f'"steps_per_hour":{sps:.0f}}}', flush=True)

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

        if (step + 1) % eval_every == 0:
            v, vt, vn, vacc, vexact = tr.run_valid(
                args.valid_subset, tok_cache)
            vbase, _bt, _bn, _ba, _be = tr.run_valid(
                args.valid_subset, tok_cache, query_memory_enabled=False)
            print(f'{{"step":{step},"split":"valid","loss":{v:.6f},'
                  f'"tokens":{vt},"samples":{vn},'
                  f'"token_acc":{vacc:.6f},"exact_acc":{vexact:.6f},'
                  f'"no_memory_loss":{vbase:.6f},'
                  f'"memory_gain":{vbase-v:.6f}}}', flush=True)
            if v < tr.best_valid - 1e-6:
                tr.best_valid = v
                tr.patience = 0
                tr.export(args.output)
            else:
                tr.patience += 1
            if tr.patience >= 5:
                print('{"early_stop":true}', flush=True)
                break

    if tr.best_valid == float("inf"):
        tr.export(args.output)
    else:
        # Report and hand off the same checkpoint selected by validation,
        # rather than evaluating the potentially overfit early-stop state.
        tr.mem.load_checkpoint(args.output)
    if args.final_output:
        if os.path.abspath(args.final_output) != os.path.abspath(args.output):
            shutil.copyfile(args.output, args.final_output)
        print(json.dumps({
            "final_output": args.final_output,
            "source": "best_validation",
        }, separators=(",", ":")), flush=True)
    v, vt, vn, vacc, vexact = tr.run_valid(
        min(args.valid_subset, 16), tok_cache)
    print(
        f'{{"result":"done","final_valid_nll":{v:.6f},'
        f'"best_valid_nll":{tr.best_valid:.6f},'
        f'"steps":{completed_steps}}}',
        flush=True)


if __name__ == "__main__":
    main()
