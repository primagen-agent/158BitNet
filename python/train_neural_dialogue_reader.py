#!/usr/bin/env python3
"""Train current full-bank memory activation on independent synthetic dialogues."""

import argparse
from collections import Counter
from copy import copy, deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import time
import torch
from torch.nn import functional as F
from neural_slot_model import (
    NeuralSlots,
    HIDDEN,
    LEXICAL,
    WIDTH,
    RELATION_HEADS,
    SEMANTIC,
    QUERY,
    identity_word_features,
    export,
    require_parameters,
    FORMAT,
    KEYS,
    PLAN_KEYS,
    QUERY_KEYS,
)
from neural_training_common import sha, load_features, extract
from neural_fit_support import semantic_relation_masks, sample_relation_queries
from neural_reader_evidence_curriculum import filter_bank_evidence_queries
from neural_reader_anchor import (
    snapshot_reader_matcher, matcher_pair_features, matcher_logits_from_parameters,
    build_reader_anchor_teacher, reader_support_anchor_loss,
)
from neural_reader_functional import (
    build_reader_functional_teacher, reader_functional_margin_loss,
    reader_functional_distillation_loss,
)
from neural_kind_correction import snapshot_kind_teacher, kind_correction_loss
from neural_query_binding import require_query_binding_fit
from neural_reader_correction import snapshot_reader_corrections, reader_correction_loss, PRESERVATION_REDUCTION
from neural_activation_curriculum import scope_fit_group, scope_development_group, lifecycle_fit_group, served_lifecycle_scope_group, served_scope_paraphrase_group, scope_relation_masks, identity_multisubject_fit_group, relation_variation_fit_group
from train_neural_dialogue import prepare, source_words, source_types, decode_references

SCOPE_QUERY_HEADS = ("scope_query_encode.", "scope_query_pool.", "scope_memory_attention_logits")
CONTEXT_HEADS = ("read_context_memory.", "read_context_query.", "read_context_delta.")
CONTEXT_QUERY_HEADS = ("read_context_query.",)
CONTEXT_MODES = ("context-query", "context")
HEADS = ("read.", "scope.", "intent.", "version.", "kind.", "kind_context.", "read_update.", "memory_attention_logits") + SCOPE_QUERY_HEADS + CONTEXT_HEADS
READER_HEADS = HEADS + (
    "encode.",
    "query_encode.",
    "query_pool.",
    "role.",
    "identity.",
    "identity_match.",
    "identity_update.",
)
IDENTITY_HEADS = ("identity_match.", "identity_update.", "role.")
IDENTITY_MATCH_HEADS = ("identity_match.",)
SCOPE_HEADS = ("scope.",) + SCOPE_QUERY_HEADS
QUERY_ENCODER_HEADS = ("query_encode.", "query_pool.")
SEMANTIC_HEADS = ("read.", "encode.", "query_encode.", "query_pool.", "read_update.", "memory_attention_logits") + CONTEXT_HEADS
MATCHER_HEADS = ("read.",)
TEMPORAL_HEADS = ("kind.", "kind_context.", "intent.", "version.")
ACTIVATION_HEADS = SEMANTIC_HEADS + ("intent.", "version.")
RELATION_MODES = ("semantic", "matcher", "activation", "query") + CONTEXT_MODES
GUARDED_READER_MODES = RELATION_MODES + ("temporal",)
FUNCTIONAL_MODES = GUARDED_READER_MODES
LOSS_REDUCTION = "query-balanced"


class DynamicFeatures:
    """Vectorized exact ragged-token computation, including padding masks."""

    def __init__(
        self,
        model,
        rows,
        cache,
        device,
        feature_dropout=0,
        *,
        freeze_memory_encoder=False,
    ):
        self.model, self.device = model, torch.device(device)
        self.feature_dropout = feature_dropout
        self.freeze_memory_encoder = freeze_memory_encoder
        required = {r["text"] for g in rows.values() for v in g.values() for r in v}
        required |= {
            r[key]
            for g in rows.values()
            for v in g.values()
            for r in v
            for key in ("reference", "initial_reference", "initial_text")
            if r.get(key)
        }
        texts = sorted(required)
        self.index = {t: i for i, t in enumerate(texts)}
        self.counts = {t: len(cache[t]["x"]) for t in texts}
        self.inputs = torch.cat([model.augment(cache[t]["x"]) for t in texts]).to(
            device
        )
        self.lengths = torch.tensor([len(cache[t]["x"]) for t in texts], device=device)
        self.offsets = self.lengths.cumsum(0) - self.lengths
        self.summaries = torch.stack([model.summary(cache[t]["x"]) for t in texts]).to(
            device
        )
        self.kind_contexts = torch.stack([model.kind_context_inputs(cache[t]["x"]) for t in texts]).to(device)
        words = [identity_word_features(t, cache[t])[0] for t in texts]
        self.word_inputs = torch.cat(
            [model.augment(w, hidden_only=True) for w in words]
        ).to(device)
        self.word_lex = torch.cat([w[:, HIDDEN:] for w in words]).to(device)
        self.word_lengths = torch.tensor([len(w) for w in words], device=device)
        self.word_offsets = self.word_lengths.cumsum(0) - self.word_lengths
        self.word_counts = {t: len(w) for t, w in zip(texts, words)}
        self.word_units = {t: identity_word_features(t, cache[t])[1] for t in texts}

    def for_model(self, model):
        """Share immutable C inputs, recomputing all learned outputs separately."""
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("functional teacher features require a frozen evaluation model")
        if any(parameter.device != self.inputs.device or parameter.dtype != self.inputs.dtype
               for parameter in model.parameters()):
            raise ValueError("functional teacher must use the same raw feature device and precision")
        result = copy(self)
        result.model = model
        result.feature_dropout = 0
        return result

    def indices(self, texts):
        return torch.tensor([self.index[t] for t in texts], device=self.device)

    def encode(self, texts, *, query=False, scope=False):
        indices = self.indices(texts)
        lengths = self.lengths[indices]
        positions = torch.arange(
            max(self.counts[t] for t in texts), device=self.device
        )[None]
        mask = positions < lengths[:, None]
        addresses = (self.offsets[indices, None] + positions).clamp_max(
            len(self.inputs) - 1
        )
        inputs = F.dropout(
            self.inputs[addresses],
            p=self.feature_dropout,
            training=(self.model.training and torch.is_grad_enabled()
                      and (query or scope or not self.freeze_memory_encoder)),
        )
        encoder = (self.model.scope_query_encode if scope else
                   self.model.query_encode if query else self.model.encode)
        return encoder(inputs).tanh(), mask, indices

    def kind_logits(self, texts, means):
        return self.model.kind_logits_from_mean(means, self.kind_contexts[self.indices(texts)])

    def memory_context_latents(self, texts):
        """Recompute each model's sidecars from the same immutable C summaries."""
        return self.model.memory_context_latents(self.kind_contexts[self.indices(texts)])

    def reference_ids(self, rows):
        texts = [r.get("reference", r["text"]) for r in rows]
        indices = self.indices(texts)
        positions = torch.arange(
            max(self.word_counts[t] for t in texts), device=self.device
        )[None]
        mask = positions < self.word_lengths[indices, None]
        addresses = (self.word_offsets[indices, None] + positions).clamp_max(
            len(self.word_inputs) - 1
        )
        logits = (
            self.model.role(self.word_inputs[addresses])
            .squeeze(-1)
            .masked_fill(~mask, -torch.inf)
        )
        lex = torch.nn.functional.normalize(
            (logits.softmax(-1)[:, :, None] * self.word_lex[addresses]).sum(1),
            dim=-1,
        )
        projected = torch.nn.functional.normalize(self.model.identity(lex), dim=-1)
        return torch.cat((lex, projected), -1)


def add_episodes(group, split):
    from prepare_neural_episodes import episode_curriculum

    facts = {f["text"]: f for f in group["facts"]}
    queries = {q["text"]: q for q in group["queries"]}
    episode_rows = episode_curriculum(split)
    supports = {}
    for row in episode_rows:
        if row["tag"] == "episode-fact":
            s = row["spans"][0]
            name = s["references"][0]
            text = name + ". " + row["body"].encode()[s["start"] : s["end"]].decode()
            for question in row["recall_questions"]:
                supports.setdefault(question, set()).add(text)
    for row in episode_rows:
        if row["tag"] == "episode-fact":
            s = row["spans"][0]
            name = s["references"][0]
            text = name + ". " + row["body"].encode()[s["start"] : s["end"]].decode()
            facts[text] = dict(
                text=text,
                entity=name,
                reference=name,
                prop=s["prop"],
                kind=s["kind"],
                episode=True,
            )
        elif row["tag"] == "episode-query":
            queries[row["body"]] = dict(
                text=row["body"],
                entities=row["entities"],
                prop=row["prop"],
                kind=row["kind"],
                **(
                    {"support_texts": sorted(supports[row["body"]])}
                    if row["body"] in supports
                    else {}
                ),
            )
    return dict(facts=list(facts.values()), queries=list(queries.values()))


def episode_observation_group(split):
    """Attach original observations to prior fit/development supervision only."""
    if split not in ("fit", "development"):
        raise ValueError("episode replay only supports fit/development")
    from prepare_neural_episodes import episode_curriculum

    observations = {}
    for row in episode_curriculum(split):
        if row["tag"] != "episode-query":
            continue
        previous = observations.get(row["body"])
        if previous and previous["context"] != row["context"]:
            raise ValueError("episode query body maps to different original contexts")
        observations[row["body"]] = row
    group = add_episodes(dict(facts=[], queries=[]), split)
    for query_row in group["queries"]:
        row = observations[query_row["text"]]
        query_row.update(
            observation=dict(row),
            observation_id=row["source_id"],
            query_class="prior_episode",
        )
    if split == "fit":
        from neural_fit_support import repair_fit_episode_support

        group = repair_fit_episode_support(group, split)
    return group


@torch.no_grad()
def bind_curriculum_queries(group, planner, cache):
    """Freeze the same learned query binding used before native activation.

    Gold sources describe fact-bank supervision only. Query text and reference
    identity come exclusively from the frozen writer's predictions over the
    original observation. Unresolved queries stay visible to component scoring.
    """
    observations = [q["observation"] for q in group["queries"]]
    required = {r[k] for r in observations for k in ("body", "context")}
    if not required <= cache.keys():
        raise ValueError("writer feature cache lacks authored query observations")
    values = prepare(dict(queries=observations), cache, "cpu")
    queries, records = [], []
    for original in group["queries"]:
        row = original["observation"]
        item = values[row["body"], row["context"]]
        encoded = planner.plan_encoding(item["body"], item["context"], item["body_words"])
        need, tags = planner.query_reference(
            encoded.mean(0),
            source_words(item, len(encoded)),
            source_types(row, item, len(encoded)),
            raw_context=planner.query_binding_context(item["body"], item["context"]),
        )
        if not torch.isfinite(need) or not torch.isfinite(tags).all():
            raise ValueError("non-finite frozen query binding")
        required_reference = bool(need > 0)
        references = (
            decode_references(row, tags.argmax(-1), item, len(encoded))
            if required_reference
            else []
        )
        reference = "; ".join(references)
        unresolved = required_reference and not reference
        query_row = dict(original)
        query_row.pop("reference", None)
        query_row.update(
            text=(reference + ". " if reference else "") + row["body"],
            unresolved=unresolved,
            binding_predicted=True,
        )
        if reference:
            # Native reference_identity overrides the first-pass identity with
            # this predicted literal source, rather than the labeled actor.
            query_row["reference"] = reference
        queries.append(query_row)
        records.append(
            dict(
                observation_id=row["source_id"],
                query_class=original.get("query_class"),
                body=row["body"],
                predicted_need=required_reference,
                predicted_reference=reference,
                unresolved=unresolved,
                binding_exact=(
                    required_reference == row["whole_need"]
                    and references == row["whole_refs"]
                ),
            )
        )
    return dict(facts=list(group["facts"]), queries=queries), records


def load_model(checkpoint):
    if checkpoint.get("format") != FORMAT:
        raise ValueError("checkpoint is not the current memory format")
    model = NeuralSlots()
    require_parameters(model, checkpoint["state_dict"])
    return model


def reader_checkpoint_payload(resume_metadata, **reader_fields):
    """Keep reader metadata and isolate the current source/selection overrides."""
    return deepcopy(dict(resume_metadata, **reader_fields))


def reader_planner_provenance(planner_checkpoint, planner_identity):
    """Record the actual frozen source checkpoint, never the resume's writer."""
    return dict(
        writer_checkpoint_sha256=planner_identity,
        frozen_query_planner_sha256=planner_identity,
        writer_training=deepcopy(planner_checkpoint.get("writer_training")),
    )


@torch.no_grad()
def sync_query_planner(model, planner):
    """Export exactly the same frozen source planner used to bind fit queries."""
    source, target = planner.state_dict(), model.state_dict()
    parameters = dict(model.named_parameters())
    for key in PLAN_KEYS + QUERY_KEYS:
        target[key].copy_(source[key])
        parameters[key].requires_grad_(False)


def supported(q, f):
    if "support_texts" in q:
        return f["text"] in q["support_texts"]
    entities = q.get("entities", [q.get("entity")])
    props = q["prop"] if isinstance(q["prop"], (list, tuple)) else [q["prop"]]
    return (
        f["entity"] in entities
        and f["prop"] in props
        and ("kind" not in q or q["kind"] == f["kind"])
    )


def active_bank_facts(facts):
    """Keep physical insertion positions when deleted slots leave training banks."""
    total = max(len(facts), max((f.get("insert_order", i) + 1 for i, f in enumerate(facts)), default=0))
    return [dict(f, insert_order=f.get("insert_order", i), bank_totalcount=total)
            for i, f in enumerate(facts) if f.get("status") != 2]


def hard_bank_groups(sources, distractors=6):
    """Fit-only banks shaped like on-topic conversation: one entity at a time.

    The supported record is often not the lexical row maximum when a bank holds
    several same-entity records plus same-property distractors. Every bank is
    built from existing fit facts and the queries keep their exact support
    texts, so no new text and no evaluation data enter training.
    """
    if not 1 <= distractors <= 16:
        raise ValueError("hard-bank distractor count must be in [1, 16]")
    pool = [f for group in sources.values() for f in group["facts"]]
    result = {}
    for source_name, group in sources.items():
        entity_of = {f["text"]: f.get("entity") for f in group["facts"]}
        entity_facts = {}
        for fact in group["facts"]:
            entity_facts.setdefault(fact.get("entity"), []).append(fact)
        queries_by_entity = {}
        for query in group["queries"]:
            entities = {entity_of[text] for text in query.get("support_texts", [])
                        if text in entity_of}
            if len(entities) == 1:
                entity = entities.pop()
                if entity:
                    queries_by_entity.setdefault(entity, []).append(query)
        for entity, facts in sorted(entity_facts.items(), key=lambda item: str(item[0])):
            if not entity or not queries_by_entity.get(entity):
                continue
            bank = list(facts)
            seen = {fact["text"] for fact in bank}
            props = {fact.get("prop") for fact in facts}
            for fact in sorted(pool, key=lambda item: item["text"]):
                if (fact.get("entity") != entity and fact.get("prop") in props
                        and fact["text"] not in seen):
                    bank.append(fact)
                    seen.add(fact["text"])
                    if sum(1 for item in bank if item.get("entity") != entity) >= distractors:
                        break
            result[f"{source_name}-{entity}"] = dict(
                facts=bank, queries=queries_by_entity[entity])
    return result


def make_bank(facts, cached):
    physical = {0: 0, 1: 1, 3: 2}
    if any("status" in f and f["status"] not in physical for f in facts):
        raise ValueError("bank must contain only current, superseded or tentative slots")
    bank, mask, _ = cached.encode([f["text"] for f in facts])
    means = (bank * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
    probabilities = cached.kind_logits([fact["text"] for fact in facts], means).softmax(-1)
    fixed = torch.tensor([physical[f["status"]] if "status" in f else -1 for f in facts], device=bank.device)
    selected = torch.where(fixed >= 0, fixed, probabilities.argmax(-1))
    status = F.one_hot(selected, 3).to(probabilities.dtype)
    status = status + (probabilities - probabilities.detach()) * (fixed < 0)[:, None]
    n = len(facts)
    total = max(n, max((f.get("bank_totalcount", n) for f in facts), default=n))
    age = torch.tensor(
        [(total - 1 - f.get("insert_order", i)) / max(total - 1, 1) for i, f in enumerate(facts)], device=bank.device
    )[:, None]
    return bank, cached.reference_ids(facts), mask, torch.cat((status, age), -1)


def raw_query(model, rows, cached, *, scope=False):
    encoded, mask, indices = cached.encode([r["text"] for r in rows], query=True, scope=scope)
    pooled = model.pool_query(encoded, mask, scope=scope)
    return torch.cat(
        (pooled, model.intent(cached.summaries[indices])), -1
    ), cached.reference_ids(rows)


def query(model, rows, cached, *, scope=False):
    current, identity = raw_query(model, rows, cached, scope=scope)
    if scope:
        # UPDATE/DELETE makes one source query in C; read feedback gates are
        # unrelated to matching every live version of a property.
        return current, identity
    original_rows = []
    for row in rows:
        initial = dict(text=row.get("initial_text", row["text"]))
        reference = row.get("initial_reference", row.get("reference"))
        if reference:
            initial["reference"] = reference
        original_rows.append(initial)
    original, original_identity = raw_query(model, original_rows, cached)
    identity = model.advance_identity(original, current, original_identity, identity)
    return model.advance_query(original, current), identity


def query_masked_mean(values, mask):
    """Give each query with eligible positions one vote, including empty banks."""
    counts = mask.sum(-1)
    per_query = values.masked_fill(~mask, 0).sum(-1) / counts.clamp_min(1)
    valid = counts > 0
    return per_query.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)


def balanced(logits, gold, positive_weight=1.0):
    loss = F.binary_cross_entropy_with_logits(logits, gold, reduction="none")
    positive = gold.bool()
    positives = query_masked_mean(loss, positive)
    negatives = query_masked_mean(loss, ~positive)
    hardest = (
        loss.masked_fill(positive, -torch.inf)
        .topk(min(8, logits.shape[1]), -1)
        .values
    )
    available = torch.isfinite(hardest)
    hard = query_masked_mean(hardest, available)
    return positive_weight * positives + negatives + 1.5 * hard


def semantic_relation_auxiliary(model, rows, facts, queries, bank, mask, *, memory_context_latents):
    """Recall relation BCE; lifecycle scope has its own teacher."""
    read = model.semantic_bank_logits(queries, bank, mask, memory_context_latents=memory_context_latents)
    positive, negative = semantic_relation_masks(rows, facts, read)
    if not read.numel():
        return read.sum()

    def mean_losses(logits):
        targets = positive.to(logits.dtype)
        losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return query_masked_mean(losses, positive) + query_masked_mean(losses, negative)

    # Compare each labelled source against bounded relation/object alternatives.
    # Absolute BCE still teaches rejection; the margin teaches discrimination
    # independently of the fixed identity and temporal probability factors.
    has_positive = positive.any(-1)
    weakest = read.masked_fill(~positive, torch.inf).min(-1).values
    weakest = torch.where(has_positive, weakest, torch.zeros_like(weakest))
    pairs = negative & has_positive[:, None]
    margin = F.softplus(1.0 + read - weakest[:, None])
    contrast = query_masked_mean(margin, pairs)
    return 0.25 * (mean_losses(read) + contrast)


def scope_relation_auxiliary(model, rows, facts, queries, bank, mask, *, memory_context_latents):
    logits = model.semantic_bank_logits(queries, bank, mask, memory_context_latents=memory_context_latents, scope=True)
    positive, negative = scope_relation_masks(rows, facts, logits)
    loss = F.binary_cross_entropy_with_logits(logits, positive.to(logits.dtype), reduction="none")
    return 0.25 * (loss.masked_fill(~positive, 0).sum() / positive.sum().clamp_min(1)
                   + loss.masked_fill(~negative, 0).sum() / negative.sum().clamp_min(1))


def configure_reader_heads(model, mode, *, freeze_memory_encoder=False):
    if mode in ("query", "identity-match") + CONTEXT_MODES and not freeze_memory_encoder:
        raise ValueError(f"{mode} refinement requires a frozen memory encoder")
    if freeze_memory_encoder and mode not in GUARDED_READER_MODES + ("identity-match",):
        raise ValueError("freezing the memory encoder requires semantic, matcher, identity-match, activation, query, context or temporal training")
    heads = {"all": READER_HEADS, "read": HEADS, "identity": IDENTITY_HEADS, "identity-match": IDENTITY_MATCH_HEADS, "scope": SCOPE_HEADS, "semantic": SEMANTIC_HEADS, "matcher": MATCHER_HEADS, "activation": ACTIVATION_HEADS, "temporal": TEMPORAL_HEADS, "query": QUERY_ENCODER_HEADS, "context-query": CONTEXT_QUERY_HEADS, "context": CONTEXT_HEADS}[mode]
    if freeze_memory_encoder:
        heads = tuple(head for head in heads if head != "encode.")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(heads))
    return heads


def reader_optimizer(model, topic_head, mode, learning_rate, attention_learning_rate):
    """Keep the four attention scalars out of the ordinary parameter group."""
    if any(not math.isfinite(rate) or rate <= 0 for rate in (learning_rate, attention_learning_rate)):
        raise ValueError("reader learning rates must be finite and positive")
    temperatures = (model.memory_attention_logits, model.scope_memory_attention_logits)
    candidates = list(model.parameters())
    if mode in ("all", "read", "semantic"):
        candidates += list(topic_head.parameters())
    ordinary = {id(p): p for p in candidates if p.requires_grad and all(p is not t for t in temperatures)}
    groups = []
    if ordinary:
        groups.append(dict(params=list(ordinary.values()), lr=learning_rate,
                           weight_decay=0.001, name="reader"))
    for name, temperature in zip(("memory_attention", "scope_memory_attention"), temperatures):
        if temperature.requires_grad:
            groups.append(dict(params=[temperature], lr=attention_learning_rate,
                               weight_decay=0.0, name=name))
    return torch.optim.AdamW(groups)


def scope_training_loss(model, group, cached, rng, positive_weight=1.0):
    """A scope-only step uses declarations/deletions and every live version."""
    facts = active_bank_facts(group["facts"])
    rows = rng.sample(group["scope_queries"], min(24, len(group["scope_queries"])))
    if not facts or not rows:
        raise ValueError("scope fit curriculum requires facts and scope queries")
    if any(row.get("evaluation_only") for row in rows):
        raise ValueError("scope development queries cannot enter a training step")
    bank, ids, mask, meta = make_bank(facts, cached)
    q, identity = query(model, rows, cached, scope=True)
    gold = torch.tensor([[supported(r, f) for f in facts] for r in rows],
                        dtype=bank.dtype, device=bank.device)
    logits = model.score_bank(q, identity.detach(), bank, ids.detach(), meta, mask, scope=True, memory_context_latents=cached.memory_context_latents([fact["text"] for fact in facts]))
    return 0.5 * balanced(logits, gold, positive_weight) + scope_relation_auxiliary(model, rows, facts, q, bank, mask, memory_context_latents=cached.memory_context_latents([fact["text"] for fact in facts]))


def semantic_training_loss(model, rows, facts, cached, topic_head, topics, positive_weight=1.0, topic_weight=0.05):
    """Recall relation learning with fixed identity and temporal parameter heads."""
    if not rows or not facts:
        raise ValueError("semantic refinement requires fit facts and queries")
    if any(row.get("evaluation_only") or row.get("split", "fit") != "fit"
           or "evidence_challenge_view" in row for row in rows + facts):
        raise ValueError("semantic refinement cannot train on development or evaluation rows")
    bank, ids, mask, meta = make_bank(facts, cached)
    q, identity = query(model, rows, cached)
    gold = torch.tensor([[supported(r, f) for f in facts] for r in rows],
                        dtype=bank.dtype, device=bank.device)
    scoring_query = torch.cat((q[:, :SEMANTIC], q[:, SEMANTIC:].detach()), -1)
    logits = model.score_bank(scoring_query, identity.detach(), bank, ids.detach(), meta.detach(), mask, memory_context_latents=cached.memory_context_latents([fact["text"] for fact in facts]))
    loss = balanced(logits, gold, positive_weight)
    loss += semantic_relation_auxiliary(model, rows, facts, q, bank, mask, memory_context_latents=cached.memory_context_latents([fact["text"] for fact in facts]))
    means = (bank * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
    relation = q[:, :SEMANTIC].reshape(-1, RELATION_HEADS, WIDTH).mean(1)
    if topic_weight:
        loss += topic_weight * F.cross_entropy(topic_head(means), torch.tensor([topics.index(f["prop"]) for f in facts], device=bank.device))
        loss += topic_weight * F.cross_entropy(topic_head(relation), torch.tensor([topics.index(r["prop"]) for r in rows], device=bank.device))
    return loss


def activation_training_loss(model, rows, facts, cached, positive_weight=1.0):
    """Learn relation and time jointly without cutting the intent gradient path."""
    if not rows or not facts:
        raise ValueError("activation refinement requires fit facts and queries")
    if any(row.get("evaluation_only") or row.get("split", "fit") != "fit"
           or "evidence_challenge_view" in row
           for row in rows + facts):
        raise ValueError("activation refinement cannot train on development or evaluation rows")
    bank, ids, mask, meta = make_bank(facts, cached)
    q, identity = query(model, rows, cached)
    gold = torch.tensor([[supported(row, fact) for fact in facts] for row in rows],
                        dtype=bank.dtype, device=bank.device)
    logits = model.score_bank(q, identity.detach(), bank, ids.detach(), meta.detach(), mask, memory_context_latents=cached.memory_context_latents([fact["text"] for fact in facts]))
    return (balanced(logits, gold, positive_weight)
            + semantic_relation_auxiliary(model, rows, facts, q, bank, mask, memory_context_latents=cached.memory_context_latents([fact["text"] for fact in facts]))
            + .25 * F.binary_cross_entropy_with_logits(
                q[:, SEMANTIC:SEMANTIC + 3], intent_targets(rows, bank.device)))


def matcher_correction_training_loss(model, rows, facts, cached, parent, parent_cached,
                                     *, margin=1.):
    """Repair fit pair decisions with fixed inputs and an independent parent."""
    if (not rows or not facts or cached.model is not model or parent_cached.model is not parent
            or not cached.freeze_memory_encoder or not parent_cached.freeze_memory_encoder
            or cached.feature_dropout or parent_cached.feature_dropout
            or parent is model or parent.training
            or any(parameter.requires_grad for parameter in parent.parameters())
            or any(parameter.requires_grad and not name.startswith(MATCHER_HEADS)
                   for name, parameter in model.named_parameters())):
        raise ValueError("matcher corrections require independent frozen inputs and read4 only")
    if any(row.get("evaluation_only") or row.get("split", "fit") != "fit"
           or "evidence_challenge_view" in row for row in rows + facts):
        raise ValueError("matcher corrections require complete fit-only banks")
    with torch.no_grad():
        old = functional_reader_trace(parent, rows, facts, parent_cached)
        gold = torch.tensor([[supported(row, fact) for fact in facts] for row in rows],
                            device=old["activation"].device, dtype=old["activation"].dtype)
        teacher = snapshot_reader_corrections(old["semantic"], old["activation"], gold,
                                             fixed_log_gate=old["fixed_log_gate"],
                                             split="fit", margin=margin)
    live = functional_reader_trace(model, rows, facts, cached)
    return reader_correction_loss(live["semantic"], teacher)[0]


def functional_reader_trace(model, rows, facts, cached):
    """Execute the whole reader with its own encoders and live input gradients."""
    if cached.model is not model:
        raise ValueError("functional reader must own its learned feature encoders")
    bank, ids, mask, metadata = make_bank(facts, cached)
    q, identity = query(model, rows, cached)
    semantic = model.semantic_bank_logits(q, bank, mask, memory_context_latents=cached.memory_context_latents([fact["text"] for fact in facts]))
    entity = model.identity_logits(identity, ids)
    intent = q[:, SEMANTIC:]
    if metadata.ndim == 2:
        metadata = metadata[None].expand(len(q), -1, -1)
    expanded_intent = intent[:, None].expand(-1, len(facts), -1)
    version = model.version(torch.cat((expanded_intent, metadata), -1)).squeeze(-1)
    temporal_log_gate = (F.logsigmoid(version)
                         + (F.logsigmoid(expanded_intent[:, :, :3]) * metadata[:, :, :3]).sum(-1))
    temporal = temporal_log_gate.exp()
    activation = model.score_bank_from_semantic(semantic, q, identity, ids, metadata)
    return dict(activation=activation, semantic=semantic, identity=entity,
                intent=intent, temporal_gate=temporal,
                fixed_log_gate=F.logsigmoid(entity) + temporal_log_gate)


def snapshot_functional_reader(model, cached):
    """Freeze all memory parameters without loading another 1B backbone."""
    parent = deepcopy(model).eval()
    parent.requires_grad_(False)
    return parent, cached.for_model(parent)


def functional_replay_loss(model, parent, rows, facts, cached, parent_cached,
                           *, margin=1., positive_weight=1., intent_weight=0.,
                           temporal_only=False, semantic_corrections=False):
    if not math.isfinite(intent_weight) or intent_weight < 0:
        raise ValueError("functional replay intent weight must be finite and nonnegative")
    if parent.training or any(parameter.requires_grad for parameter in parent.parameters()):
        raise ValueError("functional replay needs an independently frozen complete parent")
    if parent is model:
        raise ValueError("functional replay parent and student must be independent")
    if type(semantic_corrections) is not bool or (semantic_corrections and (
            temporal_only or intent_weight or cached.model is not model
            or parent_cached.model is not parent or not cached.freeze_memory_encoder
            or not parent_cached.freeze_memory_encoder
            or cached.feature_dropout or parent_cached.feature_dropout
            or any(parameter.requires_grad and not name.startswith(MATCHER_HEADS)
                   for name, parameter in model.named_parameters()))):
        raise ValueError("semantic correction replay requires frozen read4-only refinement")
    if any(row.get("evaluation_only") or row.get("split", "fit") != "fit"
           or row.get("binding_fit") or "evidence_challenge_view" in row
           for row in rows + facts):
        raise ValueError("functional replay needs old fit rows, excluding evaluation and new bindings")
    with torch.no_grad():
        old = functional_reader_trace(parent, rows, facts, parent_cached)
        gold = torch.tensor([[supported(row, fact) for fact in facts] for row in rows],
                            device=old["activation"].device, dtype=old["activation"].dtype)
        teacher = build_reader_functional_teacher(
            old["activation"], old["semantic"], old["identity"], old["intent"],
            old["temporal_gate"], gold, split="fit", margin=margin,
        )
    live = functional_reader_trace(model, rows, facts, cached)
    if semantic_corrections:
        correction = snapshot_reader_corrections(old["semantic"], old["activation"], gold,
                                                 fixed_log_gate=old["fixed_log_gate"],
                                                 split="fit", margin=margin)
        supervised, _ = reader_correction_loss(live["semantic"], correction)
    elif temporal_only:
        supervised, _ = temporal_activation_correction(
            live["activation"], gold, rows, facts, old["semantic"], old["identity"],
            positive_weight=positive_weight)
    else:
        supervised = balanced(live["activation"], gold, positive_weight)
    protected = reader_functional_margin_loss(live["activation"], teacher)
    distilled = reader_functional_distillation_loss(
        live["activation"], live["semantic"], live["identity"], live["intent"],
        live["temporal_gate"], teacher,
    )
    intent_supervision = (intent_weight * F.binary_cross_entropy_with_logits(
        live["intent"][:, :3], intent_targets(rows, live["intent"].device))
        if intent_weight else live["activation"].new_zeros(()))
    return supervised + protected + .25 * distilled + intent_supervision, dict(
        supervised=supervised.detach(), margin=protected.detach(),
        distillation=distilled.detach(), protected=teacher.protected_queries.sum().detach(),
        intent_supervision=intent_supervision.detach(),
        queries=len(rows),
    )


def source_kind_targets(facts, device):
    """Text tense labels remain independent of physical lifecycle status."""
    kinds = [fact.get("source_kind", fact["kind"]) for fact in facts]
    if any(type(kind) is not int or kind not in (0, 1, 2) for kind in kinds):
        raise ValueError("invalid source-kind training teacher")
    return torch.tensor(kinds, device=device)


def validate_kind_prefit(enabled, temporal_focus):
    if (type(enabled) is not bool or type(temporal_focus) is not bool
            or (enabled and not temporal_focus)):
        raise ValueError("kind context prefit requires a boolean flag and focused temporal retention")
    return enabled


def validate_matcher_corrections(enabled, *, mode, resumed, frozen,
                                 feature_dropout, functional_weight, topic_weight):
    if (type(enabled) is not bool or (enabled and (
            mode != "matcher" or not resumed or not frozen or feature_dropout
            or not math.isfinite(functional_weight) or functional_weight <= 0
            or topic_weight != 0))):
        raise ValueError("matcher corrections require resumed frozen matcher training, full-parent retention and zero dropout/topic weight")
    return enabled


def validate_query_refinement(enabled, *, resumed, frozen, feature_dropout,
                              functional_weight, topic_weight, anchor_weight,
                              matcher_corrections):
    """Query-only parameters still produce live feedback identity/time gates."""
    if (type(enabled) is not bool or (enabled and (
            not resumed or not frozen or feature_dropout
            or not math.isfinite(functional_weight) or functional_weight <= 0
            or topic_weight != 0 or anchor_weight != 0
            or matcher_corrections is not False))):
        raise ValueError("query refinement requires a resumed reader, frozen undropped memory, full-parent retention, zero topic/anchor weights and no matcher corrections")
    return enabled


def validate_context_refinement(enabled, *, mode, resumed, frozen, feature_dropout,
                                functional_weight, topic_weight, anchor_weight,
                                matcher_corrections):
    """Train existing context projections against an independent full parent."""
    if (type(enabled) is not bool or (enabled and (
            mode not in CONTEXT_MODES or not resumed or not frozen or feature_dropout
            or not math.isfinite(functional_weight) or functional_weight <= 0
            or topic_weight != 0 or anchor_weight != 0
            or matcher_corrections is not False))):
        raise ValueError("context refinement requires a resumed reader, frozen undropped memory, full-parent retention, zero topic/anchor weights and no matcher corrections")
    return enabled


def context_refinement_contract(mode):
    """Current training/checkpoint schema; tensor and snapshot formats stay fixed."""
    if mode not in CONTEXT_MODES:
        return None
    names = (["read_context_query.weight"] if mode == "context-query" else
             ["read_context_memory.weight", "read_context_query.weight", "read_context_delta.weight"])
    return dict(mode=mode, trainable_parameters=4096 if mode == "context-query" else 70656,
                trainable_tensors=len(names), frozen_tensors=len(KEYS) - len(names), trainable_names=names,
                memory_context_inputs="immutable original-hidden last/mean4096; U recomputed live per score",
                teacher=f"independent frozen complete{len(KEYS)}; only immutable raw C features shared",
                fixed_parameters=f"all other tensors, including old token/read/query/attention/identity/time/scope and source{len(PLAN_KEYS) + len(QUERY_KEYS)}",
                projection_gradient="current nonzero B enables U/V gradients without warmup; zero B has no U/V gradient",
                serving_format=f"current{len(KEYS)} tensors and snapshot2; unchanged")


def prefit_temporal_kind(model, prepared, cached):
    """Calibrate the contextual residual before freezing kind for time refinement."""
    if cached.model is not model or not cached.freeze_memory_encoder:
        raise ValueError("kind prefit requires independently owned frozen memory encoding")
    pools = prepared["source_kind_facts"]
    if set(pools) != {0, 1, 2} or any(not pools[kind] for kind in pools):
        raise ValueError("kind prefit requires all original fit source classes")
    facts = [fact for kind in (0, 1, 2) for fact in pools[kind]]
    if any(row.get("evaluation_only") or row.get("split", "fit") != "fit"
           or "evidence_challenge_view" in row for row in facts):
        raise ValueError("kind prefit cannot consume development or evaluation facts")
    with torch.no_grad():
        bank, mask, _ = cached.encode([fact["text"] for fact in facts])
        means = (bank * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
    from neural_kind_context import calibrate_kind_context
    contexts = cached.kind_contexts[cached.indices([fact["text"] for fact in facts])]
    result = calibrate_kind_context(model, means, contexts,
                                    source_kind_targets(facts, means.device))
    if not result["accepted"]:
        raise RuntimeError("contextual kind prefit failed; initial head preserved")
    model.kind.requires_grad_(False)
    model.kind_context.requires_grad_(False)
    return result


def temporal_activation_correction(logits, gold, rows, facts, parent_semantic,
                                   parent_identity, *, positive_weight=1.):
    """Fit-only time correction; full-bank protection and evaluation stay intact.

    Frozen relation/identity probabilities bound which positives time can
    repair. Negative correction concerns authored same-relation records in
    disallowed physical states, rather than unrelated semantic errors.
    """
    if not rows or not facts:
        raise ValueError("temporal correction requires fit facts and queries")
    if not math.isfinite(positive_weight) or positive_weight < 0:
        raise ValueError("invalid temporal correction positive weight")
    if any(row.get("evaluation_only") or row.get("split", "fit") != "fit"
           or "evidence_challenge_view" in row for row in rows + facts):
        raise ValueError("temporal correction requires fit labels")
    if any(value.shape != (len(rows), len(facts)) or value.device != logits.device
           or not torch.isfinite(value).all()
           for value in (logits, gold, parent_semantic, parent_identity)):
        raise ValueError("invalid temporal correction geometry")
    if (not logits.is_floating_point() or any(not value.is_floating_point()
            or value.dtype != logits.dtype for value in (parent_semantic, parent_identity))
            or not ((gold == 0) | (gold == 1)).all()):
        raise ValueError("invalid temporal correction probabilities or labels")
    fixed_gate = (parent_semantic.sigmoid() * parent_identity.sigmoid()).detach()
    positive = gold.bool() & (fixed_gate > .5)
    states = []
    physical = {0: 0, 1: 1, 3: 2}
    for fact in facts:
        if "status" in fact:
            if type(fact["status"]) is not int or fact["status"] not in physical:
                raise ValueError("temporal correction needs live physical states")
            states.append(physical[fact["status"]])
        else:
            if fact.get("source_kind", fact["kind"]) != fact["kind"]:
                raise ValueError("source kind differs from physical state without explicit status")
            states.append(fact["kind"])
    if any(type(state) is not int or state not in (0, 1, 2) for state in states):
        raise ValueError("invalid temporal correction state")
    allowed = intent_targets(rows, logits.device).bool()
    permitted = allowed[:, torch.tensor(states, device=logits.device)]

    def labels(value):
        return {value} if isinstance(value, str) else set(value or ())

    same_relation = torch.tensor([
        [fact["entity"] in labels(row.get("entities", row.get("entity")))
         and fact["prop"] in labels(row["prop"]) for fact in facts]
        for row in rows], dtype=torch.bool, device=logits.device)
    negative = same_relation & ~permitted & ~gold.bool()
    losses = F.binary_cross_entropy_with_logits(logits, gold.to(logits.dtype), reduction="none")
    hard = losses.masked_fill(~negative, -torch.inf).topk(min(8, len(facts)), -1).values
    loss = (positive_weight * query_masked_mean(losses, positive)
            + query_masked_mean(losses, negative)
            + 1.5 * query_masked_mean(hard, torch.isfinite(hard)))
    return loss, dict(eligible_positive_pairs=positive.sum().detach(),
                      frozen_gate_blocked_gold_pairs=(gold.bool() & ~positive).sum().detach(),
                      temporal_negative_pairs=negative.sum().detach())


def temporal_training_loss(model, rows, facts, cached, positive_weight=1.0,
                           *, kind_facts=None, parent=None, parent_cached=None):
    """Refine learned time channels with frozen relation and identity features."""
    if any(row.get("evaluation_only") or row.get("split", "fit") != "fit"
           or "evidence_challenge_view" in row
           for row in rows + facts + (kind_facts or [])):
        raise ValueError("temporal development queries cannot enter training")
    if not rows or not facts:
        raise ValueError("temporal refinement requires fit facts and queries")
    if kind_facts is not None and not kind_facts:
        raise ValueError("temporal source-kind supervision requires facts")
    if (parent is None) != (parent_cached is None):
        raise ValueError("temporal correction requires matching frozen parent features")
    bank, ids, mask, meta = make_bank(facts, cached)
    q, identity = query(model, rows, cached)
    q = torch.cat((q[:, :SEMANTIC].detach(), q[:, SEMANTIC:]), -1)
    gold = torch.tensor([[supported(row, fact) for fact in facts] for row in rows],
                        dtype=bank.dtype, device=bank.device)
    logits = model.score_bank(q, identity.detach(), bank, ids.detach(), meta, mask, memory_context_latents=cached.memory_context_latents([fact["text"] for fact in facts]))
    if parent is not None:
        if parent is model or parent.training or any(p.requires_grad for p in parent.parameters()):
            raise ValueError("temporal correction needs an independent frozen parent")
        with torch.no_grad():
            teacher = functional_reader_trace(parent, rows, facts, parent_cached)
        activation, _ = temporal_activation_correction(
            logits, gold, rows, facts, teacher["semantic"], teacher["identity"],
            positive_weight=positive_weight)
    else:
        activation = balanced(logits, gold, positive_weight)
    if kind_facts is not None:
        if not kind_facts:
            raise ValueError("temporal source-kind supervision requires facts")
        kind_bank, kind_mask, _ = cached.encode([fact["text"] for fact in kind_facts])
        kind_means = (kind_bank * kind_mask[:, :, None]).sum(1) / kind_mask.sum(1)[:, None]
        kind_targets = source_kind_targets(kind_facts, bank.device)
    else:
        kind_means = (bank * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
        kind_targets = source_kind_targets(facts, bank.device)
    kind_texts = [fact["text"] for fact in (kind_facts if kind_facts is not None else facts)]
    kind_logits = cached.kind_logits(kind_texts, kind_means)
    if parent is not None:
        # The parent computes its own representation from immutable C inputs.
        # Original source labels remain independent of the record's status.
        with torch.no_grad():
            parent_bank, parent_mask, _ = parent_cached.encode(
                [fact["text"] for fact in (kind_facts if kind_facts is not None else facts)])
            parent_means = ((parent_bank * parent_mask[:, :, None]).sum(1)
                            / parent_mask.sum(1)[:, None])
            kind_teacher = snapshot_kind_teacher(parent_cached.kind_logits(kind_texts, parent_means), kind_targets)
        kind_loss, _ = kind_correction_loss(kind_logits, kind_teacher)
    else:
        kind_loss = F.cross_entropy(kind_logits, kind_targets)
    return (activation
            + kind_loss
            + 0.25 * F.binary_cross_entropy_with_logits(
                q[:, SEMANTIC:SEMANTIC + 3], intent_targets(rows, bank.device)))


def matcher_preservation_loss(model, rows, facts, cached, parent, *, weight=1., margin=1.):
    """Fit-only complete-set protection with the same frozen inputs and gates."""
    if any(row.get("evaluation_only") for row in rows):
        raise ValueError("reader anchors cannot consume development queries")
    bank, ids, mask, meta = make_bank(facts, cached)
    q, identity = query(model, rows, cached)
    pairs = matcher_pair_features(q, bank, mask, model.attention_scales())
    with torch.no_grad():
        parent_semantic = model.read_logits_from_pairs(pairs, q, cached.memory_context_latents([fact["text"] for fact in facts]), parameters=parent)
        parent_activation = model.score_bank_from_semantic(
            parent_semantic, q, identity, ids, meta)
        # Use the actual frozen identity/time gate. Subtracting rounded output
        # probabilities can turn the exact gate=1/2 boundary into a reachable pair.
        intent = q[:, SEMANTIC:]
        metadata = meta[None].expand(len(q), -1, -1) if meta.ndim == 2 else meta
        expanded_intent = intent[:, None].expand(-1, len(facts), -1)
        version = model.version(torch.cat((expanded_intent, metadata), -1)).squeeze(-1)
        temporal_log_gate = (F.logsigmoid(version)
                             + (F.logsigmoid(expanded_intent[:, :, :3]) * metadata[:, :, :3]).sum(-1))
        fixed_log_gate = F.logsigmoid(model.identity_logits(identity, ids)) + temporal_log_gate
        gold = torch.tensor([[supported(row, fact) for fact in facts] for row in rows],
                            dtype=bank.dtype, device=bank.device)
        teacher = build_reader_anchor_teacher(parent_semantic, parent_activation, gold,
                                              fixed_log_gate=fixed_log_gate, split="fit", margin=margin)
    live = model.read_logits_from_pairs(pairs, q, cached.memory_context_latents([fact["text"] for fact in facts]))
    return reader_support_anchor_loss(live, teacher, weight=weight)


def role_supervision(model, rows, cached):
    losses = []
    for row in rows:
        text = row["text"]
        actors = row.get("entities", [row.get("entity")])
        raw, ranges = text.encode(), []
        for actor in actors:
            if not actor:
                continue
            position = 0
            while True:
                at = raw.find(actor.encode(), position)
                if at < 0:
                    break
                ranges.append((at, at + len(actor.encode())))
                position = at + len(actor.encode())
        index = cached.index[text]
        offsets = cached.word_offsets[index] + torch.arange(cached.word_counts[text], device=cached.device)
        logits = model.role(cached.word_inputs[offsets]).squeeze(-1)
        targets = torch.tensor([float(any(b > u and a < v for u, v in ranges))
                                for a, b in cached.word_units[text]], device=logits.device)
        losses.append(F.binary_cross_entropy_with_logits(logits, targets, pos_weight=logits.new_tensor(5.0)))
    return torch.stack(losses).mean()


def identity_calibration(model, queries, memories, gold):
    """Stress only the matcher with projection collisions/noise, keeping actor labels."""
    raw = queries[:, :LEXICAL] @ memories[:, :LEXICAL].T
    projected = queries[:, LEXICAL:] @ memories[:, LEXICAL:].T
    features = torch.stack((raw, projected), -1).detach()
    noisy = features.clone()
    perturbation = torch.rand_like(projected)
    noisy[..., 1] = torch.where(gold.bool(), 2 * perturbation - 1, 0.5 + 0.5 * perturbation)
    actual = model.identity_match(features).squeeze(-1)
    perturbed = model.identity_match(noisy).squeeze(-1)
    return 0.5 * (balanced(actual, gold) + balanced(perturbed, gold))


def require_identity_match_fit(rows):
    """Keep original provenance before reader_data drops fact-level markers."""
    for row in rows:
        originals = (row, row["observation"]) if isinstance(row.get("observation"), dict) else (row,)
        for original in originals:
            try:
                require_query_binding_fit(original)
                if "evidence_challenge_view" in original:
                    raise ValueError("evaluation evidence view")
            except ValueError as error:
                raise ValueError("identity-match refinement cannot train on development or evaluation rows") from error


def identity_match_training_loss(model, rows, facts, cached):
    """Calibrate only the identity matcher over frozen model-predicted references."""
    if not rows or not facts:
        raise ValueError("identity-match refinement requires fit facts and queries")
    require_identity_match_fit(rows + facts)
    if (cached.model is not model or not cached.freeze_memory_encoder or cached.feature_dropout
            or any(parameter.requires_grad != name.startswith(IDENTITY_MATCH_HEADS)
                   for name, parameter in model.named_parameters())):
        raise ValueError("identity-match requires matcher-only parameters and frozen undropped representations")
    with torch.no_grad():
        _, identities = query(model, rows, cached)
        memories = cached.reference_ids(facts)
    # Actor labels supervise pair decisions; they never supply a query reference.
    gold = identities.new_tensor([[fact["entity"] in row.get("entities", [row.get("entity")])
                                   for fact in facts] for row in rows])
    return (balanced(model.identity_logits(identities, memories), gold)
            + .25 * identity_calibration(model, identities, memories, gold))


def intent_targets(rows, device):
    states = []
    for row in rows:
        allowed = row.get("allowed_kinds")
        if allowed is None:
            allowed = [row["kind"]] if row.get("kind") in (0, 1, 2) else [0, 1, 2]
        if any(k not in (0, 1, 2) for k in allowed):
            raise ValueError("invalid allowed memory state")
        states.append([float(k in allowed) for k in range(3)])
    return torch.tensor(states, device=device)


@torch.no_grad()
def evaluate(m, group, cached):
    return _evaluate_bank(m, group, cached, scope=False)


def expansion_evidence(group):
    """Keep the existing cue curriculum fixed while adding raw set questions."""
    from neural_reader_evidence_curriculum import augment_evidence
    additions = [row for row in group["queries"] if row.get("expansion_query_class")]
    original = dict(facts=group["facts"],
                    queries=[row for row in group["queries"] if not row.get("expansion_query_class")])
    augmented = augment_evidence(original, 5322, 400)
    augmented["queries"].extend(deepcopy(additions))
    return augmented


def sample_expansion_training(group, rng, world_rng, *, material_max_sources=2,
                              guarded=True):
    """Sample authored worlds and explicitly expose complete-set questions."""
    from prepare_neural_reader_expansion import sample_expansion_world
    if material_max_sources not in (1, 2):
        raise ValueError("material worlds allow a fixed single source or zero/one/two sources")
    world = sample_expansion_world(group, world_rng,
                                  material_counts=(1,) if material_max_sources == 1 else (0, 1, 2))
    bank = active_bank_facts(world["facts"])
    rows = world["queries"]
    present = [row for row in rows if row["expansion_world_class"] == "source_present"]
    absent = [row for row in rows if row["expansion_world_class"] == "source_absent"]
    nulls = [row for row in rows if not row["prop"]]
    broad = [row for row in rows if row.get("expansion_query_class") == "broad"
             and not row.get("augmented")]
    unsupported = [row for row in rows if row.get("expansion_query_class") == "unsupported"
                   and not row.get("augmented")]
    sampled = ((sample_relation_queries(present, rng, 24) if guarded else
                rng.sample(present, min(24, len(present))))
               + rng.sample(nulls, min(8, len(nulls)))
               + rng.sample(absent, min(4, len(absent)))
               + broad + unsupported)
    if len({row["text"] for row in sampled}) != len(sampled):
        raise ValueError("expansion sampling must not duplicate a question")
    counts = Counter(row["expansion_world_class"] for row in sampled)
    for row in broad:
        counts["broad_supports_" + str(len(row["support_texts"]))] += 1
    return bank, sampled, dict(counts)


@torch.no_grad()
def expansion_world_diagnostics(model, group, cached):
    """Fixed raw fit worlds; never checkpoint selection or runtime qualification."""
    from prepare_neural_reader_expansion import sample_expansion_world, COEXISTENCE_PROP
    output = {}
    material_groups = {tuple(fact["bank_group"]) for fact in group["facts"]
                       if fact["prop"] == COEXISTENCE_PROP}
    for count in (0, 1, 2):
        world = sample_expansion_world(group, random.Random(5304),
                                      source_counts={key: count for key in material_groups})
        material = [row for row in world["queries"]
                    if row["prop"] == COEXISTENCE_PROP and not row.get("augmented")]
        groups = {name: [row for row in material if row["expansion_world_class"] == name]
                  for name in sorted({row["expansion_world_class"] for row in material})}
        output["material_sources_" + str(count)] = {
            name: evaluate(model, dict(facts=world["facts"], queries=rows), cached)
            for name, rows in groups.items()
        }
    return output


@torch.no_grad()
def scope_evaluate(m, group, cached):
    """Measure the separate source-to-all-versions scope objective."""
    return _evaluate_bank(m, group, cached, scope=True)


@torch.no_grad()
def kind_bank_diagnostics(model, group, cached):
    """Text-kind diagnostics stay separate from physical lifecycle state."""
    facts = active_bank_facts(group["facts"])
    if not facts:
        return dict(facts=0, source_kind_accuracy=None, prediction_counts=[0, 0, 0],
                    teacher_counts=[0, 0, 0], confusion=[[0] * 3 for _ in range(3)],
                    bank_status_counts=[0, 0, 0], bank_status_order=["current", "superseded", "tentative"])
    bank, mask, _ = cached.encode([fact["text"] for fact in facts])
    means = (bank * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
    logits = cached.kind_logits([fact["text"] for fact in facts], means)
    prediction = logits.argmax(-1).cpu()
    teacher = source_kind_targets(facts, "cpu")
    confusion = torch.bincount(teacher * 3 + prediction, minlength=9).reshape(3, 3)
    physical = {0: 0, 1: 1, 3: 2}
    if any("status" in fact and fact["status"] not in physical for fact in facts):
        raise ValueError("invalid physical state in kind diagnostics")
    actual = prediction.clone()
    for index, fact in enumerate(facts):
        if "status" in fact:
            actual[index] = physical[fact["status"]]
    margin = logits.topk(2, -1).values
    margin = margin[:, 0] - margin[:, 1]
    return dict(facts=len(facts), source_kind_accuracy=float((prediction == teacher).float().mean()),
                prediction_counts=torch.bincount(prediction, minlength=3).tolist(),
                teacher_counts=torch.bincount(teacher, minlength=3).tolist(),
                confusion=confusion.tolist(), bank_status_counts=torch.bincount(actual, minlength=3).tolist(),
                bank_status_order=["current", "superseded", "tentative"],
                minimum_top2_margin=float(margin.min()))


@torch.no_grad()
def attention_bank_diagnostics(model, group, cached, max_queries=16, *, scope=False):
    """Measure actual memory-token attention; no activation labels are inspected."""
    facts = active_bank_facts(group["facts"])
    rows = group.get("scope_queries" if scope else "queries", [])[:max_queries]
    scales = model.attention_scales(scope=scope)
    result = dict(scales=scales.detach().cpu().tolist(), facts=len(facts), queries=len(rows),
                  channel="scope" if scope else "read")
    if not facts or not rows:
        return dict(result, normalized_entropy=None, head_normalized_entropy=None,
                    head_maximum_weight=None)
    bank, mask, _ = cached.encode([fact["text"] for fact in facts])
    q, _ = query(model, rows, cached, scope=scope)
    relations = q[:, :SEMANTIC].reshape(len(rows), RELATION_HEADS, WIDTH)
    logits = torch.einsum("qhd,ntd->qnht", relations, bank) * 0.125
    logits *= scales[None, None, :, None]
    weights = logits.masked_fill(~mask[None, :, None], -torch.inf).softmax(-1)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
    lengths = mask.sum(-1)
    valid = lengths > 1
    if valid.any():
        normalized = entropy[:, valid, :] / lengths[valid].log()[None, :, None]
        result.update(normalized_entropy=float(normalized.mean()),
                      head_normalized_entropy=normalized.mean((0, 1)).cpu().tolist())
    else:
        result.update(normalized_entropy=None, head_normalized_entropy=None)
    result.update(head_maximum_weight=weights.max(-1).values.mean((0, 1)).cpu().tolist(),
                  single_token_facts=int((lengths == 1).sum()))
    return result


def _evaluate_bank(m, group, cached, *, scope):
    # Complete bank: every fact variant and every entity participates.
    facts = active_bank_facts(group["facts"])
    questions = group["scope_queries"] if scope else group["queries"]
    bank, ids, mask, meta = make_bank(facts, cached)
    tp = fp = fn = exact = nullbad = nullcount = positive_count = positive_exact = 0
    blocked_positive = gate_reachable_queries = 0
    for start in range(0, len(questions), 16):
        rows = questions[start : start + 16]
        q, identity = query(m, rows, cached, scope=scope)
        predicted = m.score_bank(q, identity, bank, ids, meta, mask, scope=scope, memory_context_latents=cached.memory_context_latents([fact["text"] for fact in facts])) > 0
        predicted &= torch.tensor(
            [not r.get("unresolved", False) for r in rows], device=bank.device
        )[:, None]
        gold = torch.tensor(
            [[supported(r, f) for f in facts] for r in rows], device=bank.device
        )
        # A saturated semantic probability isolates the identity/time ceiling
        # for these current inputs. This is a scoring diagnostic, not a mask
        # used by training, serving, or checkpoint selection.
        reachable = m.score_bank_from_semantic(
            torch.full_like(gold, 100., dtype=q.dtype), q, identity, ids, meta,
            scope=scope) > 0
        blocked_positive += int((gold & ~reachable).sum())
        gate_reachable_queries += int((~(gold & ~reachable).any(-1) & gold.any(-1)).sum())
        tp += int((predicted & gold).sum())
        fp += int((predicted & ~gold).sum())
        fn += int((~predicted & gold).sum())
        matching = (predicted == gold).all(-1)
        exact += int(matching.sum())
        empty = ~gold.any(-1)
        positive_count += int((~empty).sum())
        positive_exact += int((matching & ~empty).sum())
        nullcount += int(empty.sum())
        nullbad += int((predicted.any(-1) & empty).sum())
    return dict(
        activation_f1=2 * tp / max(2 * tp + fp + fn, 1),
        support_set_exact=exact / len(questions) if questions else None,
        positive_queries=positive_count,
        positive_exact=positive_exact,
        positive_support_set_exact=(
            positive_exact / positive_count if positive_count else None
        ),
        null_false_activation=nullbad / max(nullcount, 1),
        tp=tp,
        fp=fp,
        fn=fn,
        bank_size=len(facts),
        queries=len(questions),
        fixed_input_gate_blocked_positives=blocked_positive,
        fixed_input_gate_reachable_positive_queries=gate_reachable_queries,
        gate_diagnostic_scope="current input identity/time gates with saturated semantic; not a training or serving mask",
        bank_constructed_by="training/component gold references; not native qualification",
    )


def scope_selection_score(metrics):
    """Select scope independently; keep the initial checkpoint eligible."""
    accuracy = metrics["positive_support_set_exact"]
    if accuracy is None:
        raise ValueError("scope selection requires positive development queries")
    # Strict sets take precedence; among ties prefer fewer wrong slot matches.
    return (accuracy, -metrics["fp"] / max(metrics["queries"], 1), selection_score(metrics))


def scope_refinement_safe(metrics, initial):
    """Shared query changes cannot spend the initial update/delete quality."""
    exact, baseline = metrics["positive_support_set_exact"], initial["positive_support_set_exact"]
    if exact is None or baseline is None:
        return False
    null, initial_null = metrics["null_false_activation"], initial["null_false_activation"]
    null_safe = (null is None if initial_null is None
                 else null is not None and null <= initial_null)
    return exact >= baseline and metrics["fp"] <= initial["fp"] and null_safe


def selection_score(metrics):
    """Use the same objective for the starting checkpoint and later candidates."""
    accuracy = metrics["positive_support_set_exact"]
    if accuracy is None:
        raise ValueError("checkpoint selection requires positive development queries")
    return metrics["activation_f1"] + accuracy - metrics["null_false_activation"]


def selection_views_score(metrics):
    """Give each fixed development view equal weight, regardless of its size."""
    if not metrics:
        raise ValueError("checkpoint selection requires a development view")
    return sum(selection_score(m) for m in metrics.values()) / len(metrics)


def reader_initial_selection(component, feedback, *, scope=None):
    """Capture this run's independent baseline before replacing any best point."""
    from neural_feedback_evaluation import feedback_selection_score

    return reader_checkpoint_payload({},
        initial_development_views=component,
        initial_evidence_development_views=feedback,
        initial_selection_score=(scope_selection_score(scope) if scope is not None
                                 else feedback_selection_score(component, feedback)))


def reader_output_paths(output, model_output=None):
    """Keep generated diagnostics in the run and model artifacts in models/."""
    root = Path(output).resolve()
    models = (Path(model_output).resolve() if model_output else
              Path(__file__).resolve().parents[1] / "models" / root.name)
    if root == models or root.is_relative_to(models) or models.is_relative_to(root):
        raise ValueError("reader diagnostics and model outputs must be separate directories")
    return root, models


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(5302)
    if args.steps < 1:
        raise ValueError("steps must be positive")
    eval_every = getattr(args, "eval_every", 100)
    if type(eval_every) is not int or eval_every < 1:
        raise ValueError("eval-every must be a positive integer")
    identity_match_focus = args.train_heads == "identity-match"
    if identity_match_focus and (
            not args.resume or not args.freeze_memory_encoder or args.feature_dropout
            or args.semantic_topic_weight != 0
            or getattr(args, "reader_anchor_weight", 0.) != 0
            or getattr(args, "reader_functional_weight", 0.) != 0
            or getattr(args, "identity_diversity", False)
            or getattr(args, "expansion", False) or getattr(args, "evidence_cues", False)):
        raise ValueError("identity-match is identity-only calibration: requires a resumed reader, frozen undropped representations, zero topic/anchor/functional weights, and bound fit queries without identity-diversity, expansion or evidence-cues")
    reader_anchor_weight = getattr(args, "reader_anchor_weight", 0.)
    reader_anchor_margin = getattr(args, "reader_anchor_margin", 1.)
    if (not math.isfinite(reader_anchor_weight) or reader_anchor_weight < 0
            or not math.isfinite(reader_anchor_margin) or reader_anchor_margin <= 0):
        raise ValueError("invalid reader anchor weight or margin")
    if reader_anchor_weight and (args.train_heads != "matcher" or not args.resume
                                 or not args.freeze_memory_encoder or args.feature_dropout):
        raise ValueError("reader anchors require resumed matcher training with frozen undropped inputs")
    functional_weight = getattr(args, "reader_functional_weight", 0.)
    functional_margin = getattr(args, "reader_functional_margin", 1.)
    if (not math.isfinite(functional_weight) or functional_weight < 0
            or not math.isfinite(functional_margin) or functional_margin <= 0):
        raise ValueError("invalid reader functional retention weight or margin")
    if functional_weight and (args.train_heads not in FUNCTIONAL_MODES or not args.resume
                              or not args.freeze_memory_encoder or args.feature_dropout
                              or reader_anchor_weight):
        raise ValueError("functional retention requires resumed semantic/matcher/activation/query/context/temporal with frozen undropped memory and no matcher anchor")
    temporal_focus = args.train_heads == "temporal" and functional_weight > 0
    kind_context_prefit = validate_kind_prefit(getattr(args, "kind_context_prefit", False), temporal_focus)
    matcher_corrections = getattr(args, "matcher_corrections", False)
    if type(matcher_corrections) is not bool:
        raise ValueError("matcher corrections require a boolean flag")
    if matcher_corrections:
        validate_matcher_corrections(
            matcher_corrections, mode=args.train_heads,
            resumed=bool(args.resume), frozen=args.freeze_memory_encoder,
            feature_dropout=args.feature_dropout, functional_weight=functional_weight,
            topic_weight=args.semantic_topic_weight)
    query_focus = args.train_heads == "query"
    context_focus = args.train_heads in CONTEXT_MODES
    context_contract = context_refinement_contract(args.train_heads)
    if context_focus:
        validate_context_refinement(
            True, mode=args.train_heads, resumed=bool(args.resume), frozen=args.freeze_memory_encoder,
            feature_dropout=args.feature_dropout, functional_weight=functional_weight,
            topic_weight=args.semantic_topic_weight, anchor_weight=reader_anchor_weight,
            matcher_corrections=matcher_corrections)
    if query_focus:
        validate_query_refinement(
            True, resumed=bool(args.resume), frozen=args.freeze_memory_encoder,
            feature_dropout=args.feature_dropout, functional_weight=functional_weight,
            topic_weight=args.semantic_topic_weight, anchor_weight=reader_anchor_weight,
            matcher_corrections=matcher_corrections)
    query_contract = ("query-token-encoder+four-relation-pools only; all other parameters frozen; "
                      "identity/intent/time outputs remain live through query-dependent feedback gates; "
                      "original subject references come from the frozen writer; independent full-parent encoders"
                      if query_focus else None)
    if args.train_heads == "activation" and (not args.resume or not functional_weight
                                            or not args.freeze_memory_encoder
                                            or args.feature_dropout
                                            or args.semantic_topic_weight != 0):
        raise ValueError("activation refinement requires a resumed reader, full-parent retention, frozen undropped memory and zero topic weight")
    functional_intent_weight = .25 if args.train_heads == "activation" or temporal_focus else 0.
    functional_description = "old-gold-balanced-BCE+capped-final-margin+.25-probability-functional-distillation"
    if matcher_corrections:
        functional_description = "fit-parent-fixed-gate-pair-corrections+" + PRESERVATION_REDUCTION + "+capped-final-margin+.25-probability-functional-distillation"
    if functional_intent_weight:
        functional_description += "+.25-fit-multi-state-intent-BCE"
    if temporal_focus:
        functional_description = functional_description.replace(
            "old-gold-balanced-BCE", "old-gold-feasible-time-BCE")
    if not 0 <= args.semantic_topic_weight <= 1:
        raise ValueError("semantic topic weight must be in [0, 1]")
    if args.train_heads in ("matcher", "temporal", "query") + CONTEXT_MODES and args.semantic_topic_weight != 0:
        raise ValueError(f"{args.train_heads} training requires --semantic-topic-weight 0")
    if args.train_heads == "temporal" and not args.resume:
        raise ValueError("temporal refinement requires a trained reader checkpoint")
    relation_paraphrases = getattr(args, "relation_paraphrases", False)
    if relation_paraphrases and not getattr(args, "episode_relations", False):
        raise ValueError("relation paraphrases require episode relation training")
    relation_bindings = getattr(args, "relation_bindings", False)
    if relation_bindings and (args.train_heads not in RELATION_MODES
                              or args.semantic_topic_weight != 0):
        raise ValueError("relation bindings require relation/context refinement with zero topic weight")
    root, model_root = reader_output_paths(args.output, getattr(args, "model_output", None))
    if root.exists() or model_root.exists():
        raise FileExistsError("reader run and model output directories must both be fresh")
    root.mkdir(parents=True, exist_ok=False)
    model_root.mkdir(parents=True, exist_ok=False)
    if args.curriculum:
        authored = json.loads(Path(args.curriculum).read_text())
    else:
        from prepare_neural_colloquial import colloquial_curriculum

        authored = {s: colloquial_curriculum(s) for s in ("fit", "development")}
    if not {"fit", "development"} <= authored.keys():
        raise ValueError("independent fit/development observations are required")
    if identity_match_focus:
        require_identity_match_fit(authored["fit"])
    from prepare_neural_colloquial import reader_data

    groups = {s: reader_data(s, authored[s]) for s in ("fit", "development")}
    if relation_bindings:
        from neural_reader_evidence_curriculum import reader_binding_fit_group

        groups["binding_fit"] = reader_binding_fit_group()
    if args.fit_paraphrases:
        from neural_fit_support import add_fit_relation_queries

        groups["fit"] = add_fit_relation_queries(groups["fit"], "fit")
    if args.replay_episodes:
        groups["episode_fit"] = episode_observation_group("fit")
        groups["episode_development"] = episode_observation_group("development")
    backbone = sha(args.gguf)
    cache = {}
    for path in ([args.features] if args.features else []) + args.extra_features:
        saved = load_features(path, backbone=backbone, probe=sha(args.probe))
        if saved["identity"]["backbone"] != backbone:
            raise ValueError("frozen feature backbone mismatch")
        cache.update(saved["features"])
    planner_path = args.planner_checkpoint or args.resume
    if not planner_path:
        raise ValueError(
            "train a writer first and supply --planner-checkpoint; reader weights may start from scratch"
        )
    planner_bytes = Path(planner_path).read_bytes()
    planner_identity = hashlib.sha256(planner_bytes).hexdigest()
    planner_cp = torch.load(
        io.BytesIO(planner_bytes), weights_only=False, map_location="cpu"
    )
    if planner_cp["backbone_sha256"] != backbone:
        raise ValueError("frozen query planner backbone mismatch")
    planner = load_model(planner_cp).eval()
    for parameter in planner.parameters():
        parameter.requires_grad_(False)
    required = {
        q["observation"][k]
        for g in groups.values()
        for q in g["queries"]
        for k in ("body", "context")
    }
    missing = sorted(required - cache.keys())
    if missing:
        extra, _ = extract(
            args, {"additional": [dict(body=t, context=t) for t in missing]}
        )
        cache.update(extra)
    rows, bindings = {}, {}
    for name, group in groups.items():
        rows[name], bindings[name] = bind_curriculum_queries(group, planner, cache)
    binding_group = rows.pop("binding_fit") if relation_bindings else None
    if binding_group:
        binding_group["queries"] = [q for q in binding_group["queries"] if not q["unresolved"]]
        if not binding_group["queries"]:
            raise ValueError("relation bindings require resolved frozen-writer queries")
    if args.replay_episodes:
        extra = rows.pop("episode_fit")
        rows["fit"]["facts"] += extra["facts"]
        rows["fit"]["queries"] += extra["queries"]
    rows["fit"]["queries"] = [q for q in rows["fit"]["queries"] if not q["unresolved"]]
    (root / "query_binding.json").write_text(json.dumps(bindings, indent=2))
    expansion = None
    if args.expansion:
        from prepare_neural_reader_expansion import reader_expansion

        expansion = reader_expansion()
    if args.evidence_cues:
        from neural_reader_evidence_curriculum import augment_evidence

        rows["fit"] = augment_evidence(rows["fit"], 5321, 600)
        for row in rows["fit"]["queries"]:
            if row.get("augmented"):
                row.pop("reference", None)
        if expansion:
            expansion = expansion_evidence(expansion)
    identity_group = None
    if args.identity_diversity or args.train_heads == "identity":
        from prepare_neural_identity import identity_curriculum

        identity_group = identity_curriculum()
        multi = identity_multisubject_fit_group()
        known = {f["text"] for f in identity_group["facts"]}
        identity_group["facts"] += [f for f in multi["facts"] if f["text"] not in known]
        identity_group["queries"] += multi["queries"]
    scope_group, lifecycle_group = scope_fit_group(), lifecycle_fit_group()
    served_scope_group = served_lifecycle_scope_group()
    paraphrase_scope_group = served_scope_paraphrase_group()
    scope_primary = args.train_heads == "scope"
    # Identity calibration changes both recall and source-to-all-versions scope.
    # These selection guards do not enable another mode's training objectives.
    component_guard = args.train_heads in GUARDED_READER_MODES or identity_match_focus
    scope_guard = component_guard
    scope_development = scope_development_group() if scope_primary or scope_guard else None
    variation = relation_variation_fit_group() if args.train_heads in RELATION_MODES or temporal_focus else None
    if getattr(args, "episode_relations", False):
        if variation is None:
            raise ValueError("episode relation variation requires semantic, matcher or activation training")
        from neural_fit_support import episode_relation_variation_fit_group

        extra = episode_relation_variation_fit_group()
        for key in ("facts", "queries"):
            variation[key] += extra[key]
    if getattr(args, "episode_aspects", False):
        if variation is None:
            raise ValueError("episode aspect variation requires semantic, matcher or activation training")
        from neural_fit_support import episode_aspect_variation_fit_group

        extra = episode_aspect_variation_fit_group()
        for key in ("facts", "queries"):
            variation[key] += extra[key]
    if relation_paraphrases:
        from neural_reader_evidence_curriculum import augment_relation_paraphrases

        variation = augment_relation_paraphrases(variation)
    hard_groups = {}
    hard_visits = {}
    if getattr(args, "hard_banks", False):
        sources = {"authored": rows["fit"]}
        if variation:
            sources["variation"] = variation
        hard_groups = hard_bank_groups(sources)
        if not hard_groups:
            raise ValueError("hard banks require fit facts with same-entity queries")
    all_rows = dict(rows, scope_fit=scope_group, lifecycle_fit=lifecycle_group,
                    served_lifecycle_fit=served_scope_group,
                    paraphrase_scope_fit=paraphrase_scope_group)
    if scope_development:
        all_rows["scope_development"] = scope_development
    if variation:
        all_rows["relation_variation_fit"] = variation
    if binding_group:
        all_rows["reader_binding_fit"] = binding_group
    if expansion:
        all_rows["expansion_fit"] = expansion
    if identity_group:
        all_rows["identity_fit"] = identity_group
    if hard_groups:
        all_rows.update({"hard_bank_" + name: group for name, group in hard_groups.items()})
    from neural_activation_curriculum import evidence_challenge
    from neural_feedback_evaluation import (
        qualification_observation_ids, exclude_qualification,
        evidence_view_groups, feedback_selection_score, component_nonregression_safe,
        kind_component_nonregression_safe,
    )
    excluded = (qualification_observation_ids(args.qualification_curriculum)
                if args.qualification_curriculum else set())
    temporal_prepared, temporal_manifest = None, None
    if temporal_focus:
        from neural_temporal_curriculum import prepare_temporal_fit
        temporal_prepared, temporal_manifest = prepare_temporal_fit(
            rows["fit"], lifecycle_group, excluded_observation_ids=excluded)
        (root / "temporal_curriculum_manifest.json").write_text(
            json.dumps(temporal_manifest, indent=2))
    replay_groups = {}
    replay_details = None
    if functional_weight:
        from neural_reader_replay import build_replay_groups, replay_manifest
        old_groups = dict(fit=rows["fit"], lifecycle_fit=lifecycle_group)
        if variation:
            old_groups["relation_variation_fit"] = variation
        replay_groups = build_replay_groups(old_groups, excluded_observation_ids=excluded)
        replay_details = replay_manifest(replay_groups, existing_texts=cache.keys())
        all_rows.update({"retention_" + name: group for name, group in replay_groups.items()})
        (root / "functional_replay.json").write_text(json.dumps(replay_groups, indent=2))
        (root / "functional_replay_manifest.json").write_text(json.dumps(replay_details, indent=2))
    selection = {"authored_development": exclude_qualification(rows["development"], excluded)}
    if args.replay_episodes:
        selection["episode_development"] = exclude_qualification(rows["episode_development"], excluded)
    feedback = {}
    for index, (name, group) in enumerate(selection.items()):
        challenge = evidence_challenge(group, 7421 + index, limit=24)
        all_rows["evidence_" + name] = challenge
        for view, values in evidence_view_groups(challenge).items():
            feedback[name + "_cue_" + view] = values
    required |= {
        r["text"] for g in all_rows.values() for values in g.values() for r in values
    }
    required |= {
        r[key]
        for g in all_rows.values()
        for values in g.values()
        for r in values
        for key in ("reference", "initial_reference", "initial_text")
        if r.get(key)
    }
    missing = sorted(required - cache.keys())
    print(
        json.dumps(
            dict(stage="features", required=len(required), missing=len(missing))
        ),
        flush=True,
    )
    if missing:
        extra, _ = extract(
            args, {"additional": [dict(body=t, context=t) for t in missing]}
        )
        cache.update(extra)
    provenance = dict(
        backbone=backbone,
        probe=sha(args.probe),
        texts_sha256=hashlib.sha256(json.dumps(sorted(required)).encode()).hexdigest(),
    )
    torch.save(
        dict(features={t: cache[t] for t in required}, identity=provenance),
        root / "features.pt",
    )
    (root / "curriculum.json").write_text(json.dumps(rows, indent=2))
    resume_metadata = {}
    if args.resume:
        cp = torch.load(args.resume, weights_only=False, map_location="cpu")
        if cp["backbone_sha256"] != backbone:
            raise ValueError("reader checkpoint backbone mismatch")
        model = load_model(cp)
        resume_metadata = {key: value for key, value in cp.items() if key != "state_dict"}
    else:
        model = NeuralSlots()
    if identity_match_focus and any(not torch.equal(model.state_dict()[key], planner.state_dict()[key])
                                    for key in PLAN_KEYS + QUERY_KEYS):
        raise ValueError("identity-match planner tensors must match the resumed reader; planner sync cannot change frozen representations")
    sync_query_planner(model, planner)
    planner_provenance = reader_planner_provenance(planner_cp, planner_identity)
    resume_metadata = reader_checkpoint_payload(resume_metadata, **planner_provenance)
    del planner, planner_cp, planner_bytes
    train_heads = configure_reader_heads(model, args.train_heads, freeze_memory_encoder=args.freeze_memory_encoder)
    cached = DynamicFeatures(model, all_rows, cache, args.device, args.feature_dropout,
                             freeze_memory_encoder=args.freeze_memory_encoder)
    model.to(args.device)
    reader_parent = snapshot_reader_matcher(model) if reader_anchor_weight else None
    functional_parent, functional_cached = (snapshot_functional_reader(model, cached)
                                            if functional_weight else (None, None))
    functional_initial = (tuple(parameter.detach().cpu().clone()
                                for parameter in functional_parent.parameters())
                          if functional_parent is not None else ())
    topics = sorted(
        {
            f["prop"]
            for f in rows["fit"]["facts"] + (expansion["facts"] if expansion else []) + scope_group["facts"] + lifecycle_group["facts"] + (variation["facts"] if variation else [])
        }
    ) + [""]
    topic_head = torch.nn.Linear(WIDTH, len(topics)).to(args.device)
    optimizer = reader_optimizer(model, topic_head, args.train_heads, args.learning_rate, args.attention_learning_rate)
    rng = random.Random(5301)
    facts = rows["fit"]["facts"]
    positives = [q for q in rows["fit"]["queries"] if q["prop"]]
    negatives = [q for q in rows["fit"]["queries"] if not q["prop"]]
    expansion_world_rng = random.Random(5303)
    expansion_sampling_visits = Counter()
    material_max_sources = getattr(args, "expansion_material_max_sources", 2)
    initial = {
        name: evaluate(model, group, cached) for name, group in selection.items()
    }
    initial_feedback = {name: evaluate(model, group, cached) for name, group in feedback.items()}
    initial_scope = scope_evaluate(model, scope_development, cached) if scope_development else None
    diagnostic_banks = dict(selection)
    if scope_development:
        diagnostic_banks["scope_development"] = scope_development
    initial_kind = {name: kind_bank_diagnostics(model, group, cached) for name, group in diagnostic_banks.items()}
    initial_attention = {name: attention_bank_diagnostics(model, group, cached, scope=name == "scope_development") for name, group in diagnostic_banks.items()}
    initial_binding_fit = evaluate(model, binding_group, cached) if binding_group else None
    initial_replay = ({name: evaluate(model, group, cached) for name, group in replay_groups.items()}
                      if functional_weight else None)
    initial_expansion_worlds = expansion_world_diagnostics(model, expansion, cached) if expansion else None
    initial_temporal_fit = (evaluate(model, lifecycle_group, cached) if temporal_focus else None)
    initial_temporal_kind = (kind_bank_diagnostics(model, dict(facts=temporal_prepared["fit_facts"]
        + temporal_prepared["lifecycle_facts"]), cached) if temporal_focus else None)
    initial_selection = reader_initial_selection(
        initial, initial_feedback, scope=initial_scope if scope_primary else None)
    best_score = initial_selection["initial_selection_score"]
    selected = 0
    best_feedback = initial_feedback
    best_scope = last_scope = initial_scope
    best_kind = last_kind = initial_kind
    best_attention = last_attention = initial_attention
    last_metrics, last_feedback = initial, initial_feedback
    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    kind_prefit = None

    def save(metrics):
        torch.save(
            reader_checkpoint_payload(
                resume_metadata,
                format=FORMAT,
                backbone_sha256=backbone,
                state_dict=best,
                reader_step=selected,
                diagnostic_only=False,
                eligible_for_selection=True,
                reader_heads=train_heads,
                reader_training_mode=args.train_heads,
                identity_match_refinement=identity_match_focus,
                matcher_corrections=matcher_corrections,
                query_only_refinement=query_focus,
                query_refinement_contract=query_contract,
                context_projection_refinement=context_focus,
                context_projection_contract=context_contract,
                selection_metrics=metrics,
                feedback_metrics=best_feedback,
                scope_development=best_scope,
                initial_scope_development=initial_scope,
                expansion_material_max_sources=material_max_sources if expansion else None,
                initial_expansion_worlds=initial_expansion_worlds,
                scope_guard_enabled=scope_guard,
                component_guard_enabled=component_guard,
                **initial_selection,
                selection_objective=("scope strict support sets, fewer false slot matches, scope score"
                                     if scope_primary else "worst evidence-view strict accuracy, mean strict accuracy, evidence F1"
                                     + ("; independent scope nonregression guard" if scope_guard else "")),
                frozen_query_planner_sha256=planner_identity,
                fit_paraphrases=args.fit_paraphrases,
                episode_relations=getattr(args, "episode_relations", False),
                episode_aspects=getattr(args, "episode_aspects", False),
                hard_banks=bool(hard_groups),
                hard_bank_groups=({name: dict(facts=len(group["facts"]),
                                              queries=len(group["queries"]))
                                   for name, group in hard_groups.items()}
                                  if hard_groups else None),
                hard_bank_visits=dict(hard_visits) if hard_groups else None,
                relation_paraphrases=relation_paraphrases,
                relation_bindings=relation_bindings,
                relation_binding_fit_facts=len(binding_group["facts"]) if binding_group else 0,
                relation_binding_fit_queries=len(binding_group["queries"]) if binding_group else 0,
                relation_binding_sampling="every-fourth-step-same-object-different-property" if binding_group else None,
                semantic_topic_weight=args.semantic_topic_weight,
                loss_reduction=("query-worst-positive/negative-correction+" + PRESERVATION_REDUCTION
                                if matcher_corrections else LOSS_REDUCTION),
                freeze_memory_encoder=args.freeze_memory_encoder,
                kind_development_banks=best_kind,
                attention_development_banks=best_attention,
                attention_learning_rate=args.attention_learning_rate,
                reader_anchor_weight=reader_anchor_weight,
                reader_anchor_margin=reader_anchor_margin,
                reader_anchor_teacher="frozen-fit-complete-support-sets" if reader_parent is not None else None,
                reader_functional_weight=functional_weight,
                reader_functional_margin=functional_margin,
                reader_functional_teacher="independent-frozen-complete-memory-reader" if functional_parent is not None else None,
                reader_functional_loss=functional_description if functional_weight else None,
                reader_functional_intent_weight=functional_intent_weight,
                temporal_balanced_curriculum=temporal_manifest,
                temporal_correction="frozen-parent-semantic/identity-feasibility plus same-relation disallowed states; full-bank evaluation" if temporal_focus else None,
                temporal_kind_correction="class-balanced capped original-source categorical margins from independent frozen parent" if temporal_focus else None,
                kind_component_guard_enabled=temporal_focus,
                kind_prefit=kind_prefit,
                reader_functional_replay=replay_details,
                eval_every=eval_every,
            ),
            model_root / "selected.pt",
        )

    save(initial)
    steps_to_run = args.steps
    if kind_context_prefit:
        # Preserve and score the real initial checkpoint before this fit stage.
        # Main training cannot trade source labels for easier activation gates.
        kind_prefit = prefit_temporal_kind(model, temporal_prepared, cached)
        train_heads = ("intent.", "version.")
        optimizer = reader_optimizer(model, topic_head, args.train_heads,
                                     args.learning_rate, args.attention_learning_rate)
        last_kind = {name: kind_bank_diagnostics(model, group, cached)
                     for name, group in diagnostic_banks.items()}
        kind_prefit["development_kind_guard_safe"] = kind_component_nonregression_safe(last_kind, initial_kind)
        kind_prefit["development_kind_banks"] = last_kind
        if not kind_prefit["development_kind_guard_safe"]:
            # These encodings and the kind head are frozen in the next phase,
            # so later intent/version updates cannot repair this guard failure.
            steps_to_run = 0
            last_metrics = {name: evaluate(model, group, cached) for name, group in selection.items()}
            last_feedback = {name: evaluate(model, group, cached) for name, group in feedback.items()}
            last_scope = scope_evaluate(model, scope_development, cached)
            last_attention = {name: attention_bank_diagnostics(model, group, cached, scope=name == "scope_development")
                              for name, group in diagnostic_banks.items()}
        print(json.dumps(dict(stage="kind_prefit", **kind_prefit)), flush=True)
    replay_rng = random.Random(5361)
    temporal_rng, kind_rng = random.Random(5362), random.Random(5363)
    temporal_visits = dict(main_queries=0, lifecycle_steps=0, old_fit_steps=0,
                           intent_positive=[0, 0, 0], kind_examples=[0, 0, 0])
    functional_visits = dict(steps=0, queries=0, protected_query_visits=0)
    functional_groups_visited = {}
    functional_sums = {name: torch.zeros((), device=args.device)
                       for name in ("supervised", "margin", "distillation", "protected", "intent_supervision")}
    began = time.monotonic()
    for step in range(1, steps_to_run + 1):
        if args.train_heads == "scope":
            scope_data = (lifecycle_group, scope_group, served_scope_group,
                          paraphrase_scope_group)[step % 4]
            loss = scope_training_loss(model, scope_data, cached, rng, args.activation_positive_weight)
        else:
            kind_facts = None
            if temporal_focus:
                from neural_temporal_curriculum import sample_temporal_main, sample_temporal_kind
                name, sampled, bank_facts = sample_temporal_main(temporal_prepared, step, temporal_rng)
                kind_facts = sample_temporal_kind(temporal_prepared, kind_rng)
                temporal_visits["main_queries"] += len(sampled)
                temporal_visits["lifecycle_steps" if name == "lifecycle_zero" else "old_fit_steps"] += 1
                for row in sampled:
                    allowed = row.get("allowed_kinds", [row["kind"]])
                    for state in range(3):
                        temporal_visits["intent_positive"][state] += int(state in allowed)
                for fact in kind_facts:
                    temporal_visits["kind_examples"][fact.get("source_kind", fact["kind"])] += 1
            else:
                bank_facts = active_bank_facts(facts)
                sampled = (sample_relation_queries(positives, rng, 24) if args.train_heads in GUARDED_READER_MODES
                           else rng.sample(positives, min(24, len(positives)))) + rng.sample(
                    negatives, min(8, len(negatives))
                )
                if expansion and step % 2 == 0:
                    # A later lifecycle/variation/binding branch may replace
                    # this batch. Count visits only after the final choice.
                    bank_facts, sampled, expansion_batch_visits = sample_expansion_training(
                        expansion, rng, expansion_world_rng, material_max_sources=material_max_sources,
                        guarded=args.train_heads in GUARDED_READER_MODES)
                if args.train_heads in ("all", "read", "temporal") and step % 3 == 0:
                    bank_facts = active_bank_facts(lifecycle_group["facts"])
                    sampled = rng.sample(lifecycle_group["queries"], min(32, len(lifecycle_group["queries"])))
                if variation and step % 3 == 0:
                    bank_facts = active_bank_facts(variation["facts"])
                    sampled = sample_relation_queries(variation["queries"], rng, 24)
                if binding_group and step % 4 == 0:
                    bank_facts = active_bank_facts(binding_group["facts"])
                    binding_positive = [q for q in binding_group["queries"] if q["support_texts"]]
                    binding_negative = [q for q in binding_group["queries"] if not q["support_texts"]]
                    sampled = (sample_relation_queries(binding_positive, rng, 24)
                               + rng.sample(binding_negative, min(8, len(binding_negative))))
                if hard_groups and step % 3 == 2:
                    # Every third step uses one fit-only same-entity bank, so the
                    # supported record is often not the lexical row maximum.
                    hard_name = sorted(hard_groups)[(step // 3) % len(hard_groups)]
                    hard_group = hard_groups[hard_name]
                    bank_facts = active_bank_facts(hard_group["facts"])
                    sampled = (sample_relation_queries(hard_group["queries"], rng, 24)
                               if args.train_heads in GUARDED_READER_MODES else
                               rng.sample(hard_group["queries"],
                                          min(24, len(hard_group["queries"]))))
                    hard_visits[hard_name] = hard_visits.get(hard_name, 0) + 1
                if sampled and all("expansion_world_class" in row for row in sampled):
                    expansion_sampling_visits.update(expansion_batch_visits)
                    expansion_sampling_visits["steps"] += 1
            if not sampled or not bank_facts:
                raise ValueError("fit curriculum requires facts and resolved queries")
            if identity_match_focus:
                loss = identity_match_training_loss(model, sampled, bank_facts, cached)
            elif args.train_heads == "temporal":
                loss = temporal_training_loss(model, sampled, bank_facts, cached,
                    args.activation_positive_weight, kind_facts=kind_facts,
                    parent=functional_parent if temporal_focus else None,
                    parent_cached=functional_cached if temporal_focus else None)
            elif args.train_heads == "activation":
                loss = activation_training_loss(model, sampled, bank_facts, cached,
                                                args.activation_positive_weight)
            elif matcher_corrections:
                loss = matcher_correction_training_loss(model, sampled, bank_facts, cached,
                    functional_parent, functional_cached, margin=functional_margin)
            elif args.train_heads in RELATION_MODES:
                loss = semantic_training_loss(model, sampled, bank_facts, cached, topic_head, topics, args.activation_positive_weight, args.semantic_topic_weight)
            else:
                bank, ids, mask, meta = make_bank(bank_facts, cached)
                q, identity = query(model, sampled, cached)
                gold = torch.tensor(
                    [[supported(r, f) for f in bank_facts] for r in sampled],
                    dtype=torch.float32,
                    device=bank.device,
                )
                entity_gold = torch.tensor(
                    [
                        [
                            f["entity"] in r.get("entities", [r.get("entity")])
                            for f in bank_facts
                        ]
                        for r in sampled
                    ],
                    dtype=torch.float32,
                    device=bank.device,
                )
                if args.train_heads == "identity":
                    loss = balanced(model.identity_logits(identity, ids), entity_gold)
                    loss += 0.25 * identity_calibration(model, identity, ids, entity_gold)
                    loss += role_supervision(model, sampled, cached)
                else:
                    if args.train_heads == "read":
                        identity = identity.detach()
                        ids = ids.detach()
                    logits = model.score_bank(q, identity, bank, ids, meta, mask, memory_context_latents=cached.memory_context_latents([fact["text"] for fact in bank_facts]))
                    loss = balanced(logits, gold, args.activation_positive_weight)
                    loss += semantic_relation_auxiliary(model, sampled, bank_facts, q, bank, mask, memory_context_latents=cached.memory_context_latents([fact["text"] for fact in bank_facts]))
                    if args.train_heads == "all":
                        loss += balanced(model.identity_logits(identity, ids), entity_gold)
                        loss += 0.25 * identity_calibration(model, identity, ids, entity_gold)
                        loss += role_supervision(model, sampled, cached)
            if identity_group and args.train_heads in ("all", "identity"):
                identity_rows = rng.sample(identity_group["queries"], 24)
                identity_queries = cached.reference_ids(identity_rows)
                identity_memories = cached.reference_ids(identity_group["facts"])
                identity_target = torch.tensor(
                    [
                        [f["entity"] in r["entities"] for f in identity_group["facts"]]
                        for r in identity_rows
                    ],
                    dtype=torch.float32,
                    device=bank.device,
                )
                if args.train_heads != "read":
                    loss += 0.5 * balanced(model.identity_logits(identity_queries, identity_memories), identity_target)
                    loss += 0.25 * identity_calibration(model, identity_queries, identity_memories, identity_target)
                    loss += 0.5 * role_supervision(model, identity_rows, cached)
            if args.train_heads in ("all", "read"):
                scope_data = (lifecycle_group, scope_group, served_scope_group,
                          paraphrase_scope_group)[step % 4]
                scope_facts = active_bank_facts(scope_data["facts"])
                scope_rows = rng.sample(scope_data["scope_queries"], min(24, len(scope_data["scope_queries"])))
                scope_bank, scope_ids, scope_mask, scope_meta = make_bank(scope_facts, cached)
                scope_q, scope_identity = query(model, scope_rows, cached, scope=True)
                if args.train_heads == "read":
                    scope_identity, scope_ids = scope_identity.detach(), scope_ids.detach()
                scope_gold = torch.tensor([[supported(r,f) for f in scope_facts] for r in scope_rows], dtype=torch.float32, device=bank.device)
                loss += 0.5 * balanced(model.score_bank(scope_q, scope_identity, scope_bank, scope_ids, scope_meta, scope_mask, scope=True, memory_context_latents=cached.memory_context_latents([fact["text"] for fact in scope_facts])), scope_gold)
                loss += scope_relation_auxiliary(model, scope_rows, scope_facts, scope_q, scope_bank, scope_mask, memory_context_latents=cached.memory_context_latents([fact["text"] for fact in scope_facts]))
                means = (bank * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
                loss += F.cross_entropy(cached.kind_logits([fact["text"] for fact in bank_facts], means), source_kind_targets(bank_facts, bank.device))
                loss += F.cross_entropy(topic_head(means), torch.tensor([topics.index(f["prop"]) for f in bank_facts], device=bank.device))
                loss += F.cross_entropy(topic_head(q[:, :SEMANTIC].reshape(-1, RELATION_HEADS, WIDTH).mean(1)), torch.tensor([topics.index(r["prop"]) for r in sampled], device=bank.device))
                loss += 0.25 * F.binary_cross_entropy_with_logits(q[:, SEMANTIC:SEMANTIC + 3], intent_targets(sampled, bank.device))
            if reader_parent is not None:
                loss += matcher_preservation_loss(model, sampled, bank_facts, cached,
                                                  reader_parent, weight=reader_anchor_weight,
                                                  margin=reader_anchor_margin)
            if functional_parent is not None:
                from neural_reader_replay import sample_replay_group, sample_replay_queries
                replay_name, replay_group = sample_replay_group(replay_groups, replay_rng)
                replay_rows = sample_replay_queries(replay_group, replay_rng)
                replay_facts = active_bank_facts(replay_group["facts"])
                replay_loss, replay_stats = functional_replay_loss(
                    model, functional_parent, replay_rows, replay_facts, cached,
                    functional_cached, margin=functional_margin,
                    positive_weight=args.activation_positive_weight,
                    intent_weight=functional_intent_weight,
                    temporal_only=temporal_focus,
                    semantic_corrections=matcher_corrections,
                )
                loss += functional_weight * replay_loss
                functional_visits["steps"] += 1
                functional_visits["queries"] += replay_stats["queries"]
                functional_groups_visited[replay_name] = functional_groups_visited.get(replay_name, 0) + 1
                for name in functional_sums:
                    functional_sums[name] += replay_stats[name]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step % eval_every == 0 or step == steps_to_run:
            metrics = {
                name: evaluate(model, group, cached)
                for name, group in selection.items()
            }
            feedback_metrics = {name: evaluate(model, group, cached) for name, group in feedback.items()}
            scope_metrics = scope_evaluate(model, scope_development, cached) if scope_development else None
            kind_metrics = {name: kind_bank_diagnostics(model, group, cached) for name, group in diagnostic_banks.items()}
            attention_metrics = {name: attention_bank_diagnostics(model, group, cached, scope=name == "scope_development") for name, group in diagnostic_banks.items()}
            last_kind, last_metrics, last_feedback = kind_metrics, metrics, feedback_metrics
            last_scope = scope_metrics
            last_attention = attention_metrics
            score = (scope_selection_score(scope_metrics) if scope_primary
                     else feedback_selection_score(metrics, feedback_metrics))
            scope_safe = not scope_guard or scope_refinement_safe(scope_metrics, initial_scope)
            component_safe = not component_guard or component_nonregression_safe(metrics, initial)
            kind_safe = not temporal_focus or kind_component_nonregression_safe(kind_metrics, initial_kind)
            print(
                json.dumps(
                    dict(
                        step=step,
                        loss=float(loss.detach()),
                        development_views=metrics,
                        evidence_development_views=feedback_metrics,
                        scope_development=scope_metrics,
                        scope_guard_safe=scope_safe if scope_guard else None,
                        component_guard_safe=component_safe,
                        kind_component_guard_safe=kind_safe if temporal_focus else None,
                        kind_development_banks=kind_metrics,
                        attention_development_banks=attention_metrics,
                        selection_score=score,
                        functional_replay_means={name: float(value) / max(functional_visits["steps"], 1)
                                                 for name, value in functional_sums.items()} if functional_weight else None,
                        seconds=time.monotonic() - began,
                    )
                ),
                flush=True,
            )
            if scope_safe and component_safe and kind_safe and score > best_score:
                best_score, selected = score, step
                best_feedback = feedback_metrics
                best_scope = scope_metrics
                best_kind = kind_metrics
                best_attention = attention_metrics
                best = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                save(metrics)
    # Fit diagnostics describe the last trained model as well as the selected
    # checkpoint. Neither measurement participates in checkpoint selection.
    last_binding_fit = evaluate(model, binding_group, cached) if binding_group else None
    last_replay = ({name: evaluate(model, group, cached) for name, group in replay_groups.items()}
                   if functional_weight else None)
    last_temporal_fit = evaluate(model, lifecycle_group, cached) if temporal_focus else None
    last_temporal_kind = (kind_bank_diagnostics(model, dict(facts=temporal_prepared["fit_facts"]
        + temporal_prepared["lifecycle_facts"]), cached) if temporal_focus else None)
    last_parameter_changes = ({name: dict(changed=not torch.equal(parameter.detach().cpu(),
        functional_parent.state_dict()[name].detach().cpu()),
        maximum_absolute_delta=float((parameter.detach() - functional_parent.state_dict()[name]).abs().max()),
        trainable=parameter.requires_grad) for name, parameter in model.named_parameters()}
        if functional_parent is not None else None)
    last_expansion_worlds = expansion_world_diagnostics(model, expansion, cached) if expansion else None
    teacher_unchanged = (all(torch.equal(parameter.detach().cpu(), best_parent)
                            for parameter, best_parent in zip(functional_parent.parameters(), functional_initial))
                        if functional_parent is not None else None)
    torch.save(reader_checkpoint_payload(
        resume_metadata, format=FORMAT, backbone_sha256=backbone,
        state_dict={name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()},
        reader_step=steps_to_run, reader_training_mode=args.train_heads,
        reader_heads=train_heads,
        freeze_memory_encoder=args.freeze_memory_encoder,
        semantic_topic_weight=args.semantic_topic_weight,
        matcher_corrections=matcher_corrections,
        reader_anchor_weight=reader_anchor_weight,
        reader_anchor_margin=reader_anchor_margin,
        reader_anchor_teacher="frozen-fit-complete-support-sets" if reader_parent is not None else None,
        reader_functional_weight=functional_weight,
        reader_functional_margin=functional_margin,
        reader_functional_teacher="independent-frozen-complete-memory-reader" if functional_parent is not None else None,
        reader_functional_loss=functional_description if functional_weight else None,
        reader_functional_intent_weight=functional_intent_weight,
        reader_functional_replay=replay_details,
        identity_match_refinement=identity_match_focus,
        query_only_refinement=query_focus, query_refinement_contract=query_contract,
        context_projection_refinement=context_focus, context_projection_contract=context_contract,
        expansion_material_max_sources=material_max_sources if expansion else None,
        initial_expansion_worlds=initial_expansion_worlds,
        expansion_sampling_visits=dict(expansion_sampling_visits) if expansion else None,
        diagnostic_only=True, eligible_for_selection=False,
        last_step_development_views=last_metrics,
        last_step_evidence_development_views=last_feedback,
        initial_scope_development=initial_scope,
        last_step_scope_development=last_scope,
        scope_guard_enabled=scope_guard,
        component_guard_enabled=component_guard,
        **initial_selection), model_root / "last-step.pt")
    model.load_state_dict(best)
    report = {name: evaluate(model, group, cached) for name, group in rows.items()}
    report.update(
        **planner_provenance,
        selected_step=selected,
        training_steps_executed=steps_to_run,
        reader_training_mode=args.train_heads,
        identity_match_refinement=identity_match_focus,
        matcher_corrections=matcher_corrections,
        query_only_refinement=query_focus,
        query_refinement_contract=query_contract,
        context_projection_refinement=context_focus,
        context_projection_contract=context_contract,
        expansion_material_max_sources=material_max_sources if expansion else None,
        expansion_sampling_visits=dict(expansion_sampling_visits) if expansion else None,
        initial_expansion_worlds=initial_expansion_worlds,
        last_step_expansion_worlds=last_expansion_worlds,
        selected_expansion_worlds=(initial_expansion_worlds if selected == 0 else
                                  expansion_world_diagnostics(model, expansion, cached)) if expansion else None,
        expansion_worlds_used_for_selection=False,
        expansion_world_scope="Raw authored fit material sources; learned feature-role identities, not native frozen-writer binding or feedback rollout" if expansion else None,
        selection_score=best_score,
        evidence_development_views=best_feedback,
        scope_development=best_scope,
        initial_scope_development=initial_scope,
        last_step_scope_development=last_scope,
        scope_guard_enabled=scope_guard,
        component_guard_enabled=component_guard,
        kind_component_guard_enabled=temporal_focus,
        kind_prefit=kind_prefit,
        main_trainable_parameters=[name for name, parameter in model.named_parameters() if parameter.requires_grad],
        **initial_selection,
        scope_fit=scope_evaluate(model, scope_group, cached) if scope_development else None,
        relation_variation_fit=evaluate(model, variation, cached) if variation else None,
        relation_binding_fit=evaluate(model, binding_group, cached) if binding_group else None,
        initial_relation_binding_fit=initial_binding_fit,
        last_step_relation_binding_fit=last_binding_fit,
        relation_binding_fit_used_for_selection=False,
        qualification_observations_excluded=len(excluded),
        selection_objective=("independent scope development; recall/evidence metrics are diagnostics"
                             if scope_primary else "fixed evidence challenges; not native rollout qualification"
                             + ("; independent scope nonregression guard" if scope_guard else "")),
        fit_paraphrases=args.fit_paraphrases,
        episode_relations=getattr(args, "episode_relations", False),
        episode_aspects=getattr(args, "episode_aspects", False),
        hard_banks=bool(hard_groups),
        hard_bank_groups=({name: dict(facts=len(group["facts"]),
                                      queries=len(group["queries"]))
                           for name, group in hard_groups.items()}
                          if hard_groups else None),
        hard_bank_visits=dict(hard_visits) if hard_groups else None,
        relation_paraphrases=relation_paraphrases,
        relation_bindings=relation_bindings,
        relation_binding_fit_facts=len(binding_group["facts"]) if binding_group else 0,
        relation_binding_fit_queries=len(binding_group["queries"]) if binding_group else 0,
        relation_binding_sampling="every-fourth-step-same-object-different-property" if binding_group else None,
        semantic_topic_weight=args.semantic_topic_weight,
        loss_reduction=("query-worst-positive/negative-correction+" + PRESERVATION_REDUCTION
                        if matcher_corrections else LOSS_REDUCTION),
        freeze_memory_encoder=args.freeze_memory_encoder,
        kind_development_banks=best_kind,
        initial_kind_development_banks=initial_kind,
        last_step_kind_development_banks=last_kind,
        last_step_development_views=last_metrics,
        last_step_evidence_development_views=last_feedback,
        attention_learning_rate=args.attention_learning_rate,
        reader_anchor_weight=reader_anchor_weight,
        reader_anchor_margin=reader_anchor_margin,
        reader_anchor_teacher="frozen-fit-complete-support-sets" if reader_parent is not None else None,
        reader_functional_weight=functional_weight,
        reader_functional_margin=functional_margin,
        reader_functional_teacher="independent-frozen-complete-memory-reader" if functional_parent is not None else None,
        reader_functional_loss=functional_description if functional_weight else None,
        reader_functional_intent_weight=functional_intent_weight,
        temporal_balanced_curriculum=temporal_manifest,
        temporal_correction="frozen-parent-semantic/identity-feasibility plus same-relation disallowed states; full-bank evaluation" if temporal_focus else None,
        temporal_kind_correction="class-balanced capped original-source categorical margins from independent frozen parent" if temporal_focus else None,
        temporal_sampling_visits=temporal_visits if temporal_focus else None,
        initial_temporal_fit=initial_temporal_fit,
        last_step_temporal_fit=last_temporal_fit,
        initial_temporal_kind=initial_temporal_kind,
        last_step_temporal_kind=last_temporal_kind,
        last_step_parameter_changes_from_parent=last_parameter_changes,
        reader_functional_replay=replay_details,
        reader_functional_visits=dict(functional_visits,
                                      protected_query_visits=int(functional_sums["protected"])) if functional_weight else None,
        reader_functional_groups_visited=functional_groups_visited if functional_weight else None,
        reader_functional_teacher_all_parameters_unchanged=teacher_unchanged,
        reader_functional_teacher_parameters=len(functional_initial),
        reader_functional_shared_inputs="raw-C-features-only; independent-learned-query/bank/identity/time/feedback" if functional_weight else None,
        initial_functional_replay=initial_replay,
        last_step_functional_replay=last_replay,
        selected_functional_replay=({name: evaluate(model, group, cached) for name, group in replay_groups.items()}
                                    if functional_weight else None),
        reader_functional_replay_used_for_selection=False,
        reader_functional_rollout="same-authored-fit-cues; not automatic native evidence rollout" if functional_weight else None,
        eval_every=eval_every,
        attention_development_banks=best_attention,
        initial_attention_development_banks=initial_attention,
        last_step_attention_development_banks=last_attention,
        locomo_used=False,
        production_qualified=False,
        model_output=str(model_root),
        model_sha256=export(model, backbone, model_root / "model.bnsmodel"),
    )
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def reader_argument_parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("gguf", "probe", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--model-output", help="Fresh model directory; defaults to repository models/<output directory name>")
    p.add_argument(
        "--resume",
        help="Optional current checkpoint; omitted initializes fresh reader weights",
    )
    p.add_argument(
        "--planner-checkpoint", help="Frozen current writer; defaults to --resume"
    )
    p.add_argument("--curriculum", help="Independent fit/development observation JSON")
    p.add_argument("--features")
    p.add_argument("--extra-features", nargs="+", action="extend", default=[],
                   help="Additional identity-bound C feature caches; repeated options combine paths")
    p.add_argument("--qualification-curriculum", help="Reserve native QA observation IDs from checkpoint selection; labels are unused")
    p.add_argument("--steps", type=int, default=2200)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--kind-context-prefit", action="store_true",
                   help="Fit the contextual kind residual on complete fit banks using fixed CPU ridge, then freeze all kind parameters for temporal refinement")
    p.add_argument("--matcher-corrections", action="store_true",
                   help="Fit-only fixed-gate semantic corrections and partial-correct pair protection; resumed frozen matcher with full-parent retention only")
    p.add_argument("--reader-anchor-weight", type=float, default=0,
                   help="Fit-only complete support-set preservation; resumed frozen matcher only")
    p.add_argument("--reader-anchor-margin", type=float, default=1,
                   help="Cap the preserved parent semantic decision margin")
    p.add_argument("--reader-functional-weight", type=float, default=0,
                   help="Full frozen-parent behavior and old-gold replay; resumed semantic/matcher/activation/query/context/temporal only")
    p.add_argument("--reader-functional-margin", type=float, default=1,
                   help="Cap complete-parent final activation margins on correct old fit support sets")
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--attention-learning-rate", type=float, default=0.01)
    p.add_argument("--device", default="cpu")
    p.add_argument("--train-heads", choices=("all", "read", "identity", "identity-match", "scope", "semantic", "matcher", "activation", "temporal", "query") + CONTEXT_MODES, default="all",
                   help="identity-match calibrates only 33 identity matcher parameters from resume with frozen undropped representations and zero topic/anchor/functional weights; context-query learns only V (4096 parameters); context learns U/V/B (70656); both context modes require full-parent retention")
    p.add_argument("--feature-dropout", type=float, default=0)
    p.add_argument("--replay-episodes", action="store_true")
    p.add_argument(
        "--fit-paraphrases",
        action="store_true",
        help="Add at most 400 independent named fit questions before frozen writer binding",
    )
    p.add_argument("--expansion", action="store_true")
    p.add_argument("--expansion-material-max-sources", type=int, choices=(1, 2), default=2,
                   help="Expansion material worlds: 2 samples zero/one/two compatible sources; 1 uses a fixed single-source control")
    p.add_argument("--episode-relations", action="store_true",
                   help="Add independently authored fit content variants and broad complete-support questions")
    p.add_argument("--episode-aspects", action="store_true",
                   help="Add fit-only questions for the value, reason, place, time or participant named in each fit episode fact")
    p.add_argument("--hard-banks", action="store_true",
                   help="Every third step trains one fit-only same-entity bank with same-property distractors")
    p.add_argument("--relation-paraphrases", action="store_true",
                   help="Add fit-only supported cause/frequency/condition questions; requires --episode-relations")
    p.add_argument("--relation-bindings", action="store_true",
                   help="Add fit-only same-object different-property facts; semantic/matcher/activation with zero topic weight")
    p.add_argument("--evidence-cues", action="store_true")
    p.add_argument("--identity-diversity", action="store_true")
    p.add_argument("--activation-positive-weight", type=float, default=1.0)
    p.add_argument("--semantic-topic-weight", type=float, default=0.05,
                   help="Topic auxiliary loss weight; identity-match, matcher and temporal require 0")
    p.add_argument("--freeze-memory-encoder", action="store_true",
                   help="Guarded refinement: freeze memory representations and keep memory inputs undropped")
    return p


if __name__ == "__main__":
    train(reader_argument_parser().parse_args())
