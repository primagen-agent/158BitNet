#!/usr/bin/env python3
"""Train exact source plans on independently authored dialogue, never benchmarks."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
import torch
from torch.nn import functional as F
from neural_slot_model import (
    NeuralSlots,
    INPUT,
    BOUNDARY_CONTEXT,
    MEMBERSHIP_INPUT,
    SOURCE_CONTEXT_WIDTH,
    SOURCE_CONTEXT_INPUT,
    PLAN_KEYS,
    QUERY_KEYS,
    SEQUENCE_KEYS,
    require_parameters,
    word_features,
    decode_spans,
    decode_plan,
    export,
    FORMAT,
)
from neural_training_common import sha, load_features, extract
from neural_writer_anchor import strongest_wrong_path
from neural_query_binding import (
    QUERY_BINDING_MODES, QUERY_BINDING_CLASSES, query_binding_class,
    validate_query_binding_args, sample_query_binding_batch, query_binding_is_novel,
    require_query_binding_fit, validate_query_binding_fit_rows,
    query_binding_metrics, query_binding_score, query_binding_safe,
    snapshot_query_binding_parent, query_binding_parent_logits,
    make_query_binding_teacher, query_binding_anchor_parts,
)


_POSITION_CACHE = {}
_RESIDUAL_WRITER_MODES = ("writer-edge", "writer-raw", "writer-context", "writer-membership", "writer-content", "writer-context-residual", "writer-context-content", "writer-context-binding")
_MEMBERSHIP_WRITER_MODES = ("writer-membership", "writer-content", "writer-context-residual", "writer-context-content", "writer-context-binding")
_FACTORIZED_WRITER_MODES = ("writer-context-content", "writer-context-binding")


def _positions(count, device):
    key = (str(device), count)
    if key not in _POSITION_CACHE:
        _POSITION_CACHE[key] = torch.arange(count, device=device)
    return _POSITION_CACHE[key]


def plan_targets(row, units, device):
    labels = torch.zeros(len(units), dtype=torch.long, device=device)
    for span in row["spans"]:
        indices = [
            i for i, (a, b) in enumerate(units) if b > span["start"] and a < span["end"]
        ]
        labels[indices] = 4
        labels[indices[0]] = span["action"]
    return labels


def plan_path_scores(model, logits, paths, ends):
    """Vectorized source paths through the current native dynamic lattice."""
    start, _ = model.plan_potentials()
    terminal = model.plan_terminal_potentials(ends)
    positions = _positions(len(logits), logits.device)
    scores = logits[positions[None], paths].sum(-1) + start[paths[:, 0]] + terminal[paths[:, -1]]
    if len(logits) > 1:
        edges = model.plan_edge_potentials(ends)
        scores = scores + edges[positions[:-1][None], paths[:, :-1], paths[:, 1:]].sum(-1)
    return scores


def plan_path_score(model, logits, tags, ends):
    """Score one path with the exact potentials used by native Viterbi."""
    return plan_path_scores(model, logits, tags[None], ends)[0]


def _source_spans(tags):
    spans = []
    for first, tag in enumerate(tags):
        if tag not in (1, 2, 3):
            continue
        end = first + 1
        while end < len(tags) and tags[end] == 4:
            end += 1
        spans.append((first, end))
    return tuple(spans)


def _competitor_groups(tags, spans):
    endpoint, beginning = [], []
    for first, end in spans:
        last = end
        while last < len(tags) and tags[last] == 0:
            last += 1
        alternatives = []
        for new_end in range(first + 1, last + 1):
            if new_end == end:
                continue
            path = list(tags)
            path[first + 1:last] = [4] * (new_end - first - 1) + [0] * (last - new_end)
            alternatives.append(path)
        if alternatives:
            endpoint.append(alternatives)
        earliest = first
        while earliest > 0 and tags[earliest - 1] == 0:
            earliest -= 1
        alternatives = []
        for new_first in range(earliest, end):
            if new_first == first:
                continue
            path = list(tags)
            path[earliest:end] = ([0] * (new_first - earliest) + [tags[first]]
                                  + [4] * (end - new_first - 1))
            alternatives.append(path)
        if alternatives:
            beginning.append(alternatives)
    return endpoint, beginning


def _add_writer_contrasts(teacher):
    """Compile complete endpoint targets only for rows receiving that loss."""
    tags = teacher["tags"]
    endpoint, beginning = _competitor_groups(tags, teacher["spans"])
    paths, ranges = [tags], []
    for group in endpoint + beginning:
        first = len(paths)
        paths.extend(group)
        ranges.append((first, len(paths)))
    teacher.update(
        contrast_paths=teacher["target"].new_tensor(paths),
        contrast_ranges=tuple(ranges), endpoint_groups=len(endpoint),
    )
    return teacher


def build_writer_teacher(target, device=None, *, include_contrast=True):
    """Compile immutable fit labels once, with no text-dependent decisions."""
    tags = target.detach().cpu().tolist()
    device = target.device if device is None else torch.device(device)
    target = target.to(device)
    spans = _source_spans(tags)
    begin = [tag in (1, 2, 3) for tag in tags]
    teacher = dict(
        target=target, tags=tuple(tags), spans=spans, empty=not any(tags),
        begin_target=target.new_tensor(begin, dtype=torch.float32),
        begin_positive=target.new_tensor([i for i, value in enumerate(begin) if value]),
        begin_negative=target.new_tensor([i for i, value in enumerate(begin) if not value]),
        membership_positive=target.new_tensor([i for i, tag in enumerate(tags) if tag > 0]),
        membership_positive_tags=target.new_tensor([tag for tag in tags if tag > 0]),
        outside=target.new_tensor([i for i, value in enumerate(tags) if value == 0]),
    )
    return _add_writer_contrasts(teacher) if include_contrast else teacher


def endpoint_competitors(target):
    """All legal ends, preserving action and other source spans."""
    teacher = build_writer_teacher(target)
    return [teacher["contrast_paths"][a:b]
            for a, b in teacher["contrast_ranges"][:teacher["endpoint_groups"]]]


def beginning_competitors(target):
    """All legal shifted begins with the original action and source end."""
    teacher = build_writer_teacher(target)
    return [teacher["contrast_paths"][a:b]
            for a, b in teacher["contrast_ranges"][teacher["endpoint_groups"]:]]


def endpoint_contrast_loss(model, logits, target, ends, margin=1.0, *, teacher=None):
    """Equal-span margin against incomplete and overextended valid plans.

    Unlike token-averaged losses, a complete source endpoint keeps its weight
    when a fact or surrounding interaction grows longer. Both stop directions
    are supervised, rather than rewarding indiscriminate continuation.
    """
    teacher = build_writer_teacher(target) if teacher is None else teacher
    if not teacher["contrast_ranges"]:
        return logits.sum() * 0
    all_scores = plan_path_scores(model, logits, teacher["contrast_paths"], ends)
    gold = all_scores[0]
    losses = []
    for first, end in teacher["contrast_ranges"]:
        # Teacher paths obey the source grammar, so GPU nonzero/finite filtering
        # and its dynamic output shape are unnecessary in the hot loop.
        losses.append(F.softplus(margin + torch.logsumexp(all_scores[first:end], 0) - gold))
    return torch.stack(losses).mean()


def boundary_supervision_loss(boundaries, target, *, teacher=None):
    """Direct learned start/stop teachers, with equally weighted full spans.

    Each source end is a stop; preceding source units are continuations.
    Outside units are a separate stop group, including wholly empty messages.
    Balancing these groups prevents a long source from erasing its endpoint.
    """
    teacher = build_writer_teacher(target) if teacher is None else teacher
    begin = teacher["begin_target"].to(boundaries.dtype)
    begin_losses = F.binary_cross_entropy_with_logits(boundaries[:, 0], begin, reduction="none")
    begin_groups = [begin_losses[index].mean() for index in
                    (teacher["begin_positive"], teacher["begin_negative"]) if len(index)]
    end_losses = []
    for first, end in teacher["spans"]:
        stopping = F.softplus(-boundaries[end - 1, 1])
        if end - first > 1:
            continuing = F.softplus(boundaries[first:end - 1, 1]).mean()
            stopping = (stopping + continuing) / 2
        end_losses.append(stopping)
    if len(teacher["outside"]):
        end_losses.append(F.softplus(-boundaries[teacher["outside"], 1]).mean())
    return torch.stack(begin_groups).mean() + torch.stack(end_losses).mean()


def membership_supervision_loss(logits, target, *, teacher=None):
    """Balance source/background classes using the complete emission scores.

    A positive must support its gold tag, so a wrong begin cannot mask a weak
    inside tag. A negative rejects every source tag. Both use actual emission
    margins rather than the membership residual alone.
    """
    teacher = build_writer_teacher(target, include_contrast=False) if teacher is None else teacher
    classes = []
    positive = teacher["membership_positive"]
    if len(positive):
        selected = logits[positive]
        gold = selected.gather(1, teacher["membership_positive_tags"][:, None]).squeeze(1)
        classes.append(F.softplus(-(gold - selected[:, 0])).mean())
    negative = teacher["outside"]
    if len(negative):
        selected = logits[negative]
        margin = torch.logsumexp(selected[:, 1:5], dim=-1) - selected[:, 0]
        classes.append(F.softplus(margin).mean())
    return torch.stack(classes).mean() if classes else logits.sum() * 0


def _anchor_scores(model, logits, ends, teacher):
    target = teacher["target"]
    return (plan_path_score(model, logits, target, ends),
            strongest_wrong_path(model, logits, ends, target))


@torch.no_grad()
def path_anchor_required_gaps(model, parent_logits, parent_ends, teacher, margin):
    baseline_gold, baseline_competing = _anchor_scores(
        model, parent_logits, parent_ends, teacher
    )
    # Preserve the identical parent's numerical margin. Raising a tiny
    # negative rounding error to zero would introduce an initial gradient.
    return (baseline_gold - baseline_competing).clamp(max=margin)


def structured_path_anchor_loss(model, parent_logits, logits, target, ends, margin=1.0,
                                *, teacher=None, required_gap=None, parent_ends=None):
    """Preserve the parent's capped margin against every incorrect source plan.

    The frozen complete parent lattice supplies teacher gaps; the neural
    residual is the only trainable change. Full-lattice maximization protects
    added facts, omitted facts and boundary shifts without rival-set dilution.
    """
    teacher = build_writer_teacher(target, include_contrast=False) if teacher is None else teacher
    gold, competing = _anchor_scores(model, logits, ends, teacher)
    if required_gap is None:
        if parent_ends is None:
            raise ValueError("path anchors require the full parent's end potentials")
        required_gap = path_anchor_required_gaps(model, parent_logits, parent_ends, teacher, margin)
    violation = F.relu(required_gap - (gold - competing))
    return violation.square().max()


def writer_train_keys(mode):
    """Explicit isolation: source encoding is shared by query binding."""
    if mode == "writer-ref":
        return tuple(k for k in PLAN_KEYS
                     if k.startswith(("plan_ref_", "plan_origin.")))
    if mode == "writer-need":
        return ("plan_need.weight", "plan_need.bias")
    if mode == "writer-need-full":
        return ("plan_need.weight", "plan_need.bias")
    if mode == "writer-context-binding":
        return writer_train_keys("writer-context-content") + ("plan_need.weight", "plan_need.bias")
    if mode == "writer-context-residual":
        return tuple(k for k in PLAN_KEYS if k.startswith("plan_source_context."))
    if mode == "writer-context-content":
        return writer_train_keys("writer-content") + writer_train_keys("writer-context-residual")
    if mode in ("writer-raw", "writer-context"):
        return ("plan_boundary_raw.weight",)
    if mode == "writer-membership":
        return ("plan_membership_raw.weight",)
    if mode == "writer-content":
        return writer_train_keys("writer-edge") + writer_train_keys("writer-membership")
    if mode == "writer-edge":
        return tuple(k for k in PLAN_KEYS if k.startswith(("plan_boundary.", "plan_boundary_raw.")))
    if mode == "write-boundaries":
        return tuple(k for k in PLAN_KEYS if k.startswith("plan_tags.") or k in SEQUENCE_KEYS)
    if mode == "writer-full-boundaries":
        return (writer_train_keys("write-boundaries")
                + writer_train_keys("writer-content"))
    if mode == "query":
        return QUERY_KEYS
    if mode == "query-gate":
        return ("query_need.weight", "query_need.bias")
    if mode == "query-context":
        return ("query_context.weight",)
    if mode == "query-source":
        return tuple(key for key in QUERY_KEYS
                     if not key.startswith(("query_need.", "query_context.")))
    if mode == "references":
        return tuple(k for k in PLAN_KEYS + QUERY_KEYS
                     if k.startswith(("plan_need.", "plan_ref_", "plan_origin.", "query_")))
    if mode == "boundaries":
        return tuple(k for k in PLAN_KEYS
                     if k.startswith(("plan_tags.", "plan_local.")) or k in SEQUENCE_KEYS)
    if mode == "all":
        return PLAN_KEYS + QUERY_KEYS
    raise ValueError("unknown writer training mode")


def freeze_boundary_center_gradient(model):
    """New neighbor learning cannot change the existing current-word branch.

    Context mode uses zero optimizer decay, so these zero gradients also keep
    Adam's centre state zero. No input-dependent mask or serving rule is added.
    """
    if getattr(model, "writer_mode", None) == "writer-context":
        gradient = model.plan_boundary_raw.weight.grad
        if gradient is not None:
            gradient[:, INPUT : 2 * INPUT] = 0


def reference_targets(row, candidate, source_units, body_units, prefix, device):
    labels = torch.zeros(len(source_units) + prefix, dtype=torch.long, device=device)
    if (
        "ref_ranges" not in candidate
        and candidate["references"] == [row.get("speaker")]
        and row["context"].startswith("Speaker: " + row["speaker"] + "\n")
    ):
        # Synthetic subject supervision chooses one causal original location;
        # it need not mark every mention of the same name in the transcript.
        candidate = dict(
            candidate,
            ref_ranges=[
                dict(document="context", start=9, end=9 + len(row["speaker"].encode()))
            ],
        )
    if "ref_ranges" in candidate:
        for source in candidate["ref_ranges"]:
            units, offset = (
                (source_units, 0)
                if source["document"] == "context"
                else (body_units[:prefix], len(source_units))
            )
            selected = [
                i + offset
                for i, (a, b) in enumerate(units)
                if b > source["start"] and a < source["end"]
            ]
            if not selected:
                raise ValueError("reference supervision has no causal source")
            labels[selected] = 2
            labels[selected[0]] = 1
        return labels
    # References are original source bytes, chosen by the model at deployment.
    sources = [
        (row["context"], source_units, 0),
        (row["body"], body_units[:prefix], len(source_units)),
    ]
    for ref in candidate["references"]:
        found = False
        for text, units, offset in sources:
            raw = text.encode()
            start = 0
            while True:
                a = raw.find(ref.encode(), start)
                if a < 0:
                    break
                b = a + len(ref.encode())
                start = b
                indices = [
                    i + offset for i, (u, v) in enumerate(units) if v > a and u < b
                ]
                if indices:
                    # For synthetic labels, a referenced name must be a whole
                    # source unit, not a substring of another person's name.
                    first = units[indices[0] - offset][0]
                    last = units[indices[-1] - offset][1]
                    if raw[first:last].decode().strip(".,!?;:'\"") != ref:
                        continue
                    labels[indices] = 2
                    labels[indices[0]] = 1
                    found = True
        if not found:
            raise ValueError("reference gold lacks a causal source")
    return labels


def source_words(item, prefix):
    return torch.cat((item["context_words"], item["body_words"][:prefix]), 0)


def source_types(row, item, prefix):
    # Literal schema locations are model inputs, not inferred reference targets.
    speaker_len = len((row["speaker"] or "User").encode())
    time_start = 23 + speaker_len
    time_end = time_start + len((row["timestamp"] or "unspecified").encode())
    kinds = []
    for a, b in item["context_units"]:
        kinds.append(
            0
            if a >= 9 and b <= 9 + speaker_len
            else (1 if a >= time_start and b <= time_end else 2)
        )
    kinds += [3] * prefix
    return F.one_hot(torch.tensor(kinds, device=item["body"].device), 4).float()


def decode_references(row, tags, item, prefix):
    if isinstance(tags, torch.Tensor):
        tags = tags.detach().cpu().tolist()
    n = len(item["context_units"])
    result = []
    for text, units, labels in [
        (row["context"], item["context_units"], tags[:n]),
        (row["body"], item["body_units"][:prefix], tags[n:]),
    ]:
        raw = text.encode()
        for a, b in decode_spans(labels, units):
            reference = raw[a:b].decode().strip(".,!?;:'\"")
            if reference and reference not in result:
                result.append(reference)
    return result


def prepare(rows, cache, device):
    result = {}
    words = {}

    def features(text):
        if text not in words:
            value, units = word_features(text, cache[text])
            words[text] = (value.to(device), units)
        return words[text]

    for group in rows.values():
        for r in group:
            key = (r["body"], r["context"])
            if key in result:
                continue
            bw, bu = features(r["body"])
            cw, cu = features(r["context"])
            result[(r["body"], r["context"])] = dict(
                body=cache[r["body"]]["x"].to(device),
                context=cache[r["context"]]["x"].to(device),
                body_words=bw.to(device),
                body_units=bu,
                context_words=cw.to(device),
                context_units=cu,
            )
    return result


def writer_observation_key(row):
    """Source-plan teacher identity; references are not used to select anchors."""
    return (row["body"], row["context"],
            tuple(sorted((s["start"], s["end"], s["action"]) for s in row["spans"])))


def snapshot_parent_boundary(model):
    """Freeze boundary, raw membership and source-context weights in a plain map."""
    names = writer_train_keys("writer-content") + writer_train_keys("writer-context-residual")
    state = model.state_dict()
    return {name: state[name].detach().clone() for name in names}


def snapshot_parent_write_need(model):
    """Keep the original write gate separate from source and query teachers."""
    state = model.state_dict()
    return {name: state[name].detach().clone()
            for name in ("plan_need.weight", "plan_need.bias")}


@torch.no_grad()
def precompute_fit_correctness(model, fit_rows, cache):
    """Cache the complete resumed parent's agreement with fit on CPU once.

    Actual learned begin/end cues contribute to the parent teacher. No
    development group is accepted or visited. Only booleans survive this pass.
    """
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("fit anchor precomputation requires the CPU source model")
    flags = {}
    for row in fit_rows:
        key = writer_observation_key(row)
        if key in flags:
            continue
        body, context = cache[row["body"]], cache[row["context"]]
        if body["x"].device.type != "cpu" or context["x"].device.type != "cpu":
            raise ValueError("fit anchor precomputation requires CPU features")
        bw, units = word_features(row["body"], body)
        _, logits, ends = model.plan(body["x"], context["x"], bw)
        prediction = model.plan_labels(logits, ends)
        gold = plan_targets(row, units, torch.device("cpu"))
        flags[key] = bool(torch.equal(prediction, gold))
    correct = sum(flags[writer_observation_key(row)] for row in fit_rows)
    return flags, dict(
        baselinefit_correct_count=correct,
        baselinefit_incorrect_count=len(fit_rows) - correct,
        baselinefit_rows=len(fit_rows),
        baselinefit_unique_count=len(flags),
        baselinefit_unique_correct_count=sum(flags.values()),
        teacher="frozen-full-parent-source-plan-with-learned-boundary-and-membership",
        split="fit",
    )


@torch.no_grad()
def evaluate(model, rows, prepared):
    exact = source_exact = action_exact = reference_exact = whole_exact = tp = fp = (
        fn
    ) = ambiguous_unsafe = 0
    errors = []
    query_predictions = []
    for row in rows:
        item = prepared[row["body"], row["context"]]
        encoded, logits, ends = model.plan(item["body"], item["context"], item["body_words"])
        source_tags = model.plan_labels(logits, ends).detach().cpu().tolist()
        spans = decode_plan(source_tags, item["body_units"])
        pred = []
        for a, b, action, first, end in spans:
            need, tags = model.plan_reference(
                encoded[first:end].mean(0),
                source_words(item, first),
                source_types(row, item, first),
                encoded[first:end].amax(0),
            )
            required = bool(need > 0)
            refs = (
                decode_references(row, tags.argmax(-1), item, first) if required else []
            )
            pred.append(
                dict(start=a, end=b, action=action, need=required, references=refs)
            )
        gold = [
            {k: s[k] for k in ("start", "end", "action", "need", "references")}
            for s in row["spans"]
        ]
        source_exact += [(x["start"], x["end"]) for x in pred] == [
            (x["start"], x["end"]) for x in gold
        ]
        action_exact += [(x["start"], x["end"], x["action"]) for x in pred] == [
            (x["start"], x["end"], x["action"]) for x in gold
        ]
        reference_exact += [(x["need"], x["references"]) for x in pred] == [
            (x["need"], x["references"]) for x in gold
        ]
        exact += pred == gold
        committed = {
            (x["start"], x["end"], x["action"], tuple(x["references"]))
            for x in pred
            if not (x["need"] and not x["references"])
        }
        expected = {
            (x["start"], x["end"], x["action"], tuple(x["references"]))
            for x in gold
            if not (x["need"] and not x["references"])
        }
        tp += len(committed & expected)
        fp += len(committed - expected)
        fn += len(expected - committed)
        if row["tag"] == "ambiguous-reference" or row.get("query_class") == "ambiguous":
            ambiguous_unsafe += len(committed) > 0
        need, tags = model.query_reference(
            encoded.mean(0),
            source_words(item, len(encoded)),
            source_types(row, item, len(encoded)),
            raw_context=model.query_binding_context(item["body"], item["context"]),
        )
        refs = (
            decode_references(row, tags.argmax(-1), item, len(encoded))
            if need > 0
            else []
        )
        whole_ok = bool(need > 0) == row["whole_need"] and refs == row["whole_refs"]
        query_predictions.append(dict(need=bool(need > 0), references=refs))
        whole_exact += whole_ok
        if pred != gold or not whole_ok:
            errors.append(
                dict(
                    body=row["body"],
                    speaker=row["speaker"],
                    tag=row["tag"],
                    gold=gold,
                    predicted=pred,
                    whole_reference=refs,
                    whole_need_probability=float(need.sigmoid()),
                    whole_gold=row["whole_refs"],
                )
            )
    n = len(rows)
    return (
        dict(
            rows=n,
            plan_exact=exact / n,
            source_exact=source_exact / n,
            action_exact=action_exact / n,
            reference_exact=reference_exact / n,
            whole_reference_exact=whole_exact / n,
            committed_fact_f1=2 * tp / max(2 * tp + fp + fn, 1),
            tp=tp,
            fp=fp,
            fn=fn,
            ambiguous_unsafe_writes=ambiguous_unsafe,
            **query_binding_metrics(rows, query_predictions),
        ),
        errors,
    )


@torch.no_grad()
def writer_fit_training_diagnostic(model, fit_rows, prepared, *, step, stage):
    """Observe fit behavior without changing checkpoint eligibility or teachers."""
    metrics, _ = evaluate(model, fit_rows, prepared)
    return dict(
        split="fit", role="training-observability-only", used_for_checkpoint_selection=False,
        step=step, model_stage=stage, evaluation_device=str(next(model.parameters()).device),
        metrics=metrics,
    )


@torch.no_grad()
def evaluate_query_binding(model, rows, prepared):
    """Observe binding only, without evaluating the frozen source lattice."""
    predictions = []
    for row in rows:
        item = prepared[row["body"], row["context"]]
        encoded = model.plan_encoding(item["body"], item["context"], item["body_words"])
        kinds = source_types(row, item, len(encoded)).to(encoded.dtype)
        need, tags = model.query_reference(
            encoded.mean(0), source_words(item, len(encoded)), kinds,
            raw_context=model.query_binding_context(item["body"], item["context"]),
        )
        predictions.append(dict(need=bool(need > 0), references=(
            decode_references(row, tags.argmax(-1), item, len(encoded)) if need > 0 else [])))
    return dict(rows=len(rows), **query_binding_metrics(rows, predictions))


def writer_terminal_checkpoint_payload(model, resume_metadata, **updates):
    """Diagnostic weights remain separate from the selected checkpoint."""
    payload = dict(resume_metadata)
    payload.update(updates)
    payload.update(
        state_dict={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        diagnostic_only=True, eligible_for_selection=False)
    return payload


def observation_writer_teacher(model, row, item, *, include_paths=True, include_contrast=True):
    """Lazy fit-only teacher cache; development evaluation never calls this."""
    cache = getattr(model, "writer_fit_teachers", None)
    key = writer_observation_key(row)
    teacher = cache[key] if cache is not None and key in cache else {}
    if include_paths and "target" not in teacher:
        target = plan_targets(row, item["body_units"], torch.device("cpu"))
        teacher.update(build_writer_teacher(
            target, item["body"].device, include_contrast=include_contrast
        ))
    elif include_paths and include_contrast and "contrast_paths" not in teacher:
        _add_writer_contrasts(teacher)
    if cache is not None:
        cache[key] = teacher
    return teacher


@torch.no_grad()
def parent_boundary_teacher(model, encoded, words, teacher, *, body_encoded):
    if "parent_boundaries" not in teacher:
        boundaries = NeuralSlots.boundary_scores_from_parameters(
            encoded, words, model.boundary_parent_parameters
        ).detach()
        teacher["parent_boundaries"] = boundaries
        membership = NeuralSlots.total_membership_from_parameters(
            encoded, body_encoded, words, model.boundary_parent_parameters
        ).detach()
        teacher["parent_membership"] = membership
        teacher["parent_logits"] = (model.plan_tags(encoded)
                                    + F.pad(boundaries[:, :1].expand(-1, 3), (1, 1))
                                    + F.pad(membership[:, None].expand(-1, 4), (1, 0))).detach()
        teacher["parent_ends"] = boundaries[:, 1]
    return teacher


@torch.no_grad()
def parent_boundary_correction_teacher(model, teacher):
    """Frozen fit-only decoded error locations, never serving decisions."""
    if "correction_indices" in teacher:
        return teacher
    parent = model.plan_labels(teacher["parent_logits"], teacher["parent_ends"]).tolist()
    gold = teacher["tags"]
    gold_begin = [tag in (1, 2, 3) for tag in gold]
    parent_begin = [tag in (1, 2, 3) for tag in parent]
    gold_end = [tag > 0 and (i + 1 == len(gold) or gold[i + 1] != 4) for i, tag in enumerate(gold)]
    parent_end = [tag > 0 and (i + 1 == len(parent) or parent[i + 1] != 4) for i, tag in enumerate(parent)]
    begin_error = [a != b for a, b in zip(parent_begin, gold_begin)]
    end_error = [tag > 0 and (a != b or parent[i] == 0)
                 for i, (tag, a, b) in enumerate(zip(gold, parent_end, gold_end))]
    correction = [], []
    targets = [], []
    preserve = []
    for i in range(len(gold)):
        for channel, (error, desired) in enumerate(((begin_error[i], gold_begin[i]),
                                                   (end_error[i], gold_end[i]))):
            if error:
                correction[channel].append(i)
                targets[channel].append(float(desired))
            elif (gold[i] > 0) == (parent[i] > 0):
                # Matching Boolean cues do not imply a correct boundary when
                # one path has source and the other has none. Leave those
                # unlabelled outputs to the full path objective; in particular
                # gold outside still receives no direct end target.
                preserve.append(2 * i + channel)
    template = teacher["target"]
    teacher["correction_indices"] = tuple(template.new_tensor(indices) for indices in correction)
    teacher["correction_targets"] = tuple(template.new_tensor(values, dtype=torch.float32) for values in targets)
    # Compile the selected negative/positive classes on the fit-only cold
    # path. Counts stay Python integers, so the hot loss never inspects a
    # device label or performs a dynamic mask selection.
    classes = tuple(
        tuple([index for index, value in zip(indices, values) if value == desired]
              for desired in (0., 1.))
        for indices, values in zip(correction, targets)
    )
    teacher["correction_class_indices"] = tuple(
        tuple(template.new_tensor(indices) for indices in channel)
        for channel in classes
    )
    teacher["correction_class_counts"] = tuple(
        tuple(len(indices) for indices in channel) for channel in classes
    )
    teacher["preservation_indices"] = template.new_tensor(preserve)
    missing = [i for i, tag in enumerate(gold) if tag > 0 and parent[i] == 0]
    spurious = [i for i, tag in enumerate(gold) if tag == 0 and parent[i] > 0]
    matched = [i for i, tag in enumerate(gold) if (tag > 0) == (parent[i] > 0)]
    teacher["membership_correction"] = dict(
        membership_positive=template.new_tensor(missing),
        membership_positive_tags=template.new_tensor([gold[i] for i in missing]),
        outside=template.new_tensor(spurious),
    )
    teacher["membership_preservation_indices"] = template.new_tensor(matched)
    return teacher


def boundary_residual_supervision_loss(boundaries, teacher):
    """Correct only parent errors; keep all remaining boundary outputs fixed."""
    losses = []
    for channel, (classes, counts) in enumerate(zip(teacher["correction_class_indices"],
                                                   teacher["correction_class_counts"])):
        class_losses = []
        for desired, (indices, count) in enumerate(zip(classes, counts)):
            if count:
                selected = boundaries[indices, channel]
                class_losses.append(F.softplus(-selected if desired else selected).mean())
        if class_losses:
            # Equal weight per present class prevents a long missing tail's
            # continue words from diluting its single stop. Gold outside ends
            # remain excluded by the cached correction mask.
            losses.append(torch.stack(class_losses).mean())
    if len(teacher["preservation_indices"]):
        residual = (boundaries - teacher["parent_boundaries"]).flatten()[teacher["preservation_indices"]]
        losses.append(residual.square().mean())
    return torch.stack(losses).sum() if losses else boundaries.sum() * 0


def membership_residual_supervision_loss(logits, membership, teacher):
    """Correct parent membership errors; preserve the complete learned residual."""
    loss = membership_supervision_loss(
        logits, teacher["target"], teacher=teacher["membership_correction"]
    )
    preserved = teacher["membership_preservation_indices"]
    if len(preserved):
        residual = (membership - teacher["parent_membership"])[preserved]
        loss = loss + residual.square().mean()
    return loss


def write_need_residual_supervision_loss(logits, teacher):
    """Correct frozen-parent gate errors and preserve its correct signed margins."""
    loss = logits.sum() * 0
    correction = teacher["correction_indices"]
    if len(correction):
        loss = loss + F.binary_cross_entropy_with_logits(
            logits[correction], teacher["target"][correction]
        )
    preserved = teacher["preservation_indices"]
    if len(preserved):
        signed = logits[preserved] * teacher["sign"][preserved]
        loss = loss + F.relu(teacher["required_margin"] - signed).square().mean()
    return loss


def observation_write_need_loss(model, row, item, encoded):
    """Fit-only original full-encoding gold spans supervise the write gate alone.

    Source encoders are frozen in binding mode. Parent correctness and margins
    are captured once; current student predictions never choose loss masks.
    Reference tags and whole-query binding are not evaluated by this objective.
    """
    if getattr(model, "writer_mode", None) not in ("writer-context-binding", "writer-need"):
        return encoded.new_zeros(())
    if not row["spans"]:
        return model.plan_need.weight.sum() * 0
    key = (row["body"], row["context"], tuple(
        (span["start"], span["end"], span["action"], bool(span["need"]))
        for span in row["spans"]
    ))
    cache = model.write_need_fit_teachers
    teacher = cache.get(key)
    if teacher is None:
        ranges = []
        for span in row["spans"]:
            indices = [i for i, (a, b) in enumerate(item["body_units"])
                       if b > span["start"] and a < span["end"]]
            if not indices:
                raise ValueError("write need teacher requires a nonempty gold source span")
            ranges.append((indices[0], indices[-1] + 1))
        pooled = torch.stack([encoded.detach()[first:end].amax(0) for first, end in ranges])
        with torch.no_grad():
            parent = model.write_need_parent_parameters
            parent_logits = F.linear(pooled, parent["plan_need.weight"], parent["plan_need.bias"]).flatten()
            target = parent_logits.new_tensor([float(span["need"]) for span in row["spans"]])
            # The one-time host list prevents repeated device mask selection.
            correct = ((parent_logits > 0) == target.bool()).cpu().tolist()
            sign = target * 2 - 1
            preserved = [i for i, value in enumerate(correct) if value]
            correction = [i for i, value in enumerate(correct) if not value]
            preservation_indices = torch.tensor(preserved, dtype=torch.long, device=encoded.device)
            teacher = dict(
                ranges=tuple(ranges), target=target, sign=sign,
                correction_indices=torch.tensor(correction, dtype=torch.long, device=encoded.device),
                preservation_indices=preservation_indices,
                required_margin=(parent_logits[preservation_indices] * sign[preservation_indices]).clamp(
                    max=getattr(model, "write_need_anchor_margin", 1.0)),
            )
        cache[key] = teacher
    pooled = torch.stack([encoded.detach()[first:end].amax(0) for first, end in teacher["ranges"]])
    logits = model.plan_need(pooled).flatten()
    return getattr(model, "write_need_weight", 1.0) * write_need_residual_supervision_loss(logits, teacher)


def observation_factorized_source_loss(model, encoded, body_encoded, words, teacher):
    """Separate fit-only membership correction from conditional source structure."""
    from neural_writer_correction import build_writer_correction_teacher, writer_correction_losses

    parent_boundary_teacher(model, encoded.detach(), words, teacher,
                            body_encoded=body_encoded.detach())
    # Retain the existing immutable parent FN/FP row classes for batch reduction.
    parent_boundary_correction_teacher(model, teacher)
    base_logits = model.plan_tags(encoded).detach()
    if "source_correction" not in teacher:
        teacher["source_correction"] = build_writer_correction_teacher(
            model, base_logits, teacher["parent_boundaries"], teacher["parent_membership"],
            teacher["target"], split="fit", margin=getattr(model, "source_correction_margin", 1.0)
        )
    occupancy, structure = writer_correction_losses(
        model, base_logits, model.plan_boundary_scores(encoded, words),
        model.plan_total_membership(encoded, body_encoded, words), teacher["source_correction"]
    )
    model.factorized_source_row_parts = (occupancy.detach(), structure.detach())
    return getattr(model, "structured_weight", 1.0) * (occupancy + structure)


def source_correction_teacher_counts(model):
    """Count only visited fit teachers; no student-dependent eligibility."""
    teachers = [value["source_correction"] for value in model.writer_fit_teachers.values()
                if "source_correction" in value]
    return dict(
        visited_parent_wrong_source_rows=len(teachers),
        source_eligible_rows=sum(bool(teacher.source_eligible) for teacher in teachers),
        source_preservation_eligible_rows=sum(bool(teacher.source_preservation_eligible) for teacher in teachers),
        structure_eligible_rows=sum(bool(teacher.structure_eligible) for teacher in teachers),
        frozen_action_unreachable_rows=sum(bool(teacher.frozen_action_unreachable) for teacher in teachers),
    )


def query_binding_training_parts(model, row, item):
    """Existing query supervision without the frozen source CRF and write losses."""
    require_query_binding_fit(row)
    query_binding_class(row)
    encoded = model.plan_encoding(item["body"], item["context"], item["body_words"])
    words = source_words(item, len(encoded))
    kinds = source_types(row, item, len(encoded)).to(encoded.dtype)
    candidate = encoded.mean(0)
    raw_context = model.query_binding_context(item["body"], item["context"])
    need, tags = model.query_reference(candidate, words, kinds, raw_context=raw_context)
    device = need.device
    source = dict(references=row["whole_refs"])
    if "whole_ref_ranges" in row:
        source["ref_ranges"] = row["whole_ref_ranges"]
    labels = reference_targets(row, source, item["context_units"], item["body_units"],
                               len(encoded), device)
    loss = 0.5 * F.binary_cross_entropy_with_logits(
        need, need.new_tensor(float(row["whole_need"])))
    if row["whole_need"] and getattr(model, "writer_mode", None) != "query-gate":
        loss = loss + 0.5 * F.cross_entropy(
            tags, labels, weight=tags.new_tensor([1.0, 5.0, 5.0]))
        origin = (kinds * (labels > 0)[:, None]).sum(0).clamp_max(1)
        loss = loss + F.binary_cross_entropy_with_logits(
            model.query_origin(model.query_binding_candidate(candidate, raw_context)), origin)
    need_anchor = tag_anchor = need * 0
    if getattr(model, "query_binding_anchor_weight", 0):
        parent = getattr(model, "query_binding_parent_parameters", None)
        if parent is None:
            raise ValueError("query binding anchors require an independent resumed parent")
        teachers = model.query_binding_fit_teachers
        key = row["body"], row["context"]
        signature = (row["whole_need"], tuple(row["whole_refs"]),
                     tuple((r["document"], r["start"], r["end"])
                           for r in row.get("whole_ref_ranges", ())))
        if key not in teachers:
            old_need, old_tags = query_binding_parent_logits(
                model, candidate, words, kinds, parent, raw_context=raw_context)
            teachers[key] = dict(make_query_binding_teacher(
                old_need, old_tags, row["whole_need"], labels,
                margin=model.query_binding_anchor_margin), label_signature=signature)
        teacher = teachers[key]
        if teacher["label_signature"] != signature:
            raise ValueError("identical query observations have inconsistent binding labels")
        need_anchor, tag_anchor = query_binding_anchor_parts(need, tags, teacher)
    return loss, need_anchor, tag_anchor


def query_binding_batch_loss(model, rows, prepared):
    groups = {name: [] for name in QUERY_BINDING_CLASSES}
    need_anchors, tag_anchors = [], []
    for row in rows:
        loss, need_anchor, tag_anchor = query_binding_training_parts(
            model, row, prepared[row["body"], row["context"]])
        groups[query_binding_class(row)].append(loss)
        need_anchors.append(need_anchor)
        tag_anchors.append(tag_anchor)
    nonempty = [torch.stack(values).mean() for values in groups.values() if values]
    if not nonempty:
        raise ValueError("query binding batch must not be empty")
    supervised = torch.stack(nonempty).mean()
    anchor = torch.stack(need_anchors).amax() + torch.stack(tag_anchors).amax()
    model.query_binding_batch_stats = dict(
        classes={name: len(values) for name, values in groups.items()},
        supervised_loss=supervised.detach(), anchor_loss=anchor.detach())
    return supervised + getattr(model, "query_binding_anchor_weight", 0) * anchor


def observation_loss(model, row, item):
    if getattr(model, "writer_mode", None) in QUERY_BINDING_MODES:
        supervised, need_anchor, tag_anchor = query_binding_training_parts(model, row, item)
        return supervised + getattr(model, "query_binding_anchor_weight", 0) * (need_anchor + tag_anchor)
    device = item["body"].device
    loss = 0
    encoded, body_encoded, logits, ends = model.plan_with_encodings(
        item["body"], item["context"], item["body_words"]
    )
    need_loss = observation_write_need_loss(model, row, item, encoded)
    if getattr(model, "writer_mode", None) == "writer-need-full":
        # Directly supervise every gold span's write gate.  Unlike the frozen
        # parent residual mode this revisits parent-correct margins as well,
        # so a subject cue absent from the parent's few fit errors can learn.
        positive, negative = [], []
        for span in row["spans"]:
            indices = [i for i, (a, b) in enumerate(item["body_units"])
                       if b > span["start"] and a < span["end"]]
            if not indices:
                raise ValueError("write need supervision requires a nonempty gold source span")
            first, end = indices[0], indices[-1] + 1
            gate = model.plan_need(encoded[first:end].amax(0)).flatten()
            value = F.binary_cross_entropy_with_logits(
                gate, gate.new_tensor([float(span["need"])])
            )
            (positive if span["need"] else negative).append(value)
        # Equal class means: self-contained named facts are a minority of the
        # fit spans, and an unweighted mean would leave them unseen.
        if positive and negative:
            return 0.5 * (torch.stack(positive).mean() + torch.stack(negative).mean())
        classes = positive or negative
        return torch.stack(classes).mean() if classes else encoded.new_zeros(())
    if getattr(model, "writer_mode", None) == "writer-ref":
        # Reference decoding is trained alone: the write gate and source plan
        # stay frozen, and need=False spans contribute no reference target.
        reference_loss = encoded.new_zeros(())
        for span in row["spans"]:
            if not span["need"]:
                continue
            indices = [i for i, (a, b) in enumerate(item["body_units"])
                       if b > span["start"] and a < span["end"]]
            first, end = indices[0], indices[-1] + 1
            need, tags = model.plan_reference(
                encoded[first:end].mean(0),
                source_words(item, first),
                source_types(row, item, first),
                encoded[first:end].amax(0),
            )
            rgold = reference_targets(
                row, span, item["context_units"], item["body_units"], first, device
            )
            reference_loss = reference_loss + F.cross_entropy(
                tags, rgold, weight=torch.tensor([1.0, 5.0, 5.0], device=device)
            )
            kinds = source_types(row, item, first)
            origin_gold = (kinds * (rgold > 0)[:, None]).sum(0).clamp_max(1)
            reference_loss = reference_loss + 2 * F.binary_cross_entropy_with_logits(
                model.plan_origin(encoded[first:end].mean(0)), origin_gold
            )
        return reference_loss
    if getattr(model, "writer_mode", None) == "writer-need":
        # The isolated need gate consumes gold span means only. Every source,
        # reference and query head stays frozen and no unused loss is built.
        return need_loss
    anchor_weight = getattr(model, "boundary_anchor_weight", 0)
    path_weight = getattr(model, "boundary_path_anchor_weight", 0)
    if (anchor_weight or path_weight) and model.boundary_fit_correct[writer_observation_key(row)]:
        # Preserve the resumed parent's learned boundary cues. Correct rows do
        # not receive a competing task teacher or a zero-output target.
        teacher = observation_writer_teacher(
            model, row, item, include_paths=bool(path_weight), include_contrast=False
        )
        parent_boundary_teacher(model, encoded.detach(), item["body_words"], teacher,
                                body_encoded=body_encoded.detach())
        current_boundaries = model.plan_boundary_scores(encoded, item["body_words"])
        raw_mse = (current_boundaries - teacher["parent_boundaries"]).square().mean()
        if getattr(model, "writer_mode", None) in _MEMBERSHIP_WRITER_MODES:
            raw_mse = raw_mse + (model.plan_total_membership(encoded, body_encoded, item["body_words"])
                                 - teacher["parent_membership"]).square().mean()
        path_loss = logits.new_zeros(())
        if path_weight:
            target = teacher["target"]
            anchor_margin = getattr(model, "boundary_path_anchor_margin", 1.0)
            if teacher.get("required_gap_margin") != anchor_margin:
                teacher["required_gap"] = path_anchor_required_gaps(
                    model, teacher["parent_logits"], teacher["parent_ends"], teacher, anchor_margin
                )
                teacher["required_gap_margin"] = anchor_margin
            path_loss = structured_path_anchor_loss(
                model, None, logits, target, ends,
                anchor_margin,
                teacher=teacher, required_gap=teacher["required_gap"],
            )
        model.boundary_row_anchor_parts = (raw_mse.detach(), path_loss.detach())
        return anchor_weight * raw_mse + path_weight * path_loss + need_loss
    if getattr(model, "factorized_source_corrections", False):
        teacher = observation_writer_teacher(model, row, item, include_contrast=False)
        return observation_factorized_source_loss(
            model, encoded, body_encoded, item["body_words"], teacher
        ) + need_loss
    teacher = observation_writer_teacher(model, row, item)
    gold = teacher["target"]
    residual_mode = getattr(model, "writer_mode", None) in _RESIDUAL_WRITER_MODES
    if not residual_mode:
        loss = loss + F.cross_entropy(
            logits,
            gold,
            weight=torch.tensor([1.0, 2.0, 3.0, 3.0, 1.0], device=device),
        )
    loss = loss + getattr(model, "structured_weight", 1.0) * model.sequence_loss(
        logits, gold, ends
    )
    if getattr(model, "endpoint_weight", 0):
        loss = loss + model.endpoint_weight * endpoint_contrast_loss(
            model, logits, gold, ends, getattr(model, "endpoint_margin", 1.0), teacher=teacher
        )
    if getattr(model, "writer_mode", None) in _MEMBERSHIP_WRITER_MODES:
        parent_boundary_teacher(model, encoded.detach(), item["body_words"], teacher,
                                body_encoded=body_encoded.detach())
        parent_boundary_correction_teacher(model, teacher)
        loss = loss + getattr(model, "membership_weight", 1.0) * membership_residual_supervision_loss(
            logits, model.plan_total_membership(encoded, body_encoded, item["body_words"]), teacher
        )
    if getattr(model, "boundary_weight", 0):
        boundaries = model.plan_boundary_scores(encoded, item["body_words"])
        if residual_mode:
            parent_boundary_teacher(model, encoded.detach(), item["body_words"], teacher,
                                    body_encoded=body_encoded.detach())
            parent_boundary_correction_teacher(model, teacher)
            boundary_loss = boundary_residual_supervision_loss(boundaries, teacher)
        else:
            boundary_loss = boundary_supervision_loss(boundaries, gold, teacher=teacher)
        loss = loss + model.boundary_weight * boundary_loss
    if getattr(model, "writer_mode", None) in ("write-boundaries",) + _RESIDUAL_WRITER_MODES:
        # Binding mode trains only the write need gate above. Other reference
        # and query heads are frozen and their unused losses are not evaluated.
        return loss + need_loss
    for span in row["spans"]:
        indices = [
            i
            for i, (a, b) in enumerate(item["body_units"])
            if b > span["start"] and a < span["end"]
        ]
        first, end = indices[0], indices[-1] + 1
        need, tags = model.plan_reference(
            encoded[first:end].mean(0),
            source_words(item, first),
            source_types(row, item, first),
            encoded[first:end].amax(0),
        )
        loss = loss + F.binary_cross_entropy_with_logits(
            need, torch.tensor(float(span["need"]), device=device)
        )
        # Self-contained writes do not use BIO/origin outputs. Ambiguous
        # need=True writes still supervise all-O sources and zero origins.
        if not span["need"]:
            continue
        rgold = reference_targets(
            row, span, item["context_units"], item["body_units"], first, device
        )
        loss = loss + F.cross_entropy(
            tags, rgold, weight=torch.tensor([1.0, 5.0, 5.0], device=device)
        )
        kinds = source_types(row, item, first)
        origin_gold = (kinds * (rgold > 0)[:, None]).sum(0).clamp_max(1)
        loss = loss + 2 * F.binary_cross_entropy_with_logits(
            model.plan_origin(encoded[first:end].mean(0)), origin_gold
        )
    raw_context = model.query_binding_context(item["body"], item["context"])
    need, tags = model.query_reference(
        encoded.mean(0),
        source_words(item, len(encoded)),
        source_types(row, item, len(encoded)),
        raw_context=raw_context,
    )
    wg = dict(references=row["whole_refs"])
    if "whole_ref_ranges" in row:
        wg["ref_ranges"] = row["whole_ref_ranges"]
    rgold = reference_targets(
        row, wg, item["context_units"], item["body_units"], len(encoded), device
    )
    loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
        need, torch.tensor(float(row["whole_need"]), device=device)
    )
    # Self-contained queries never use the source decoder. Training their
    # unused all-O tags erases useful write/context binding. Genuine ambiguous
    # queries still teach need=True and all-O sources, preserving abstention.
    if row["whole_need"]:
        loss = loss + 0.5 * F.cross_entropy(
            tags, rgold, weight=torch.tensor([1.0, 5.0, 5.0], device=device)
        )
        kinds = source_types(row, item, len(encoded))
        origin_gold = (kinds * (rgold > 0)[:, None]).sum(0).clamp_max(1)
        loss = loss + F.binary_cross_entropy_with_logits(
            model.query_origin(model.query_binding_candidate(encoded.mean(0), raw_context)), origin_gold
        )
    return loss


def batch_loss(model, rows, prepared):
    if not rows:
        raise ValueError("writer batch must not be empty")
    if getattr(model, "writer_mode", None) in QUERY_BINDING_MODES:
        return query_binding_batch_loss(model, rows, prepared)
    anchored_mode = (getattr(model, "boundary_anchor_weight", 0)
                     or getattr(model, "boundary_path_anchor_weight", 0))
    if getattr(model, "factorized_source_corrections", False) and (
            not anchored_mode or model.writer_mode not in _FACTORIZED_WRITER_MODES):
        raise ValueError("factorized source corrections require an anchored source-context content or binding mode")
    error_balance = getattr(model, "source_error_balance", False)
    if error_balance and (not anchored_mode or model.writer_mode not in
                          ("writer-context-residual", "writer-context-content", "writer-context-binding")):
        raise ValueError("source error balancing requires source-context residual training with anchors")
    per_row, mse_parts, path_parts = [], [], []
    for row in rows:
        per_row.append(observation_loss(model, row, prepared[row["body"], row["context"]]))
        if anchored_mode and model.boundary_fit_correct[writer_observation_key(row)]:
            mse, path = model.boundary_row_anchor_parts
            mse_parts.append(mse)
            path_parts.append(path)
    losses = torch.stack(per_row)
    error_groups = ([], [], [])
    overlap = 0
    if anchored_mode:
        anchored = [model.boundary_fit_correct[writer_observation_key(row)] for row in rows]
        mask = torch.tensor(anchored, dtype=torch.bool, device=losses.device)
        if error_balance:
            for index, (row, correct) in enumerate(zip(rows, anchored)):
                if correct:
                    continue
                # These locations belong to the immutable resumed parent;
                # current student predictions never select fit-loss classes.
                teacher = model.writer_fit_teachers[writer_observation_key(row)]
                correction = teacher["membership_correction"]
                missing = bool(len(correction["membership_positive"]))
                spurious = bool(len(correction["outside"]))
                if missing:
                    error_groups[0].append(index)
                if spurious:
                    error_groups[1].append(index)
                if not missing and not spurious:
                    error_groups[2].append(index)
                overlap += int(missing and spurious)
        model.boundary_batch_stats = dict(
            anchored_rows=sum(anchored),
            learning_rows=len(rows) - sum(anchored),
            anchor_loss_sum=(
                getattr(model, "boundary_anchor_weight", 0) * (torch.stack(mse_parts).sum() if mse_parts else losses.new_zeros(()))
                + getattr(model, "boundary_path_anchor_weight", 0) * (torch.stack(path_parts).sum() if path_parts else losses.new_zeros(()))
            ),
            anchor_raw_mse_sum=torch.stack(mse_parts).sum() if mse_parts else losses.new_zeros(()),
            path_anchor_loss_sum=torch.stack(path_parts).sum() if path_parts else losses.new_zeros(()),
            source_missing_rows=len(error_groups[0]),
            source_spurious_rows=len(error_groups[1]),
            source_other_rows=len(error_groups[2]),
            source_overlap_rows=overlap,
        )
    if error_balance:
        zero = losses.sum() * 0
        correct_loss = losses[mask].mean() if any(anchored) else zero
        class_means = [losses[indices].mean() for indices in error_groups if indices]
        correction_loss = torch.stack(class_means).mean() if class_means else zero
        return correct_loss + correction_loss
    return losses.mean()


def writer_score(metrics):
    return (
        metrics["committed_fact_f1"]
        + metrics["plan_exact"]
        + metrics["whole_reference_exact"]
        - metrics["ambiguous_unsafe_writes"] / max(metrics["rows"], 1)
    )


def writer_reference_safe(metrics, initial):
    """Writer improvement cannot spend the initial reference/safety quality."""
    return (
        metrics["reference_exact"] >= initial["reference_exact"]
        and metrics["whole_reference_exact"] >= initial["whole_reference_exact"]
        and metrics["ambiguous_unsafe_writes"] <= initial["ambiguous_unsafe_writes"]
    )


def writer_edge_candidate_safe(metrics, initial):
    """Preserve complete source plans, committed facts and reference quality."""
    return (writer_reference_safe(metrics, initial)
            and metrics["committed_fact_f1"] >= initial["committed_fact_f1"]
            and all(metrics[key] >= initial[key]
                    for key in ("plan_exact", "source_exact", "action_exact")))


def writer_development_score(metrics, boundary=None, narrative=None, *,
                             query_binding=False, query_development=None):
    """Score attached public development groups after their quality guards."""
    if query_binding:
        return query_binding_score([metrics] + ([query_development] if query_development is not None else []))
    if narrative is not None:
        scores = [narrative["committed_fact_f1"]]
        if boundary is not None:
            scores.append(boundary["committed_fact_f1"])
        return sum(scores) / len(scores)
    return boundary["committed_fact_f1"] if boundary is not None else writer_score(metrics)


def writer_development_safe(metrics, initial, boundary=None, boundary_initial=None,
                            narrative=None, narrative_initial=None, *,
                            query_binding=False, query_development=None, query_initial=None):
    """Every attached group retains its complete source and reference quality."""
    groups = [(metrics, initial)]
    for current, baseline in ((boundary, boundary_initial), (narrative, narrative_initial)):
        if current is not None:
            if baseline is None:
                raise ValueError("public development guards require initial metrics")
            groups.append((current, baseline))
    if query_development is not None:
        if query_initial is None:
            raise ValueError("query binding development guard requires initial metrics")
        groups.append((query_development, query_initial))
    return all(writer_edge_candidate_safe(current, baseline)
               and (not query_binding or query_binding_safe(current, baseline))
               for current, baseline in groups)


def bounded_dialogue_replay(rows, budget=512, seed=13013):
    """Keep rare action/reference supervision inside a bounded fit replay."""
    rng = random.Random(seed)
    required_tags = {
        "lifecycle", "reference-update", "ambiguous-reference",
        "literal-provenance-prefix", "causal-intra-message", "mixed-subjects",
        "explicit-other", "anonymous-self",
    }
    selected = {
        i for i, row in enumerate(rows)
        if row["tag"] in required_tags
        or any(span["action"] in (2, 3) for span in row["spans"])
    }
    if len(selected) > budget:
        raise ValueError("replay budget cannot preserve required lifecycle/reference cases")
    groups = {}
    for i, row in enumerate(rows):
        if i not in selected:
            groups.setdefault(row["tag"], []).append(i)
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    while len(selected) < budget and keys:
        remaining = []
        for key in keys:
            if len(selected) == budget:
                break
            selected.add(groups[key].pop())
            if groups[key]:
                remaining.append(key)
        keys = remaining
    return [rows[i] for i in sorted(selected)]


def sample_action_balanced_batch(positives, negatives, rng):
    """Allocate rare DELETE/UPDATE rows before WRITE, using labels only in training."""
    selected = set()
    def is_diverse(row):
        return row.get("boundary_diversity") or row.get("boundary_generalization")

    def is_narrative(row):
        return bool(row.get("narrative_diversity"))

    diverse = [i for i, row in enumerate(positives) if is_diverse(row)]
    selected.update(rng.sample(diverse, min(4, len(diverse))))
    for action, quota in ((3, 2), (2, 2), (1, 8 - len(selected))):
        if action == 1:
            assertions = [i for i, row in enumerate(positives) if i not in selected
                          and row.get("assertion_generalization")
                          and not is_diverse(row) and not is_narrative(row)]
            reserved = rng.sample(assertions, min(2, len(assertions), max(12 - len(selected), 0)))
            selected.update(reserved)
            quota = max(quota - len(reserved), 0)
        choices = [
            i for i, row in enumerate(positives) if i not in selected
            and not is_diverse(row) and not is_narrative(row)
            and any(span["action"] == action for span in row["spans"])
        ]
        selected.update(rng.sample(choices, min(quota, len(choices))))
    remaining = [i for i, row in enumerate(positives) if i not in selected
                 and not is_diverse(row) and not is_narrative(row)]
    selected.update(rng.sample(remaining, min(max(12 - len(selected), 0), len(remaining))))
    batch = [positives[i] for i in sorted(selected)]
    diverse_negative = []
    for tags in (("writer-complex-fiction", "writer-edge-fiction"),
                 ("writer-complex-greeting", "writer-edge-greeting")):
        choices = [r for r in negatives if is_diverse(r) and r["tag"] in tags]
        if choices:
            diverse_negative.append(rng.choice(choices))
    regular_negative = [r for r in negatives if not is_diverse(r) and not is_narrative(r)]
    assertion_negative = [r for r in regular_negative if r.get("assertion_generalization")]
    selected_assertion_negative = rng.sample(assertion_negative, min(1, len(assertion_negative)))
    remaining_negative = [r for r in regular_negative if r not in selected_assertion_negative]
    batch += diverse_negative + selected_assertion_negative + rng.sample(
        remaining_negative, min(8 - len(diverse_negative) - len(selected_assertion_negative),
                                len(remaining_negative)))
    narrative_positive = [r for r in positives if is_narrative(r)]
    batch += rng.sample(narrative_positive, min(2, len(narrative_positive)))
    for tag in ("writer-narrative-questions", "writer-narrative-fiction"):
        choices = [r for r in negatives if is_narrative(r) and r["tag"] == tag]
        if choices:
            batch.append(rng.choice(choices))
    rng.shuffle(batch)
    return batch


def train(args):
    torch.set_num_threads(1)
    torch.manual_seed(5201)
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.eval_every < 1:
        raise ValueError("eval-every must be positive")
    validate_query_binding_args(args)
    if (args.endpoint_weight < 0 or args.endpoint_margin < 0 or args.boundary_weight < 0 or args.membership_weight < 0
            or args.boundary_anchor_weight < 0 or args.boundary_path_anchor_weight < 0):
        raise ValueError("boundary loss weights and margin must be nonnegative")
    need_weight = getattr(args, "write_need_weight", 1.0)
    need_margin = getattr(args, "write_need_anchor_margin", 1.0)
    if not math.isfinite(need_weight) or not math.isfinite(need_margin) or need_weight < 0 or need_margin <= 0:
        raise ValueError("write need weight and anchor margin must be finite, with nonnegative weight and positive margin")
    if args.train_heads in ("writer-context-binding", "writer-need", "writer-need-full") and not args.resume:
        raise ValueError("write need training requires a trained current-format writer --resume")
    factorized = getattr(args, "factorized_source_corrections", False)
    correction_margin = getattr(args, "source_correction_margin", 1.0)
    if not math.isfinite(correction_margin) or correction_margin <= 0:
        raise ValueError("source correction margin must be finite and positive")
    if factorized and (args.train_heads not in _FACTORIZED_WRITER_MODES
                       or not (args.boundary_anchor_weight or args.boundary_path_anchor_weight)):
        raise ValueError("factorized source corrections require anchored writer-context-content or writer-context-binding")
    if factorized and not args.resume:
        raise ValueError("factorized source corrections require a trained current-format writer --resume")
    if (args.boundary_anchor_weight or args.boundary_path_anchor_weight) and args.train_heads not in _RESIDUAL_WRITER_MODES:
        raise ValueError("boundary anchors require isolated boundary-residual training")
    if args.boundary_path_anchor_margin <= 0:
        raise ValueError("boundary-path-anchor-margin must be positive")
    if args.source_error_balance and (args.train_heads not in
                                     ("writer-context-residual", "writer-context-content", "writer-context-binding")
                                     or not (args.boundary_anchor_weight or args.boundary_path_anchor_weight)):
        raise ValueError("source error balancing requires source-context residual training with anchors")
    if args.boundary_diversity and args.boundary_generalization:
        raise ValueError("independent boundary curriculum must not mix the earlier diversity course")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    if args.curriculum:
        source = json.loads(Path(args.curriculum).read_text())
    else:
        from prepare_neural_colloquial import colloquial_curriculum

        source = {s: colloquial_curriculum(s) for s in ("fit", "development")}
    if not {"fit", "development"} <= source.keys():
        raise ValueError("independent fit/development observations are required")
    rows = {s: source[s] for s in ("fit", "development")}
    # Action/reference examples include remembered facts, corrections and
    # deletion requests. Dataset replay does not change the model architecture.
    if args.replay_dialogue:
        from prepare_neural_dialogue import dialogue_curriculum

        replay = dialogue_curriculum("fit")
        rows["fit"] = rows["fit"] + bounded_dialogue_replay(replay)
    if args.replay_episodes:
        from prepare_neural_episodes import episode_curriculum

        rows["fit"] = rows["fit"] + episode_curriculum("fit")
    if getattr(args, "episode_development", False):
        if not args.replay_episodes:
            raise ValueError("episode development selection requires episode replay training")
        from prepare_neural_episodes import episode_curriculum

        rows["development"] = rows["development"] + episode_curriculum("development")
    if args.boundary_diversity:
        from neural_writer_curriculum import boundary_diversity_fit_rows

        rows["fit"] = rows["fit"] + boundary_diversity_fit_rows()
    if args.boundary_generalization:
        from neural_writer_curriculum import with_boundary_generalization

        rows = with_boundary_generalization(rows)
    if args.narrative_diversity:
        from neural_writer_curriculum import with_narrative_diversity

        rows = with_narrative_diversity(rows)
    source_context_enabled = getattr(args, "source_context_generalization", False)
    source_context_added_fit_rows = source_context_added_context_texts = 0
    if source_context_enabled:
        from neural_writer_curriculum import with_source_context_generalization

        fit_count = len(rows["fit"])
        contexts = {row["context"] for group in rows.values() for row in group}
        rows = with_source_context_generalization(rows)
        source_context_added_fit_rows = len(rows["fit"]) - fit_count
        source_context_added_context_texts = len(
            {row["context"] for row in rows["fit"]} - contexts
        )
    assertion_enabled = getattr(args, "assertion_generalization", False)
    assertion_added_fit_rows = 0
    if assertion_enabled:
        from neural_writer_curriculum import with_assertion_generalization

        fit_count = len(rows["fit"])
        rows = with_assertion_generalization(rows)
        assertion_added_fit_rows = len(rows["fit"]) - fit_count
    query_focus = args.train_heads in QUERY_BINDING_MODES
    query_generalization = getattr(args, "query_binding_generalization", False)
    if query_generalization:
        from neural_query_binding_curriculum import (
            explicit_question_rows, query_binding_curriculum)

        rows["fit"] = rows["fit"] + query_binding_curriculum("fit")
        rows["fit"] = rows["fit"] + explicit_question_rows("fit")
        if getattr(args, "query_binding_dialogue", False):
            from neural_query_binding_curriculum import query_binding_dialogue_fit_augmentation

            rows["fit"] = rows["fit"] + query_binding_dialogue_fit_augmentation()
        rows["query_binding_development"] = (
            query_binding_curriculum("development")
            + explicit_question_rows("development"))
    if query_focus:
        validate_query_binding_fit_rows(args.train_heads, rows["fit"])
    (root / "curriculum.json").write_text(json.dumps(rows, indent=2))
    required = {
        r[k] for group in rows.values() for r in group for k in ("body", "context")
    }
    backbone = sha(args.gguf)
    if args.features:
        saved = load_features(args.features, backbone=backbone, probe=sha(args.probe))
        if saved["identity"]["backbone"] != backbone:
            raise ValueError("frozen feature backbone mismatch")
        cache = saved["features"]
        missing = sorted(required - cache.keys())
        if missing:
            extra, _ = extract(
                args, {"additional": [dict(body=t, context=t) for t in missing]}
            )
            cache.update(extra)
        provenance = dict(
            backbone=backbone,
            probe=sha(args.probe),
            texts_sha256=hashlib.sha256(
                json.dumps(sorted(required)).encode()
            ).hexdigest(),
        )
        torch.save(
            dict(features={t: cache[t] for t in required}, identity=provenance),
            root / "features.pt",
        )
    else:
        cache, provenance = extract(args, rows)
    model = NeuralSlots()
    resume_metadata = {}
    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False, map_location="cpu")
        if checkpoint["format"] != FORMAT or checkpoint["backbone_sha256"] != backbone:
            raise ValueError(
                "checkpoint identity differs from the current model/backbone"
            )
        require_parameters(model, checkpoint["state_dict"])
        resume_metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    anchor_stats = None
    anchored_mode = bool(args.boundary_anchor_weight or args.boundary_path_anchor_weight)
    residual_mode = args.train_heads in _RESIDUAL_WRITER_MODES
    membership_mode = args.train_heads in _MEMBERSHIP_WRITER_MODES
    membership_only = args.train_heads in ("writer-membership", "writer-context-residual")
    parent_parameters = snapshot_parent_boundary(model) if anchored_mode or residual_mode else None
    need_parent_parameters = snapshot_parent_write_need(model) if args.train_heads in ("writer-context-binding", "writer-need") else None
    if anchored_mode:
        model.boundary_fit_correct, anchor_stats = precompute_fit_correctness(model, rows["fit"], cache)
        print(json.dumps(dict(boundary_anchor_precomputation=anchor_stats)), flush=True)
    device = torch.device(args.device)
    model.to(device)
    if parent_parameters is not None:
        model.boundary_parent_parameters = {name: value.to(device) for name, value in parent_parameters.items()}
    if need_parent_parameters is not None:
        model.write_need_parent_parameters = {name: value.to(device) for name, value in need_parent_parameters.items()}
    model.write_need_fit_teachers = {}
    model.write_need_weight = getattr(args, "write_need_weight", 1.0)
    model.write_need_anchor_margin = getattr(args, "write_need_anchor_margin", 1.0)
    model.structured_weight = args.structured_weight
    model.endpoint_weight = args.endpoint_weight
    model.endpoint_margin = args.endpoint_margin
    model.boundary_weight = args.boundary_weight if args.train_heads in ("writer-edge", "writer-raw", "writer-context", "writer-content", "writer-context-content", "writer-context-binding") else 0
    model.membership_weight = args.membership_weight if membership_mode else 0
    model.boundary_anchor_weight = args.boundary_anchor_weight
    model.boundary_path_anchor_weight = args.boundary_path_anchor_weight
    model.boundary_path_anchor_margin = args.boundary_path_anchor_margin
    model.writer_mode = args.train_heads
    model.factorized_source_corrections = factorized
    model.source_correction_margin = correction_margin
    model.source_error_balance = args.source_error_balance
    model.writer_fit_teachers = {}
    allowed = writer_train_keys(args.train_heads)
    if getattr(args, "train_plan_context", False):
        allowed = allowed + ("plan_context.weight", "plan_context.bias",
                             "plan_local.weight", "plan_local.bias")
    model.query_binding_anchor_weight = getattr(args, "query_binding_anchor_weight", 0.0)
    model.query_binding_anchor_margin = getattr(args, "query_binding_anchor_margin", 1.0)
    model.query_binding_parent_parameters = (snapshot_query_binding_parent(model)
                                            if model.query_binding_anchor_weight else None)
    model.query_binding_fit_teachers = {}
    query_parent_initial = ({name: value.detach().cpu().clone()
                             for name, value in model.query_binding_parent_parameters.items()}
                            if model.query_binding_parent_parameters is not None else None)
    query_training = dict(
        query_binding_refinement=query_focus,
        query_binding_generalization=query_generalization,
        query_binding_dialogue_augmentation=(getattr(args, "query_binding_dialogue", False)
                                             if query_generalization else None),
        query_binding_trainable_parameters=(sum(p.numel() for name, p in model.named_parameters() if name in allowed) if query_focus else None),
        query_binding_anchor_weight=model.query_binding_anchor_weight,
        query_binding_anchor_margin=model.query_binding_anchor_margin,
        query_binding_anchor_teacher=(f"independent-frozen-query{len(QUERY_KEYS)}; shared-source{len(PLAN_KEYS)}-frozen" if model.query_binding_parent_parameters is not None else None),
        query_binding_supervision=("query-need-only" if args.train_heads == "query-gate" else
                                   "used-BIO+origin; need-gate-frozen" if args.train_heads == "query-source" else
                                   "query-need+used-BIO+origin" if query_focus else None),
        query_binding_reduction="nonempty-binding-class-means+global-worst-capped-parent-margins" if query_focus else None,
    )
    need_training = dict(
        write_need_trainable="plan_need.weight" in allowed,
        write_need_parent_snapshot=need_parent_parameters is not None,
        write_need_weight=model.write_need_weight if need_parent_parameters is not None else None,
        write_need_anchor_margin=model.write_need_anchor_margin if need_parent_parameters is not None else None,
        write_need_objective="frozen-parent-wrong-gate-BCE-plus-capped-correct-signed-margin" if need_parent_parameters is not None else None,
        write_need_pooling="gold-span-original-full-encoding-mean-fit-only" if need_parent_parameters is not None else None,
        query_binding_trainable=any(name in allowed for name in QUERY_KEYS),
        write_reference_tags_trainable=any(name.startswith(("plan_ref_", "plan_origin.")) for name in allowed),
        factorized_source_corrections=factorized,
        source_correction_margin=correction_margin if factorized else None,
        source_correction_teacher="frozen-parent-fit-only-component-gaps-and-eligibility" if factorized else None,
        source_correction_objective="capped-occupancy-and-conditional-structure-path-margins" if factorized else None,
        source_correction_gradient_isolation="occupancy-membership-only-with-detached-boundaries;structure-boundaries-only-without-membership" if factorized else None,
        source_correction_weight=model.structured_weight if factorized else None,
        legacy_wrong_row_source_losses_replaced=factorized,
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowed)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0 if args.train_heads in ("writer-context",) + _MEMBERSHIP_WRITER_MODES else 0.001,
    )
    prepared = prepare(rows, cache, device)
    initial, _ = evaluate(model, rows["development"], prepared)
    boundary_initial = (evaluate(model, rows["boundary_development"], prepared)[0]
                        if args.boundary_generalization else None)
    narrative_initial = (evaluate(model, rows["narrative_development"], prepared)[0]
                         if args.narrative_diversity else None)
    query_initial = (evaluate(model, rows["query_binding_development"], prepared)[0]
                     if query_generalization else None)
    query_fit_initial = evaluate_query_binding(model, rows["fit"], prepared) if query_focus else None
    initial_fit_diagnostic = (writer_fit_training_diagnostic(
        model, rows["fit"], prepared, step=0, stage="resumed-parent-before-training"
    ) if factorized else None)
    best_score = writer_development_score(initial, boundary_initial, narrative_initial,
        query_binding=query_focus, query_development=query_initial)
    initial_score = best_score
    selected = 0
    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def save(metrics, boundary_metrics=None, narrative_metrics=None, query_metrics=None):
        torch.save(
            dict(
                resume_metadata,
                format=FORMAT,
                backbone_sha256=backbone,
                state_dict=best,
                selected_step=selected,
                diagnostic_only=False,
                eligible_for_selection=True,
                metrics=metrics,
                boundary_development=boundary_metrics,
                narrative_development=narrative_metrics,
                query_binding_development=query_metrics,
                query_binding_development_initial=query_initial,
                selection_score=best_score,
                initial_selection_score=initial_score,
                writer_training=dict(mode=args.train_heads, trainable_tensors=list(allowed),
                                     raw_context_words=BOUNDARY_CONTEXT,
                                     plan_context_trainable=bool(getattr(args, "train_plan_context", False)),
                                     center_branch_frozen=args.train_heads == "writer-context",
                                     endpoint_weight=args.endpoint_weight, endpoint_margin=args.endpoint_margin,
                                     boundary_diversity=args.boundary_diversity,
                                     boundary_generalization=args.boundary_generalization,
                                     narrative_diversity=args.narrative_diversity,
                                     source_context_generalization=source_context_enabled,
                                     source_context_added_fit_rows=source_context_added_fit_rows,
                                     source_context_added_context_texts=source_context_added_context_texts,
                                     assertion_generalization=assertion_enabled,
                                     assertion_added_fit_rows=assertion_added_fit_rows,
                                     boundary_weight=model.boundary_weight,
                                     membership_weight=model.membership_weight,
                                     membership_input_order=("encoded", "previous_raw", "current_raw", "next_raw"),
                                     membership_input_width=MEMBERSHIP_INPUT,
                                     membership_encoded_normalized=False,
                                     source_context_residual_input_order=("body_encoding", "full_minus_body_encoding", "normalized_current_hidden"),
                                     source_context_residual_input_width=SOURCE_CONTEXT_INPUT,
                                     source_context_residual_hidden_width=SOURCE_CONTEXT_WIDTH,
                                     source_context_residual_output="unbounded-common-source-membership",
                                     source_context_residual_trainable="plan_source_context.0.weight" in allowed,
                                     source_context_residual_parent_snapshot=parent_parameters is not None,
                                     source_error_balance=args.source_error_balance,
                                     batch_loss_reduction=("nonempty-frozen-parent-FN-FP-other-row-class-means-plus-correct-anchor-row-mean"
                                                           if args.source_error_balance else "all-row-mean"),
                                     membership_objective="capped-source-occupancy-path-margin" if factorized else "parent-error-class-balanced-gold-source-or-all-reject-vs-background" if membership_mode else None,
                                     membership_preservation="capped-parent-component-gap" if factorized else "unchanged-parent-total-membership-MSE" if membership_mode else None,
                                     boundary_anchor_weight=args.boundary_anchor_weight,
                                     boundary_path_anchor_weight=args.boundary_path_anchor_weight,
                                     boundary_path_anchor_margin=args.boundary_path_anchor_margin,
                                     anchor_teacher="full-resumed-parent" if anchored_mode else None,
                                     path_anchor_competitors="all-source-paths-except-gold" if args.boundary_path_anchor_weight else None,
                                     wrong_row_boundary_objective=("frozen-write-gate-only" if args.train_heads == "writer-need" else "capped-conditional-structure-path-margin" if factorized else "frozen" if membership_only else "parent-error-residual" if residual_mode else "complete-token-supervision"),
                                     correction_reduction=("frozen-parent-wrong-need-BCE" if args.train_heads == "writer-need" else "sum of isolated source-occupancy and conditional-structure margin losses" if factorized else "mean over frozen-parent membership FN/FP emission classes" if membership_only else "separate class means for parent boundary and membership errors" if membership_mode else "mean over each nonempty target class, then mean classes" if residual_mode else None),
                                     wrong_row_preservation=("capped-frozen-parent-correct-signed-margins" if args.train_heads == "writer-need" else "capped-frozen-parent-component-gaps" if factorized else "boundary heads frozen; matched total-membership MSE" if membership_only else "matched source-membership boundary and total-membership MSE" if membership_mode else "no selected correction and equal parent/gold source membership" if residual_mode else None),
                                     eval_every=args.eval_every, **need_training, **query_training),
                source_correction_fit_teacher_counts=source_correction_teacher_counts(model) if factorized else None,
                boundary_anchor_precomputation=anchor_stats,
                reference_quality_floor={key: initial[key] for key in ("reference_exact", "whole_reference_exact", "ambiguous_unsafe_writes")},
                fixed_development_initial=initial,
                boundary_development_initial=boundary_initial,
                narrative_development_initial=narrative_initial,
                narrative_development_role="public-development-not-independent-acceptance" if args.narrative_diversity else None,
            ),
            root / "selected.pt",
        )

    save(initial, boundary_initial, narrative_initial, query_initial)
    rng = random.Random(5202)
    positives = [r for r in rows["fit"] if r["spans"]]
    negatives = [r for r in rows["fit"] if not r["spans"]]
    began = time.monotonic()
    interval_anchor_sum = torch.zeros((), device=device)
    interval_anchor_mse_sum = torch.zeros((), device=device)
    interval_path_anchor_sum = torch.zeros((), device=device)
    interval_anchor_rows = interval_learning_rows = 0
    interval_source_rows = dict(missing=0, spurious=0, other=0, overlap=0)
    query_sampling_visits = dict(steps=0, **{name: 0 for name in QUERY_BINDING_CLASSES},
        novel={name: 0 for name in QUERY_BINDING_CLASSES},
        replay={name: 0 for name in QUERY_BINDING_CLASSES})
    last_metrics, last_boundary, last_narrative, last_query = initial, boundary_initial, narrative_initial, query_initial
    for step in range(1, args.steps + 1):
        sampled = (sample_query_binding_batch(rows["fit"], rng) if query_focus
                   else sample_action_balanced_batch(positives, negatives, rng))
        if query_focus:
            query_sampling_visits["steps"] += 1
            for row in sampled:
                name = query_binding_class(row)
                query_sampling_visits[name] += 1
                query_sampling_visits["novel" if query_binding_is_novel(row) else "replay"][name] += 1
        if not sampled:
            raise ValueError("fit curriculum is empty")
        loss = batch_loss(model, sampled, prepared)
        if anchored_mode:
            interval_anchor_sum += model.boundary_batch_stats["anchor_loss_sum"]
            interval_anchor_mse_sum += model.boundary_batch_stats["anchor_raw_mse_sum"]
            interval_path_anchor_sum += model.boundary_batch_stats["path_anchor_loss_sum"]
            interval_anchor_rows += model.boundary_batch_stats["anchored_rows"]
            interval_learning_rows += model.boundary_batch_stats["learning_rows"]
            for name in interval_source_rows:
                interval_source_rows[name] += model.boundary_batch_stats["source_" + name + "_rows"]
        optimizer.zero_grad()
        loss.backward()
        freeze_boundary_center_gradient(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            metrics, _ = evaluate(model, rows["development"], prepared)
            boundary_metrics = (evaluate(model, rows["boundary_development"], prepared)[0]
                                if args.boundary_generalization else None)
            narrative_metrics = (evaluate(model, rows["narrative_development"], prepared)[0]
                                 if args.narrative_diversity else None)
            query_metrics = (evaluate(model, rows["query_binding_development"], prepared)[0]
                             if query_generalization else None)
            last_metrics, last_boundary, last_narrative, last_query = metrics, boundary_metrics, narrative_metrics, query_metrics
            safe = writer_development_safe(metrics, initial, boundary_metrics, boundary_initial,
                narrative_metrics, narrative_initial, query_binding=query_focus,
                query_development=query_metrics, query_initial=query_initial)
            print(
                json.dumps(
                    dict(
                        step=step,
                        loss=float(loss.detach()),
                        development=metrics,
                        boundary_development=boundary_metrics,
                        narrative_development=narrative_metrics,
                        query_binding_development=query_metrics,
                        query_binding_safe=(query_binding_safe(metrics, initial) if query_focus else None),
                        query_binding_sampling_visits=query_sampling_visits if query_focus else None,
                        fixed_development_safe=safe,
                        reference_quality_safe=writer_reference_safe(metrics, initial),
                        boundary_anchor_loss=float(interval_anchor_sum) / max(interval_anchor_rows, 1),
                        boundary_anchor_raw_mse=float(interval_anchor_mse_sum) / max(interval_anchor_rows, 1),
                        boundary_path_anchor_loss=float(interval_path_anchor_sum) / max(interval_anchor_rows, 1),
                        boundary_anchored_rows=interval_anchor_rows,
                        boundary_learning_rows=interval_learning_rows,
                        boundary_learning_fraction=interval_learning_rows / max(interval_anchor_rows + interval_learning_rows, 1),
                        source_error_balance=args.source_error_balance,
                        source_error_rows=dict(interval_source_rows),
                        seconds=time.monotonic() - began,
                    )
                ),
                flush=True,
            )
            score = writer_development_score(metrics, boundary_metrics, narrative_metrics,
                query_binding=query_focus, query_development=query_metrics)
            if safe and score > best_score:
                best_score, selected = score, step
                best = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                save(metrics, boundary_metrics, narrative_metrics, query_metrics)
            interval_anchor_sum.zero_()
            interval_anchor_mse_sum.zero_()
            interval_path_anchor_sum.zero_()
            interval_anchor_rows = interval_learning_rows = 0
            for name in interval_source_rows:
                interval_source_rows[name] = 0
    last_trained_fit_diagnostic = (writer_fit_training_diagnostic(
        model, rows["fit"], prepared, step=args.steps, stage="last-trained-before-best-restoration"
    ) if factorized else None)
    query_fit_terminal = (evaluate_query_binding(model, rows["fit"], prepared)
                          if query_focus else None)
    query_parent_unchanged = (all(torch.equal(value.detach().cpu(), query_parent_initial[name])
        for name, value in model.query_binding_parent_parameters.items())
        if query_parent_initial is not None else None)
    if query_focus:
        torch.save(writer_terminal_checkpoint_payload(
            model, resume_metadata, format=FORMAT, backbone_sha256=backbone,
            training_step=args.steps, writer_training_mode=args.train_heads,
            query_binding_training=query_training,
            last_step_development=last_metrics,
            last_step_boundary_development=last_boundary,
            last_step_narrative_development=last_narrative,
            last_step_query_binding_development=last_query,
            initial_selection_score=initial_score,
            last_step_query_binding_fit=query_fit_terminal), root / "last-step.pt")
    if getattr(args, "save_last_diagnostic", False):
        # Diagnostic only: retain the terminal state to explain a blocked
        # selection.  It is never a candidate checkpoint.
        torch.save(
            dict(
                format=FORMAT,
                backbone_sha256=backbone,
                training_step=args.steps,
                writer_training_mode=args.train_heads,
                diagnostic_only=True,
                state_dict={k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()},
            ),
            root / "last-diagnostic.pt",
        )
    model.load_state_dict(best)
    operator_cache_stats = dict(
        unique_fit_teachers=len(model.writer_fit_teachers),
        frozen_margin_entries=sum("required_gap" in t for t in model.writer_fit_teachers.values()),
        unique_write_need_fit_teachers=len(model.write_need_fit_teachers),
        frozen_parent_wrong_write_need_gates=sum(len(t["correction_indices"]) for t in model.write_need_fit_teachers.values()),
        frozen_parent_correct_write_need_gates=sum(len(t["preservation_indices"]) for t in model.write_need_fit_teachers.values()),
        source_correction_fit_teacher_counts=source_correction_teacher_counts(model) if factorized else None,
    )
    model.writer_fit_teachers.clear()
    model.write_need_fit_teachers.clear()
    model.cpu()
    del prepared
    prepared = prepare(rows, cache, torch.device("cpu"))
    report = {s: evaluate(model, data, prepared)[0] for s, data in rows.items()}
    if factorized:
        report["fit_training_diagnostics"] = dict(
            initial=initial_fit_diagnostic,
            last_trained=last_trained_fit_diagnostic,
            existing_report_fit_role="selected-checkpoint-after-best-restoration",
        )
    report.update(
        selected_step=selected,
        training_steps_executed=args.steps,
        episode_development_selection=bool(getattr(args, "episode_development", False)),
        initial_selection_score=initial_score,
        selection_score=best_score,
        query_binding_training=query_training,
        query_binding_sampling_visits=query_sampling_visits if query_focus else None,
        query_binding_parent_all13_unchanged=query_parent_unchanged,
        query_binding_fit_initial=query_fit_initial,
        query_binding_fit_terminal=query_fit_terminal,
        query_binding_development_initial=query_initial,
        last_step_development=last_metrics,
        last_step_boundary_development=last_boundary,
        last_step_narrative_development=last_narrative,
        last_step_query_binding_development=last_query,
        query_binding_fit_used_for_selection=False,
        writer_training=dict(mode=args.train_heads, trainable_tensors=list(allowed),
                             raw_context_words=BOUNDARY_CONTEXT,
                             plan_context_trainable=bool(getattr(args, "train_plan_context", False)),
                             center_branch_frozen=args.train_heads == "writer-context",
                             endpoint_weight=args.endpoint_weight, endpoint_margin=args.endpoint_margin,
                             boundary_diversity=args.boundary_diversity,
                             boundary_generalization=args.boundary_generalization,
                             narrative_diversity=args.narrative_diversity,
                             source_context_generalization=source_context_enabled,
                             source_context_added_fit_rows=source_context_added_fit_rows,
                             source_context_added_context_texts=source_context_added_context_texts,
                             assertion_generalization=assertion_enabled,
                             assertion_added_fit_rows=assertion_added_fit_rows,
                             boundary_weight=model.boundary_weight,
                             membership_weight=model.membership_weight,
                             membership_input_order=("encoded", "previous_raw", "current_raw", "next_raw"),
                             membership_input_width=MEMBERSHIP_INPUT,
                             membership_encoded_normalized=False,
                             source_context_residual_input_order=("body_encoding", "full_minus_body_encoding", "normalized_current_hidden"),
                             source_context_residual_input_width=SOURCE_CONTEXT_INPUT,
                             source_context_residual_hidden_width=SOURCE_CONTEXT_WIDTH,
                             source_context_residual_output="unbounded-common-source-membership",
                             source_context_residual_trainable="plan_source_context.0.weight" in allowed,
                             source_context_residual_parent_snapshot=parent_parameters is not None,
                             source_error_balance=args.source_error_balance,
                             batch_loss_reduction=("nonempty-frozen-parent-FN-FP-other-row-class-means-plus-correct-anchor-row-mean"
                                                   if args.source_error_balance else "all-row-mean"),
                             membership_objective="capped-source-occupancy-path-margin" if factorized else "parent-error-class-balanced-gold-source-or-all-reject-vs-background" if membership_mode else None,
                             membership_preservation="capped-parent-component-gap" if factorized else "unchanged-parent-total-membership-MSE" if membership_mode else None,
                             boundary_anchor_weight=args.boundary_anchor_weight,
                             boundary_path_anchor_weight=args.boundary_path_anchor_weight,
                             boundary_path_anchor_margin=args.boundary_path_anchor_margin,
                             anchor_teacher="full-resumed-parent" if anchored_mode else None,
                             path_anchor_competitors="all-source-paths-except-gold" if args.boundary_path_anchor_weight else None,
                             wrong_row_boundary_objective="capped-conditional-structure-path-margin" if factorized else "frozen" if membership_only else "parent-error-residual" if residual_mode else "complete-token-supervision",
                             correction_reduction="sum of isolated source-occupancy and conditional-structure margin losses" if factorized else "mean over frozen-parent membership FN/FP emission classes" if membership_only else "separate class means for parent boundary and membership errors" if membership_mode else "mean over each nonempty target class, then mean classes" if residual_mode else None,
                             wrong_row_preservation="capped-frozen-parent-component-gaps" if factorized else "boundary heads frozen; matched total-membership MSE" if membership_only else "matched source-membership boundary and total-membership MSE" if membership_mode else "no selected correction and equal parent/gold source membership" if residual_mode else None,
                             eval_every=args.eval_every, **need_training, **query_training),
        boundary_anchor_precomputation=anchor_stats,
        training_operator_cache=operator_cache_stats,
        final_report_device="cpu",
        reference_quality_floor={key: initial[key] for key in ("reference_exact", "whole_reference_exact", "ambiguous_unsafe_writes")},
        fixed_development_initial=initial,
        boundary_development_initial=boundary_initial,
        narrative_development_initial=narrative_initial,
        narrative_development_role="public-development-not-independent-acceptance" if args.narrative_diversity else None,
        selection=("query-binding-class-worst/mean/full; per-group-source-reference-binding-safety-floors" if query_focus else
                   "every-group-complete-source-reference-f1-floors-mean-public-f1" if args.narrative_diversity else
                   "fixed-and-boundary-complete-source-reference-f1-floors" if args.boundary_generalization else
                   "fixed-complete-source-reference-f1-floors"),
        locomo_used=False,
        production_qualified=False,
        model_sha256=export(model, backbone, root / "model.bnsmodel"),
    )
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def writer_argument_parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("gguf", "probe", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument(
        "--resume",
        help="Optional current-format checkpoint; omitted initializes from scratch",
    )
    p.add_argument("--curriculum", help="Independent fit/development observation JSON")
    p.add_argument("--features", help="Identity-checked frozen C feature cache")
    p.add_argument("--steps", type=int, default=1800)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--device", default="cpu")
    p.add_argument("--learning-rate", type=float, default=0.002)
    p.add_argument("--structured-weight", type=float, default=1)
    p.add_argument("--endpoint-weight", type=float, default=0.5)
    p.add_argument("--endpoint-margin", type=float, default=1.0)
    p.add_argument("--boundary-weight", type=float, default=1.0)
    p.add_argument("--membership-weight", type=float, default=1.0)
    p.add_argument("--write-need-weight", type=float, default=1.0,
                   help="Binding-mode parent-error write gate BCE and correct-gate margin loss weight")
    p.add_argument("--write-need-anchor-margin", type=float, default=1.0,
                   help="Cap the preserved signed margin of each parent-correct write gate")
    p.add_argument("--boundary-anchor-weight", type=float, default=0.0)
    p.add_argument("--boundary-path-anchor-weight", type=float, default=0.0)
    p.add_argument("--boundary-path-anchor-margin", type=float, default=1.0)
    p.add_argument("--source-error-balance", action="store_true",
                   help="Balance frozen-parent missing/spurious/other fit-row classes separately from correct-row anchors")
    p.add_argument("--save-last-diagnostic", action="store_true",
                   help="Write the terminal training state to last-diagnostic.pt for blocked-selection diagnosis; never a candidate")
    p.add_argument("--factorized-source-corrections", action="store_true",
                   help="Replace wrong-parent source losses with capped occupancy and conditional-structure margins")
    p.add_argument("--source-correction-margin", type=float, default=1.0,
                   help="Required wrong-parent component margin and cap on correct-parent component preservation")
    p.add_argument("--query-binding-anchor-weight", type=float, default=0.0,
                   help="Protect frozen-parent correct fit need and used BIO margins during resumed query binding")
    p.add_argument("--query-binding-anchor-margin", type=float, default=1.0,
                   help="Cap parent-correct query need and BIO margins")
    p.add_argument("--query-binding-generalization", action="store_true",
                   help="Add independent query binding fit worlds and a separate development guard")
    p.add_argument("--query-binding-dialogue", action="store_true",
                   help="Add fit-only dialogue-history binding rows where the addressed name follows the original mention")
    p.add_argument(
        "--train-heads",
        choices=("all", "query", "query-gate", "query-source", "query-context", "references", "writer-ref", "boundaries", "write-boundaries", "writer-full-boundaries", "writer-need", "writer-need-full", "writer-edge", "writer-raw", "writer-context", "writer-membership", "writer-content", "writer-context-residual", "writer-context-content", "writer-context-binding"),
        default="all",
    )
    p.add_argument("--replay-dialogue", action="store_true")
    p.add_argument("--replay-episodes", action="store_true")
    p.add_argument("--episode-development", action="store_true",
                   help="Append episode development observations to checkpoint selection; requires --replay-episodes")
    p.add_argument("--boundary-diversity", action="store_true")
    p.add_argument("--boundary-generalization", action="store_true")
    p.add_argument("--narrative-diversity", action="store_true",
                   help="Add independent multi-fact fit turns and a public narrative development guard")
    p.add_argument("--source-context-generalization", action="store_true",
                   help="Cross irrelevant histories for the authored boundary/narrative fit worlds only")
    p.add_argument("--assertion-generalization", action="store_true",
                   help="Add fit-only real quoted reports and factual question premises with nonassertion controls")
    p.add_argument("--train-plan-context", action="store_true",
                   help="Include the shared plan condition and local-context layers in boundary writer training")
    return p


if __name__ == "__main__":
    train(writer_argument_parser().parse_args())
