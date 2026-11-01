"""Exact-source endpoint objectives and isolation of frozen NN query binding."""

from pathlib import Path
import re
import random
import hashlib
import json
import argparse
import time
import sys
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import NeuralSlots, INPUT, SEQUENCE_KEYS, observation_context, word_features, decode_plan
from train_neural_dialogue import (
    endpoint_competitors,
    endpoint_contrast_loss,
    plan_path_score,
    beginning_competitors,
    boundary_supervision_loss,
    writer_train_keys,
    freeze_boundary_center_gradient,
    observation_loss,
    source_words,
    source_types,
    sample_action_balanced_batch,
    writer_reference_safe,
    writer_edge_candidate_safe,
    precompute_fit_correctness,
    writer_observation_key,
    batch_loss,
    structured_path_anchor_loss,
    build_writer_teacher,
    path_anchor_required_gaps,
    observation_writer_teacher,
    snapshot_parent_boundary,
    parent_boundary_correction_teacher,
    boundary_residual_supervision_loss,
    plan_targets,
    writer_development_safe,
    writer_development_score,
    writer_score,
)
from neural_writer_curriculum import (
    boundary_diversity_fit_rows, FACTS, FIT_NAMES,
    boundary_generalization_curriculum, BOUNDARY_NAMES, BOUNDARY_FACTS,
    with_boundary_generalization,
    narrative_curriculum, with_narrative_diversity, NARRATIVE_NAMES, NARRATIVE_PACKS,
)


def oracle_endpoint_loss(model, logits, target, ends, margin=1.):
    groups = endpoint_competitors(target) + beginning_competitors(target)
    if not groups:
        return logits.sum() * 0
    gold = plan_path_score(model, logits, target, ends)
    losses = []
    for paths in groups:
        scores = torch.stack([plan_path_score(model, logits, path, ends) for path in paths])
        losses.append(F.softplus(margin + torch.logsumexp(scores, 0) - gold))
    return torch.stack(losses).mean()


def oracle_boundary_loss(boundaries, target):
    begin = ((target > 0) & (target < 4)).to(boundaries.dtype)
    losses = F.binary_cross_entropy_with_logits(boundaries[:, 0], begin, reduction="none")
    groups = [losses[begin.bool()].mean()] if begin.bool().any() else []
    if (~begin.bool()).any():
        groups.append(losses[~begin.bool()].mean())
    ending = []
    tags = target.detach().cpu().tolist()
    for first, tag in enumerate(tags):
        if tag not in (1, 2, 3):
            continue
        end = first + 1
        while end < len(tags) and tags[end] == 4:
            end += 1
        stop = F.softplus(-boundaries[end - 1, 1])
        if end - first > 1:
            stop = (stop + F.softplus(boundaries[first:end - 1, 1]).mean()) / 2
        ending.append(stop)
    if (target == 0).any():
        ending.append(F.softplus(-boundaries[target == 0, 1]).mean())
    return torch.stack(groups).mean() + torch.stack(ending).mean()


def oracle_wrong_path(model, logits, target, ends):
    """Independent sequential DP, tracking whether a path ever differs from gold."""
    start, _ = model.plan_potentials()
    tags = torch.arange(5, device=logits.device)
    first = start + logits[0]
    same = first.masked_fill(tags != target[0], -torch.inf)
    different = first.masked_fill(tags == target[0], -torch.inf)
    for index, edge in enumerate(model.plan_edge_potentials(ends), 1):
        incoming_same = (same[:, None] + edge).amax(0)
        incoming_different = (different[:, None] + edge).amax(0)
        different = torch.maximum(incoming_different,
                                  incoming_same.masked_fill(tags == target[index], -torch.inf)) + logits[index]
        same = (incoming_same + logits[index]).masked_fill(tags != target[index], -torch.inf)
    return (different + model.plan_terminal_potentials(ends)).amax()


def oracle_path_anchor(model, raw, logits, target, ends, margin, parent_ends):
    current_gold = plan_path_score(model, logits, target, ends)
    competing = oracle_wrong_path(model, logits, target, ends)
    with torch.no_grad():
        baseline_gold = plan_path_score(model, raw, target, parent_ends)
        baseline_competing = oracle_wrong_path(model, raw, target, parent_ends)
        required = (baseline_gold - baseline_competing).clamp(max=margin)
    return F.relu(required - (current_gold - competing)).square().max()


def sequential_sequence_loss(model, logits, target, ends):
    """Numerical reference for the original left-to-right CRF scan."""
    start, _ = model.plan_potentials()
    alpha = start + logits[0]
    for row, edge in zip(logits[1:], model.plan_edge_potentials(ends)):
        alpha = torch.logsumexp(alpha[:, None] + edge, 0) + row
    partition = torch.logsumexp(alpha + model.plan_terminal_potentials(ends), 0)
    return (partition - plan_path_score(model, logits, target, ends)) / len(target)


def benchmark_writer_operators(args):
    """Synthetic forward/backward timing; no model fitting or evaluation data."""
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process")
    torch.set_num_threads(1)
    torch.manual_seed(60017)
    device = torch.device(args.device)
    model = NeuralSlots().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    def gradients(loss, inputs):
        result = torch.autograd.grad(loss, inputs, allow_unused=True)
        return [torch.zeros_like(x) if g is None else g for x, g in zip(inputs, result)]

    results = []
    for count in (32, 96):
        target = torch.zeros(count, dtype=torch.long, device=device)
        target[3], target[4:count - 5] = 1, 4
        empty = torch.zeros_like(target)
        positive = build_writer_teacher(target)
        negative = build_writer_teacher(empty)
        raw = torch.randn(count, 5, device=device) / 3
        raw[torch.arange(count, device=device), target] += 4.
        negative_raw = torch.randn(count, 5, device=device) / 3
        negative_raw[:, 0] += 4.
        logits = (raw + torch.randn_like(raw) * .02).detach().requires_grad_()
        negative_logits = (negative_raw + torch.randn_like(raw) * .02).detach().requires_grad_()
        parent_ends = torch.randn(count, device=device) * .03
        ends = (parent_ends + torch.randn(count, device=device) * .02).requires_grad_()
        positive_gap = path_anchor_required_gaps(model, raw, parent_ends, positive, 1000.)
        negative_gap = path_anchor_required_gaps(model, negative_raw, parent_ends, negative, 1000.)
        cases = (
            ("endpoint", (logits, ends),
             lambda: endpoint_contrast_loss(model, logits, target, ends),
             lambda: endpoint_contrast_loss(model, logits, target, ends, teacher=positive)),
            ("boundary", (logits, ends),
             lambda: oracle_boundary_loss(torch.stack((logits[:, 1] - raw[:, 1], ends), 1), target),
             lambda: boundary_supervision_loss(torch.stack((logits[:, 1] - raw[:, 1], ends), 1), target, teacher=positive)),
            ("positive_anchor", (logits, ends),
             lambda: structured_path_anchor_loss(model, raw, logits, target, ends, 1000., parent_ends=parent_ends),
             lambda: structured_path_anchor_loss(model, None, logits, target, ends, 1000., teacher=positive, required_gap=positive_gap)),
            ("negative_anchor", (negative_logits, ends),
             lambda: structured_path_anchor_loss(model, negative_raw, negative_logits, empty, ends, 1000., parent_ends=parent_ends),
             lambda: structured_path_anchor_loss(model, None, negative_logits, empty, ends, 1000., teacher=negative, required_gap=negative_gap)),
            ("sequence", (logits, ends),
             lambda: sequential_sequence_loss(model, logits, target, ends),
             lambda: model.sequence_loss(logits, target, ends)),
        )
        for name, inputs, reference, optimized in cases:
            expected, actual = reference(), optimized()
            loss_error = float((expected.detach() - actual.detach()).abs().cpu())
            expected_grad, actual_grad = gradients(expected, inputs), gradients(actual, inputs)
            gradient_error = max(float((a - b).abs().max().cpu()) for a, b in zip(expected_grad, actual_grad))
            if loss_error > 1e-4 or gradient_error > 2e-5:
                raise AssertionError(f"{name} value/gradient oracle failed: {loss_error}/{gradient_error}")
            timings = {}
            for mode, function in (("uncached_or_sequential", reference), ("cached_or_parallel", optimized)):
                for _ in range(2):
                    gradients(function(), inputs)
                sync()
                began = time.perf_counter()
                for _ in range(args.iterations):
                    gradients(function(), inputs)
                sync()
                timings[mode] = (time.perf_counter() - began) * 1000 / args.iterations
            results.append(dict(operator=name, units=count, forward_backward_ms=timings,
                                speedup=timings["uncached_or_sequential"] / timings["cached_or_parallel"],
                                loss_error=loss_error, gradient_error=gradient_error))
    report = dict(scope="synthetic writer loss/gradient microbenchmark", device=str(device),
                  torch_version=torch.__version__, iterations=args.iterations,
                  training_run=False, quality_evaluated=False, results=results)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


class WriterEndpointContract(unittest.TestCase):
    def test_context_learning_updates_neighbors_and_preserves_current_word_weights(self):
        model = NeuralSlots()
        model.writer_mode = "writer-context"
        allowed = writer_train_keys("writer-context")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in allowed)
        with torch.no_grad():
            model.plan_boundary_raw.weight[:, INPUT:2 * INPUT].normal_(0, .1)
        centre = model.plan_boundary_raw.weight[:, INPUT:2 * INPUT].detach().clone()
        encoded, words = torch.randn(4, 64), torch.randn(4, INPUT)
        optimizer = torch.optim.AdamW([model.plan_boundary_raw.weight], lr=.01, weight_decay=0)
        for _ in range(3):
            optimizer.zero_grad()
            scores = model.plan_boundary_scores(encoded, words)
            F.softplus(-scores).mean().backward()
            freeze_boundary_center_gradient(model)
            self.assertTrue(torch.equal(model.plan_boundary_raw.weight.grad[:, INPUT:2 * INPUT],
                                        torch.zeros_like(centre)))
            self.assertGreater(float(model.plan_boundary_raw.weight.grad[:, :INPUT].abs().sum()), 0)
            self.assertGreater(float(model.plan_boundary_raw.weight.grad[:, 2 * INPUT:].abs().sum()), 0)
            optimizer.step()
        self.assertTrue(torch.equal(model.plan_boundary_raw.weight[:, INPUT:2 * INPUT], centre))
        self.assertGreater(float(model.plan_boundary_raw.weight[:, :INPUT].abs().sum()), 0)

    def setUp(self):
        torch.manual_seed(31031)
        torch.set_num_threads(1)

    def test_path_score_uses_native_emission_start_transition_end_formula(self):
        model = NeuralSlots()
        with torch.no_grad():
            model.plan_start.copy_(torch.arange(5).float() / 10)
            model.plan_end.copy_(torch.arange(5).float() / 7)
            model.plan_transition.copy_(torch.arange(25).reshape(5, 5).float() / 13)
        logits = torch.arange(25).reshape(5, 5).float() / 9
        tags = torch.tensor([0, 1, 4, 4, 0])
        expected = (logits[0, 0] + logits[1, 1] + logits[2, 4] + logits[3, 4] + logits[4, 0]
                    + model.plan_start[0] + model.plan_end[0]
                    + model.plan_transition[0, 1] + model.plan_transition[1, 4]
                    + model.plan_transition[4, 4] + model.plan_transition[4, 0])
        self.assertTrue(torch.equal(plan_path_score(model, logits, tags, torch.zeros(5)), expected))

    def test_competitors_preserve_actions_other_spans_and_valid_inside_grammar(self):
        model = NeuralSlots()
        target = torch.tensor([0, 1, 4, 4, 0, 2, 4, 0, 0, 0, 0])
        before = target.clone()
        groups = endpoint_competitors(target)
        self.assertEqual(len(groups), 2)
        for group in groups:
            for alternative in group:
                self.assertEqual(alternative[1], 1)
                self.assertEqual(alternative[5], 2)
                self.assertTrue(torch.isfinite(plan_path_score(model, torch.zeros(11, 5), alternative, torch.zeros(11))))
                values = alternative.tolist()
                self.assertNotEqual(values[0], 4)
                self.assertFalse(any(a == 0 and b == 4 for a, b in zip(values, values[1:])))
        self.assertTrue(torch.equal(target, before))

    def test_endpoint_gradients_penalize_both_truncation_and_greeting_overrun(self):
        model = NeuralSlots()
        target = torch.tensor([1, 4, 4, 4, 0, 0, 0, 0])
        logits = torch.zeros(8, 5)
        logits[3, 0] = 3.0  # Premature stop at the actual final source word.
        logits[4, 4] = 2.0  # Absorbing the first word outside the source.
        logits.requires_grad_()
        endpoint_contrast_loss(model, logits, target, torch.zeros(8)).backward()
        self.assertLess(float(logits.grad[3, 4]), 0)
        self.assertGreater(float(logits.grad[3, 0]), 0)
        self.assertLess(float(logits.grad[4, 0]), 0)
        self.assertGreater(float(logits.grad[4, 4]), 0)

    def test_endpoint_weight_does_not_dilute_with_longer_complete_facts(self):
        model = NeuralSlots()
        def gradient(length, objective):
            target = torch.tensor([1] + [4] * (length - 1) + [0] * 8)
            logits = torch.zeros(len(target), 5, requires_grad=True)
            loss = (endpoint_contrast_loss(model, logits, target, torch.zeros(len(target)))
                    if objective == "endpoint" else F.cross_entropy(logits, target))
            loss.backward()
            return float(logits.grad[length - 1, 4])
        short, long = gradient(6, "endpoint"), gradient(40, "endpoint")
        self.assertGreaterEqual(abs(long), abs(short) / 2)
        self.assertLess(abs(gradient(40, "token")), abs(gradient(6, "token")) / 3)

    def test_negative_observations_do_not_gain_continuation_pressure(self):
        model = NeuralSlots()
        logits = torch.randn(12, 5, requires_grad=True)
        target = torch.zeros(12, dtype=torch.long)
        loss = endpoint_contrast_loss(model, logits, target, torch.zeros(12))
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_write_boundaries_step_changes_only_five_tensors_and_no_query_binding(self):
        model = NeuralSlots()
        allowed = writer_train_keys("write-boundaries")
        self.assertEqual(set(allowed), {"plan_tags.weight", "plan_tags.bias", *SEQUENCE_KEYS})
        self.assertEqual(len(allowed), 5)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in allowed)
        model.writer_mode = "write-boundaries"
        model.endpoint_weight = 0.5
        body = "Hello. I store screws in a brass tin. Goodbye."
        context = observation_context("Iris", "2039-04-19 10:00 UTC")
        def units(text):
            return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        bu, cu = units(body), units(context)
        bx, cx = torch.randn(len(bu), INPUT) / 30, torch.randn(len(cu), INPUT) / 30
        item = dict(body=bx, context=cx, body_words=bx, context_words=cx,
                    body_units=bu, context_units=cu)
        row = dict(body=body, context=context, speaker="Iris", timestamp="2039-04-19 10:00 UTC",
                   spans=[dict(start=body.index("I store"), end=body.index(" Goodbye"), action=1)])
        def binding():
            encoded, _, _ = model.plan(bx, cx, bx)
            need, tags = model.query_reference(encoded.mean(0), source_words(item, len(encoded)), source_types(row, item, len(encoded)), raw_context=model.query_binding_context(item["body"], item["context"]))
            return encoded.detach().clone(), need.detach().clone(), tags.detach().clone()
        before_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        before_binding = binding()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        observation_loss(model, row, item).backward()
        optimizer.step()
        after_binding = binding()
        for old, new in zip(before_binding, after_binding):
            self.assertTrue(torch.equal(old, new))
        changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before_state[k])}
        self.assertTrue(changed)
        self.assertTrue(changed <= set(allowed))
        self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters() if name not in allowed))

    def test_diversity_sources_are_complete_fit_only_and_negative_quotes_stay_empty(self):
        from prepare_neural_dialogue import NAMES
        rows = boundary_diversity_fit_rows()
        positives = [r for r in rows if r["spans"]]
        fiction = [r for r in rows if r["tag"] == "writer-complex-fiction"]
        greetings = [r for r in rows if r["tag"] == "writer-complex-greeting"]
        self.assertEqual((len(positives), len(fiction), len(greetings)), (32, 32, 16))
        self.assertEqual(len(set(FACTS)), 32)
        self.assertTrue(set(FIT_NAMES) <= set(NAMES["fit"]))
        self.assertTrue(all(set(FIT_NAMES).isdisjoint(names) for split, names in NAMES.items() if split != "fit"))
        selected = set()
        for row in positives:
            span = row["spans"][0]
            raw = row["body"].encode()
            fragment = raw[span["start"]:span["end"]].decode()
            self.assertIn(fragment, FACTS)
            self.assertTrue(fragment.endswith("."))
            selected.add(fragment)
            self.assertTrue(span["start"] > 0 and span["end"] < len(raw))
            ref = span["ref_ranges"][0]
            self.assertEqual(row["context"].encode()[ref["start"]:ref["end"]].decode(), row["speaker"])
            self.assertEqual(span["references"], [row["speaker"]])
            self.assertEqual(row["whole_refs"], [row["speaker"]])
        self.assertEqual(selected, set(FACTS))
        self.assertTrue(all(r["spans"] == [] and not r["whole_need"] and r["whole_refs"] == [] for r in fiction + greetings))
        self.assertEqual(rows, boundary_diversity_fit_rows())

    def test_diversity_quota_survives_large_replay_without_losing_rare_actions(self):
        diversity = boundary_diversity_fit_rows()
        positive = [r for r in diversity if r["spans"]]
        negative = [r for r in diversity if not r["spans"]]
        for action in (1, 2, 3):
            positive += [dict(tag="regular", body=f"regular-{action}-{i}", spans=[dict(action=action)]) for i in range(1000)]
        negative += [dict(tag="regular-negative", body=f"negative-{i}", spans=[]) for i in range(1000)]
        rng = random.Random(3701)
        for _ in range(10):
            batch = sample_action_balanced_batch(positive, negative, rng)
            self.assertEqual(len(batch), 20)
            self.assertEqual(sum(bool(r.get("boundary_diversity")) and bool(r["spans"]) for r in batch), 4)
            self.assertEqual(sum(r["tag"] == "writer-complex-fiction" for r in batch), 1)
            self.assertEqual(sum(r["tag"] == "writer-complex-greeting" for r in batch), 1)
            self.assertGreaterEqual(sum(any(s["action"] == 2 for s in r["spans"]) for r in batch), 2)
            self.assertGreaterEqual(sum(any(s["action"] == 3 for s in r["spans"]) for r in batch), 2)

    def test_checkpoint_gate_preserves_both_reference_metrics_and_unsafe_floor(self):
        initial = dict(reference_exact=.98, whole_reference_exact=1., ambiguous_unsafe_writes=0)
        safe = dict(reference_exact=.99, whole_reference_exact=1., ambiguous_unsafe_writes=0)
        self.assertTrue(writer_reference_safe(safe, initial))
        self.assertFalse(writer_reference_safe(dict(safe, reference_exact=.97), initial))
        self.assertFalse(writer_reference_safe(dict(safe, whole_reference_exact=.99), initial))
        self.assertFalse(writer_reference_safe(dict(safe, ambiguous_unsafe_writes=1), initial))

    def test_dynamic_path_score_applies_edges_only_after_source_and_terminal_source(self):
        model = NeuralSlots()
        logits = torch.zeros(4, 5)
        ends = torch.tensor([2.0, -1.0, 3.0, -2.0], requires_grad=True)
        tags = torch.tensor([0, 1, 4, 0])
        static = plan_path_score(model, logits, tags, torch.zeros(4))
        continuing, stopping = model.plan_boundary_potentials(ends)
        expected = static + continuing[1] + stopping[2]
        actual = plan_path_score(model, logits, tags, ends)
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertEqual(float(ends.grad[0]), 0)
        self.assertEqual(float(ends.grad[3]), 0)
        self.assertLess(float(ends.grad[1]), 0)
        self.assertGreater(float(ends.grad[2]), 0)
        single = torch.tensor([1])
        terminal = plan_path_score(model, torch.zeros(1, 5), single, torch.tensor([2.0]))
        zero_terminal = plan_path_score(model, torch.zeros(1, 5), single, torch.zeros(1))
        self.assertTrue(torch.allclose(terminal - zero_terminal, model.plan_boundary_potentials(torch.tensor([2.0]))[1][0]))

    def test_full_competitors_include_long_clause_trims_and_beginnings_both_directions(self):
        target = torch.tensor([0, 0, 1] + [4] * 10 + [0] * 8 + [2, 4, 0])
        ends = endpoint_competitors(target)[0]
        self.assertTrue(any(path[5:13].eq(0).all() for path in ends))
        self.assertTrue(any(path[13:21].eq(4).all() for path in ends))
        begins = beginning_competitors(target)[0]
        self.assertTrue(any(int(path[0]) == 1 for path in begins))
        self.assertTrue(any(int(path[9]) == 1 for path in begins))
        model = NeuralSlots()
        for group_index, group in enumerate(endpoint_competitors(target) + beginning_competitors(target)):
            for path in group:
                self.assertTrue(torch.isfinite(plan_path_score(model, torch.zeros(len(target), 5), path, torch.zeros(len(target)))))
                other_begin, action = (21, 2) if group_index in (0, 2) else (2, 1)
                self.assertEqual(int(path[other_begin]), action)

    def test_boundary_teacher_has_stop_continue_outside_and_begin_gradients(self):
        target = torch.tensor([0, 1, 4, 4, 0, 2, 0])
        boundaries = torch.zeros(7, 2, requires_grad=True)
        boundary_supervision_loss(boundaries, target).backward()
        for index in (1, 5):
            self.assertLess(float(boundaries.grad[index, 0]), 0)
        for index in (0, 2, 3, 4, 6):
            self.assertGreater(float(boundaries.grad[index, 0]), 0)
        for index in (1, 2):
            self.assertGreater(float(boundaries.grad[index, 1]), 0)
        for index in (0, 3, 4, 5, 6):
            self.assertLess(float(boundaries.grad[index, 1]), 0)
        def endpoint_gradient(length):
            labels = torch.tensor([1] + [4] * (length - 1) + [0] * 8)
            values = torch.zeros(len(labels), 2, requires_grad=True)
            boundary_supervision_loss(values, labels).backward()
            return float(values.grad[length - 1, 1])
        self.assertAlmostEqual(endpoint_gradient(6), endpoint_gradient(40), places=6)
        empty = torch.zeros(8, 2, requires_grad=True)
        boundary_supervision_loss(empty, torch.zeros(8, dtype=torch.long)).backward()
        self.assertTrue((empty.grad[:, 0] > 0).all())
        self.assertTrue((empty.grad[:, 1] < 0).all())

    def test_writer_edge_step_changes_only_five_tensors_and_preserves_both_bindings(self):
        model = NeuralSlots()
        allowed = writer_train_keys("writer-edge")
        self.assertEqual(set(allowed), {"plan_boundary.0.weight", "plan_boundary.0.bias", "plan_boundary.2.weight", "plan_boundary.2.bias", "plan_boundary_raw.weight"})
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in allowed)
        model.writer_mode, model.endpoint_weight, model.boundary_weight = "writer-edge", .5, 1.
        model.boundary_parent_parameters = snapshot_parent_boundary(model)
        body = "Hello. I keep bolts in a small tray until they are clean. Goodbye."
        context = observation_context("Ada", "2047-03-18 08:45 UTC")
        def units(text):
            return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        bu, cu = units(body), units(context)
        bx, cx = torch.randn(len(bu), INPUT) / 30, torch.randn(len(cu), INPUT) / 30
        item = dict(body=bx, context=cx, body_words=bx, context_words=cx, body_units=bu, context_units=cu)
        row = dict(body=body, context=context, speaker="Ada", timestamp="2047-03-18 08:45 UTC",
                   spans=[dict(start=body.index("I keep"), end=body.index(" Goodbye"), action=1)])
        def binding():
            encoded, _, _ = model.plan(bx, cx, bx)
            query = model.query_reference(encoded.mean(0), source_words(item, len(encoded)), source_types(row, item, len(encoded)), raw_context=model.query_binding_context(item["body"], item["context"]))
            write = model.plan_reference(encoded[1:-1].mean(0), source_words(item, 1), source_types(row, item, 1))
            return [encoded.detach().clone()] + [x.detach().clone() for x in query + write]
        before_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        before = binding()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.002)
        for _ in range(2):
            optimizer.zero_grad()
            observation_loss(model, row, item).backward()
            optimizer.step()
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, binding())))
        changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before_state[k])}
        self.assertEqual(changed, set(allowed))
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if name not in allowed))

    def test_independent_boundary_worlds_have_exact_full_spans_and_empty_controls(self):
        groups = {split: boundary_generalization_curriculum(split) for split in ("fit", "development")}
        self.assertTrue(set(BOUNDARY_NAMES["fit"]).isdisjoint(BOUNDARY_NAMES["development"]))
        self.assertTrue(set(BOUNDARY_FACTS["fit"]).isdisjoint(BOUNDARY_FACTS["development"]))
        self.assertTrue({r["body"] for r in groups["fit"]}.isdisjoint(r["body"] for r in groups["development"]))
        for split, rows in groups.items():
            self.assertEqual(len(rows), 72)
            self.assertEqual(sum(bool(r["spans"]) for r in rows), 48)
            self.assertEqual(len({r["source_id"] for r in rows}), 72)
            for row in rows:
                self.assertEqual(row["boundary_world"], split)
                if not row["spans"]:
                    self.assertEqual(row["whole_refs"], [])
                    continue
                span = row["spans"][0]
                fact = row["body"].encode()[span["start"]:span["end"]].decode()
                self.assertIn(fact, BOUNDARY_FACTS[split])
                self.assertEqual(span["references"], [row["speaker"]])
                ref = span["ref_ranges"][0]
                self.assertEqual(row["context"].encode()[ref["start"]:ref["end"]].decode(), row["speaker"])
            self.assertEqual(rows, boundary_generalization_curriculum(split))

    def test_independent_selector_cannot_trade_away_original_fact_f1(self):
        initial = dict(reference_exact=1., whole_reference_exact=1., ambiguous_unsafe_writes=0,
                       committed_fact_f1=.83, plan_exact=.9, source_exact=.9, action_exact=.9)
        self.assertTrue(writer_edge_candidate_safe(initial, initial))
        self.assertFalse(writer_edge_candidate_safe(dict(initial, committed_fact_f1=.82), initial))
        self.assertFalse(writer_edge_candidate_safe(dict(initial, reference_exact=.99), initial))

    def test_boundary_attachment_preserves_original_development_and_never_trains_new_dev(self):
        from prepare_neural_colloquial import colloquial_curriculum, NAMES
        from prepare_neural_dialogue import NAMES as DIALOGUE_NAMES
        from prepare_neural_identity import ACTORS
        source = {"fit": [dict(source_id="original-fit")], "development": colloquial_curriculum("development")}
        before = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        augmented = with_boundary_generalization(source)
        after = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        self.assertEqual(before, after)
        self.assertEqual(augmented["development"], source["development"])
        self.assertEqual(len(augmented["fit"]), 73)
        self.assertEqual(len(augmented["boundary_development"]), 72)
        fit_ids = {r["source_id"] for r in augmented["fit"]}
        self.assertTrue(fit_ids.isdisjoint(r["source_id"] for r in augmented["boundary_development"]))
        known_fit = set(NAMES["fit"]) | set(DIALOGUE_NAMES["fit"]) | set(ACTORS)
        self.assertTrue(set(BOUNDARY_NAMES["development"]).isdisjoint(known_fit))

    def test_narrative_sources_roundtrip_complete_three_facts_and_exclude_question_gaps(self):
        source_counts = {"fit": (24, 8, 8), "development": (8, 4, 0)}
        for split, expected in source_counts.items():
            rows = narrative_curriculum(split)
            self.assertEqual(rows, narrative_curriculum(split))
            self.assertEqual((sum(r["tag"] == "writer-narrative-facts" for r in rows),
                              sum(r["tag"] == "writer-narrative-questions" for r in rows),
                              sum(r["tag"] == "writer-narrative-fiction" for r in rows)), expected)
            for row in rows:
                with self.subTest(split=split, source_id=row["source_id"]):
                    raw = row["body"].encode()
                    _, units = word_features(row["body"], dict(x=torch.zeros(1, INPUT), bounds=[(0, len(raw))]))
                    target = plan_targets(row, units, torch.device("cpu"))
                    predicted = decode_plan(target, units)
                    self.assertEqual([(a, b, action) for a, b, action, _, _ in predicted],
                                     [(s["start"], s["end"], s["action"]) for s in row["spans"]])
                    self.assertLessEqual(len(row["body"].split()), 140)
                    if row["spans"]:
                        self.assertEqual(len(row["spans"]), 3)
                        self.assertEqual(sorted(s["narrative_fact_index"] for s in row["spans"]), [0, 1, 2])
                        pack = int(row["source_id"].split("-")[-3])
                        facts, questions = NARRATIVE_PACKS[split][pack]
                        for span in row["spans"]:
                            self.assertEqual(raw[span["start"]:span["end"]].decode(), facts[span["narrative_fact_index"]])
                            ref = span["ref_ranges"][0]
                            self.assertEqual(row["context"].encode()[ref["start"]:ref["end"]].decode(), row["speaker"])
                            self.assertEqual(span["references"], [row["speaker"]])
                        other = NARRATIVE_NAMES[split][(pack + 1) % len(NARRATIVE_NAMES[split])]
                        checked = 0
                        for question in questions:
                            question = question.format(other=other).encode()
                            start = raw.find(question)
                            if start < 0:
                                continue
                            labels = [int(target[i]) for i, (a, b) in enumerate(units)
                                      if b > start and a < start + len(question)]
                            self.assertTrue(labels)
                            self.assertEqual(set(labels), {0})
                            checked += 1
                        self.assertGreaterEqual(checked, 2)
                    else:
                        self.assertTrue(torch.equal(target, torch.zeros_like(target)))
        # UTF-8 preambles exercise byte offsets rather than character counts.
        self.assertIn("—", narrative_curriculum("fit")[0]["body"])

    def test_narrative_fit_attachment_keeps_public_development_separate_and_inputs_unchanged(self):
        from prepare_neural_colloquial import NAMES
        from prepare_neural_dialogue import NAMES as DIALOGUE_NAMES
        from prepare_neural_identity import ACTORS
        source = dict(fit=[dict(source_id="original-fit")],
                      development=[dict(source_id="fixed-development")],
                      boundary_development=[dict(source_id="existing-boundary")])
        before = json.dumps(source, sort_keys=True)
        attached = with_narrative_diversity(source)
        self.assertEqual(json.dumps(source, sort_keys=True), before)
        self.assertEqual(attached["development"], source["development"])
        self.assertEqual(attached["boundary_development"], source["boundary_development"])
        self.assertEqual(len(attached["fit"]), 41)
        self.assertEqual(len(attached["narrative_development"]), 12)
        self.assertTrue({r["source_id"] for r in attached["fit"]}.isdisjoint(
            r["source_id"] for r in attached["narrative_development"]))
        fit_facts = {text for facts, questions in NARRATIVE_PACKS["fit"] for text in facts + questions}
        dev_facts = {text for facts, questions in NARRATIVE_PACKS["development"] for text in facts + questions}
        self.assertTrue(fit_facts.isdisjoint(dev_facts))
        known_fit = set(NAMES["fit"]) | set(DIALOGUE_NAMES["fit"]) | set(ACTORS)
        self.assertTrue(set(NARRATIVE_NAMES["fit"]) <= known_fit)
        self.assertTrue(set(NARRATIVE_NAMES["development"]).isdisjoint(known_fit))
        self.assertTrue(all(r["evaluation_role"] == "public-development-not-independent-acceptance"
                            for r in attached["narrative_development"]))
        with self.assertRaisesRegex(ValueError, "already attached"):
            with_narrative_diversity(attached)

    def test_narrative_batch_quota_preserves_boundary_and_lifecycle_replay(self):
        curriculum = narrative_curriculum("fit") + boundary_generalization_curriculum("fit")
        positives = [r for r in curriculum if r["spans"]]
        negatives = [r for r in curriculum if not r["spans"]]
        for action in (1, 2, 3):
            positives += [dict(tag="regular", body=f"regular-{action}-{i}", spans=[dict(action=action)]) for i in range(1000)]
        negatives += [dict(tag="regular-negative", body=f"negative-{i}", spans=[]) for i in range(1000)]
        for seed in range(10):
            batch = sample_action_balanced_batch(positives, negatives, random.Random(seed))
            self.assertEqual(len(batch), 24)
            self.assertEqual(sum(r["tag"] == "writer-narrative-facts" for r in batch), 2)
            self.assertEqual(sum(r["tag"] == "writer-narrative-questions" for r in batch), 1)
            self.assertEqual(sum(r["tag"] == "writer-narrative-fiction" for r in batch), 1)
            self.assertEqual(sum(bool(r.get("boundary_generalization")) and bool(r["spans"]) for r in batch), 4)
            self.assertGreaterEqual(sum(any(s["action"] == 2 for s in r["spans"]) for r in batch), 2)
            self.assertGreaterEqual(sum(any(s["action"] == 3 for s in r["spans"]) for r in batch), 2)
            self.assertEqual(len({id(r) for r in batch}), len(batch))

    def test_public_narrative_selection_rejects_regression_even_when_boundary_improves(self):
        initial = dict(committed_fact_f1=.85, plan_exact=.9, source_exact=.9, action_exact=.9, reference_exact=.98,
                       whole_reference_exact=1., ambiguous_unsafe_writes=0, rows=180)
        boundary_initial = dict(initial, committed_fact_f1=.6)
        narrative_initial = dict(initial, committed_fact_f1=.4)
        boundary = dict(boundary_initial, committed_fact_f1=.8)
        narrative = dict(narrative_initial, committed_fact_f1=.5)
        self.assertTrue(writer_development_safe(initial, initial, boundary, boundary_initial, narrative, narrative_initial))
        self.assertGreater(writer_development_score(initial, boundary, narrative),
                           writer_development_score(initial, boundary_initial, narrative_initial))
        self.assertFalse(writer_development_safe(initial, initial, boundary, boundary_initial,
                                                 dict(narrative, committed_fact_f1=.3), narrative_initial))
        self.assertFalse(writer_development_safe(initial, initial, dict(boundary, committed_fact_f1=.5), boundary_initial,
                                                 dict(narrative, committed_fact_f1=.9), narrative_initial))
        self.assertFalse(writer_development_safe(dict(initial, committed_fact_f1=.84), initial,
                                                 boundary, boundary_initial, narrative, narrative_initial))
        self.assertFalse(writer_development_safe(dict(initial, whole_reference_exact=.99), initial,
                                                 boundary, boundary_initial, narrative, narrative_initial))
        self.assertEqual(writer_development_score(initial), writer_score(initial))
        self.assertEqual(writer_development_score(initial, boundary), boundary["committed_fact_f1"])
        self.assertEqual(writer_development_safe(initial, initial, boundary, boundary_initial),
                         writer_edge_candidate_safe(initial, initial))

    def test_fit_anchor_precomputation_uses_full_parent_and_caches_unique_teachers(self):
        model = NeuralSlots()
        with torch.no_grad():
            model.plan_tags.weight.zero_()
            model.plan_tags.bias.copy_(torch.tensor([-1., 4., -2., -2., 5.]))
            model.plan_transition.zero_()
            model.plan_start.zero_()
            model.plan_end.zero_()
            model.plan_boundary[2].bias.copy_(torch.tensor([10., 10.]))
        body = "I keep twine."
        context = observation_context("Ada", "2047-03-18 08:45 UTC")
        def item(text):
            bounds = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
            return dict(x=torch.zeros(len(bounds), INPUT), bounds=torch.tensor(bounds))
        cache = {body: item(body), context: item(context)}
        correct = dict(body=body, context=context,
                       spans=[dict(start=m.start(), end=m.end(), action=1) for m in re.finditer(r"\S+", body)])
        incorrect = dict(body=body, context=context, spans=[dict(start=0, end=len(body), action=1)])
        with mock.patch.object(model, "plan_labels", wraps=model.plan_labels) as decoder:
            flags, counts = precompute_fit_correctness(model, [correct, correct, incorrect], cache)
        self.assertEqual(decoder.call_count, 2)
        self.assertTrue(flags[writer_observation_key(correct)])
        self.assertFalse(flags[writer_observation_key(incorrect)])
        self.assertEqual(counts["baselinefit_correct_count"], 2)
        self.assertEqual(counts["baselinefit_incorrect_count"], 1)
        self.assertEqual(counts["baselinefit_unique_count"], 2)
        self.assertEqual(counts["split"], "fit")
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_correct_anchor_suppresses_task_teacher_and_training_does_not_decode(self):
        model = NeuralSlots()
        for name, p in model.named_parameters():
            p.requires_grad_(name in writer_train_keys("writer-edge"))
        model.writer_mode, model.endpoint_weight, model.boundary_weight = "writer-edge", .5, 1.
        model.boundary_anchor_weight = 2.
        model.writer_fit_teachers = {}
        model.boundary_parent_parameters = snapshot_parent_boundary(model)
        with torch.no_grad():
            model.plan_boundary[2].bias.copy_(torch.tensor([2., -3.]))
        body = "Hello. I keep twine. Goodbye."
        context = observation_context("Ada", "2047-03-18 08:45 UTC")
        def units(text):
            return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        bu, cu = units(body), units(context)
        bx, cx = torch.randn(len(bu), INPUT) / 30, torch.randn(len(cu), INPUT) / 30
        item = dict(body=bx, context=cx, body_words=bx, context_words=cx, body_units=bu, context_units=cu)
        row = dict(body=body, context=context, spans=[dict(start=body.index("I keep"), end=body.index(" Goodbye"), action=1)])
        key = writer_observation_key(row)
        model.boundary_fit_correct = {key: True}
        with mock.patch("train_neural_dialogue.plan_targets", side_effect=AssertionError("correct rows must not repeat task teaching")), mock.patch.object(model, "plan_labels", side_effect=AssertionError("fit flags must be reused")):
            loss = batch_loss(model, [row], {(body, context): item})
        self.assertAlmostEqual(float(loss.detach()), 13., places=5)
        self.assertEqual(model.boundary_batch_stats["anchored_rows"], 1)
        self.assertEqual(model.boundary_batch_stats["learning_rows"], 0)
        loss.backward()
        self.assertTrue(torch.allclose(model.plan_boundary[2].bias.grad, torch.tensor([4., -6.])))
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if name not in writer_train_keys("writer-edge")))
        model.boundary_fit_correct[key] = False
        # Sparse correction decodes the frozen parent once on a cold teacher;
        # the training hot loop must reuse that result, never decode current.
        observation_loss(model, row, item)
        with mock.patch.object(model, "plan_labels", side_effect=AssertionError("training must not re-decode the baseline")):
            task = observation_loss(model, row, item)
        model.boundary_anchor_weight = 0.
        ordinary = observation_loss(model, row, item)
        self.assertTrue(torch.equal(task, ordinary))
        self.assertGreater(float(task.detach()), 0.)

    def test_sparse_correction_targets_missing_active_tail_but_never_gold_outside_end(self):
        model = NeuralSlots()
        gold = torch.tensor([0, 1, 4, 4, 4, 4, 0])
        teacher = build_writer_teacher(gold)
        # The action subtype differs while the beginning is aligned. The
        # scalar new beginning cue cannot teach the frozen action classifier.
        parent = torch.tensor([0, 2, 4, 0, 0, 0, 0])
        teacher.update(parent_logits=torch.zeros(7, 5), parent_ends=torch.zeros(7),
                       parent_boundaries=torch.randn(7, 2) / 10)
        with mock.patch.object(model, "plan_labels", return_value=parent) as decode:
            parent_boundary_correction_teacher(model, teacher)
            parent_boundary_correction_teacher(model, teacher)
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(teacher["correction_indices"][0].tolist(), [])
        self.assertEqual(teacher["correction_indices"][1].tolist(), [2, 3, 4, 5])
        self.assertEqual(teacher["correction_targets"][1].tolist(), [0., 0., 0., 1.])
        current = teacher["parent_boundaries"].clone().requires_grad_()
        loss = boundary_residual_supervision_loss(current, teacher)
        grad = torch.autograd.grad(loss, current)[0]
        self.assertTrue(torch.equal(grad[:, 0], torch.zeros(7)))
        self.assertTrue((grad[2:5, 1] > 0).all())
        self.assertLess(float(grad[5, 1]), 0.)
        self.assertTrue(torch.equal(grad[[0, 1, 6], 1], torch.zeros(3)))
        changed = teacher["parent_boundaries"].clone()
        changed[1, 0] += .25
        changed[6, 1] += .1
        changed.requires_grad_()
        retained = torch.autograd.grad(boundary_residual_supervision_loss(changed, teacher), changed)[0]
        self.assertGreater(float(retained[1, 0]), 0.)
        self.assertGreater(float(retained[6, 1]), 0.)

    def test_sparse_begin_uses_decoded_mismatch_and_all_matched_outputs_are_preserved(self):
        model = NeuralSlots()
        gold = torch.tensor([0, 1, 4, 0])
        teacher = build_writer_teacher(gold)
        teacher.update(parent_logits=torch.zeros(4, 5), parent_ends=torch.zeros(4),
                       parent_boundaries=torch.randn(4, 2) / 10)
        with mock.patch.object(model, "plan_labels", return_value=torch.tensor([1, 4, 4, 0])):
            parent_boundary_correction_teacher(model, teacher)
        self.assertEqual(teacher["correction_indices"][0].tolist(), [0, 1])
        self.assertEqual(teacher["correction_targets"][0].tolist(), [0., 1.])
        current = teacher["parent_boundaries"].clone().requires_grad_()
        grad = torch.autograd.grad(boundary_residual_supervision_loss(current, teacher), current)[0]
        self.assertGreater(float(grad[0, 0]), 0.)
        self.assertLess(float(grad[1, 0]), 0.)
        self.assertTrue(torch.equal(grad[2:, 0], torch.zeros(2)))
        self.assertTrue(torch.equal(grad[:, 1], torch.zeros(4)))

    def test_missing_tail_stop_gradient_is_not_diluted_by_continue_length(self):
        model = NeuralSlots()
        for count in (8, 64, 128):
            with self.subTest(selected_tail_length=count):
                gold = torch.tensor([1] + [4] * count)
                parent = torch.tensor([1, 4] + [0] * (count - 1))
                teacher = build_writer_teacher(gold)
                teacher.update(parent_logits=torch.zeros(count + 1, 5),
                               parent_ends=torch.zeros(count + 1),
                               parent_boundaries=torch.zeros(count + 1, 2, dtype=torch.double))
                with mock.patch.object(model, "plan_labels", return_value=parent):
                    parent_boundary_correction_teacher(model, teacher)
                self.assertEqual(teacher["correction_class_counts"], ((0, 0), (count - 1, 1)))
                current = teacher["parent_boundaries"].clone().requires_grad_()
                loss = boundary_residual_supervision_loss(current, teacher)
                grad = torch.autograd.grad(loss, current, retain_graph=True)[0]
                old = F.binary_cross_entropy_with_logits(
                    current[teacher["correction_indices"][1], 1],
                    teacher["correction_targets"][1].to(current.dtype))
                old_grad = torch.autograd.grad(old, current)[0]
                self.assertAlmostEqual(float(loss.detach()), float(torch.log(torch.tensor(2., dtype=torch.double))), places=12)
                self.assertAlmostEqual(float(old_grad[-1, 1]), -.5 / count, places=12)
                self.assertAlmostEqual(float(grad[-1, 1]), -.25, places=12)
                self.assertAlmostEqual(float(grad[1:-1, 1].sum()), .25, places=12)
                self.assertTrue(torch.equal(grad[:, 0], torch.zeros(count + 1, dtype=torch.double)))
                self.assertEqual(float(grad[0, 1]), 0.)

    def test_sparse_begin_positive_and_negative_classes_have_equal_weight(self):
        model = NeuralSlots()
        for count in (8, 64, 128):
            with self.subTest(selected_begin_count=count):
                gold = torch.tensor([1] + [0] * (count - 1))
                parent = torch.tensor([0] + [1] * (count - 1))
                teacher = build_writer_teacher(gold)
                teacher.update(parent_logits=torch.zeros(count, 5), parent_ends=torch.zeros(count),
                               parent_boundaries=torch.zeros(count, 2, dtype=torch.double))
                with mock.patch.object(model, "plan_labels", return_value=parent):
                    parent_boundary_correction_teacher(model, teacher)
                self.assertEqual(teacher["correction_class_counts"][0], (count - 1, 1))
                current = teacher["parent_boundaries"].clone().requires_grad_()
                grad = torch.autograd.grad(boundary_residual_supervision_loss(current, teacher), current)[0]
                self.assertAlmostEqual(float(grad[0, 0]), -.25, places=12)
                self.assertAlmostEqual(float(grad[1:, 0].sum()), .25, places=12)
                # False writes on gold outside never receive an end BCE target.
                self.assertTrue(torch.equal(grad[1:, 1], torch.zeros(count - 1, dtype=torch.double)))

    def test_sparse_single_class_uses_ordinary_mean_without_half_weight(self):
        model = NeuralSlots()
        cases = (
            ("begin_positive", [1, 0, 1], [0, 0, 0], 0, 1),
            ("begin_negative", [0, 0, 0], [1, 0, 1], 0, 0),
            ("end_positive", [1, 4, 0], [1, 4, 4], 1, 1),
            ("end_negative", [1, 4, 4], [1, 0, 1], 1, 0),
        )
        for name, gold, parent, channel, desired in cases:
            with self.subTest(case=name):
                teacher = build_writer_teacher(torch.tensor(gold))
                teacher.update(parent_logits=torch.zeros(3, 5), parent_ends=torch.zeros(3),
                               parent_boundaries=torch.randn(3, 2, dtype=torch.double))
                with mock.patch.object(model, "plan_labels", return_value=torch.tensor(parent)):
                    parent_boundary_correction_teacher(model, teacher)
                counts = teacher["correction_class_counts"][channel]
                self.assertEqual(counts[1 - desired], 0)
                self.assertGreater(counts[desired], 0)
                current = teacher["parent_boundaries"].clone().requires_grad_()
                indices = teacher["correction_class_indices"][channel][desired]
                actual = boundary_residual_supervision_loss(current, teacher)
                expected = F.binary_cross_entropy_with_logits(
                    current[indices, channel], torch.full_like(current[indices, channel], desired))
                actual_grad = torch.autograd.grad(actual, current, retain_graph=True)[0]
                expected_grad = torch.autograd.grad(expected, current)[0]
                self.assertTrue(torch.allclose(actual_grad[indices, channel],
                                               expected_grad[indices, channel], atol=1e-15, rtol=1e-15))

    def test_empty_sparse_classes_keep_parent_mse_and_empty_row_zero(self):
        model = NeuralSlots()
        for tags in ([0, 1, 4, 0], []):
            with self.subTest(tags=tags):
                target = torch.tensor(tags, dtype=torch.long)
                teacher = build_writer_teacher(target)
                parent = torch.randn(len(tags), 2, dtype=torch.double)
                teacher.update(parent_logits=torch.zeros(len(tags), 5), parent_ends=torch.zeros(len(tags)),
                               parent_boundaries=parent)
                with mock.patch.object(model, "plan_labels", return_value=target):
                    parent_boundary_correction_teacher(model, teacher)
                self.assertEqual(teacher["correction_class_counts"], ((0, 0), (0, 0)))
                current = parent.clone().requires_grad_()
                loss = boundary_residual_supervision_loss(current, teacher)
                grad = torch.autograd.grad(loss, current)[0]
                self.assertEqual(float(loss.detach()), 0.)
                self.assertTrue(torch.equal(grad, torch.zeros_like(current)))
                if tags:
                    changed = (parent + torch.randn_like(parent) / 10).requires_grad_()
                    actual = boundary_residual_supervision_loss(changed, teacher)
                    expected = (changed - parent).square().mean()
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertTrue(torch.equal(torch.autograd.grad(actual, changed)[0],
                                                2 * (changed.detach() - parent) / changed.numel()))

    def test_cached_sparse_loss_never_reads_or_transfers_device_labels(self):
        model = NeuralSlots()
        teacher = build_writer_teacher(torch.tensor([0, 1, 4, 4, 4, 0]))
        teacher.update(parent_logits=torch.zeros(6, 5), parent_ends=torch.zeros(6),
                       parent_boundaries=torch.zeros(6, 2))
        with mock.patch.object(model, "plan_labels", return_value=torch.tensor([1, 4, 4, 0, 0, 0])):
            parent_boundary_correction_teacher(model, teacher)
        self.assertTrue(all(isinstance(count, int) for channel in teacher["correction_class_counts"] for count in channel))
        current = (torch.randn(6, 2) / 10).requires_grad_()
        expected = boundary_residual_supervision_loss(current, teacher)
        expected_grad = torch.autograd.grad(expected, current)[0]
        with mock.patch.object(model, "plan_labels", side_effect=AssertionError("parent decode is cold-only")), \
             mock.patch.object(torch.Tensor, "tolist", side_effect=AssertionError("no device label list")), \
             mock.patch.object(torch.Tensor, "item", side_effect=AssertionError("no device scalar read")), \
             mock.patch.object(torch.Tensor, "any", side_effect=AssertionError("no device class inspection")), \
             mock.patch.object(torch.Tensor, "nonzero", side_effect=AssertionError("class indices are cached")), \
             mock.patch.object(torch.Tensor, "cpu", side_effect=AssertionError("no hot transfer")):
            self.assertIs(parent_boundary_correction_teacher(model, teacher), teacher)
            actual = boundary_residual_supervision_loss(current, teacher)
            actual_grad = torch.autograd.grad(actual, current)[0]
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_grad, expected_grad))

    def test_sparse_preservation_requires_parent_gold_source_membership_agreement(self):
        model = NeuralSlots()
        gold = torch.tensor([0, 1, 4, 4, 0, 0])
        parent = torch.tensor([1, 4, 4, 0, 1, 4])
        teacher = build_writer_teacher(gold)
        teacher.update(parent_logits=torch.zeros(6, 5), parent_ends=torch.zeros(6),
                       parent_boundaries=torch.randn(6, 2, dtype=torch.double))
        with mock.patch.object(model, "plan_labels", return_value=parent):
            parent_boundary_correction_teacher(model, teacher)
        self.assertEqual(teacher["correction_indices"][0].tolist(), [0, 1, 4])
        self.assertEqual(teacher["correction_indices"][1].tolist(), [2, 3])
        self.assertEqual(teacher["preservation_indices"].tolist(), [3, 4])
        # These Boolean cues are not selected BCE errors, but their source
        # membership disagrees. Neither false outside source nor missing
        # inside source may receive a misleading matched-output clamp.
        unlabelled = [1, 6, 9, 10, 11]
        current = teacher["parent_boundaries"].clone()
        current.flatten()[unlabelled] += .75
        current[1, 1] += .2
        current[2, 0] += .3
        current.requires_grad_()
        grad = torch.autograd.grad(boundary_residual_supervision_loss(current, teacher), current)[0]
        self.assertTrue(torch.equal(grad.flatten()[unlabelled], torch.zeros(len(unlabelled), dtype=current.dtype)))
        self.assertAlmostEqual(float(grad[1, 1]), .2, places=12)
        self.assertAlmostEqual(float(grad[2, 0]), .3, places=12)
        # Gold outside ends never gain a BCE label as a result of this fix.
        self.assertTrue(torch.equal(grad[[0, 4, 5], 1], torch.zeros(3, dtype=current.dtype)))

    def test_direct_raw_has_first_step_gradient_with_zero_old_output_weights(self):
        model = NeuralSlots()
        self.assertEqual(tuple(model.plan_boundary_raw.weight.shape), (2, 3 * INPUT))
        model.plan_boundary[2].weight.data.zero_()
        model.plan_boundary[2].bias.data.zero_()
        words = torch.randn(5, INPUT)
        encoded = torch.randn(5, 64)
        output = model.plan_boundary_scores(encoded, words)
        loss = F.binary_cross_entropy_with_logits(output[:, 0], torch.ones(5))
        grad = torch.autograd.grad(loss, model.plan_boundary_raw.weight)[0]
        self.assertGreater(float(grad.abs().sum()), 0.)

    def test_residual_modes_skip_token_ce_while_other_writer_modes_keep_it(self):
        model = NeuralSlots()
        with torch.no_grad():
            model.plan_tags.weight.zero_()
            model.plan_tags.bias.copy_(torch.tensor([0., 4., -2., -2., 2.]))
        model.boundary_parent_parameters = snapshot_parent_boundary(model)
        model.writer_fit_teachers = {}
        model.structured_weight = model.endpoint_weight = 0.
        model.boundary_weight = 1.
        body, context = "Thanks for listening.", observation_context("Ada")
        bu = [(m.start(), m.end()) for m in re.finditer(r"\S+", body)]
        bx, cx = torch.randn(len(bu), INPUT) / 30, torch.randn(5, INPUT) / 30
        item = dict(body=bx, context=cx, body_words=bx, body_units=bu)
        row = dict(body=body, context=context, spans=[])
        for mode in ("writer-edge", "writer-raw"):
            model.writer_mode = mode
            with mock.patch("train_neural_dialogue.F.cross_entropy", side_effect=AssertionError("residual mode cannot duplicate begin CE")):
                self.assertGreater(float(observation_loss(model, row, item).detach()), 0.)
        model.writer_mode, model.boundary_weight = "write-boundaries", 0.
        original_ce = F.cross_entropy
        with mock.patch("train_neural_dialogue.F.cross_entropy", wraps=original_ce) as ce:
            observation_loss(model, row, item)
        self.assertEqual(ce.call_count, 1)

    def test_structured_anchor_is_zero_with_zero_gradient_at_frozen_source_baseline(self):
        model = NeuralSlots()
        with torch.no_grad():
            model.plan_start.zero_()
            model.plan_end.zero_()
            model.plan_transition.zero_()
        for values in ([1], [0, 0], [1, 4, 0], [0, 1, 4]):
            target = torch.tensor(values)
            raw = torch.zeros(len(target), 5)
            raw[torch.arange(len(target)), target] = 4.
            current = raw.clone().requires_grad_()
            parent_ends = torch.randn(len(target)) / 10
            ends = parent_ends.clone().requires_grad_()
            loss = structured_path_anchor_loss(model, raw, current, target, ends, parent_ends=parent_ends)
            self.assertEqual(float(loss.detach()), 0.)
            gradients = torch.autograd.grad(loss, (current, ends))
            self.assertTrue(all(torch.equal(g, torch.zeros_like(g)) for g in gradients))

    def test_structured_anchor_rejects_false_writes_and_truncation_without_safe_rival_dilution(self):
        model = NeuralSlots()
        with torch.no_grad():
            model.plan_start.zero_()
            model.plan_end.zero_()
            model.plan_transition.zero_()
        penalties = []
        for count in (2, 20):
            target = torch.zeros(count, dtype=torch.long)
            raw = torch.zeros(count, 5)
            raw[:, 0] = 2.
            current = raw.clone()
            current[0, 1] = 4.
            current.requires_grad_()
            ends = torch.zeros(count, requires_grad=True)
            loss = structured_path_anchor_loss(model, raw, current, target, ends, parent_ends=torch.zeros_like(ends))
            penalties.append(float(loss.detach()))
            grad_logits, grad_ends = torch.autograd.grad(loss, (current, ends))
            self.assertGreater(float(grad_logits[0, 1]), 0.)
            self.assertLess(float(grad_logits[0, 0]), 0.)
        self.assertAlmostEqual(penalties[0], 9., places=5)
        self.assertAlmostEqual(penalties[0], penalties[1], places=5)
        target = torch.tensor([1, 4, 4, 0])
        raw = torch.zeros(4, 5)
        raw[torch.arange(4), target] = 4.
        ends = torch.tensor([0., 8., 0., 0.], requires_grad=True)
        loss = structured_path_anchor_loss(model, raw, raw, target, ends, parent_ends=torch.zeros_like(ends))
        self.assertGreater(float(loss.detach()), 0.)
        self.assertGreater(float(torch.autograd.grad(loss, ends)[0][1]), 0.)

    def test_cached_losses_and_gradients_match_independent_path_oracles(self):
        # Float64 distinguishes the algorithms' gradients from float32
        # cancellation when a capped parent margin is nearly unchanged.
        model = NeuralSlots().double()
        for p in model.parameters():
            p.requires_grad_(False)
        for values in ([1], [0, 0, 0], [0, 1, 4, 0, 2, 4, 4, 0]):
            target = torch.tensor(values)
            teacher = build_writer_teacher(target)
            raw = torch.randn(len(target), 5, dtype=torch.float64) / 3
            raw[torch.arange(len(target)), target] += 4.
            current = (raw + torch.randn_like(raw) / 10).requires_grad_()
            ends = (torch.randn(len(target), dtype=torch.float64) / 10).requires_grad_()
            boundaries = torch.stack((current[:, 1] - raw[:, 1], ends), 1)
            parent_ends = torch.randn(len(target), dtype=torch.float64) / 10
            pairs = [
                (oracle_endpoint_loss(model, current, target, ends),
                 endpoint_contrast_loss(model, current, target, ends, teacher=teacher)),
                (oracle_boundary_loss(boundaries, target),
                 boundary_supervision_loss(boundaries, target, teacher=teacher)),
            ]
            for margin in (1., 1000.):
                gap = path_anchor_required_gaps(model, raw, parent_ends, teacher, margin)
                pairs.append((oracle_path_anchor(model, raw, current, target, ends, margin, parent_ends),
                              structured_path_anchor_loss(model, None, current, target, ends, margin,
                                                          teacher=teacher, required_gap=gap)))
            for expected, actual in pairs:
                self.assertTrue(torch.allclose(expected, actual, atol=2e-5))
                expected_grads = torch.autograd.grad(expected, (current, ends), retain_graph=True, allow_unused=True)
                actual_grads = torch.autograd.grad(actual, (current, ends), retain_graph=True, allow_unused=True)
                for expected_grad, actual_grad in zip(expected_grads, actual_grads):
                    if expected_grad is None:
                        self.assertIsNone(actual_grad)
                    else:
                        self.assertTrue(torch.allclose(expected_grad, actual_grad, atol=2e-6))

    def test_cached_hot_losses_do_not_rebuild_or_transfer_teacher_labels(self):
        model = NeuralSlots()
        for values in ([0, 1, 4, 0], [0, 0, 0, 0]):
            target = torch.tensor(values)
            teacher = build_writer_teacher(target)
            raw = torch.randn(len(target), 5)
            current = raw.clone().requires_grad_()
            ends = torch.zeros(len(target), requires_grad=True)
            required = path_anchor_required_gaps(model, raw, torch.zeros_like(ends), teacher, 1.)
            with mock.patch("train_neural_dialogue.build_writer_teacher", side_effect=AssertionError("teacher already compiled")), mock.patch.object(torch.Tensor, "cpu", side_effect=AssertionError("hot path cannot transfer labels")):
                endpoint_contrast_loss(model, current, target, ends, teacher=teacher)
                boundary_supervision_loss(torch.stack((current[:, 1], ends), 1), target, teacher=teacher)
                structured_path_anchor_loss(model, None, current, target, ends,
                                            teacher=teacher, required_gap=required)

    def test_observation_teacher_reuses_cache_without_mutating_or_aliasing_gold(self):
        model = NeuralSlots()
        model.writer_fit_teachers = {}
        row = dict(body="Hello. I keep string. Goodbye.", context="Speaker: Ada",
                   spans=[dict(start=7, end=21, action=1)])
        before = json.dumps(row, sort_keys=True)
        units = [(m.start(), m.end()) for m in re.finditer(r"\S+", row["body"])]
        item = dict(body=torch.zeros(len(units), INPUT), body_units=units)
        with mock.patch("train_neural_dialogue.plan_targets", wraps=__import__("train_neural_dialogue").plan_targets) as targets:
            first = observation_writer_teacher(model, row, item)
            self.assertIs(first, observation_writer_teacher(model, row, item))
            other = dict(row, spans=[dict(start=7, end=21, action=2)])
            second = observation_writer_teacher(model, other, item)
        self.assertEqual(targets.call_count, 2)
        self.assertIsNot(first, second)
        self.assertEqual(json.dumps(row, sort_keys=True), before)

    def test_correct_anchor_only_teacher_storage_is_linear(self):
        model = NeuralSlots()
        model.writer_mode = "writer-context"
        model.boundary_anchor_weight = model.boundary_path_anchor_weight = 1.
        model.boundary_path_anchor_margin = 1000.
        model.boundary_parent_parameters = snapshot_parent_boundary(model)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in writer_train_keys("writer-context"))
        sizes = []
        for count in (32, 64, 128):
            body = " ".join(["word"] * count)
            units = [(m.start(), m.end()) for m in re.finditer(r"\S+", body)]
            row = dict(body=body, context="Speaker: Ada",
                       spans=[dict(start=units[3][0], end=units[-4][1], action=1)])
            words = torch.zeros(count, INPUT)
            item = dict(body=words, body_words=words, body_units=units,
                        context=torch.zeros(1, INPUT))
            key = writer_observation_key(row)
            model.writer_fit_teachers = {}
            model.boundary_fit_correct = {key: True}
            with mock.patch("train_neural_dialogue._competitor_groups", side_effect=AssertionError("anchor rows cannot enumerate endpoint paths")):
                loss = observation_loss(model, row, item)
            self.assertEqual(float(loss.detach()), 0.)
            gradient = torch.autograd.grad(loss, model.plan_boundary_raw.weight)[0]
            self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))
            teacher = model.writer_fit_teachers[key]
            self.assertNotIn("contrast_paths", teacher)
            self.assertNotIn("contrast_ranges", teacher)
            self.assertNotIn("anchor_paths", teacher)
            tensors = [value for value in teacher.values() if isinstance(value, torch.Tensor)]
            self.assertLessEqual(max(value.numel() for value in tensors), 5 * count)
            sizes.append(sum(value.numel() for value in tensors))
            self.assertLessEqual(sizes[-1], 16 * count)
        self.assertEqual(sizes[2] - sizes[1], 2 * (sizes[1] - sizes[0]))

    def test_light_teacher_can_add_full_endpoint_targets_without_changing_parent_cache(self):
        model = NeuralSlots().double()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.writer_fit_teachers = {}
        body = "Hello. I keep string. I save ribbon. Goodbye."
        units = [(m.start(), m.end()) for m in re.finditer(r"\S+", body)]
        row = dict(body=body, context="Speaker: Ada", spans=[
            dict(start=body.index("I keep"), end=body.index(" I save"), action=1),
            dict(start=body.index("I save"), end=body.index(" Goodbye"), action=2),
        ])
        item = dict(body=torch.zeros(len(units), INPUT), body_units=units)
        with mock.patch("train_neural_dialogue._competitor_groups", side_effect=AssertionError("light teacher only")):
            teacher = observation_writer_teacher(model, row, item, include_contrast=False)
        teacher.update(parent_boundaries=torch.randn(len(units), 2),
                       parent_logits=torch.randn(len(units), 5),
                       parent_ends=torch.randn(len(units)), required_gap=torch.tensor(.75))
        parent = {key: teacher[key] for key in
                  ("parent_boundaries", "parent_logits", "parent_ends", "required_gap")}
        parent_values = {key: value.clone() for key, value in parent.items()}
        with mock.patch("train_neural_dialogue.plan_targets", side_effect=AssertionError("gold already cached")), mock.patch("train_neural_dialogue._competitor_groups", wraps=__import__("train_neural_dialogue")._competitor_groups) as contrasts:
            self.assertIs(observation_writer_teacher(model, row, item), teacher)
            self.assertIs(observation_writer_teacher(model, row, item), teacher)
        self.assertEqual(contrasts.call_count, 1)
        for key, value in parent.items():
            self.assertIs(teacher[key], value)
            self.assertTrue(torch.equal(value, parent_values[key]))
            self.assertFalse(value.requires_grad)
        fresh = build_writer_teacher(teacher["target"])
        self.assertNotIn("anchor_paths", fresh)
        self.assertTrue(torch.equal(teacher["contrast_paths"], fresh["contrast_paths"]))
        self.assertEqual(teacher["contrast_ranges"], fresh["contrast_ranges"])
        logits = torch.randn(len(units), 5, dtype=torch.float64, requires_grad=True)
        ends = torch.randn(len(units), dtype=torch.float64, requires_grad=True)
        actual = endpoint_contrast_loss(model, logits, teacher["target"], ends, teacher=teacher)
        expected = endpoint_contrast_loss(model, logits, fresh["target"], ends, teacher=fresh)
        self.assertTrue(torch.equal(actual, expected))
        actual_gradients = torch.autograd.grad(actual, (logits, ends), retain_graph=True)
        expected_gradients = torch.autograd.grad(expected, (logits, ends))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual_gradients, expected_gradients)))

    def test_no_rival_endpoint_teacher_is_compiled_only_once(self):
        model = NeuralSlots()
        model.writer_fit_teachers = {}
        for count, positive in ((1, True), (3, False)):
            body = " ".join(["word"] * count)
            units = [(m.start(), m.end()) for m in re.finditer(r"\S+", body)]
            row = dict(body=body, context="Speaker: Ada", spans=[
                dict(start=0, end=len(body), action=1)
            ] if positive else [])
            item = dict(body=torch.zeros(count, INPUT), body_units=units)
            teacher = observation_writer_teacher(model, row, item, include_contrast=False)
            with mock.patch("train_neural_dialogue._competitor_groups", wraps=__import__("train_neural_dialogue")._competitor_groups) as contrasts:
                for _ in range(3):
                    self.assertIs(observation_writer_teacher(model, row, item), teacher)
            self.assertEqual(contrasts.call_count, 1)
            self.assertEqual(teacher["contrast_ranges"], ())
            self.assertEqual(tuple(teacher["contrast_paths"].shape), (1, count))

    def test_cached_full_boundary_parameter_gradients_reuse_frozen_margin(self):
        model = NeuralSlots()
        with torch.no_grad():
            model.plan_tags.weight.zero_()
            model.plan_tags.bias.copy_(torch.tensor([-1., 4., -2., -2., 5.]))
            model.plan_transition.zero_()
            model.plan_start.zero_()
            model.plan_end.zero_()
            model.plan_boundary[2].weight.normal_(0, .04)
        allowed = writer_train_keys("writer-edge")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in allowed)
        model.writer_mode = "writer-edge"
        model.boundary_anchor_weight, model.boundary_path_anchor_weight = 1., 10.
        model.boundary_path_anchor_margin = 1000.
        model.boundary_parent_parameters = snapshot_parent_boundary(model)
        with torch.no_grad():
            model.plan_boundary[2].bias.add_(torch.tensor([.03, .06]))
        model.writer_fit_teachers = {}
        body = "I keep cotton string in a pouch."
        context = observation_context("Ada", "2047-03-18 08:45 UTC")
        def units(text):
            return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        bu, cu = units(body), units(context)
        bx, cx = torch.randn(len(bu), INPUT) / 30, torch.randn(len(cu), INPUT) / 30
        item = dict(body=bx, context=cx, body_words=bx, body_units=bu)
        row = dict(body=body, context=context, spans=[dict(start=0, end=len(body), action=1)])
        model.boundary_fit_correct = {writer_observation_key(row): True}
        parameters = [p for p in model.parameters() if p.requires_grad]
        full_teacher = observation_writer_teacher(model, row, item)
        self.assertIn("contrast_paths", full_teacher)
        full = observation_loss(model, row, item)
        full_grads = torch.autograd.grad(full, parameters)
        model.writer_fit_teachers = {}
        first = observation_loss(model, row, item)
        first_grads = torch.autograd.grad(first, parameters)
        self.assertTrue(torch.equal(full, first))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(full_grads, first_grads)))
        self.assertNotIn("contrast_paths", model.writer_fit_teachers[writer_observation_key(row)])
        with mock.patch("train_neural_dialogue.plan_targets", side_effect=AssertionError("labels cached")), mock.patch("train_neural_dialogue.path_anchor_required_gaps", side_effect=AssertionError("frozen gaps cached")):
            second = observation_loss(model, row, item)
        second_grads = torch.autograd.grad(second, parameters)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first_grads, second_grads)))
        model.boundary_path_anchor_margin = 1.
        with mock.patch("train_neural_dialogue.path_anchor_required_gaps", wraps=path_anchor_required_gaps) as gaps:
            observation_loss(model, row, item)
        self.assertEqual(gaps.call_count, 1)

    def test_nonzero_full_parent_snapshot_is_exact_at_init_and_immutable_through_updates(self):
        model = NeuralSlots()
        with torch.no_grad():
            model.plan_tags.weight.zero_()
            model.plan_tags.bias.copy_(torch.tensor([-1., 4., -2., -2., 5.]))
            model.plan_transition.zero_()
            model.plan_start.zero_()
            model.plan_end.zero_()
            model.plan_boundary[2].weight.normal_(0, .02)
            model.plan_boundary[2].bias.copy_(torch.tensor([.2, .3]))
        state_keys = set(model.state_dict())
        model.boundary_parent_parameters = snapshot_parent_boundary(model)
        parent_before = {key: value.clone() for key, value in model.boundary_parent_parameters.items()}
        allowed = writer_train_keys("writer-edge")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in allowed)
        model.writer_mode = "writer-edge"
        model.boundary_anchor_weight, model.boundary_path_anchor_weight = 1., 10.
        model.boundary_path_anchor_margin = 1000.
        model.writer_fit_teachers = {}
        body = "I keep cotton string in a pouch."
        timestamp = "2047-03-18 08:45 UTC"
        context = observation_context("Ada", timestamp)
        def units(text):
            return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        bu, cu = units(body), units(context)
        bx, cx = torch.randn(len(bu), INPUT) / 30, torch.randn(len(cu), INPUT) / 30
        item = dict(body=bx, context=cx, body_words=bx, context_words=cx, body_units=bu, context_units=cu)
        row = dict(body=body, context=context, speaker="Ada", timestamp=timestamp,
                   spans=[dict(start=0, end=len(body), action=1)])
        key = writer_observation_key(row)
        model.boundary_fit_correct = {key: True}
        initial = observation_loss(model, row, item)
        self.assertEqual(float(initial.detach()), 0.)
        parts = model.boundary_row_anchor_parts
        self.assertTrue(all(float(value) == 0 for value in parts))
        teacher = model.writer_fit_teachers[key]
        self.assertGreater(float(teacher["parent_boundaries"].abs().mean()), .05)
        self.assertTrue(torch.equal(teacher["parent_ends"], teacher["parent_boundaries"][:, 1]))
        fixed_encoded, _, _ = model.plan(bx, cx, bx)
        before_refs = tuple(x.detach().clone() for x in model.query_reference(
            fixed_encoded.mean(0), source_words(item, len(bu)), source_types(row, item, len(bu)), raw_context=model.query_binding_context(item["body"], item["context"])))
        trainable = [p for p in model.parameters() if p.requires_grad]
        initial_grads = torch.autograd.grad(initial, trainable)
        self.assertTrue(all(torch.equal(g, torch.zeros_like(g)) for g in initial_grads))
        optimizer = torch.optim.SGD(trainable, lr=.01)
        for _ in range(2):
            optimizer.zero_grad()
            current = model.plan_boundary_scores(fixed_encoded.detach(), bx)
            # A small analytic task supplies an update; the parent remains the
            # immutable baseline for the resumed model's protection objective.
            loss = observation_loss(model, row, item) + F.softplus(-current[:, 1]).mean()
            loss.backward()
            self.assertGreater(float(model.plan_boundary_raw.weight.grad.abs().sum()), 0.)
            optimizer.step()
        self.assertTrue(all(torch.equal(value, parent_before[key])
                            for key, value in model.boundary_parent_parameters.items()))
        self.assertEqual(set(model.state_dict()), state_keys)
        self.assertTrue(all(not value.requires_grad for value in model.boundary_parent_parameters.values()))
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if name not in allowed))
        after_encoded, _, _ = model.plan(bx, cx, bx)
        after_refs = model.query_reference(after_encoded.mean(0), source_words(item, len(bu)), source_types(row, item, len(bu)), raw_context=model.query_binding_context(item["body"], item["context"]))
        self.assertTrue(torch.equal(fixed_encoded, after_encoded))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before_refs, after_refs)))

    def test_writer_raw_updates_only_one_tensor_and_keeps_full_parent_teacher(self):
        model = NeuralSlots()
        with torch.no_grad():
            model.plan_tags.weight.zero_()
            model.plan_tags.bias.copy_(torch.tensor([-1., 4., -2., -2., 5.]))
            model.plan_transition.zero_()
            model.plan_start.zero_()
            model.plan_end.zero_()
            model.plan_boundary[2].weight.normal_(0, .02)
            model.plan_boundary[2].bias.copy_(torch.tensor([.2, .3]))
        allowed = writer_train_keys("writer-raw")
        self.assertEqual(allowed, ("plan_boundary_raw.weight",))
        model.boundary_parent_parameters = snapshot_parent_boundary(model)
        self.assertEqual(set(model.boundary_parent_parameters),
                         set(writer_train_keys("writer-edge") + writer_train_keys("writer-membership")
                             + writer_train_keys("writer-context-residual")))
        self.assertEqual(len(model.boundary_parent_parameters), 10)
        parent_before = {key: value.clone() for key, value in model.boundary_parent_parameters.items()}
        state_before = {key: value.clone() for key, value in model.state_dict().items()}
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in allowed)
        model.writer_mode = "writer-raw"
        model.boundary_weight, model.endpoint_weight = 1., .5
        model.boundary_anchor_weight, model.boundary_path_anchor_weight = 1., 10.
        model.boundary_path_anchor_margin = 1000.
        model.writer_fit_teachers = {}
        body = "I keep cotton string in a pouch."
        context = observation_context("Ada", "2047-03-18 08:45 UTC")
        def units(text):
            return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        bu, cu = units(body), units(context)
        bx, cx = torch.randn(len(bu), INPUT) / 30, torch.randn(len(cu), INPUT) / 30
        item = dict(body=bx, context=cx, body_words=bx, body_units=bu)
        row = dict(body=body, context=context, spans=[dict(start=0, end=len(body), action=1)])
        model.boundary_fit_correct = {writer_observation_key(row): True}
        initial = observation_loss(model, row, item)
        self.assertEqual(float(initial.detach()), 0.)
        self.assertTrue(torch.equal(torch.autograd.grad(initial, model.plan_boundary_raw.weight)[0], torch.zeros_like(model.plan_boundary_raw.weight)))
        optimizer = torch.optim.SGD([model.plan_boundary_raw.weight], lr=.01)
        for _ in range(2):
            optimizer.zero_grad()
            encoded, _, _ = model.plan(bx, cx, bx)
            current = model.plan_boundary_scores(encoded, bx)
            loss = observation_loss(model, row, item) + F.softplus(-current[:, 1]).mean()
            loss.backward()
            self.assertGreater(float(model.plan_boundary_raw.weight.grad.abs().sum()), 0.)
            optimizer.step()
        changed = {key for key, value in model.state_dict().items() if not torch.equal(value, state_before[key])}
        self.assertEqual(changed, {"plan_boundary_raw.weight"})
        self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters() if name not in allowed))
        self.assertTrue(all(torch.equal(value, parent_before[key]) for key, value in model.boundary_parent_parameters.items()))
        # An incorrect row still gets the full boundary task without evaluating
        # the frozen reference heads; raw-only isolation applies to both paths.
        model.boundary_fit_correct[writer_observation_key(row)] = False
        with mock.patch.object(model, "plan_reference", side_effect=AssertionError("reference head is frozen")), mock.patch.object(model, "query_reference", side_effect=AssertionError("query head is frozen")):
            self.assertGreater(float(observation_loss(model, row, item).detach()), 0.)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        parser = argparse.ArgumentParser(description="Writer loss/gradient operator microbenchmark")
        parser.add_argument("--bench", action="store_true")
        parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
        parser.add_argument("--iterations", type=int, default=20)
        parser.add_argument("--output")
        benchmark_writer_operators(parser.parse_args())
    else:
        unittest.main()
