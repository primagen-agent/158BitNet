"""Query-binding supervision and frozen-parent safety on synthetic CPU inputs."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import NeuralSlots, QUERY_KEYS, INPUT, observation_context
from neural_query_binding import (
    QUERY_BINDING_CLASSES, query_binding_class, sample_query_binding_batch, query_binding_is_novel,
    query_binding_metrics, query_binding_score, query_binding_safe,
    snapshot_query_binding_parent, query_binding_parent_logits,
    make_query_binding_teacher, query_binding_anchor_parts, validate_query_binding_args,
    validate_query_binding_fit_rows,
)
import train_neural_dialogue as trainer


def binding_row(kind, index=0):
    return dict(
        body=f"Synthetic {kind} question {index}", context="Original literal context",
        source_id=f"binding-{kind}-{index}", query_class=kind,
        whole_need=kind != "self_contained", whole_refs=["陈瑜"] if kind == "resolved" else [],
    )


def correct_prediction(row):
    return dict(need=row["whole_need"], references=list(row["whole_refs"]))


class QueryBindingSamplingTests(unittest.TestCase):
    def test_fit_validation_rejects_source_only_data_without_trainable_supervision(self):
        self_contained = [binding_row("self_contained")]
        for mode in ("query", "query-gate"):
            validate_query_binding_fit_rows(mode, self_contained)
        with self.assertRaisesRegex(ValueError, "reference-source supervision"):
            validate_query_binding_fit_rows("query-source", self_contained)
        for kind in ("resolved", "ambiguous"):
            validate_query_binding_fit_rows("query-source", self_contained + [binding_row(kind)])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            validate_query_binding_fit_rows("query", [])
        with self.assertRaisesRegex(ValueError, "authored fit"):
            validate_query_binding_fit_rows("query", [dict(binding_row("resolved"), evaluation_only=True)])

    def test_three_classes_distinguish_missing_sources_from_no_need(self):
        self.assertEqual(QUERY_BINDING_CLASSES, ("self_contained", "resolved", "ambiguous"))
        for kind in QUERY_BINDING_CLASSES:
            with self.subTest(kind=kind):
                self.assertEqual(query_binding_class(binding_row(kind)), kind)

    def test_sampler_balances_classes_without_replacement_and_is_reproducible(self):
        rows = [binding_row(kind, i) for kind in QUERY_BINDING_CLASSES for i in range(11)]
        before = deepcopy(rows)
        first = sample_query_binding_batch(rows, random.Random(281), per_class=3)
        second = sample_query_binding_batch(rows, random.Random(281), per_class=3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 9)
        self.assertEqual(len({row["source_id"] for row in first}), len(first))
        for kind in QUERY_BINDING_CLASSES:
            self.assertEqual(sum(query_binding_class(row) == kind for row in first), 3)
        self.assertEqual(rows, before)

    def test_sampler_uses_small_nonempty_classes_without_duplicating_rows(self):
        rows = [binding_row("resolved", i) for i in range(2)] + [binding_row("ambiguous")]
        batch = sample_query_binding_batch(rows, random.Random(14), per_class=8)
        self.assertEqual(len(batch), 3)
        self.assertEqual({row["source_id"] for row in batch}, {row["source_id"] for row in rows})

    def test_sampler_rejects_evaluation_only_rows(self):
        for marker in (dict(evaluation_only=True), dict(split="sealed"),
                       dict(query_binding_split="development"),
                       dict(evaluation_role="evaluation-only")):
            row = dict(binding_row("resolved"), **marker)
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                sample_query_binding_batch([row], random.Random(8))
            with self.assertRaises(ValueError):
                trainer.query_binding_training_parts(None, row, None)

    def test_sampler_preserves_new_and_replay_examples_in_every_available_class(self):
        rows = []
        for kind in QUERY_BINDING_CLASSES:
            rows.extend(binding_row(kind, i) for i in range(10))
            rows.extend(dict(binding_row(kind, i + 10), query_binding_split="fit") for i in range(10))
        batch = sample_query_binding_batch(rows, random.Random(17), per_class=8)
        self.assertEqual(len(batch), 24)
        self.assertEqual(len({row["source_id"] for row in batch}), 24)
        for kind in QUERY_BINDING_CLASSES:
            members = [row for row in batch if query_binding_class(row) == kind]
            self.assertEqual(sum(query_binding_is_novel(row) for row in members), 4)
            self.assertEqual(sum(not query_binding_is_novel(row) for row in members), 4)

    def test_sampler_fills_quota_without_losing_a_rare_new_example(self):
        rows = [binding_row("resolved", i) for i in range(10)]
        rows.append(dict(binding_row("resolved", 10), query_binding_split="fit"))
        batch = sample_query_binding_batch(rows, random.Random(18), per_class=8)
        self.assertEqual(len(batch), 8)
        self.assertEqual(sum(query_binding_is_novel(row) for row in batch), 1)

    def test_binding_classes_reject_inconsistent_literal_labels(self):
        for row in (dict(binding_row("resolved"), whole_need=False),
                    dict(binding_row("resolved"), whole_refs=[""]),
                    dict(binding_row("resolved"), whole_need=1)):
            with self.subTest(row=row), self.assertRaises(ValueError):
                query_binding_class(row)


class QueryBindingMetricsTests(unittest.TestCase):
    def rows_and_predictions(self):
        rows = [binding_row(kind) for kind in QUERY_BINDING_CLASSES]
        return rows, [correct_prediction(row) for row in rows]

    def test_perfect_classes_have_perfect_score_and_are_safe(self):
        rows, predictions = self.rows_and_predictions()
        metrics = query_binding_metrics(rows, predictions)
        self.assertEqual(query_binding_score(metrics), (1., 1., 1.))
        self.assertTrue(query_binding_safe(metrics, metrics))
        self.assertEqual(metrics["query_ambiguous_unsafe"], 0)
        self.assertEqual(metrics["query_binding_wrong_subjects"], 0)

    def test_false_resolution_of_ambiguous_query_is_unsafe(self):
        rows, predictions = self.rows_and_predictions()
        initial = query_binding_metrics(rows, predictions)
        predictions[2]["references"] = ["Other person"]
        current = query_binding_metrics(rows, predictions)
        self.assertEqual(current["query_ambiguous_unsafe"], 1)
        self.assertFalse(query_binding_safe(current, initial))

    def test_no_need_does_not_make_an_ambiguous_query_safe(self):
        rows, predictions = self.rows_and_predictions()
        initial = query_binding_metrics(rows, predictions)
        predictions[2]["need"] = False
        current = query_binding_metrics(rows, predictions)
        self.assertEqual(current["query_ambiguous_unsafe"], 1)
        self.assertFalse(query_binding_safe(current, initial))

    def test_wrong_subject_is_distinct_from_need_accuracy(self):
        rows, predictions = self.rows_and_predictions()
        initial = query_binding_metrics(rows, predictions)
        predictions[1]["references"] = ["Other person"]
        current = query_binding_metrics(rows, predictions)
        self.assertEqual(current["query_binding_need_exact"], 1.)
        self.assertEqual(current["query_binding_wrong_subjects"], 1)
        self.assertFalse(query_binding_safe(current, initial))

    def test_group_score_merges_counts_before_class_and_row_averaging(self):
        first = [binding_row("self_contained")]
        second = [binding_row("self_contained", i + 1) for i in range(3)]
        a = query_binding_metrics(first, [dict(need=True, references=[])])
        b = query_binding_metrics(second, [correct_prediction(row) for row in second])
        self.assertEqual(query_binding_score([a, b]), (.75, .75, .75))

    def test_source_quality_cannot_approve_unsafe_query_binding(self):
        rows, predictions = self.rows_and_predictions()
        source = dict(reference_exact=1., whole_reference_exact=1., ambiguous_unsafe_writes=0,
                      committed_fact_f1=1., plan_exact=1., source_exact=1., action_exact=1.)
        initial = dict(source, **query_binding_metrics(rows, predictions))
        predictions[2]["references"] = ["Other person"]
        current = dict(source, **query_binding_metrics(rows, predictions))
        self.assertFalse(trainer.writer_development_safe(current, initial, query_binding=True))
        self.assertTrue(trainer.writer_development_safe(initial, initial, query_binding=True))

    def test_changed_class_denominators_or_invalid_counts_are_rejected(self):
        rows, predictions = self.rows_and_predictions()
        initial = query_binding_metrics(rows, predictions)
        fewer = query_binding_metrics(rows[:2], predictions[:2])
        self.assertFalse(query_binding_safe(fewer, initial))
        broken = deepcopy(initial)
        broken["query_binding_classes"]["resolved"]["full_correct"] = 2
        self.assertFalse(query_binding_safe(broken, initial))
        with self.assertRaises(ValueError):
            query_binding_score(broken)

    def test_added_development_group_cannot_spend_a_binding_safety_floor(self):
        rows, predictions = self.rows_and_predictions()
        source = dict(reference_exact=1., whole_reference_exact=1., ambiguous_unsafe_writes=0,
                      committed_fact_f1=1., plan_exact=1., source_exact=1., action_exact=1.)
        baseline = dict(source, **query_binding_metrics(rows, predictions))
        predictions[2]["need"] = False
        changed = dict(source, **query_binding_metrics(rows, predictions))
        self.assertFalse(trainer.writer_development_safe(
            baseline, baseline, query_binding=True, query_development=changed, query_initial=baseline))
        with self.assertRaisesRegex(ValueError, "requires initial metrics"):
            trainer.writer_development_safe(
                baseline, baseline, query_binding=True, query_development=baseline)


class QueryBindingTeacherTests(unittest.TestCase):
    def teacher(self, dtype=torch.float64, *, gold_need=True):
        need = torch.tensor(.4 if gold_need else -.4, dtype=dtype, requires_grad=True)
        tags = torch.tensor([[2., 0., -1.], [0., 3., 1.], [0., 2., 4.]],
                            dtype=dtype, requires_grad=True)
        gold = torch.tensor([0, 1, 2])
        teacher = make_query_binding_teacher(need, tags, gold_need, gold, margin=1.)
        return need, tags, gold, teacher

    def test_parent_equality_has_exact_zero_loss_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for required in (False, True):
                with self.subTest(dtype=dtype, required=required):
                    need, tags, _, teacher = self.teacher(dtype, gold_need=required)
                    nloss, tloss = query_binding_anchor_parts(need, tags, teacher)
                    self.assertEqual(float(nloss.detach()), 0.)
                    self.assertEqual(float(tloss.detach()), 0.)
                    (nloss + tloss).backward()
                    self.assertTrue(torch.equal(need.grad, torch.zeros_like(need)))
                    self.assertTrue(torch.equal(tags.grad, torch.zeros_like(tags)))

    def test_teacher_has_no_aliases_or_gradients(self):
        need, tags, gold, teacher = self.teacher()
        saved = {name: value.clone() for name, value in teacher.items()}
        self.assertTrue(all(not value.requires_grad for value in teacher.values()))
        with torch.no_grad():
            need.fill_(100.)
            tags.fill_(-100.)
            gold.fill_(2)
        self.assertTrue(all(torch.equal(teacher[name], value) for name, value in saved.items()))

    def test_no_need_teachers_do_not_preserve_unused_span_logits(self):
        need, tags, _, teacher = self.teacher(gold_need=False)
        changed = (-tags.detach()).requires_grad_()
        _, loss = query_binding_anchor_parts(need, changed, teacher)
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.equal(changed.grad, torch.zeros_like(changed)))

    def test_ambiguous_all_outside_sources_are_preserved(self):
        need = torch.tensor(.5)
        parent = torch.tensor([[2., 0., -1.], [3., -2., 0.]])
        teacher = make_query_binding_teacher(need, parent, True, torch.zeros(2, dtype=torch.long))
        current = parent.clone()
        current[1, 1] = 4.
        current.requires_grad_()
        _, loss = query_binding_anchor_parts(need, current, teacher)
        self.assertGreater(float(loss.detach()), 0.)
        loss.backward()
        self.assertLess(float(current.grad[1, 0]), 0.)
        self.assertGreater(float(current.grad[1, 1]), 0.)

    def test_incorrect_parent_gate_and_span_are_not_preserved(self):
        parent_need = torch.tensor(-.5)
        parent_tags = torch.tensor([[3., 0., -2.]])
        teacher = make_query_binding_teacher(parent_need, parent_tags, True, torch.tensor([1]))
        need = torch.tensor(.5, requires_grad=True)
        tags = torch.tensor([[0., 3., -2.]], requires_grad=True)
        nloss, tloss = query_binding_anchor_parts(need, tags, teacher)
        self.assertEqual(float((nloss + tloss).detach()), 0.)
        (nloss + tloss).backward()
        self.assertEqual(float(need.grad), 0.)
        self.assertTrue(torch.equal(tags.grad, torch.zeros_like(tags)))

    def test_gate_margin_is_finite_capped_and_has_restoring_direction(self):
        for parent_logit, current_logit, target in ((5., .2, True), (-5., -.2, False), (0., .2, False)):
            with self.subTest(parent_logit=parent_logit, target=target):
                parent = torch.tensor(parent_logit)
                tags = torch.tensor([[2., 0., -1.]])
                teacher = make_query_binding_teacher(parent, tags, target, torch.tensor([0]), margin=1.)
                current = torch.tensor(current_logit, requires_grad=True)
                loss, _ = query_binding_anchor_parts(current, tags, teacher)
                self.assertGreater(float(loss.detach()), 0.)
                self.assertLessEqual(float(loss.detach()), 1.)
                loss.backward()
                self.assertLess(float(current.grad) * (1 if target else -1), 0.)

    def test_empty_source_sequence_has_finite_gate_protection(self):
        need = torch.tensor(.5, requires_grad=True)
        tags = torch.empty(0, 3, requires_grad=True)
        teacher = make_query_binding_teacher(need, tags, True, torch.empty(0, dtype=torch.long))
        changed_need = torch.tensor(-.5, requires_grad=True)
        nloss, tloss = query_binding_anchor_parts(changed_need, tags, teacher)
        self.assertEqual(float(nloss.detach()), 1.)
        self.assertEqual(float(tloss.detach()), 0.)
        (nloss + tloss).backward()
        self.assertLess(float(changed_need.grad), 0.)
        self.assertEqual(tags.grad.shape, (0, 3))

    def test_teacher_rejects_nonfinite_or_misaligned_parent_logits(self):
        for need, tags, gold in (
            (torch.tensor(float("nan")), torch.zeros(1, 3), torch.zeros(1, dtype=torch.long)),
            (torch.tensor(1.), torch.full((1, 3), float("inf")), torch.zeros(1, dtype=torch.long)),
            (torch.tensor(1.), torch.zeros(1, 4), torch.zeros(1, dtype=torch.long)),
            (torch.tensor(1.), torch.zeros(1, 3), torch.zeros(2, dtype=torch.long)),
        ):
            with self.subTest(shape=tags.shape, need=need), self.assertRaises(ValueError):
                make_query_binding_teacher(need, tags, True, gold)


class QueryBindingParameterTests(unittest.TestCase):
    def test_modes_select_exact_binding_parameters(self):
        self.assertEqual(set(trainer.writer_train_keys("query")), set(QUERY_KEYS))
        self.assertEqual(set(trainer.writer_train_keys("query-gate")),
                         {"query_need.weight", "query_need.bias"})
        self.assertEqual(set(trainer.writer_train_keys("query-source")),
                         set(QUERY_KEYS) - {"query_need.weight", "query_need.bias", "query_context.weight"})
        self.assertEqual(trainer.writer_train_keys("query-context"), ("query_context.weight",))
        self.assertEqual(len(QUERY_KEYS), 13)

    def test_parent_logits_match_current_query_binding_without_sharing_parameters(self):
        torch.manual_seed(8391)
        model = NeuralSlots()
        with torch.no_grad():
            model.query_context.weight.normal_(std=.005)
        parent = snapshot_query_binding_parent(model)
        candidate = torch.randn(64) / 10
        raw_context = torch.randn(4 * INPUT) / 10
        words = torch.randn(4, 4096) / 10
        types = torch.eye(4)
        expected = model.query_reference(candidate, words, types, raw_context=raw_context)
        actual = query_binding_parent_logits(model, candidate, words, types, parent, raw_context=raw_context)
        for x, y in zip(expected, actual):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        before = {name: value.clone() for name, value in parent.items()}
        with torch.no_grad():
            model.query_need.bias.add_(3.)
            model.query_origin.bias.add_(1.)
            model.query_context.weight.add_(.1)
        changed = model.query_reference(candidate, words, types, raw_context=raw_context)
        self.assertFalse(torch.equal(expected[0], changed[0]))
        repeated = query_binding_parent_logits(model, candidate, words, types, parent, raw_context=raw_context)
        for x, y in zip(expected, repeated):
            torch.testing.assert_close(x, y, atol=0, rtol=0)
        self.assertEqual(len(parent), 13)
        self.assertTrue(all(torch.equal(parent[name], value) for name, value in before.items()))
        self.assertTrue(all(not value.requires_grad for value in parent.values()))

    def test_query_gate_and_anchor_validation_require_resumed_binding_mode(self):
        def args(**changes):
            return SimpleNamespace(**(dict(train_heads="query", resume="current.pt",
                query_binding_anchor_weight=1., query_binding_anchor_margin=1.,
                query_binding_generalization=False) | changes))
        validate_query_binding_args(args())
        validate_query_binding_args(args(train_heads="query-gate"))
        validate_query_binding_args(args(train_heads="query-context"))
        for changes in (dict(resume=None), dict(train_heads="all"),
                        dict(query_binding_anchor_weight=-1.), dict(query_binding_anchor_margin=0.),
                        dict(query_binding_anchor_margin=float("nan")),
                        dict(train_heads="query-gate", resume=None, query_binding_anchor_weight=0.),
                        dict(train_heads="query-context", resume=None, query_binding_anchor_weight=0.)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_query_binding_args(args(**changes))

    def test_cli_exposes_query_gate_and_parent_protection_options(self):
        args = trainer.writer_argument_parser().parse_args([
            "--gguf", "backbone.gguf", "--probe", "probe", "--output", "output",
            "--resume", "current.pt", "--train-heads", "query-gate",
            "--query-binding-anchor-weight", "1.5", "--query-binding-anchor-margin", ".8",
            "--query-binding-generalization",
        ])
        self.assertEqual(args.train_heads, "query-gate")
        self.assertEqual(args.query_binding_anchor_weight, 1.5)
        self.assertEqual(args.query_binding_anchor_margin, .8)
        self.assertTrue(args.query_binding_generalization)
        validate_query_binding_args(args)

    def test_cli_exposes_context_only_resumed_training(self):
        args = trainer.writer_argument_parser().parse_args([
            "--gguf", "backbone.gguf", "--probe", "probe", "--output", "output",
            "--resume", "current.pt", "--train-heads", "query-context",
        ])
        self.assertEqual(args.train_heads, "query-context")
        validate_query_binding_args(args)


def training_fixture(device, mode):
    torch.manual_seed(3057)
    model = NeuralSlots().to(device)
    with torch.no_grad():
        model.query_need.weight.zero_()
        model.query_need.bias.fill_(-1.)
    model.writer_mode = mode
    allowed = set(trainer.writer_train_keys(mode))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowed)
    model.query_binding_anchor_weight = 1.
    model.query_binding_anchor_margin = 1.
    model.query_binding_parent_parameters = snapshot_query_binding_parent(model)
    model.query_binding_fit_teachers = {}
    rows, prepared = [], {}
    for kind in QUERY_BINDING_CLASSES:
        body = "Where does 陈瑜 keep the sketchpad?" if kind == "self_contained" else (
            "Where does she keep the sketchpad?" if kind == "resolved" else "Where does she work?")
        past = "user: 陈瑜 is the only person under discussion.\n" if kind == "resolved" else (
            "user: 陈瑜 and Bram are both under discussion.\n")
        context = observation_context("User", "", past)
        row = dict(binding_row(kind), body=body, context=context, speaker="User", timestamp="", spans=[])
        if kind == "resolved":
            start = context.encode().index("陈瑜".encode())
            row["whole_ref_ranges"] = [dict(document="context", start=start,
                                            end=start + len("陈瑜".encode()))]
        def units(text):
            return [(len(text[:m.start()].encode()), len(text[:m.end()].encode()))
                    for m in re.finditer(r"\S+", text)]
        body_units, context_units = units(body), units(context)
        body_words = (torch.randn(len(body_units), INPUT) / 20).to(device)
        context_words = (torch.randn(len(context_units), INPUT) / 20).to(device)
        item = dict(body=body_words, context=context_words, body_words=body_words,
                    context_words=context_words, body_units=body_units, context_units=context_units)
        rows.append(row)
        prepared[body, context] = item
    return model, rows, prepared, allowed


def check_training_parameters(test, device):
    for mode in ("query", "query-gate", "query-source", "query-context"):
        with test.subTest(device=device, mode=mode):
            model, rows, prepared, allowed = training_fixture(device, mode)
            before = {name: value.detach().clone() for name, value in model.state_dict().items()}
            parent = {name: value.clone() for name, value in model.query_binding_parent_parameters.items()}
            test.assertEqual(len(before), 91)
            test.assertEqual(len(parent), 13)
            if mode == "query-context":
                test.assertEqual(len(before.keys() - allowed), 90)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.005)
            with patch.object(model, "plan_with_encodings", side_effect=AssertionError("source CRF must stay unused")), \
                 patch.object(model, "sequence_loss", side_effect=AssertionError("source sequence loss must stay unused")):
                for _ in range(2):
                    optimizer.zero_grad()
                    loss = trainer.batch_loss(model, rows, prepared)
                    test.assertTrue(bool(torch.isfinite(loss)))
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        if name in allowed:
                            test.assertIsNotNone(parameter.grad, name)
                            test.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                        else:
                            test.assertIsNone(parameter.grad, name)
                    optimizer.step()
            test.assertTrue(any(not torch.equal(before[name], model.state_dict()[name]) for name in allowed))
            for name, value in model.state_dict().items():
                if name not in allowed:
                    test.assertTrue(torch.equal(value, before[name]), name)
            test.assertTrue(all(torch.equal(model.query_binding_parent_parameters[name], value)
                                for name, value in parent.items()))
            test.assertEqual(model.query_binding_batch_stats["classes"],
                             dict(self_contained=1, resolved=1, ambiguous=1))
            test.assertEqual(len(model.query_binding_fit_teachers), 3)


class QueryBindingTrainingTests(unittest.TestCase):
    def test_cpu_query_modes_preserve_frozen_tensors_and_independent_parent(self):
        check_training_parameters(self, "cpu")

    def test_class_supervision_and_global_anchor_are_not_diluted_by_repeated_rows(self):
        rows = [binding_row(kind) for kind in QUERY_BINDING_CLASSES]
        model = SimpleNamespace(query_binding_anchor_weight=1.)
        prepared = {(row["body"], row["context"]): None for row in rows}
        def parts(_model, row, item):
            kind = query_binding_class(row)
            return (torch.tensor(dict(self_contained=1., resolved=3., ambiguous=5.)[kind]),
                    torch.tensor(2. if kind == "resolved" else 0.),
                    torch.tensor(4. if kind == "ambiguous" else 0.))
        with patch.object(trainer, "query_binding_training_parts", side_effect=parts):
            first = trainer.query_binding_batch_loss(model, rows, prepared)
            repeated = trainer.query_binding_batch_loss(model, rows + [rows[0]] * 20, prepared)
        self.assertEqual(float(first), 9.)
        self.assertEqual(float(repeated), 9.)

    def test_fit_teacher_cache_rejects_changed_labels_for_the_same_observation(self):
        model, rows, prepared, _ = training_fixture("cpu", "query")
        row = rows[1]
        item = prepared[row["body"], row["context"]]
        trainer.query_binding_training_parts(model, row, item)
        changed = dict(row, whole_need=False, whole_refs=[], whole_ref_ranges=[])
        with self.assertRaisesRegex(ValueError, "inconsistent binding labels"):
            trainer.query_binding_training_parts(model, changed, item)

    def test_late_fit_teacher_uses_nonzero_resumed_parent_context(self):
        model, rows, prepared, _ = training_fixture("cpu", "query-context")
        with torch.no_grad():
            model.query_context.weight.normal_(std=.005)
        model.query_binding_parent_parameters = snapshot_query_binding_parent(model)
        row = rows[1]
        item = prepared[row["body"], row["context"]]
        trainer.query_binding_training_parts(model, row, item)
        key = row["body"], row["context"]
        expected = {name: value.clone() if isinstance(value, torch.Tensor) else value
                    for name, value in model.query_binding_fit_teachers[key].items()}
        model.query_binding_fit_teachers.clear()
        with torch.no_grad():
            model.query_context.weight.add_(.1)
        trainer.query_binding_training_parts(model, row, item)
        actual = model.query_binding_fit_teachers[key]
        self.assertEqual(set(actual), set(expected))
        for name, value in expected.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(actual[name], value, atol=0, rtol=0)
            else:
                self.assertEqual(actual[name], value)

    def test_terminal_payload_keeps_last_weights_after_selected_state_is_restored(self):
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        selected = {name: value.detach().clone() for name, value in model.state_dict().items()}
        metadata = dict(diagnostic_only=False, eligible_for_selection=True,
                        selected_step=0, marker="current-parent")
        with torch.no_grad():
            model.bias.fill_(2.)
        terminal = trainer.writer_terminal_checkpoint_payload(
            model, metadata, training_steps_executed=1)
        model.load_state_dict(selected)
        self.assertEqual(float(terminal["state_dict"]["bias"]), 2.)
        self.assertEqual(float(model.bias.detach()), 0.)
        self.assertTrue(terminal["diagnostic_only"])
        self.assertFalse(terminal["eligible_for_selection"])
        self.assertEqual(terminal["training_steps_executed"], 1)
        self.assertEqual(terminal["marker"], "current-parent")
        self.assertEqual(metadata, dict(diagnostic_only=False, eligible_for_selection=True,
                                        selected_step=0, marker="current-parent"))

    def test_training_saves_terminal_before_restoring_rejected_selected_checkpoint(self):
        self.check_terminal_restoration("query-gate")

    def test_context_training_saves_terminal_before_restoring_rejected_selected_checkpoint(self):
        self.check_terminal_restoration("query-context")

    def check_terminal_restoration(self, mode):
        # A small substitute tests the real train/save/restore control
        # flow. Model geometry and actual binder gradients are tested separately.
        class TinyBinder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.query_need = torch.nn.Linear(1, 1)
                self.query_context = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    self.query_need.weight.zero_()
                    self.query_need.bias.zero_()
                    self.query_context.weight.zero_()
        train_key = "query_need.bias" if mode == "query-gate" else "query_context.weight"

        def trained_value(model):
            return dict(model.named_parameters())[train_key]
        rows = [dict(binding_row(kind), spans=[], tag="synthetic-query") for kind in QUERY_BINDING_CLASSES]
        source_quality = dict(reference_exact=1., whole_reference_exact=1., ambiguous_unsafe_writes=0,
                              committed_fact_f1=1., plan_exact=1., source_exact=1., action_exact=1.)
        def evaluate(model, observations, prepared):
            predictions = [correct_prediction(row) for row in observations]
            if float(trained_value(model).detach()) > .01:
                for row, prediction in zip(observations, predictions):
                    if query_binding_class(row) == "ambiguous":
                        prediction["references"] = ["Other person"]
            return dict(source_quality, **query_binding_metrics(observations, predictions)), []
        def prepare(groups, cache, device):
            return {(row["body"], row["context"]): None for values in groups.values() for row in values}
        backbone = "b" * 64
        cache = {row[k]: None for row in rows for k in ("body", "context")}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            curriculum = root / "curriculum-input.json"
            curriculum.write_text(json.dumps(dict(fit=rows, development=rows)))
            resume = root / "diagnostic-parent.pt"
            torch.save(dict(format=trainer.FORMAT, backbone_sha256=backbone,
                state_dict=TinyBinder().state_dict(), diagnostic_only=True, eligible_for_selection=False), resume)
            output = root / "output"
            args = trainer.writer_argument_parser().parse_args([
                "--gguf", "backbone.gguf", "--probe", "probe", "--output", str(output),
                "--resume", str(resume), "--curriculum", str(curriculum), "--features", "features.pt",
                "--train-heads", mode, "--steps", "1", "--eval-every", "1",
                "--learning-rate", ".1", "--query-binding-anchor-weight", "1",
            ])
            with patch.object(trainer, "NeuralSlots", TinyBinder), \
                 patch.object(trainer, "sha", return_value=backbone), \
                 patch.object(trainer, "load_features", return_value=dict(features=cache, identity=dict(backbone=backbone))), \
                 patch.object(trainer, "extract", side_effect=AssertionError("no backbone extraction")), \
                 patch.object(trainer, "require_parameters", side_effect=lambda model, state: model.load_state_dict(state)), \
                 patch.object(trainer, "snapshot_query_binding_parent", side_effect=lambda model: {
                     name: value.detach().clone() for name, value in model.state_dict().items()}), \
                 patch.object(trainer, "prepare", side_effect=prepare), \
                 patch.object(trainer, "evaluate", side_effect=evaluate), \
                 patch.object(trainer, "evaluate_query_binding", side_effect=lambda *values: evaluate(*values)[0]), \
                 patch.object(trainer, "batch_loss", side_effect=lambda model, *_: (trained_value(model) - 2.).square().sum()), \
                 patch.object(trainer, "export", return_value="c" * 64), redirect_stdout(io.StringIO()):
                trainer.train(args)
            selected = torch.load(output / "selected.pt", weights_only=False)
            terminal = torch.load(output / "last-step.pt", weights_only=False)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(selected["selected_step"], 0)
            self.assertTrue(selected["eligible_for_selection"])
            self.assertFalse(selected["diagnostic_only"])
            self.assertFalse(terminal["eligible_for_selection"])
            self.assertTrue(terminal["diagnostic_only"])
            self.assertEqual(float(selected["state_dict"][train_key]), 0.)
            self.assertGreater(float(terminal["state_dict"][train_key]), .01)
            self.assertEqual(terminal["writer_training_mode"], mode)
            self.assertEqual(report["last_step_development"]["query_ambiguous_unsafe"], 1)
            self.assertEqual(report["development"]["query_ambiguous_unsafe"], 0)
            self.assertEqual(report["query_binding_sampling_visits"]["steps"], 1)
            self.assertFalse(report["query_binding_fit_used_for_selection"])
            self.assertFalse(report["locomo_used"])


@unittest.skipUnless(torch.backends.mps.is_available(), "requires actual Apple MPS")
class QueryBindingMPSTests(unittest.TestCase):
    def test_actual_mps_query_modes_preserve_frozen_tensors_and_independent_parent(self):
        try:
            check_training_parameters(self, "mps")
        finally:
            torch.mps.synchronize()
            torch.mps.empty_cache()


if __name__ == "__main__":
    unittest.main()
