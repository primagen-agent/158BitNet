"""Frozen-parent write gate refinement preserves source and query isolation."""

from copy import deepcopy
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import NeuralSlots, INPUT, WIDTH, QUERY_KEYS, observation_context
from neural_writer_curriculum import assertion_generalization_fit_rows
from train_neural_dialogue import (
    batch_loss, observation_loss, observation_write_need_loss, plan_targets,
    sample_action_balanced_batch, snapshot_parent_boundary, snapshot_parent_write_need,
    source_types, source_words, train, writer_development_safe,
    writer_observation_key, writer_train_keys,
)


NEED_KEYS = ("plan_need.weight", "plan_need.bias")


def fixture(model, *, need=True, short=False):
    body = "cup" if short else "The ceramic cup rests beside the lamp."
    speaker, timestamp = "Arden", "2058-08-16 10:45 UTC"
    context = observation_context(speaker, timestamp, "user: Please continue.\n")
    units = lambda text: [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    body_units, context_units = units(body), units(context)
    dtype = next(model.parameters()).dtype
    words = torch.randn(len(body_units), INPUT, dtype=dtype) / 20
    context_words = torch.randn(len(context_units), INPUT, dtype=dtype) / 20
    item = dict(body=words, body_words=words, body_units=body_units,
                context=context_words, context_words=context_words, context_units=context_units)
    span = dict(start=0, end=len(body.encode()), action=1, need=need,
                references=[speaker] if need else [])
    row = dict(body=body, context=context, speaker=speaker, timestamp=timestamp,
               spans=[span], whole_need=True, whole_refs=[speaker])
    return row, item


def configure(model, *, anchored=False):
    model.writer_mode = "writer-context-binding"
    model.structured_weight, model.endpoint_weight = 1., .5
    model.boundary_weight, model.membership_weight = 1., .5
    model.boundary_anchor_weight = 1. if anchored else 0.
    model.boundary_path_anchor_weight = 10. if anchored else 0.
    model.boundary_path_anchor_margin = 1000.
    model.write_need_weight, model.write_need_anchor_margin = 1., 1.
    model.boundary_parent_parameters = snapshot_parent_boundary(model)
    model.write_need_parent_parameters = snapshot_parent_write_need(model)
    model.writer_fit_teachers, model.write_need_fit_teachers = {}, {}
    model.source_error_balance = False
    allowed = writer_train_keys(model.writer_mode)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowed)


def plain_batches_without_assertions(positives, negatives, rng):
    """Pre-assertion default sampler for nondiverse, nonnarrative rows."""
    selected = set()
    for action, quota in ((3, 2), (2, 2), (1, 8)):
        choices = [i for i, row in enumerate(positives) if i not in selected
                   and any(span["action"] == action for span in row["spans"])]
        selected.update(rng.sample(choices, min(quota, len(choices))))
    remaining = [i for i in range(len(positives)) if i not in selected]
    selected.update(rng.sample(remaining, min(max(12 - len(selected), 0), len(remaining))))
    result = [positives[i] for i in sorted(selected)]
    result += rng.sample(negatives, min(8, len(negatives)))
    rng.shuffle(result)
    return result


READ_CONTEXT_KEYS = {"read_context_memory.weight", "read_context_query.weight", "read_context_delta.weight"}


class WriterBindingContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(37102)

    def test_twelve_parameters_change_while_all_other_79_and_query_outputs_stay_fixed(self):
        model = NeuralSlots().double()
        with torch.no_grad():
            model.plan_tags.bias.copy_(torch.tensor([8., -3., -3., -3., -3.]))
            model.plan_boundary[2].weight.normal_(0, .03)
            model.plan_membership_raw.weight.normal_(0, .02)
            model.plan_source_context[2].weight.normal_(0, .03)
            model.plan_need.weight.normal_(0, .03)
            model.plan_need.bias.fill_(-3.)
        configure(model)
        allowed = set(writer_train_keys(model.writer_mode))
        self.assertEqual(allowed, set(writer_train_keys("writer-context-content")) | set(NEED_KEYS))
        self.assertEqual(len(allowed), 12)
        self.assertTrue(allowed.isdisjoint(QUERY_KEYS))
        self.assertEqual(len(model.state_dict()), 91)
        row, item = fixture(model)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        parents = deepcopy((model.boundary_parent_parameters, model.write_need_parent_parameters))
        full = model.plan_encoding(item["body"], item["context"], item["body_words"])
        kinds = source_types(row, item, len(full)).to(full.dtype)
        query_refs = tuple(value.clone() for value in model.query_reference(
            full.mean(0), source_words(item, len(full)), kinds, raw_context=model.query_binding_context(item["body"], item["context"])))
        recall = tuple(value.clone() for value in model.query((item["body"], item["body_words"])))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=0)
        observation_loss(model, row, item).backward()
        for name, parameter in model.named_parameters():
            if name in allowed:
                self.assertIsNotNone(parameter.grad, name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
            else:
                self.assertIsNone(parameter.grad, name)
        optimizer.step()
        changed = {name for name, value in model.state_dict().items() if not torch.equal(value, before[name])}
        self.assertEqual(changed, allowed)
        self.assertEqual(len(before) - len(changed), 79)
        frozen = {key: value for key, value in before.items() if key not in changed}
        self.assertTrue(READ_CONTEXT_KEYS <= frozen.keys())
        parameters = dict(model.named_parameters())
        for key in READ_CONTEXT_KEYS:
            self.assertFalse(parameters[key].requires_grad, key)
            self.assertIsNone(parameters[key].grad, key)
            self.assertEqual(parameters[key].detach().cpu().numpy().tobytes(),
                             frozen[key].cpu().numpy().tobytes(), key)
        new_full = model.plan_encoding(item["body"], item["context"], item["body_words"])
        self.assertTrue(torch.equal(full, new_full))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(query_refs, model.query_reference(
            new_full.mean(0), source_words(item, len(full)), kinds, raw_context=model.query_binding_context(item["body"], item["context"])))))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(recall, model.query((item["body"], item["body_words"])))))
        for current, parent in zip((model.boundary_parent_parameters, model.write_need_parent_parameters), parents):
            self.assertTrue(all(torch.equal(current[name], value) for name, value in parent.items()))

    def test_correct_gate_initial_anchor_is_exact_zero_and_uses_capped_parent_margin(self):
        for dtype in (torch.float32, torch.float64):
            for desired, parent_logit in ((True, .2), (True, 5.), (False, -.2), (False, -5.), (False, 0.)):
                with self.subTest(dtype=dtype, desired=desired, parent_logit=parent_logit):
                    model = NeuralSlots().to(dtype)
                    with torch.no_grad():
                        model.plan_need.weight.zero_()
                        model.plan_need.bias.fill_(parent_logit)
                    configure(model)
                    row, item = fixture(model, need=desired)
                    full = torch.randn(len(item["body_units"]), WIDTH, dtype=dtype, requires_grad=True)
                    value = observation_write_need_loss(model, row, item, full)
                    self.assertEqual(float(value.detach()), 0.)
                    value.backward()
                    self.assertIsNone(full.grad)
                    for name in NEED_KEYS:
                        gradient = dict(model.named_parameters())[name].grad
                        self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))
                    teacher = next(iter(model.write_need_fit_teachers.values()))
                    self.assertEqual(len(teacher["correction_indices"]), 0)
                    self.assertAlmostEqual(float(teacher["required_margin"][0]), min(abs(parent_logit), 1.))
                    # Relaxing a strong gate to the capped margin remains safe.
                    with torch.no_grad():
                        model.plan_need.bias.fill_((1 if desired else -1) * max(min(abs(parent_logit), 1.), .5))
                    self.assertEqual(float(observation_write_need_loss(model, row, item, full).detach()), 0.)
                    with torch.no_grad():
                        model.plan_need.bias.fill_(-1 if desired else 1)
                    self.assertGreater(float(observation_write_need_loss(model, row, item, full).detach()), 0.)

    def test_wrong_gate_bce_uses_original_full_span_and_gradients_only_need_weights(self):
        model = NeuralSlots().double()
        with torch.no_grad():
            model.plan_need.weight.normal_(0, .05)
            model.plan_need.bias.fill_(-3.)
        configure(model)
        row, item = fixture(model)
        row["spans"][0].update(start=item["body_units"][1][0], end=item["body_units"][3][1])
        full = torch.randn(len(item["body_units"]), WIDTH, dtype=torch.float64, requires_grad=True)
        pool = full.detach()[1:4].mean(0)
        logit = model.plan_need(pool).flatten()
        expected = F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit))
        value = observation_write_need_loss(model, row, item, full)
        torch.testing.assert_close(value, expected, atol=0, rtol=0)
        value.backward()
        self.assertIsNone(full.grad)
        torch.testing.assert_close(model.plan_need.weight.grad, (logit.detach().sigmoid() - 1)[:, None] * pool[None])
        self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters() if name not in NEED_KEYS))
        teacher = next(iter(model.write_need_fit_teachers.values()))
        self.assertEqual(teacher["ranges"], ((1, 4),))
        self.assertEqual(teacher["correction_indices"].tolist(), [0])
        # Student correctness cannot move a gate into the parent-correct mask.
        with torch.no_grad():
            model.plan_need.bias.fill_(3.)
        observation_write_need_loss(model, row, item, full)
        self.assertEqual(teacher["correction_indices"].tolist(), [0])

    def test_spurious_need_gate_receives_bce_without_training_source_or_other_refs(self):
        model = NeuralSlots().double()
        with torch.no_grad():
            model.plan_need.weight.zero_()
            model.plan_need.bias.fill_(3.)
        configure(model)
        row, item = fixture(model, need=False)
        full = torch.randn(len(item["body_units"]), WIDTH, dtype=torch.float64, requires_grad=True)
        expected = F.binary_cross_entropy_with_logits(model.plan_need(full.detach().mean(0)).flatten(), torch.zeros(1, dtype=torch.float64))
        value = observation_write_need_loss(model, row, item, full)
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        value.backward()
        self.assertGreater(float(model.plan_need.bias.grad[0]), 0.)
        self.assertIsNone(full.grad)
        self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters() if name not in NEED_KEYS))
        teacher = next(iter(model.write_need_fit_teachers.values()))
        with torch.no_grad():
            model.plan_need.bias.fill_(-3.)
        observation_write_need_loss(model, row, item, full)
        self.assertEqual(teacher["correction_indices"].tolist(), [0])

    def test_gate_targets_and_span_order_have_distinct_teacher_keys(self):
        model = NeuralSlots()
        with torch.no_grad():
            model.plan_need.weight.zero_()
            model.plan_need.bias.fill_(-2.)
        configure(model)
        row, item = fixture(model)
        row["spans"] = [dict(row["spans"][0], end=item["body_units"][1][1]),
                        dict(row["spans"][0], start=item["body_units"][3][0], need=False)]
        full = torch.randn(len(item["body_units"]), WIDTH)
        observation_write_need_loss(model, row, item, full)
        reversed_row = deepcopy(row)
        reversed_row["spans"].reverse()
        observation_write_need_loss(model, reversed_row, item, full)
        changed_target = deepcopy(row)
        changed_target["spans"][0]["need"] = False
        observation_write_need_loss(model, changed_target, item, full)
        self.assertEqual(len(model.write_need_fit_teachers), 3)

    def test_source_correct_full_plan_anchor_still_corrects_wrong_need_without_other_refs(self):
        model = NeuralSlots().double()
        with torch.no_grad():
            model.plan_tags.weight.zero_()
            model.plan_tags.bias.copy_(torch.tensor([-10., 10., -10., -10., -10.]))
            model.plan_need.weight.zero_()
            model.plan_need.bias.fill_(-3.)
        configure(model, anchored=True)
        row, item = fixture(model, short=True)
        encoded, logits, ends = model.plan(item["body"], item["context"], item["body_words"])
        self.assertTrue(torch.equal(model.plan_labels(logits, ends), plan_targets(row, item["body_units"], torch.device("cpu"))))
        model.boundary_fit_correct = {writer_observation_key(row): True}
        model.source_error_balance = True
        with mock.patch.object(model, "plan_reference", side_effect=AssertionError("unused reference tags")), \
             mock.patch.object(model, "query_reference", side_effect=AssertionError("unused whole binding")):
            value = batch_loss(model, [row], {(row["body"], row["context"]): item})
        self.assertGreater(float(value.detach()), 0.)
        self.assertEqual(float(model.boundary_batch_stats["anchor_loss_sum"]), 0.)
        self.assertEqual(tuple(float(x) for x in model.boundary_row_anchor_parts), (0., 0.))
        value.backward()
        self.assertGreater(float(model.plan_need.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.plan_need.bias.grad.abs().sum()), 0.)
        for name, parameter in model.named_parameters():
            if name not in NEED_KEYS and parameter.grad is not None:
                self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)), name)

    def test_parent_gate_map_is_independent_unregistered_and_restore_leaves_it_unchanged(self):
        model = NeuralSlots()
        configure(model)
        parent = {name: value.clone() for name, value in model.write_need_parent_parameters.items()}
        self.assertEqual(set(parent), set(NEED_KEYS))
        self.assertEqual(len(model.state_dict()), 91)
        self.assertFalse(any("parent" in name for name, _ in model.named_parameters()))
        for name, value in model.write_need_parent_parameters.items():
            self.assertFalse(value.requires_grad)
            self.assertNotEqual(value.data_ptr(), model.state_dict()[name].data_ptr())
        original = {name: value.clone() for name, value in model.state_dict().items()}
        with torch.no_grad():
            model.plan_need.weight.add_(1)
            model.plan_need.bias.add_(2)
        model.load_state_dict(original)
        self.assertTrue(all(torch.equal(value, parent[name]) for name, value in model.write_need_parent_parameters.items()))

    def test_development_guards_keep_every_source_reference_and_unsafe_floor(self):
        baseline = dict(committed_fact_f1=.8, plan_exact=.7, source_exact=.8, action_exact=.9,
                        reference_exact=.9, whole_reference_exact=.9, ambiguous_unsafe_writes=0, rows=10)
        for group in range(3):
            for metric in baseline.keys() - {"rows"}:
                groups = [deepcopy(baseline) for _ in range(3)]
                groups[group][metric] += 1 if metric == "ambiguous_unsafe_writes" else -.01
                self.assertFalse(writer_development_safe(groups[0], baseline, groups[1], baseline, groups[2], baseline), (group, metric))
        self.assertTrue(writer_development_safe(baseline, baseline, baseline, baseline, baseline, baseline))

    def test_assertion_minimum_quota_keeps_update_delete_and_default_sampler(self):
        positives = [dict(source_id=f"{action}-{i}", spans=[dict(action=action)])
                     for action, count in ((1, 1000), (2, 6), (3, 6)) for i in range(count)]
        negatives = [dict(source_id=f"negative-{i}", spans=[]) for i in range(20)]
        assertions = assertion_generalization_fit_rows()
        assertion_positive = [row for row in assertions if row["spans"]]
        assertion_negative = [row for row in assertions if not row["spans"]]
        for seed in range(20):
            for default_rows in (positives, positives[:1000], positives[:1006], positives[1000:], positives[:2]):
                default_rng, old_rng = random.Random(seed), random.Random(seed)
                current = sample_action_balanced_batch(default_rows, negatives, default_rng)
                expected = plain_batches_without_assertions(default_rows, negatives, old_rng)
                self.assertEqual(current, expected)
                self.assertEqual(default_rng.getstate(), old_rng.getstate())
            batch = sample_action_balanced_batch(positives + assertion_positive, negatives + assertion_negative, random.Random(seed))
            self.assertEqual(len(batch), 20)
            self.assertGreaterEqual(sum(bool(row.get("assertion_generalization")) and bool(row["spans"]) for row in batch), 2)
            self.assertGreaterEqual(sum(bool(row.get("assertion_generalization")) and not row["spans"] for row in batch), 1)
            for action in (2, 3):
                self.assertEqual(sum(any(span["action"] == action for span in row["spans"]) for row in batch), 2)

    def test_binding_requires_resume_and_finite_need_settings_before_loading_data(self):
        valid = dict(steps=1, eval_every=1, endpoint_weight=.5, endpoint_margin=1.,
                     boundary_weight=1., membership_weight=.5, boundary_anchor_weight=1.,
                     boundary_path_anchor_weight=10., boundary_path_anchor_margin=1000.,
                     train_heads="writer-context-binding", source_error_balance=True,
                     write_need_weight=1., write_need_anchor_margin=1., resume=None)
        with self.assertRaisesRegex(ValueError, "requires.*--resume"):
            train(SimpleNamespace(**valid))
        for name in ("write_need_weight", "write_need_anchor_margin"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, "finite"):
                    train(SimpleNamespace(**dict(valid, **{name: value, "resume": "unopened.pt"})))


if __name__ == "__main__":
    unittest.main()
