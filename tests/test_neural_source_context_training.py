"""Learned source-context residuals, frozen bindings and complete parent anchors."""

from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import NeuralSlots, INPUT, HIDDEN, WIDTH, observation_context
from train_neural_dialogue import (
    build_writer_teacher,
    batch_loss,
    membership_residual_supervision_loss,
    observation_loss,
    parent_boundary_teacher,
    snapshot_parent_boundary,
    source_types,
    source_words,
    writer_observation_key,
    writer_train_keys,
)


CONTEXT_KEYS = (
    "plan_source_context.0.weight", "plan_source_context.0.bias",
    "plan_source_context.2.weight", "plan_source_context.2.bias",
)
MODES = ("writer-context-residual", "writer-content", "writer-context-content", "writer-membership")


def fixture(model, *, positive):
    body = "I keep a ceramic cup beside the blue lamp."
    speaker, timestamp = "Arden", "2058-08-16 10:45 UTC"
    context = observation_context(speaker, timestamp, "user: Hello there.\nassistant: Please continue.\n")
    units = lambda text: [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    body_units, context_units = units(body), units(context)
    dtype = next(model.parameters()).dtype
    words = torch.randn(len(body_units), INPUT, dtype=dtype) / 20
    context_words = torch.randn(len(context_units), INPUT, dtype=dtype) / 20
    item = dict(body=words, body_words=words, body_units=body_units,
                context=context_words, context_words=context_words, context_units=context_units)
    row = dict(body=body, context=context, speaker=speaker, timestamp=timestamp,
               spans=[dict(start=0, end=len(body), action=1)] if positive else [])
    return row, item


def configure(model, *, anchored=False, mode="writer-context-residual"):
    model.writer_mode = mode
    model.structured_weight, model.endpoint_weight = 1., .5
    model.boundary_weight = 1. if mode in ("writer-content", "writer-context-content") else 0.
    model.membership_weight = 1.
    model.boundary_anchor_weight = 1. if anchored else 0.
    model.boundary_path_anchor_weight = 10. if anchored else 0.
    model.boundary_path_anchor_margin = 1000.
    model.boundary_parent_parameters = snapshot_parent_boundary(model)
    model.writer_fit_teachers = {}
    model.source_error_balance = False
    allowed = writer_train_keys(mode)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowed)


def initialize_nonzero_parent(model):
    with torch.no_grad():
        model.plan_tags.bias.copy_(torch.tensor([8., -3., -3., -3., -3.]))
        model.plan_membership_raw.weight.normal_(0, .03)
        # Make this parent depend explicitly on current-word backbone hidden
        # values, so equality cannot hide an omitted raw-H teacher input.
        model.plan_source_context[0].weight[:, :2 * WIDTH].zero_()
        model.plan_source_context[0].weight[:, 2 * WIDTH:].normal_(0, .3)
        model.plan_source_context[0].bias.zero_()
        model.plan_source_context[2].weight.normal_(0, .2)
        model.plan_source_context[2].bias.fill_(.3)
        model.plan_boundary[2].weight.normal_(0, .03)
        model.plan_boundary[2].bias.copy_(torch.tensor([.1, .2]))


READ_CONTEXT_KEYS = {"read_context_memory.weight", "read_context_query.weight", "read_context_delta.weight"}


class SourceContextTrainingContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(148203)
        torch.set_num_threads(1)

    def test_context_mode_isolates_four_parameters_and_snapshots_complete_parent(self):
        model = NeuralSlots()
        configure(model)
        self.assertEqual(writer_train_keys("writer-context-residual"), CONTEXT_KEYS)
        self.assertEqual(tuple(name for name, p in model.named_parameters() if p.requires_grad), CONTEXT_KEYS)
        self.assertEqual(len(writer_train_keys("writer-content")), 6)
        self.assertTrue(set(CONTEXT_KEYS).isdisjoint(writer_train_keys("writer-content")))
        parent = model.boundary_parent_parameters
        self.assertEqual(len(parent), 10)
        self.assertEqual(set(parent), set(writer_train_keys("writer-content") + CONTEXT_KEYS))
        self.assertTrue(all(not value.requires_grad for value in parent.values()))
        self.assertTrue(all(value.data_ptr() != model.state_dict()[name].data_ptr()
                            for name, value in parent.items()))
        self.assertEqual(len(model.state_dict()), 91)
        self.assertEqual(model.plan_source_context[0].in_features, 2 * WIDTH + HIDDEN)
        with self.assertRaisesRegex(ValueError, "unknown writer training mode"):
            writer_train_keys("unknown")

    def test_zero_output_learns_final_layer_then_hidden_layer_with_old_bindings_frozen(self):
        model = NeuralSlots().double()
        with torch.no_grad():
            model.plan_tags.bias.copy_(torch.tensor([8., -3., -3., -3., -3.]))
            model.plan_membership_raw.weight.normal_(0, .03)
        configure(model)
        row, item = fixture(model, positive=True)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        full, body, logits, ends = model.plan_with_encodings(item["body"], item["context"], item["body_words"])
        self.assertTrue(torch.equal(model.plan_context_residual(full, body, item["body_words"]), torch.zeros(len(full), dtype=full.dtype)))
        self.assertTrue(torch.equal(model.plan_labels(logits, ends), torch.zeros(len(full), dtype=torch.long)))
        kinds = source_types(row, item, len(full)).to(full.dtype)
        refs = tuple(value.clone() for value in model.query_reference(
            full.mean(0), source_words(item, len(full)), kinds, raw_context=model.query_binding_context(item["body"], item["context"])))
        read_query = tuple(value.clone() for value in model.query((item["body"], item["body_words"])))
        scope_query = tuple(value.clone() for value in model.query((item["body"], item["body_words"]), scope=True))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=0)
        for step in range(2):
            optimizer.zero_grad()
            with mock.patch.object(model, "plan_reference", side_effect=AssertionError("frozen source references")), mock.patch.object(model, "query_reference", side_effect=AssertionError("frozen whole-query references")), mock.patch("train_neural_dialogue.boundary_residual_supervision_loss", side_effect=AssertionError("frozen boundaries")):
                observation_loss(model, row, item).backward()
            for name, parameter in model.named_parameters():
                if name not in CONTEXT_KEYS:
                    self.assertIsNone(parameter.grad)
                else:
                    self.assertIsNotNone(parameter.grad)
                    if step == 0 and name.startswith("plan_source_context.0."):
                        self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)))
                    else:
                        self.assertGreater(float(parameter.grad.abs().sum()), 0.)
            optimizer.step()
            changed = {name for name, value in model.state_dict().items() if not torch.equal(value, before[name])}
            self.assertEqual(changed, set(CONTEXT_KEYS[2:] if step == 0 else CONTEXT_KEYS))
        self.assertEqual(len(before) - len(changed), 87)
        frozen = {key: value for key, value in before.items() if key not in changed}
        self.assertTrue(READ_CONTEXT_KEYS <= frozen.keys())
        parameters = dict(model.named_parameters())
        for key in READ_CONTEXT_KEYS:
            self.assertFalse(parameters[key].requires_grad, key)
            self.assertIsNone(parameters[key].grad, key)
            self.assertEqual(parameters[key].detach().cpu().numpy().tobytes(),
                             frozen[key].cpu().numpy().tobytes(), key)
        after_full, after_body, after_logits, after_ends = model.plan_with_encodings(
            item["body"], item["context"], item["body_words"])
        self.assertTrue(torch.equal(full, after_full))
        self.assertTrue(torch.equal(body, after_body))
        self.assertTrue(torch.equal(ends, after_ends))
        self.assertTrue(torch.equal(logits[:, 0], after_logits[:, 0]))
        for tag in (2, 3, 4):
            self.assertTrue(torch.allclose(logits[:, tag] - logits[:, 1],
                                           after_logits[:, tag] - after_logits[:, 1], atol=1e-12, rtol=1e-12))
        after_refs = model.query_reference(after_full.mean(0), source_words(item, len(full)), kinds, raw_context=model.query_binding_context(item["body"], item["context"]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(refs, after_refs)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(read_query, model.query((item["body"], item["body_words"])))))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(scope_query, model.query((item["body"], item["body_words"]), scope=True))))

    def test_nonzero_parent_has_exact_zero_correct_anchor_in_all_modes_and_precisions(self):
        for dtype in (torch.float32, torch.float64):
            for mode in MODES:
                with self.subTest(dtype=dtype, mode=mode):
                    model = NeuralSlots().to(dtype=dtype)
                    initialize_nonzero_parent(model)
                    configure(model, anchored=True, mode=mode)
                    row, item = fixture(model, positive=False)
                    key = writer_observation_key(row)
                    model.boundary_fit_correct = {key: True}
                    full, body, logits, ends = model.plan_with_encodings(
                        item["body"], item["context"], item["body_words"])
                    residual = model.plan_context_residual(full, body, item["body_words"])
                    self.assertGreater(float(residual.detach().abs().sum()), 0.)
                    altered = item["body_words"].clone()
                    altered[:, :HIDDEN].neg_()
                    self.assertFalse(torch.equal(residual,
                                                 model.plan_context_residual(full, body, altered)))
                    self.assertTrue(torch.equal(model.plan_labels(logits, ends), torch.zeros(len(full), dtype=torch.long)))
                    loss = observation_loss(model, row, item)
                    self.assertEqual(float(loss.detach()), 0.)
                    self.assertTrue(all(float(part) == 0. for part in model.boundary_row_anchor_parts))
                    gradients = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad])
                    self.assertTrue(all(torch.equal(g, torch.zeros_like(g)) for g in gradients))
                    teacher = model.writer_fit_teachers[key]
                    self.assertTrue(torch.equal(logits, teacher["parent_logits"]))
                    self.assertTrue(torch.equal(ends, teacher["parent_ends"]))
                    self.assertTrue(torch.equal(model.plan_total_membership(full, body, item["body_words"]),
                                                teacher["parent_membership"]))
                    self.assertFalse(torch.equal(model.plan_membership_scores(full, item["body_words"]),
                                                 teacher["parent_membership"]))

    def test_nonzero_parent_matched_membership_preservation_starts_at_zero(self):
        for dtype in (torch.float32, torch.float64):
            for mode in MODES:
                with self.subTest(dtype=dtype, mode=mode):
                    model = NeuralSlots().to(dtype=dtype)
                    initialize_nonzero_parent(model)
                    configure(model, mode=mode)
                    _, item = fixture(model, positive=False)
                    full, body, logits, _ = model.plan_with_encodings(item["body"], item["context"], item["body_words"])
                    teacher = build_writer_teacher(torch.zeros(len(full), dtype=torch.long), include_contrast=False)
                    parent_boundary_teacher(model, full, item["body_words"], teacher, body_encoded=body)
                    empty = torch.empty(0, dtype=torch.long)
                    teacher["membership_correction"] = dict(membership_positive=empty, membership_positive_tags=empty, outside=empty)
                    teacher["membership_preservation_indices"] = torch.arange(len(full))
                    total = model.plan_total_membership(full, body, item["body_words"])
                    loss = membership_residual_supervision_loss(logits, total, teacher)
                    self.assertEqual(float(loss.detach()), 0.)
                    selected = [parameter for name, parameter in model.named_parameters()
                                if parameter.requires_grad and (name in CONTEXT_KEYS or name == "plan_membership_raw.weight")]
                    gradients = torch.autograd.grad(loss, selected)
                    self.assertTrue(all(torch.equal(g, torch.zeros_like(g)) for g in gradients))
                    incomplete = membership_residual_supervision_loss(
                        logits, model.plan_membership_scores(full, item["body_words"]), teacher)
                    self.assertGreater(float(incomplete.detach()), 0.)

    def test_parent_reconstruction_uses_frozen_network_and_raw_membership_after_child_updates(self):
        for dtype in (torch.float32, torch.float64):
            model = NeuralSlots().to(dtype=dtype)
            initialize_nonzero_parent(model)
            configure(model, mode="writer-context-content")
            _, item = fixture(model, positive=False)
            full, body, _, _ = model.plan_with_encodings(item["body"], item["context"], item["body_words"])
            parent = model.boundary_parent_parameters
            parent_before = {name: value.clone() for name, value in parent.items()}
            reference = model.plan_total_membership(full, body, item["body_words"]).detach().clone()
            with torch.no_grad():
                model.plan_membership_raw.weight.add_(.03)
                model.plan_source_context[0].weight.add_(.1)
                model.plan_source_context[2].bias.add_(.5)
            teacher = {}
            parent_boundary_teacher(model, full, item["body_words"], teacher, body_encoded=body)
            self.assertTrue(torch.equal(teacher["parent_membership"], reference))
            self.assertFalse(torch.equal(model.plan_total_membership(full, body, item["body_words"]), reference))
            again = NeuralSlots.total_membership_from_parameters(full, body, item["body_words"], parent)
            self.assertTrue(torch.equal(again, reference))
            self.assertTrue(all(torch.equal(value, parent_before[name]) for name, value in parent.items()))

    def test_joint_context_content_updates_ten_tensors_and_keeps_eighty_one_frozen(self):
        model = NeuralSlots().double()
        with torch.no_grad():
            model.plan_tags.bias.copy_(torch.tensor([8., -3., -3., -3., -3.]))
            model.plan_boundary[2].weight.normal_(0, .03)
            model.plan_membership_raw.weight.normal_(0, .03)
        configure(model, mode="writer-context-content")
        allowed = set(writer_train_keys("writer-context-content"))
        self.assertEqual(allowed, set(writer_train_keys("writer-content")) | set(CONTEXT_KEYS))
        self.assertEqual(len(allowed), 10)
        self.assertEqual(len(writer_train_keys("writer-content")), 6)
        row, item = fixture(model, positive=True)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        parent_before = {name: value.clone() for name, value in model.boundary_parent_parameters.items()}
        full, body, _, _ = model.plan_with_encodings(item["body"], item["context"], item["body_words"])
        kinds = source_types(row, item, len(full)).to(full.dtype)
        refs = tuple(value.clone() for value in model.query_reference(
            full.mean(0), source_words(item, len(full)), kinds, raw_context=model.query_binding_context(item["body"], item["context"])))
        read_query = tuple(value.clone() for value in model.query((item["body"], item["body_words"])))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.00025, weight_decay=0)
        for step in range(2):
            optimizer.zero_grad()
            with mock.patch.object(model, "plan_reference", side_effect=AssertionError("frozen source references")), mock.patch.object(model, "query_reference", side_effect=AssertionError("frozen whole-query references")):
                observation_loss(model, row, item).backward()
            for name, parameter in model.named_parameters():
                if name not in allowed:
                    self.assertIsNone(parameter.grad)
                elif step == 0 and name.startswith("plan_source_context.0."):
                    self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)))
                else:
                    self.assertGreater(float(parameter.grad.abs().sum()), 0.)
            optimizer.step()
        changed = {name for name, value in model.state_dict().items() if not torch.equal(value, before[name])}
        self.assertEqual(changed, allowed)
        self.assertEqual(len(before) - len(changed), 81)
        frozen = {key: value for key, value in before.items() if key not in changed}
        self.assertTrue(READ_CONTEXT_KEYS <= frozen.keys())
        parameters = dict(model.named_parameters())
        for key in READ_CONTEXT_KEYS:
            self.assertFalse(parameters[key].requires_grad, key)
            self.assertIsNone(parameters[key].grad, key)
            self.assertEqual(parameters[key].detach().cpu().numpy().tobytes(),
                             frozen[key].cpu().numpy().tobytes(), key)
        after_full, after_body, _, _ = model.plan_with_encodings(item["body"], item["context"], item["body_words"])
        self.assertTrue(torch.equal(full, after_full))
        self.assertTrue(torch.equal(body, after_body))
        after_refs = model.query_reference(after_full.mean(0), source_words(item, len(full)), kinds, raw_context=model.query_binding_context(item["body"], item["context"]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(refs, after_refs)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(read_query, model.query((item["body"], item["body_words"])))))
        self.assertTrue(all(torch.equal(value, parent_before[name])
                            for name, value in model.boundary_parent_parameters.items()))

    def test_residual_can_exceed_the_removed_delta_bound_without_changing_actions(self):
        model = NeuralSlots().double()
        configure(model)
        _, item = fixture(model, positive=False)
        full, body, initial, ends = model.plan_with_encodings(item["body"], item["context"], item["body_words"])
        with torch.no_grad():
            model.plan_source_context[2].bias.fill_(12.)
        residual = model.plan_context_residual(full, body, item["body_words"])
        self.assertTrue(torch.equal(residual, torch.full_like(residual, 12.)))
        _, _, changed, changed_ends = model.plan_with_encodings(item["body"], item["context"], item["body_words"])
        self.assertTrue(torch.equal(ends, changed_ends))
        self.assertTrue(torch.equal(initial[:, 0], changed[:, 0]))
        self.assertTrue(torch.allclose(changed[:, 1:] - initial[:, 1:], torch.full_like(initial[:, 1:], 12.), atol=1e-12, rtol=1e-12))

    def test_correct_replay_density_does_not_shrink_real_initial_correction_gradient(self):
        model = NeuralSlots().double()
        with torch.no_grad():
            model.plan_tags.bias.copy_(torch.tensor([8., -3., -3., -3., -3.]))
        configure(model, anchored=True)
        model.source_error_balance = True
        missing, item = fixture(model, positive=True)
        correct = dict(missing, spans=[])
        model.boundary_fit_correct = {writer_observation_key(missing): False,
                                      writer_observation_key(correct): True}
        prepared = {(missing["body"], missing["context"]): item}
        parameters = [p for p in model.parameters() if p.requires_grad]
        first = batch_loss(model, [missing, correct], prepared)
        first_gradient = torch.autograd.grad(first, parameters)
        repeated = batch_loss(model, [missing] + [correct] * 7, prepared)
        repeated_gradient = torch.autograd.grad(repeated, parameters)
        self.assertTrue(torch.equal(first, repeated))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first_gradient, repeated_gradient)))
        self.assertTrue(any(float(g.abs().sum()) > 0. for g in first_gradient))
        self.assertEqual(model.boundary_batch_stats["source_missing_rows"], 1)
        self.assertEqual(model.boundary_batch_stats["source_spurious_rows"], 0)
        self.assertEqual(model.boundary_batch_stats["anchored_rows"], 7)

    @staticmethod
    def reduction_fixture(kinds, *, enabled=True):
        parameter = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        rows, teachers, correctness, prepared = [], {}, {}, {}
        for index, kind in enumerate(kinds):
            row = dict(body="fit row " + str(index), context="fit context", spans=[])
            key = writer_observation_key(row)
            rows.append(row)
            correctness[key] = kind == "correct"
            teachers[key] = dict(membership_correction=dict(
                membership_positive=torch.tensor([0] if kind in ("FN", "both") else [], dtype=torch.long),
                outside=torch.tensor([0] if kind in ("FP", "both") else [], dtype=torch.long)))
            prepared[row["body"], row["context"]] = dict(kind=kind)
        model = SimpleNamespace(writer_mode="writer-context-residual", source_error_balance=enabled,
                                boundary_anchor_weight=1., boundary_path_anchor_weight=10.,
                                boundary_fit_correct=correctness, writer_fit_teachers=teachers)

        def loss(_, row, item):
            kind = item["kind"]
            if kind == "correct":
                model.boundary_row_anchor_parts = (parameter.detach().square(), parameter.detach() * 0)
                return parameter.square()
            targets = {"FN": 2., "FP": -1., "both": .2, "other": .4}
            return (parameter - targets[kind]).square()

        return model, rows, prepared, parameter, loss

    def test_repeating_spurious_rows_does_not_overpower_missing_class_gradient(self):
        results = []
        for kinds in (("FN", "FP", "correct"), ("FN",) + ("FP",) * 9 + ("correct",)):
            model, rows, prepared, parameter, loss = self.reduction_fixture(kinds)
            with mock.patch("train_neural_dialogue.observation_loss", side_effect=loss):
                value = batch_loss(model, rows, prepared)
            results.append((value, torch.autograd.grad(value, parameter)[0]))
        self.assertTrue(torch.equal(results[0][0], results[1][0]))
        self.assertTrue(torch.allclose(results[0][1], results[1][1], atol=1e-14, rtol=1e-14))
        self.assertAlmostEqual(float(results[0][1]), -1., places=12)
        self.assertEqual(model.boundary_batch_stats["source_spurious_rows"], 9)
        # The unchanged default would give the numerous FP rows greater weight.
        model, rows, prepared, parameter, loss = self.reduction_fixture(
            ("FN",) + ("FP",) * 9 + ("correct",), enabled=False)
        with mock.patch("train_neural_dialogue.observation_loss", side_effect=loss):
            value = batch_loss(model, rows, prepared)
        self.assertGreater(float(torch.autograd.grad(value, parameter)[0]), 0.)

    def test_overlap_and_absent_error_classes_have_finite_differentiable_losses(self):
        for kind, gradient in (("both", -.4), ("FN", -4.), ("FP", 2.),
                               ("other", -.8), ("correct", 0.)):
            with self.subTest(kind=kind):
                model, rows, prepared, parameter, loss = self.reduction_fixture((kind,))
                with mock.patch("train_neural_dialogue.observation_loss", side_effect=loss):
                    value = batch_loss(model, rows, prepared)
                actual = torch.autograd.grad(value, parameter)[0]
                self.assertTrue(torch.isfinite(value))
                self.assertAlmostEqual(float(actual), gradient, places=12)
                self.assertEqual(model.boundary_batch_stats["source_overlap_rows"], int(kind == "both"))

    def test_disabled_balancing_preserves_default_all_row_mean_and_gradient(self):
        model, rows, prepared, parameter, loss = self.reduction_fixture(
            ("FN", "FP", "correct", "correct"), enabled=False)
        with mock.patch("train_neural_dialogue.observation_loss", side_effect=loss):
            actual = batch_loss(model, rows, prepared)
        expected = torch.stack([loss(model, row, prepared[row["body"], row["context"]])
                                for row in rows]).mean()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(torch.autograd.grad(actual, parameter)[0],
                                    torch.autograd.grad(expected, parameter)[0]))

    def test_balancing_requires_isolated_context_modes_and_anchors_and_nonempty_batch(self):
        model, rows, prepared, _, loss = self.reduction_fixture(("FN",))
        with mock.patch("train_neural_dialogue.observation_loss", side_effect=loss):
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                batch_loss(model, [], prepared)
            for mode, anchors in (("references", True), ("writer-membership", True),
                                  ("writer-context-residual", False)):
                model.writer_mode = mode
                model.boundary_anchor_weight = float(anchors)
                model.boundary_path_anchor_weight = 10. * anchors
                with self.assertRaisesRegex(ValueError, "requires source-context residual training with anchors"):
                    batch_loss(model, rows, prepared)


if __name__ == "__main__":
    unittest.main()
