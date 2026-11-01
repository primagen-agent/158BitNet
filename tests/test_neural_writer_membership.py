"""Fit-only membership losses and immutable full-parent anchor contracts."""

from pathlib import Path
import math
import re
import sys
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import NeuralSlots, INPUT, WIDTH, MEMBERSHIP_INPUT, observation_context
from train_neural_dialogue import (
    build_writer_teacher,
    endpoint_contrast_loss,
    boundary_residual_supervision_loss,
    membership_residual_supervision_loss,
    membership_supervision_loss,
    observation_loss,
    parent_boundary_correction_teacher,
    plan_targets,
    snapshot_parent_boundary,
    source_types,
    source_words,
    writer_observation_key,
    writer_train_keys,
)


def fixture(model, positive=True):
    body = "I place the empty jar beside the wooden rack."
    speaker, timestamp = "Junia", "2059-05-14 09:20 UTC"
    context = observation_context(speaker, timestamp)
    units = lambda text: [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    body_units, context_units = units(body), units(context)
    dtype = next(model.parameters()).dtype
    words = torch.randn(len(body_units), INPUT, dtype=dtype) / 30
    context_words = torch.randn(len(context_units), INPUT, dtype=dtype) / 30
    item = dict(body=words, body_words=words, body_units=body_units,
                context=context_words, context_words=context_words, context_units=context_units)
    row = dict(body=body, context=context, speaker=speaker, timestamp=timestamp,
               spans=[dict(start=0, end=len(body), action=1)] if positive else [])
    return row, item


def configure(model, anchored=False, mode="writer-membership"):
    model.writer_mode = mode
    model.structured_weight, model.endpoint_weight = 1., .5
    model.boundary_weight, model.membership_weight = (1. if mode == "writer-content" else 0.), 1.
    model.boundary_anchor_weight = 1. if anchored else 0.
    model.boundary_path_anchor_weight = 10. if anchored else 0.
    model.boundary_path_anchor_margin = 1000.
    model.boundary_parent_parameters = snapshot_parent_boundary(model)
    model.writer_fit_teachers = {}
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in writer_train_keys(mode))


READ_CONTEXT_KEYS = {"read_context_memory.weight", "read_context_query.weight", "read_context_delta.weight"}


class WriterMembershipContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81073)
        torch.set_num_threads(1)

    def test_only_membership_tensor_is_trainable(self):
        model = NeuralSlots()
        self.assertEqual(len(model.state_dict()), 91)
        self.assertEqual(writer_train_keys("writer-membership"), ("plan_membership_raw.weight",))
        self.assertNotIn("plan_membership_raw.weight", writer_train_keys("writer-edge"))
        configure(model)
        self.assertEqual([name for name, p in model.named_parameters() if p.requires_grad],
                         ["plan_membership_raw.weight"])
        parent = model.boundary_parent_parameters
        self.assertEqual(set(parent), set(writer_train_keys("writer-edge")) | {"plan_membership_raw.weight"}
                         | set(writer_train_keys("writer-context-residual")))
        self.assertTrue(all(not value.requires_grad for value in parent.values()))

    def test_encoded_context_coefficients_receive_gradients_with_source_encoder_frozen(self):
        model = NeuralSlots().double()
        configure(model)
        row, item = fixture(model)
        encoded = model.plan_encoding(item["body"], item["context"], item["body_words"])
        self.assertFalse(encoded.requires_grad)
        self.assertEqual(tuple(model.plan_membership_raw.weight.shape), (1, MEMBERSHIP_INPUT))
        self.assertEqual(MEMBERSHIP_INPUT, WIDTH + 3 * INPUT)
        scores = model.plan_membership_scores(encoded, item["body_words"])
        F.softplus(scores).mean().backward()
        gradient = model.plan_membership_raw.weight.grad[:, :WIDTH]
        self.assertGreater(float(gradient.abs().sum()), 0.)
        self.assertTrue(torch.allclose(gradient[0], encoded.mean(0) / 2,
                                       atol=1e-12, rtol=1e-12))
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters()
                            if name != "plan_membership_raw.weight"))

    def test_joint_content_trains_exactly_six_boundary_and_membership_parameters(self):
        model = NeuralSlots().double()
        with torch.no_grad():
            model.plan_boundary[2].weight.normal_(0, .02)
            model.plan_membership_raw.weight.normal_(0, .02)
        configure(model, mode="writer-content")
        allowed = set(writer_train_keys("writer-content"))
        self.assertEqual(allowed, set(writer_train_keys("writer-edge")) | {"plan_membership_raw.weight"})
        self.assertEqual(len(allowed), 6)
        row, item = fixture(model)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        parent = {k: v.clone() for k, v in model.boundary_parent_parameters.items()}
        encoded, _, _ = model.plan(item["body"], item["context"], item["body_words"])
        kinds = source_types(row, item, len(encoded)).to(encoded.dtype)
        refs = tuple(v.clone() for v in model.query_reference(
            encoded.mean(0), source_words(item, len(encoded)), kinds, raw_context=model.query_binding_context(item["body"], item["context"])))
        read_query = tuple(v.clone() for v in model.query((item["body"], item["body_words"])))
        scope_query = tuple(v.clone() for v in model.query((item["body"], item["body_words"]), scope=True))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=.001, weight_decay=0)
        optimizer.zero_grad()
        observation_loss(model, row, item).backward()
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters() if name not in allowed))
        self.assertTrue(all(p.grad is not None and float(p.grad.abs().sum()) > 0
                            for name, p in model.named_parameters() if name in allowed))
        optimizer.step()
        changed = {k for k, v in model.state_dict().items() if not torch.equal(before[k], v)}
        self.assertEqual(changed, allowed)
        self.assertEqual(len(before) - len(changed), 85)
        frozen = {key: value for key, value in before.items() if key not in changed}
        self.assertTrue(READ_CONTEXT_KEYS <= frozen.keys())
        parameters = dict(model.named_parameters())
        for key in READ_CONTEXT_KEYS:
            self.assertFalse(parameters[key].requires_grad, key)
            self.assertIsNone(parameters[key].grad, key)
            self.assertEqual(parameters[key].detach().cpu().numpy().tobytes(),
                             frozen[key].cpu().numpy().tobytes(), key)
        new_encoded, _, _ = model.plan(item["body"], item["context"], item["body_words"])
        self.assertTrue(torch.equal(encoded, new_encoded))
        new_refs = model.query_reference(new_encoded.mean(0), source_words(item, len(encoded)), kinds, raw_context=model.query_binding_context(item["body"], item["context"]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(refs, new_refs)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(read_query, model.query((item["body"], item["body_words"])))))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(scope_query, model.query((item["body"], item["body_words"]), scope=True))))
        self.assertTrue(all(torch.equal(model.boundary_parent_parameters[k], v) for k, v in parent.items()))

    def test_joint_content_correct_nonzero_parent_has_zero_loss_and_all_six_zero_gradients(self):
        for dtype, positive in ((torch.float32, False), (torch.float32, True),
                                (torch.float64, False), (torch.float64, True)):
            model = NeuralSlots().to(dtype=dtype)
            with torch.no_grad():
                model.plan_tags.weight.zero_()
                model.plan_tags.bias.copy_(torch.tensor([-2., 4., -3., -3., 5.] if positive
                                                         else [4., -3., -3., -3., -3.]))
                model.plan_start.zero_()
                model.plan_end.zero_()
                model.plan_transition.zero_()
                model.plan_boundary[2].weight.normal_(0, .02)
                model.plan_boundary[2].bias.copy_(torch.tensor([.1, .2]))
                model.plan_membership_raw.weight.normal_(0, .02)
            configure(model, anchored=True, mode="writer-content")
            row, item = fixture(model, positive)
            key = writer_observation_key(row)
            model.boundary_fit_correct = {key: True}
            encoded, logits, ends = model.plan(item["body"], item["context"], item["body_words"])
            target = plan_targets(row, item["body_units"], logits.device)
            self.assertTrue(torch.equal(model.plan_labels(logits, ends), target))
            loss = observation_loss(model, row, item)
            self.assertEqual(float(loss.detach()), 0.)
            trainable = [p for p in model.parameters() if p.requires_grad]
            self.assertEqual(len(trainable), 6)
            gradients = torch.autograd.grad(loss, trainable)
            self.assertTrue(all(torch.equal(g, torch.zeros_like(g)) for g in gradients))
            teacher = model.writer_fit_teachers[key]
            self.assertTrue(torch.equal(logits, teacher["parent_logits"]))
            self.assertTrue(torch.equal(ends, teacher["parent_ends"]))
            self.assertTrue(torch.equal(model.plan_membership_scores(encoded, item["body_words"]), teacher["parent_membership"]))
            self.assertNotIn("contrast_paths", teacher)

    def test_joint_content_wrong_row_contains_boundary_correction_and_membership_objectives(self):
        model = NeuralSlots().double()
        configure(model, anchored=True, mode="writer-content")
        row, item = fixture(model)
        key = writer_observation_key(row)
        model.boundary_fit_correct = {key: False}
        with mock.patch("train_neural_dialogue.F.cross_entropy", side_effect=AssertionError("frozen tag loss")), mock.patch.object(model, "plan_reference", side_effect=AssertionError("frozen reference")), mock.patch.object(model, "query_reference", side_effect=AssertionError("frozen source query")):
            actual = observation_loss(model, row, item)
        encoded, logits, ends = model.plan(item["body"], item["context"], item["body_words"])
        teacher = model.writer_fit_teachers[key]
        self.assertIn("correction_indices", teacher)
        self.assertIn("preservation_indices", teacher)
        expected = (model.sequence_loss(logits, teacher["target"], ends)
                    + .5 * endpoint_contrast_loss(model, logits, teacher["target"], ends, teacher=teacher)
                    + membership_residual_supervision_loss(logits, model.plan_membership_scores(encoded, item["body_words"]), teacher)
                    + boundary_residual_supervision_loss(model.plan_boundary_scores(encoded, item["body_words"]), teacher))
        self.assertTrue(torch.equal(actual, expected))
        trainable = [p for p in model.parameters() if p.requires_grad]
        actual_grad = torch.autograd.grad(actual, trainable)
        expected_grad = torch.autograd.grad(expected, trainable)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual_grad, expected_grad)))

    def test_correct_zero_warm_parent_has_exact_zero_anchor_loss_and_gradient(self):
        for dtype, positive in ((torch.float32, False), (torch.float32, True),
                                (torch.float64, False), (torch.float64, True)):
            model = NeuralSlots().to(dtype=dtype)
            with torch.no_grad():
                model.plan_tags.weight.zero_()
                model.plan_tags.bias.copy_(torch.tensor([-2., 4., -3., -3., 5.] if positive
                                                         else [4., -3., -3., -3., -3.]))
                model.plan_start.zero_()
                model.plan_end.zero_()
                model.plan_transition.zero_()
            configure(model, anchored=True)
            row, item = fixture(model, positive)
            _, logits, ends = model.plan(item["body"], item["context"], item["body_words"])
            target = plan_targets(row, item["body_units"], logits.device)
            self.assertTrue(torch.equal(model.plan_labels(logits, ends), target))
            model.boundary_fit_correct = {writer_observation_key(row): True}
            loss = observation_loss(model, row, item)
            self.assertEqual(float(loss.detach()), 0.)
            gradient = torch.autograd.grad(loss, model.plan_membership_raw.weight)[0]
            self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))
            teacher = model.writer_fit_teachers[writer_observation_key(row)]
            self.assertNotIn("contrast_paths", teacher)
            self.assertTrue(torch.equal(teacher["parent_logits"], logits))

    def test_nonzero_parent_membership_is_in_cached_logits_and_anchor(self):
        model = NeuralSlots().double()
        with torch.no_grad():
            model.plan_membership_raw.weight.normal_(0, .2)
            model.plan_boundary[2].weight.normal_(0, .02)
        configure(model, anchored=True)
        row, item = fixture(model)
        key = writer_observation_key(row)
        model.boundary_fit_correct = {key: True}
        encoded, logits, ends = model.plan(item["body"], item["context"], item["body_words"])
        loss = observation_loss(model, row, item)
        self.assertEqual(float(loss.detach()), 0.)
        gradient = torch.autograd.grad(loss, model.plan_membership_raw.weight)[0]
        self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))
        teacher = model.writer_fit_teachers[key]
        self.assertTrue(torch.equal(logits, teacher["parent_logits"]))
        self.assertTrue(torch.equal(ends, teacher["parent_ends"]))
        self.assertTrue(torch.equal(model.plan_membership_scores(encoded, item["body_words"]),
                                    teacher["parent_membership"]))
        before = {k: v.clone() for k, v in teacher.items() if isinstance(v, torch.Tensor)}
        parent_before = {k: v.clone() for k, v in model.boundary_parent_parameters.items()}
        with torch.no_grad():
            model.plan_membership_raw.weight.add_(.03)
        current_membership = model.plan_membership_scores(encoded, item["body_words"])
        expected_mse = (current_membership - teacher["parent_membership"]).square().mean()
        later = observation_loss(model, row, item)
        self.assertGreater(float(later.detach()), 0.)
        self.assertTrue(torch.equal(model.boundary_row_anchor_parts[0], expected_mse))
        self.assertTrue(all(torch.equal(teacher[k], v) for k, v in before.items()))
        self.assertTrue(all(torch.equal(model.boundary_parent_parameters[k], v)
                            for k, v in parent_before.items()))
        self.assertTrue(all(not value.requires_grad for value in teacher.values()
                            if isinstance(value, torch.Tensor)))

    def test_total_emission_loss_matches_independent_class_balanced_binary_loss(self):
        target = torch.tensor([1, 4, 0, 0, 2, 0, 3, 4])
        logits = torch.randn(len(target), 5, dtype=torch.float64, requires_grad=True)
        positive, negative = target > 0, target == 0
        gold_margin = logits[torch.arange(len(target)), target] - logits[:, 0]
        reject_margin = torch.logsumexp(logits[:, 1:5], -1) - logits[:, 0]
        expected = (F.binary_cross_entropy_with_logits(gold_margin[positive], torch.ones_like(gold_margin[positive]))
                    + F.binary_cross_entropy_with_logits(reject_margin[negative], torch.zeros_like(reject_margin[negative]))) / 2
        actual = membership_supervision_loss(logits, target)
        self.assertTrue(torch.equal(actual, expected))
        expected_gradient = torch.autograd.grad(expected, logits, retain_graph=True)[0]
        actual_gradient = torch.autograd.grad(actual, logits)[0]
        self.assertTrue(torch.allclose(actual_gradient, expected_gradient, atol=1e-15, rtol=1e-15))

    def test_positive_and_negative_classes_are_balanced_at_zero_margin(self):
        for count in (8, 64, 128):
            delta = torch.zeros(count, dtype=torch.float64, requires_grad=True)
            logits = torch.cat((torch.full((count, 1), math.log(4), dtype=torch.float64),
                                delta[:, None].expand(-1, 4)), 1)
            # Gold B has margin0 at the positive; the negative all-source
            # rejection margin is0 when O equals log(4).
            logits = logits + F.pad(torch.tensor([-math.log(4)] + [0.] * (count - 1),
                                                 dtype=torch.float64)[:, None], (0, 4))
            target = torch.zeros(count, dtype=torch.long)
            target[0] = 1
            loss = membership_supervision_loss(logits, target)
            gradient = torch.autograd.grad(loss, delta)[0]
            self.assertAlmostEqual(float(loss.detach()), math.log(2), places=14)
            self.assertAlmostEqual(float(gradient[0]), -.25, places=14)
            self.assertAlmostEqual(float(gradient[1:].sum()), .25, places=14)

    def test_single_negative_class_mean_is_invariant_to_repeated_length(self):
        for count in (1, 8, 128):
            delta = torch.zeros((), dtype=torch.float64, requires_grad=True)
            logits = torch.cat((torch.full((count, 1), math.log(4), dtype=torch.float64),
                                delta.expand(count, 4)), 1)
            loss = membership_supervision_loss(logits, torch.zeros(count, dtype=torch.long))
            gradient = torch.autograd.grad(loss, delta)[0]
            self.assertAlmostEqual(float(loss.detach()), math.log(2), places=14)
            self.assertAlmostEqual(float(gradient), .5, places=14)

    def test_negative_residual_does_not_hide_a_strong_frozen_inside_emission(self):
        delta = torch.tensor(-8., dtype=torch.float64, requires_grad=True)
        frozen = torch.tensor([[0., -30., -30., -30., 16.]], dtype=torch.float64)
        logits = frozen + F.pad(delta.expand(1, 4), (1, 0))
        loss = membership_supervision_loss(logits, torch.tensor([0]))
        gradient = torch.autograd.grad(loss, delta)[0]
        residual_only = F.softplus(delta)
        residual_gradient = torch.autograd.grad(residual_only, delta)[0]
        self.assertGreater(float(loss.detach()), 8.)
        self.assertGreater(float(gradient), .99)
        self.assertLess(float(residual_gradient), .001)
        inside_gradient = torch.autograd.grad(membership_supervision_loss(logits, torch.tensor([0])), logits)[0]
        self.assertGreater(float(inside_gradient[0, 4]), .99)

    def test_wrong_high_begin_cannot_saturate_the_gold_inside_positive_target(self):
        # Legal gold path B/I: word1 belongs to a fact, but a wrong B logit8
        # masks its gold I logit-4 in the old any-source positive margin.
        frozen = torch.tensor([[0., 8., -30., -30., -30.],
                               [0., 8., -30., -30., -4.]], dtype=torch.float64)
        target = torch.tensor([1, 4])
        delta = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        logits = frozen + F.pad(delta[:, None].expand(-1, 4), (1, 0))
        old = F.softplus(-(torch.logsumexp(logits[:, 1:5], -1) - logits[:, 0])).mean()
        new = membership_supervision_loss(logits, target)
        old_delta = torch.autograd.grad(old, delta, retain_graph=True)[0]
        new_delta = torch.autograd.grad(new, delta, retain_graph=True)[0]
        new_logits = torch.autograd.grad(new, logits)[0]
        self.assertLess(float(old_delta[1].abs()), .001)
        self.assertLess(float(new_delta[1]), -.49)
        self.assertLess(float(new_logits[1, 4]), -.49)
        self.assertEqual(float(new_logits[1, 1]), 0.)
        self.assertGreater(float(new_logits[1, 0]), .49)

    def test_cached_positive_gold_tags_do_not_transfer_or_rebuild_in_hot_loss(self):
        target = torch.tensor([0, 1, 4, 0, 2, 0, 3, 4])
        teacher = build_writer_teacher(target, include_contrast=False)
        self.assertEqual(teacher["membership_positive"].tolist(), [1, 2, 4, 6, 7])
        self.assertEqual(teacher["membership_positive_tags"].tolist(), [1, 4, 2, 3, 4])
        logits = torch.randn(len(target), 5, dtype=torch.float64, requires_grad=True)
        expected = membership_supervision_loss(logits, target, teacher=teacher)
        with mock.patch("train_neural_dialogue.build_writer_teacher", side_effect=AssertionError("labels already cached")), mock.patch.object(torch.Tensor, "cpu", side_effect=AssertionError("hot loss cannot transfer tags")), mock.patch.object(torch.Tensor, "tolist", side_effect=AssertionError("hot loss cannot read tags")):
            actual = membership_supervision_loss(logits, target, teacher=teacher)
        self.assertTrue(torch.equal(actual, expected))
        actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, logits)[0]
        self.assertTrue(torch.equal(actual_grad, expected_grad))

    def test_one_missing_inside_is_not_diluted_by_many_correct_source_words(self):
        model = NeuralSlots().double()
        gradients = []
        losses = []
        for count in (4, 64, 128):
            gold = torch.tensor([1] + [4] * (count - 1))
            parent_logits = torch.full((count, 5), -30., dtype=torch.float64)
            parent_logits[:, 0] = 0.
            parent_logits[torch.arange(count - 1), gold[:-1]] = 8.
            parent_logits[-1, 4] = -4.
            teacher = build_writer_teacher(gold, include_contrast=False)
            teacher.update(parent_logits=parent_logits, parent_ends=torch.zeros(count, dtype=torch.float64),
                           parent_boundaries=torch.zeros(count, 2, dtype=torch.float64),
                           parent_membership=torch.zeros(count, dtype=torch.float64))
            parent_boundary_correction_teacher(model, teacher)
            error = teacher["membership_correction"]
            self.assertEqual(error["membership_positive"].tolist(), [count - 1])
            self.assertEqual(error["membership_positive_tags"].tolist(), [4])
            delta = torch.zeros(count, dtype=torch.float64, requires_grad=True)
            logits = parent_logits + F.pad(delta[:, None].expand(-1, 4), (1, 0))
            loss = membership_residual_supervision_loss(logits, delta, teacher)
            gradient = torch.autograd.grad(loss, delta, retain_graph=True)[0]
            full_gradient = torch.autograd.grad(membership_supervision_loss(logits, gold), delta)[0]
            self.assertTrue(torch.equal(gradient[:-1], torch.zeros_like(gradient[:-1])))
            self.assertLess(float(gradient[-1]), -.98)
            self.assertAlmostEqual(float(gradient[-1]), count * float(full_gradient[-1]), places=14)
            gradients.append(float(gradient[-1]))
            losses.append(float(loss.detach()))
        self.assertEqual(gradients, [gradients[0]] * len(gradients))
        self.assertEqual(losses, [losses[0]] * len(losses))

    def test_spurious_write_update_delete_and_inside_membership_all_receive_rejection(self):
        model = NeuralSlots().double()
        parent_tags = torch.tensor([1, 4, 2, 4, 3, 4])
        gold = torch.zeros_like(parent_tags)
        frozen = torch.full((len(gold), 5), -30., dtype=torch.float64)
        frozen[:, 0] = 0.
        frozen[torch.arange(len(gold)), parent_tags] = 8.
        teacher = build_writer_teacher(gold, include_contrast=False)
        teacher.update(parent_logits=frozen, parent_ends=torch.zeros(len(gold), dtype=torch.float64),
                       parent_boundaries=torch.zeros(len(gold), 2, dtype=torch.float64),
                       parent_membership=torch.zeros(len(gold), dtype=torch.float64))
        parent_boundary_correction_teacher(model, teacher)
        self.assertEqual(teacher["membership_correction"]["outside"].tolist(), list(range(len(gold))))
        self.assertEqual(len(teacher["membership_correction"]["membership_positive"]), 0)
        delta = torch.zeros(len(gold), dtype=torch.float64, requires_grad=True)
        logits = frozen + F.pad(delta[:, None].expand(-1, 4), (1, 0))
        loss = membership_residual_supervision_loss(logits, delta, teacher)
        delta_gradient, logits_gradient = torch.autograd.grad(loss, (delta, logits))
        self.assertGreater(float(loss.detach()), 8.)
        self.assertTrue((delta_gradient > .16).all())
        self.assertTrue((logits_gradient[torch.arange(len(gold)), parent_tags] > .16).all())
        self.assertTrue((logits_gradient[:, 0] < -.16).all())

    def test_membership_error_mask_is_immutable_and_matched_residual_is_preserved(self):
        model = NeuralSlots().double()
        gold = torch.tensor([1, 4, 4, 0, 0])
        parent = torch.tensor([1, 4, 0, 1, 0])
        old_member = torch.tensor([.2, -.3, .4, -.5, .6], dtype=torch.float64)
        frozen = torch.full((len(gold), 5), -8., dtype=torch.float64)
        frozen[torch.arange(len(gold)), parent] = 8.
        frozen = frozen + F.pad(old_member[:, None].expand(-1, 4), (1, 0))
        teacher = build_writer_teacher(gold, include_contrast=False)
        teacher.update(parent_logits=frozen, parent_ends=torch.zeros(len(gold), dtype=torch.float64),
                       parent_boundaries=torch.zeros(len(gold), 2, dtype=torch.float64),
                       parent_membership=old_member.clone())
        with mock.patch.object(model, "plan_labels", wraps=model.plan_labels) as decoder:
            parent_boundary_correction_teacher(model, teacher)
        self.assertEqual(decoder.call_count, 1)
        error = teacher["membership_correction"]
        self.assertEqual(error["membership_positive"].tolist(), [2])
        self.assertEqual(error["membership_positive_tags"].tolist(), [4])
        self.assertEqual(error["outside"].tolist(), [3])
        self.assertEqual(teacher["membership_preservation_indices"].tolist(), [0, 1, 4])
        before = {key: value.clone() for key, value in error.items()}
        preserved_before = teacher["membership_preservation_indices"].clone()
        current = (old_member + .1).requires_grad_()
        logits = frozen + F.pad((current - old_member)[:, None].expand(-1, 4), (1, 0))
        expected = (membership_supervision_loss(logits, gold, teacher=error)
                    + (current[[0, 1, 4]] - old_member[[0, 1, 4]]).square().mean())
        with mock.patch.object(model, "plan_labels", side_effect=AssertionError("never decode current or cached parent")), mock.patch.object(torch.Tensor, "cpu", side_effect=AssertionError("no hot labels to CPU")), mock.patch.object(torch.Tensor, "tolist", side_effect=AssertionError("no hot labels to host")):
            self.assertIs(parent_boundary_correction_teacher(model, teacher), teacher)
            actual = membership_residual_supervision_loss(logits, current, teacher)
        self.assertTrue(torch.equal(actual, expected))
        gradient = torch.autograd.grad(actual, current)[0]
        self.assertTrue(torch.allclose(gradient[[0, 1, 4]], torch.full((3,), .2 / 3, dtype=torch.float64),
                                       atol=1e-15, rtol=1e-15))
        self.assertLess(float(gradient[2]), 0.)
        self.assertGreater(float(gradient[3]), 0.)
        self.assertTrue(all(torch.equal(error[key], value) for key, value in before.items()))
        self.assertTrue(torch.equal(teacher["membership_preservation_indices"], preserved_before))
        self.assertTrue(torch.equal(teacher["parent_membership"], old_member))

    def test_wrong_observation_keeps_crf_and_endpoint_and_skips_frozen_losses(self):
        model = NeuralSlots().double()
        configure(model, anchored=True)
        row, item = fixture(model)
        key = writer_observation_key(row)
        model.boundary_fit_correct = {key: False}
        with mock.patch("train_neural_dialogue.F.cross_entropy", side_effect=AssertionError("no frozen tag loss")), mock.patch("train_neural_dialogue.boundary_residual_supervision_loss", side_effect=AssertionError("boundaries are frozen")), mock.patch.object(model, "plan_reference", side_effect=AssertionError("reference head frozen")), mock.patch.object(model, "query_reference", side_effect=AssertionError("query head frozen")):
            actual = observation_loss(model, row, item)
        encoded, logits, ends = model.plan(item["body"], item["context"], item["body_words"])
        teacher = model.writer_fit_teachers[key]
        target = teacher["target"]
        expected = (model.sequence_loss(logits, target, ends)
                    + .5 * endpoint_contrast_loss(model, logits, target, ends, teacher=teacher)
                    + membership_residual_supervision_loss(logits, model.plan_membership_scores(encoded, item["body_words"]), teacher))
        self.assertTrue(torch.equal(actual, expected))
        actual_gradient = torch.autograd.grad(actual, model.plan_membership_raw.weight)[0]
        expected_gradient = torch.autograd.grad(expected, model.plan_membership_raw.weight)[0]
        self.assertTrue(torch.equal(actual_gradient, expected_gradient))
        self.assertGreater(float(actual_gradient.abs().sum()), 0.)

    def test_one_unit_optimizer_step_changes_only_membership_and_preserves_action_differences(self):
        model = NeuralSlots().double()
        configure(model)
        row, item = fixture(model, positive=False)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        encoded, logits, ends = model.plan(item["body"], item["context"], item["body_words"])
        kinds = source_types(row, item, len(encoded)).to(encoded.dtype)
        refs = tuple(v.clone() for v in model.query_reference(
            encoded.mean(0), source_words(item, len(encoded)), kinds, raw_context=model.query_binding_context(item["body"], item["context"])))
        read_query = tuple(v.clone() for v in model.query((item["body"], item["body_words"])))
        scope_query = tuple(v.clone() for v in model.query((item["body"], item["body_words"]), scope=True))
        optimizer = torch.optim.AdamW([model.plan_membership_raw.weight], lr=.01, weight_decay=0)
        optimizer.zero_grad()
        observation_loss(model, row, item).backward()
        self.assertTrue(all(p.grad is None for name, p in model.named_parameters()
                            if name != "plan_membership_raw.weight"))
        optimizer.step()
        changed = {key for key, value in model.state_dict().items()
                   if not torch.equal(before[key], value)}
        self.assertEqual(changed, {"plan_membership_raw.weight"})
        new_encoded, new_logits, new_ends = model.plan(item["body"], item["context"], item["body_words"])
        self.assertTrue(torch.equal(encoded, new_encoded))
        self.assertTrue(torch.equal(ends, new_ends))
        self.assertTrue(torch.equal(logits[:, 0], new_logits[:, 0]))
        for tag in (2, 3, 4):
            self.assertTrue(torch.allclose(logits[:, tag] - logits[:, 1],
                                           new_logits[:, tag] - new_logits[:, 1], atol=1e-12, rtol=1e-12))
        new_refs = model.query_reference(new_encoded.mean(0), source_words(item, len(encoded)),
                                         kinds, raw_context=model.query_binding_context(item["body"], item["context"]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(refs, new_refs)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(read_query, model.query((item["body"], item["body_words"])))))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(scope_query, model.query((item["body"], item["body_words"]), scope=True))))


if __name__ == "__main__":
    unittest.main()
