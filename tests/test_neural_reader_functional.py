"""Full-parent fit retention remains connected through frozen reader gates."""

import copy
from dataclasses import fields
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_reader_functional import (
    build_reader_functional_teacher,
    reader_functional_distillation_loss,
    reader_functional_margin_loss,
)
from neural_slot_model import INPUT, LEXICAL, NeuralSlots, QUERY, SEMANTIC, WIDTH


def scalar(value):
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)


def combined_activation(semantic, identity, temporal):
    probability = (F.logsigmoid(semantic) + F.logsigmoid(identity)
                   + temporal.log()).clamp_max(-1e-6)
    return probability - torch.log(-torch.expm1(probability))


def outputs(activation):
    """Gate-neutral logits make capped final-margin assertions exact."""
    return (activation, activation.clone(), torch.full_like(activation, 1000.),
            activation.new_zeros((len(activation), WIDTH)), torch.ones_like(activation))


def teacher_for(activation, gold, **kwargs):
    return build_reader_functional_teacher(*outputs(activation), gold,
                                           split="fit", **kwargs)


def model_outputs(model, original_features, evidence_features, bank, ids, metadata, mask):
    original, original_identity = model.query(original_features)
    evidence, evidence_identity = model.query(evidence_features)
    original, evidence = original[None], evidence[None]
    identity = model.advance_identity(original, evidence, original_identity[None], evidence_identity[None])
    q = model.advance_query(original, evidence)
    semantic = model.semantic_bank_logits(q, bank, mask, memory_context_latents=(q).new_zeros((len(mask), 16)))
    entity = model.identity_logits(identity, ids)
    intent = q[:, SEMANTIC:]
    expanded = intent[:, None].expand(-1, len(bank), -1)
    time = (F.logsigmoid(model.version(torch.cat((expanded, metadata[None]), -1)).squeeze(-1))
            + (F.logsigmoid(expanded[:, :, :3]) * metadata[None, :, :3]).sum(-1)).exp()
    activation = model.score_bank_from_semantic(semantic, q, identity, ids, metadata)
    return activation, semantic, entity, intent, time


class ReaderFunctionalContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71073)
        torch.set_num_threads(1)

    def test_initial_losses_and_all_input_gradients_are_exactly_zero(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                values = tuple(value.requires_grad_() for value in
                               outputs(torch.tensor([[.2, -.3, .8], [-.5, -.4, -.1]], dtype=dtype)))
                gold = torch.tensor([[1, 0, 1], [0, 0, 0]])
                teacher = build_reader_functional_teacher(*values, gold, split="fit")
                self.assertEqual(teacher.correct_queries.tolist(), [True, True])
                self.assertEqual(scalar(reader_functional_margin_loss(values[0], teacher)), 0.)
                loss = (reader_functional_margin_loss(values[0], teacher)
                        + reader_functional_distillation_loss(*values, teacher))
                self.assertEqual(scalar(loss), 0.)
                grads = torch.autograd.grad(loss, values)
                self.assertTrue(all(torch.equal(gradient, torch.zeros_like(gradient)) for gradient in grads))

    def test_teacher_has_no_gradient_or_alias_to_mutable_inputs(self):
        values = tuple(value.requires_grad_() for value in outputs(torch.tensor([[.5, -.2]])))
        gold = torch.tensor([[1., 0.]], requires_grad=True)
        teacher = build_reader_functional_teacher(*values, gold, split="fit")
        before = {field.name: getattr(teacher, field.name).clone()
                  for field in fields(teacher) if isinstance(getattr(teacher, field.name), torch.Tensor)}
        for value in before.values():
            self.assertFalse(value.requires_grad)
        for original, snapshot in zip(values, (teacher.parent_activation, teacher.parent_semantic,
                                               teacher.parent_identity, teacher.parent_intent,
                                               teacher.parent_temporal_gate)):
            self.assertNotEqual(original.data_ptr(), snapshot.data_ptr())
        with torch.no_grad():
            for value in values:
                value.add_(1.)
            gold.fill_(0)
        for name, value in before.items():
            self.assertTrue(torch.equal(getattr(teacher, name), value))
        with self.assertRaises(AttributeError):
            teacher.margin = 4.

    def test_final_margin_is_capped_and_restores_a_lost_complete_support(self):
        parent = torch.tensor([[10., -10.]], dtype=torch.float64)
        teacher = teacher_for(parent, torch.tensor([[1, 0]]))
        relaxed = torch.tensor([[2., -2.]], dtype=torch.float64, requires_grad=True)
        self.assertEqual(scalar(reader_functional_margin_loss(relaxed, teacher)), 0.)
        changed = torch.tensor([[-.5, -2.]], dtype=torch.float64, requires_grad=True)
        loss = reader_functional_margin_loss(changed, teacher)
        self.assertEqual(scalar(loss), 2.25)
        self.assertTrue(torch.equal(torch.autograd.grad(loss, changed)[0],
                                    torch.tensor([[-3., 0.]], dtype=torch.float64)))

    def test_worst_slot_loss_is_not_diluted_by_large_banks(self):
        results = []
        for negatives in (1, 99):
            parent = torch.tensor([[2.] + [-2.] * negatives], dtype=torch.float64)
            teacher = teacher_for(parent, torch.tensor([[1] + [0] * negatives]))
            changed = torch.tensor([[.5] + [-2.] * negatives], dtype=torch.float64, requires_grad=True)
            loss = reader_functional_margin_loss(changed, teacher)
            results.append((scalar(loss), scalar(torch.autograd.grad(loss, changed)[0][0, 0])))
        self.assertEqual(results, [(.25, -1.), (.25, -1.)])

    def test_parent_errors_are_excluded_without_student_changing_the_teacher_mask(self):
        parent = torch.tensor([[.7, -.8], [-.3, -.9], [.7, -.8]], dtype=torch.float64)
        teacher = teacher_for(parent, torch.tensor([[1, 0], [0, 0], [0, 1]]))
        self.assertEqual(teacher.protected_queries.tolist(), [True, True, False])
        unchanged_correct = torch.tensor([[.7, -.8], [-.3, -.9], [-100., 100.]],
                                         dtype=torch.float64, requires_grad=True)
        self.assertEqual(scalar(reader_functional_margin_loss(unchanged_correct, teacher)), 0.)
        changed = unchanged_correct.detach().clone().requires_grad_()
        with torch.no_grad():
            changed[1, 0] = .4
        loss = reader_functional_margin_loss(changed, teacher)
        self.assertAlmostEqual(scalar(loss), .245)
        gradient = torch.autograd.grad(loss, changed)[0]
        self.assertGreater(scalar(gradient[1, 0]), 0.)
        self.assertTrue(torch.equal(gradient[2], torch.zeros(2, dtype=torch.float64)))
        wrong_only = teacher_for(parent[2:], torch.tensor([[0, 1]]))
        live = tuple(value.clone().requires_grad_() for value in outputs(changed.detach()[2:]))
        loss = reader_functional_margin_loss(live[0], wrong_only) + reader_functional_distillation_loss(*live, wrong_only)
        self.assertEqual(scalar(loss), 0.)
        self.assertTrue(all(torch.equal(grad, torch.zeros_like(grad))
                            for grad in torch.autograd.grad(loss, live)))

    def test_probability_masks_downweight_irrelevant_components_and_saturated_logits(self):
        semantic = torch.tensor([[3., -10., 3.]], dtype=torch.float64)
        identity = torch.tensor([[3., 3., -10.]], dtype=torch.float64)
        time = torch.ones_like(semantic)
        activation = combined_activation(semantic, identity, time)
        intent = torch.zeros(1, WIDTH, dtype=torch.float64)
        teacher = build_reader_functional_teacher(activation, semantic, identity, intent, time,
                                                  torch.tensor([[1, 0, 0]]), split="fit")
        self.assertLess(scalar(teacher.semantic_mask[0, 2]), .0001)
        self.assertLess(scalar(teacher.identity_mask[0, 1]), .0001)
        self.assertLess(scalar(teacher.temporal_mask[0, 1]), .0001)
        live_semantic = semantic.clone().requires_grad_()
        with torch.no_grad():
            live_semantic[0, 1] -= 100
        loss = reader_functional_distillation_loss(activation, live_semantic, identity, intent, time, teacher)
        self.assertLess(scalar(loss), 1e-8)
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, live_semantic)[0]).all())

    def test_query_drift_through_frozen_identity_gate_has_restoring_gradient(self):
        model = NeuralSlots().double()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            model.identity_update.weight.zero_()
            model.identity_update.weight[0, QUERY] = 1.
            model.identity_update.bias.zero_()
            model.identity_match[0].weight.zero_()
            model.identity_match[0].weight[0, 0] = 2.
            model.identity_match[0].bias.zero_()
            model.identity_match[2].weight.zero_()
            model.identity_match[2].weight[0, 0] = 2.
            model.identity_match[2].bias.zero_()
        original_q = torch.zeros(1, QUERY, dtype=torch.float64)
        original_id = torch.zeros(1, LEXICAL + WIDTH, dtype=torch.float64)
        evidence_id = torch.zeros_like(original_id)
        original_id[0, 0] = original_id[0, LEXICAL] = 1.
        evidence_id[0, 1] = evidence_id[0, LEXICAL + 1] = 1.
        semantic = torch.tensor([[3.]], dtype=torch.float64)
        time = torch.tensor([[.88]], dtype=torch.float64)
        initial_q = original_q.clone()
        parent_id = model.advance_identity(original_q, initial_q, original_id, evidence_id)
        parent_entity = model.identity_logits(parent_id, original_id)
        parent = combined_activation(semantic, parent_entity, time)
        teacher = build_reader_functional_teacher(parent, semantic, parent_entity,
                                                  initial_q[:, SEMANTIC:], time,
                                                  torch.ones_like(parent), split="fit")
        self.assertTrue(teacher.protected_queries.all())
        live_q = original_q.clone().requires_grad_()
        with torch.no_grad():
            live_q[0, 0] = 2.
        live_id = model.advance_identity(original_q, live_q, original_id, evidence_id)
        live_entity = model.identity_logits(live_id, original_id)
        live = combined_activation(semantic, live_entity, time)
        loss = reader_functional_margin_loss(live, teacher)
        self.assertGreater(scalar(loss), 0.)
        gradient = torch.autograd.grad(loss, live_q)[0]
        self.assertGreater(scalar(gradient[0, 0]), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_independent_parent_survives_joint_student_drift_and_connects_semantic_heads(self):
        model = NeuralSlots().double()
        with torch.no_grad():
            model.identity_update.weight.normal_(0, .05)
            model.identity_update.bias.zero_()
            model.read_update.weight.normal_(0, .05)
        parent = copy.deepcopy(model)
        for parameter in parent.parameters():
            parameter.requires_grad_(False)
        trainable = ("query_encode.", "query_pool.", "read.", "read_update.", "memory_attention_logits")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(trainable))
        features = (torch.randn(4, INPUT, dtype=torch.float64) * .2,
                    torch.randn(3, INPUT, dtype=torch.float64) * .2)
        evidence = (torch.randn(5, INPUT, dtype=torch.float64) * .2,
                    torch.randn(2, INPUT, dtype=torch.float64) * .2)
        bank = torch.randn(3, 4, WIDTH, dtype=torch.float64)
        ids = torch.randn(3, LEXICAL + WIDTH, dtype=torch.float64)
        ids = torch.cat((F.normalize(ids[:, :LEXICAL], dim=-1),
                         F.normalize(ids[:, LEXICAL:], dim=-1)), -1)
        metadata = torch.tensor([[1., 0, 0, 0], [0., 1., 0, 0], [0, 0, 1., 0]], dtype=torch.float64)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
        with torch.no_grad():
            parent_values = model_outputs(parent, features, evidence, bank, ids, metadata, mask)
        teacher = build_reader_functional_teacher(*parent_values, parent_values[0] > 0, split="fit")
        initial_values = model_outputs(model, features, evidence, bank, ids, metadata, mask)
        self.assertEqual(scalar(reader_functional_distillation_loss(*initial_values, teacher)), 0.)
        with torch.no_grad():
            model.query_encode.weight[0].add_(.2)
        values = model_outputs(model, features, evidence, bank, ids, metadata, mask)
        self.assertFalse(torch.equal(values[2], parent_values[2]))
        self.assertFalse(torch.equal(values[4], parent_values[4]))
        loss = reader_functional_distillation_loss(*values, teacher)
        self.assertGreater(scalar(loss), 0.)
        loss.backward()
        for names in (("query_encode.weight",), ("query_pool.weight",),
                      ("read_update.weight",), ("memory_attention_logits",)):
            self.assertGreater(sum(scalar(dict(model.named_parameters())[name].grad.abs().sum())
                                   for name in names), 0.)
        self.assertTrue(all(parameter.grad is None for name, parameter in model.named_parameters()
                            if not name.startswith(trainable)))
        with torch.no_grad():
            unchanged = model_outputs(parent, features, evidence, bank, ids, metadata, mask)
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(parent_values, unchanged)))

    def test_temporal_input_drift_is_protected_even_when_version_parameters_are_frozen(self):
        model = NeuralSlots().double()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            model.version[0].weight.zero_()
            model.version[0].weight[0, 3] = 1.
            model.version[0].bias.zero_()
            model.version[2].weight.zero_()
            model.version[2].weight[0, 0] = 1.
            model.version[2].bias.zero_()
        semantic = torch.tensor([[5.]], dtype=torch.float64)
        identity = torch.tensor([[5.]], dtype=torch.float64)
        metadata = torch.zeros(1, 4, dtype=torch.float64)
        intent = torch.zeros(1, WIDTH, dtype=torch.float64)

        def time_gate(values):
            return model.version(torch.cat((values, metadata), -1)).sigmoid()

        parent_time = time_gate(intent)
        parent = combined_activation(semantic, identity, parent_time)
        teacher = build_reader_functional_teacher(parent, semantic, identity, intent, parent_time,
                                                  torch.zeros_like(parent), split="fit")
        self.assertTrue(teacher.protected_queries.all())
        live_intent = intent.clone().requires_grad_()
        with torch.no_grad():
            live_intent[0, 3] = 1.
        live = combined_activation(semantic, identity, time_gate(live_intent))
        loss = reader_functional_margin_loss(live, teacher)
        self.assertGreater(scalar(loss), 0.)
        self.assertGreater(scalar(torch.autograd.grad(loss, live_intent)[0][0, 3]), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_empty_banks_and_query_groups_return_connected_zero_losses(self):
        for shape in ((0, 3), (2, 0)):
            values = tuple(value.requires_grad_() for value in outputs(torch.empty(shape)))
            teacher = build_reader_functional_teacher(*values, torch.zeros(shape), split="fit")
            loss = (reader_functional_margin_loss(values[0], teacher)
                    + reader_functional_distillation_loss(*values, teacher))
            self.assertEqual(scalar(loss), 0.)
            self.assertTrue(all(gradient.shape == value.shape
                                for gradient, value in zip(torch.autograd.grad(loss, values), values)))

    def test_rejects_evaluation_labels_bad_geometry_dtypes_and_nonfinite_controls(self):
        values = outputs(torch.tensor([[.2, -.3]], dtype=torch.float64))
        gold = torch.tensor([[1, 0]])
        for split in ("development", "qualification", "runtime", "source"):
            with self.assertRaisesRegex(ValueError, "must come from fit"):
                build_reader_functional_teacher(*values, gold, split=split)
        with self.assertRaisesRegex(ValueError, "must come from fit"):
            build_reader_functional_teacher(*values, gold, split="fit", evaluation_only=True)
        for index, invalid in ((1, values[1].float()), (2, values[2][:, :1]),
                               (3, torch.zeros(1, WIDTH - 1, dtype=torch.float64)),
                               (4, torch.full_like(values[4], 1.1)),
                               (0, torch.full_like(values[0], scalar("nan")))):
            bad = list(values)
            bad[index] = invalid
            with self.assertRaises(ValueError):
                build_reader_functional_teacher(*bad, gold, split="fit")
        for labels in (torch.tensor([[2, 0]]), torch.tensor([[1.]]), torch.tensor([[scalar("nan"), 0]])):
            with self.assertRaises(ValueError):
                build_reader_functional_teacher(*values, labels, split="fit")
        for margin in (-1., 0., scalar("inf"), scalar("nan")):
            with self.assertRaisesRegex(ValueError, "margin must be finite and positive"):
                build_reader_functional_teacher(*values, gold, split="fit", margin=margin)
        teacher = build_reader_functional_teacher(*values, gold, split="fit")
        for weight in (-1., scalar("inf"), scalar("nan")):
            with self.assertRaisesRegex(ValueError, "weight must be finite and nonnegative"):
                reader_functional_margin_loss(values[0], teacher, weight)
            with self.assertRaisesRegex(ValueError, "weight must be finite and nonnegative"):
                reader_functional_distillation_loss(*values, teacher, weight)
        with self.assertRaises(ValueError):
            reader_functional_margin_loss(values[0].float(), teacher)
        with self.assertRaisesRegex(ValueError, "cannot exceed any learned gate"):
            build_reader_functional_teacher(values[0] + 100, *values[1:], gold, split="fit")


if __name__ == "__main__":
    unittest.main()
