"""Trainer integration of isolated fit-only occupancy and structure margins."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import NeuralSlots
from test_neural_writer_binding import configure, fixture
from train_neural_dialogue import (
    batch_loss, observation_loss, source_correction_teacher_counts, train,
    writer_fit_training_diagnostic, writer_observation_key, writer_train_keys,
)


def model_and_row(*, structure=False, correct=False, need_wrong=False):
    model = NeuralSlots().double()
    with torch.no_grad():
        model.plan_tags.weight.zero_()
        if structure or correct:
            model.plan_tags.bias.copy_(torch.tensor([-1000., 8., -20., -20., -10.]))
        else:
            model.plan_tags.bias.copy_(torch.tensor([8., -3., -20., -20., -3.]))
        model.plan_need.weight.zero_()
        model.plan_need.bias.fill_(-3. if need_wrong else 3.)
    configure(model, anchored=True)
    model.factorized_source_corrections = True
    model.source_correction_margin = 1.
    row, item = fixture(model, short=not structure)
    model.boundary_fit_correct = {writer_observation_key(row): correct}
    return model, row, item


class FactorizedWriterTraining(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(61281)

    def test_occupancy_correction_routes_only_to_membership_and_replaces_old_losses(self):
        model, row, item = model_and_row()
        model.source_error_balance = True
        with mock.patch.object(model, "sequence_loss", side_effect=AssertionError("old CRF")), \
             mock.patch.object(model, "plan_reference", side_effect=AssertionError("unused reference tags")), \
             mock.patch.object(model, "query_reference", side_effect=AssertionError("unused query binding")), \
             mock.patch("train_neural_dialogue.endpoint_contrast_loss", side_effect=AssertionError("old endpoints")), \
             mock.patch("train_neural_dialogue.membership_residual_supervision_loss", side_effect=AssertionError("old membership BCE")), \
             mock.patch("train_neural_dialogue.boundary_residual_supervision_loss", side_effect=AssertionError("old boundary BCE")):
            loss = batch_loss(model, [row], {(row["body"], row["context"]): item})
        self.assertGreater(float(loss.detach()), 0.)
        self.assertEqual(float(model.factorized_source_row_parts[1]), 0.)
        loss.backward()
        self.assertGreater(float(model.plan_membership_raw.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.plan_source_context[2].weight.grad.abs().sum()), 0.)
        allowed = set(writer_train_keys("writer-membership")) | set(writer_train_keys("writer-context-residual"))
        for name, parameter in model.named_parameters():
            if name not in allowed and parameter.grad is not None:
                self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)), name)
        self.assertEqual(model.boundary_batch_stats["source_missing_rows"], 1)
        self.assertEqual(source_correction_teacher_counts(model), dict(
            visited_parent_wrong_source_rows=1, source_eligible_rows=1,
            source_preservation_eligible_rows=0,
            structure_eligible_rows=0, frozen_action_unreachable_rows=0))

    def test_structure_correction_omits_membership_and_routes_only_to_edge_heads(self):
        model, row, item = model_and_row(structure=True)
        loss = observation_loss(model, row, item)
        self.assertEqual(float(model.factorized_source_row_parts[0]), 0.)
        self.assertGreater(float(model.factorized_source_row_parts[1]), 0.)
        loss.backward()
        allowed = set(writer_train_keys("writer-edge"))
        self.assertGreater(float(model.plan_boundary[2].weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.plan_boundary_raw.weight.grad.abs().sum()), 0.)
        for name, parameter in model.named_parameters():
            if name not in allowed and parameter.grad is not None:
                self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)), name)
        teacher = model.writer_fit_teachers[writer_observation_key(row)]["source_correction"]
        self.assertFalse(teacher.source_eligible)
        self.assertTrue(teacher.source_preservation_eligible)
        self.assertTrue(teacher.structure_eligible)
        self.assertEqual(source_correction_teacher_counts(model)["source_preservation_eligible_rows"], 1)
        self.assertEqual(len(model.state_dict()), 91)
        self.assertEqual(len(writer_train_keys(model.writer_mode)), 12)

    def test_component_teacher_is_cached_from_parent_and_independent_of_student_drift(self):
        model, row, item = model_and_row()
        import neural_writer_correction
        with mock.patch("neural_writer_correction.build_writer_correction_teacher", wraps=neural_writer_correction.build_writer_correction_teacher) as build:
            observation_loss(model, row, item)
            teacher = model.writer_fit_teachers[writer_observation_key(row)]["source_correction"]
            parent_gap = teacher.parent_source_gap.clone()
            required = teacher.source_required_gap.clone()
            with torch.no_grad():
                model.plan_source_context[2].bias.add_(100.)
            loss = observation_loss(model, row, item)
        self.assertEqual(build.call_count, 1)
        self.assertIs(teacher, model.writer_fit_teachers[writer_observation_key(row)]["source_correction"])
        self.assertTrue(teacher.source_eligible)
        self.assertTrue(torch.equal(teacher.parent_source_gap, parent_gap))
        self.assertTrue(torch.equal(teacher.source_required_gap, required))
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad)) for p in model.parameters()))

    def test_correct_source_anchor_and_wrong_need_loss_are_exactly_preserved(self):
        model, row, item = model_and_row(correct=True, need_wrong=True)
        values, gradients = [], []
        for enabled in (False, True):
            model.factorized_source_corrections = enabled
            model.writer_fit_teachers, model.write_need_fit_teachers = {}, {}
            model.zero_grad(set_to_none=True)
            with mock.patch("neural_writer_correction.build_writer_correction_teacher", side_effect=AssertionError("correct-source rows keep their original anchor")):
                loss = observation_loss(model, row, item)
            values.append(loss.detach().clone())
            loss.backward()
            gradients.append({name: parameter.grad.clone() if parameter.grad is not None else None
                              for name, parameter in model.named_parameters()})
        self.assertTrue(torch.equal(*values))
        self.assertGreater(float(values[0]), 0.)
        for name in gradients[0]:
            if gradients[0][name] is None:
                self.assertIsNone(gradients[1][name])
            else:
                self.assertTrue(torch.equal(gradients[0][name], gradients[1][name]), name)
        self.assertEqual(source_correction_teacher_counts(model)["visited_parent_wrong_source_rows"], 0)

    def test_disabled_flag_matches_legacy_loss_gradients_and_rng_exactly(self):
        model, row, item = model_and_row(structure=True)
        losses, gradients, random_states = [], [], []
        for explicit_false in (False, True):
            if explicit_false:
                model.factorized_source_corrections = False
            else:
                del model.factorized_source_corrections
            model.writer_fit_teachers, model.write_need_fit_teachers = {}, {}
            model.zero_grad(set_to_none=True)
            losses.append(observation_loss(model, row, item))
            losses[-1].backward()
            gradients.append({name: parameter.grad.clone() if parameter.grad is not None else None
                              for name, parameter in model.named_parameters()})
            random_states.append(torch.get_rng_state())
        self.assertTrue(torch.equal(losses[0].detach(), losses[1].detach()))
        self.assertTrue(torch.equal(*random_states))
        for name in gradients[0]:
            if gradients[0][name] is None:
                self.assertIsNone(gradients[1][name])
            else:
                self.assertTrue(torch.equal(gradients[0][name], gradients[1][name]), name)

    def test_factorized_validation_requires_anchored_resumed_modes_and_finite_margin(self):
        settings = dict(steps=1, eval_every=1, endpoint_weight=.5, endpoint_margin=1.,
                        boundary_weight=1., membership_weight=.5, boundary_anchor_weight=1.,
                        boundary_path_anchor_weight=10., boundary_path_anchor_margin=1000.,
                        train_heads="writer-context-content", source_error_balance=True,
                        write_need_weight=1., write_need_anchor_margin=1., resume="unopened.pt",
                        factorized_source_corrections=True, source_correction_margin=1.)
        for updates, message in ((dict(resume=None), "require.*--resume"),
                                 (dict(train_heads="writer-context-residual"), "anchored"),
                                 (dict(boundary_anchor_weight=0., boundary_path_anchor_weight=0.), "anchored")):
            with self.assertRaisesRegex(ValueError, message):
                train(SimpleNamespace(**dict(settings, **updates)))
        for margin in (0., -1., float("nan"), float("inf")):
            with self.subTest(margin=margin), self.assertRaisesRegex(ValueError, "finite and positive"):
                train(SimpleNamespace(**dict(settings, source_correction_margin=margin)))

    def test_fit_diagnostics_capture_last_trained_weights_before_selected_state_restore(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.)
        selected = {name: value.clone() for name, value in model.state_dict().items()}
        fit_rows, prepared = [dict(body="fit fixture")], object()

        def fit_metrics(actual_model, actual_rows, actual_prepared):
            self.assertIs(actual_model, model)
            self.assertIs(actual_rows, fit_rows)
            self.assertIs(actual_prepared, prepared)
            self.assertFalse(torch.is_grad_enabled())
            return dict(rows=1, observed_weight=float(model.weight.detach()[0, 0])), []

        with mock.patch("train_neural_dialogue.evaluate", side_effect=fit_metrics) as evaluation:
            initial = writer_fit_training_diagnostic(
                model, fit_rows, prepared, step=0, stage="resumed-parent-before-training")
            with torch.no_grad():
                model.weight.fill_(3.)
            trained = writer_fit_training_diagnostic(
                model, fit_rows, prepared, step=800, stage="last-trained-before-best-restoration")
            model.load_state_dict(selected)
        self.assertEqual(evaluation.call_count, 2)
        self.assertEqual(initial["metrics"]["observed_weight"], 1.)
        self.assertEqual(trained["metrics"]["observed_weight"], 3.)
        self.assertEqual(float(model.weight.detach()[0, 0]), 1.)
        for diagnostic in (initial, trained):
            self.assertEqual(diagnostic["split"], "fit")
            self.assertEqual(diagnostic["role"], "training-observability-only")
            self.assertFalse(diagnostic["used_for_checkpoint_selection"])
            self.assertEqual(diagnostic["evaluation_device"], "cpu")

    def test_fit_diagnostic_does_not_mutate_parameters_gradients_teachers_or_rng(self):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.grad = torch.ones_like(model.weight) * 7
        model.writer_fit_teachers = {"fixed-fit": object()}
        teacher = model.writer_fit_teachers["fixed-fit"]
        state, grad, rng = model.weight.detach().clone(), model.weight.grad.clone(), torch.get_rng_state()
        with mock.patch("train_neural_dialogue.evaluate", return_value=(dict(rows=1, plan_exact=1.), [])):
            diagnostic = writer_fit_training_diagnostic(
                model, [dict(body="fit fixture")], {}, step=1, stage="last-trained-before-best-restoration")
        self.assertEqual(diagnostic["metrics"]["plan_exact"], 1.)
        self.assertTrue(torch.equal(model.weight, state))
        self.assertTrue(torch.equal(model.weight.grad, grad))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertIs(model.writer_fit_teachers["fixed-fit"], teacher)


if __name__ == "__main__":
    unittest.main()
