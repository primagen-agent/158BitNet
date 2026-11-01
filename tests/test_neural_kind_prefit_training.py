"""Deterministic kind calibration precedes six-parameter time refinement."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from neural_slot_model import FORMAT, INPUT, NeuralSlots, WIDTH
from neural_kind_context import calibrate_kind_context
import train_neural_dialogue_reader as reader
import test_neural_temporal_retention as retention
from test_neural_temporal_curriculum import small_fit


class ControlledKindCache:
    """Tiny controlled memory/context vectors; no backbone feature cache."""
    def __init__(self, model, vectors, contexts):
        self.model, self.vectors = model, vectors
        self.freeze_memory_encoder = True
        self.index = {text: i for i, text in enumerate(sorted(contexts))}
        self.kind_contexts = torch.stack([contexts[text] for text in sorted(contexts)]).to(model.kind.weight)

    def indices(self, texts):
        return torch.tensor([self.index[text] for text in texts], device=self.model.kind.weight.device)

    def encode(self, texts):
        device = self.model.kind.weight.device
        values = torch.stack([self.vectors[text] for text in texts]).to(device)
        return values[:, None], torch.ones(len(texts), 1, dtype=torch.bool, device=device), torch.arange(len(texts), device=device)


class KindPrefitTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def controlled_fixture(self, device="cpu"):
        torch.manual_seed(73103)
        model = NeuralSlots().to(device).eval()
        reader.configure_reader_heads(model, "temporal", freeze_memory_encoder=True)
        pools, vectors, contexts = {0: [], 1: [], 2: []}, {}, {}
        for kind, count in ((0, 536), (1, 64), (2, 28)):
            for index in range(count):
                fact = dict(text=f"Ada. Original class {kind} source {index}.",
                            entity="Ada", reference="Ada", prop="detail", kind=kind)
                if kind == 0 and index == 0:
                    fact.update(kind=1, source_kind=0, status=1)
                pools[kind].append(fact)
                vector = torch.zeros(WIDTH)
                vector[3 if kind == 2 and index >= 16 else kind] = 1
                vectors[fact["text"]] = vector
                context = torch.zeros(4096)
                context[kind] = 1.
                contexts[fact["text"]] = context
        with torch.no_grad():
            model.kind.weight.zero_();model.kind.bias.zero_()
            for kind in (0, 1, 2):
                model.kind.weight[kind, kind] = 5
        return model, dict(source_kind_facts=pools), ControlledKindCache(model, vectors, contexts)

    def test_complete_628_calibration_preserves_kind64_and_freezes_kind_context(self):
        model, prepared, cached = self.controlled_fixture()
        parent = deepcopy(model).eval().requires_grad_(False)
        frozen = {name: value.detach().clone() for name, value in model.named_parameters() if not name.startswith("kind_context.")}
        before_parent = deepcopy(parent.state_dict())
        with patch("neural_kind_context.calibrate_kind_context", wraps=calibrate_kind_context) as calibrate:
            result = reader.prefit_temporal_kind(model, prepared, cached)
        self.assertTrue(result["accepted"])
        self.assertEqual(calibrate.call_args.args[1].shape, (628, 64))
        self.assertEqual(calibrate.call_args.args[2].shape, (628, 4096))
        self.assertEqual(calibrate.call_args.args[3].tolist(), [0] * 536 + [1] * 64 + [2] * 28)
        self.assertEqual(result["initial"]["correct"], 616)
        self.assertEqual(result["final"]["correct"], 628)
        self.assertFalse(model.kind.weight.requires_grad)
        self.assertFalse(model.kind.bias.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in model.kind_context.parameters()))
        self.assertEqual({name for name, p in model.named_parameters() if p.requires_grad},
                         {"intent.weight", "intent.bias", "version.0.weight", "version.0.bias", "version.2.weight", "version.2.bias"})
        for name, parameter in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(parameter, frozen[name], atol=0, rtol=0)
        for name, value in parent.state_dict().items():
            torch.testing.assert_close(value, before_parent[name], atol=0, rtol=0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in parent.parameters()))

    def exercise_six_parameter_refinement(self, device="cpu"):
        model, facts, rows, kinds, cached = retention.TemporalRetentionContract().fixture(device)
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        pools = {kind: [f for f in kinds if f.get("source_kind", f["kind"]) == kind] for kind in (0, 1, 2)}
        result = reader.prefit_temporal_kind(model, dict(source_kind_facts=pools), cached)
        self.assertTrue(result["accepted"])
        after_prefit = {name: p.detach().clone() for name, p in model.named_parameters()}
        self.assertTrue(any(not torch.equal(before[name], after_prefit[name]) for name in ("kind_context.weight", "kind_context.bias")))
        for name in before:
            if not name.startswith("kind_context."):
                torch.testing.assert_close(before[name], after_prefit[name], atol=0, rtol=0)
        optimizer = reader.reader_optimizer(model, torch.nn.Linear(WIDTH, 3).to(device), "temporal", .001, .01)
        active = {name: p for name, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(len(active), 6)
        self.assertEqual({id(p) for group in optimizer.param_groups for p in group["params"]}, {id(p) for p in active.values()})
        loss = reader.temporal_training_loss(model, rows, facts, cached, kind_facts=kinds,
                                            parent=parent, parent_cached=parent_cached)
        replay, _ = reader.functional_replay_loss(model, parent, rows, facts, cached, parent_cached,
                                                 temporal_only=True, intent_weight=.25)
        (loss + replay).backward()
        for name, p in model.named_parameters():
            if name in active:
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
                self.assertGreater(float(p.grad.abs().sum()), 0, name)
            else:
                self.assertIsNone(p.grad, name)
        optimizer.step()
        for name, p in model.named_parameters():
            if name not in active:
                torch.testing.assert_close(p, after_prefit[name], atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in parent.parameters()))

    def test_main_optimizer_updates_six_and_preserves_calibrated_kind_and_other81(self):
        self.exercise_six_parameter_refinement()

    def test_configuration_and_fit_ownership_fail_before_fitting(self):
        for enabled, focus in ((-1, True), (0, True), (1, True), (1.5, True),
                               (None, True), ("true", True), (True, False),
                               (True, 1), (False, None)):
            with self.assertRaises(ValueError):
                reader.validate_kind_prefit(enabled, focus)
        self.assertIs(reader.validate_kind_prefit(False, False), False)
        self.assertIs(reader.validate_kind_prefit(False, True), False)
        self.assertIs(reader.validate_kind_prefit(True, True), True)
        model, prepared, cached = self.controlled_fixture()
        invalid = []
        for changes in (dict(evaluation_only=True), dict(split="development"), dict(evidence_challenge_view=0)):
            damaged = deepcopy(prepared);damaged["source_kind_facts"][0][0].update(changes)
            invalid.append((damaged, cached))
        invalid.append((dict(source_kind_facts={0: [], 1: [], 2: []}), cached))
        invalid.append((prepared, SimpleNamespace(model=model, freeze_memory_encoder=False)))
        invalid.append((prepared, SimpleNamespace(model=deepcopy(model), freeze_memory_encoder=True)))
        with patch("neural_kind_context.calibrate_kind_context", side_effect=AssertionError("invalid fit reached solver")):
            for pools, owner in invalid:
                with self.assertRaises(ValueError):
                    reader.prefit_temporal_kind(model, pools, owner)

    def test_rejected_prefit_keeps_real_initial_head_and_does_not_freeze_it(self):
        model, prepared, cached = self.controlled_fixture()
        before = deepcopy(model.kind_context.state_dict())
        # Controlled rejection verifies caller behavior; exact helper rollback
        # is independently exercised in the pure calibration suite.
        with patch("neural_kind_context.calibrate_kind_context", return_value=dict(accepted=False)):
            with self.assertRaisesRegex(RuntimeError, "initial head preserved"):
                reader.prefit_temporal_kind(model, prepared, cached)
        for name, value in model.kind_context.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        self.assertTrue(all(p.requires_grad for p in model.kind_context.parameters()))

    def exercise_baseline_and_early_veto(self, veto=False):
        fit = small_fit()
        for q in fit["queries"]:
            q.setdefault("observation", dict(body=q["text"], context="Original fit context", source_id=q["observation_id"]))
            q.update(binding_predicted=True, unresolved=False)
        model = NeuralSlots()
        identity = "a" * 64
        class NoEvaluationCache:
            def __init__(self, owner, *args, **kwargs):
                self.model = owner
                self.freeze_memory_encoder = True
            def for_model(self, owner):
                return NoEvaluationCache(owner)
        metrics = dict(activation_f1=.8, support_set_exact=.75, positive_queries=4,
                       positive_exact=3, positive_support_set_exact=.75, null_false_activation=0.,
                       tp=3, fp=1, fn=1, bank_size=36, queries=4)
        captured = {}
        def extracted(_args, groups):
            texts = {r[k] for rows in groups.values() for r in rows for k in ("body", "context")}
            return {text: dict(x=torch.zeros(1, INPUT), bounds=[(0, len(text))]) for text in texts}, {}
        def kind_metrics(_model, group, _cached):
            counts = [sum(f.get("source_kind", f["kind"]) == i for f in group["facts"]) for i in range(3)]
            confusion = [[counts[i] if i == j else 0 for j in range(3)] for i in range(3)]
            if veto and "kind_before" in captured:
                confusion[0][0] -= 1;confusion[0][2] += 1
            predictions = [sum(row[column] for row in confusion) for column in range(3)]
            return dict(facts=sum(counts), teacher_counts=counts, prediction_counts=predictions,
                        confusion=confusion)
        with tempfile.TemporaryDirectory(dir=ROOT / "build") as temporary:
            temporary = Path(temporary)
            checkpoint = temporary / "initial.pt"
            torch.save(dict(format=FORMAT, backbone_sha256=identity, state_dict=model.state_dict()), checkpoint)
            output = temporary / "run"
            args = SimpleNamespace(steps=1, eval_every=1, train_heads="temporal", resume=str(checkpoint),
                planner_checkpoint=str(checkpoint), output=str(output), kind_context_prefit=True,
                reader_functional_weight=1., reader_functional_margin=1., reader_anchor_weight=0.,
                reader_anchor_margin=1., freeze_memory_encoder=True, feature_dropout=0.,
                semantic_topic_weight=0., curriculum=None, fit_paraphrases=False, replay_episodes=False,
                relation_paraphrases=False, episode_relations=False, relation_bindings=False,
                gguf="mock-backbone", probe="mock-probe", features="mock-features", extra_features=[],
                evidence_cues=False, expansion=False, identity_diversity=False, qualification_curriculum=None,
                device="cpu", learning_rate=.001, attention_learning_rate=.01, activation_positive_weight=1.)
            def stop_after_baseline(student, _prepared, _cached, **kwargs):
                saved = torch.load(output / "selected.pt", weights_only=False, map_location="cpu")
                self.assertEqual(saved["reader_step"], 0)
                self.assertIsNone(saved["kind_prefit"])
                captured["kind_before"] = student.kind_context.bias.detach().clone()
                torch.testing.assert_close(saved["state_dict"]["kind_context.bias"], captured["kind_before"], atol=0, rtol=0)
                with torch.no_grad():
                    student.kind_context.bias.add_(7)
                self.assertFalse(torch.equal(student.kind_context.bias, saved["state_dict"]["kind_context.bias"]))
                if veto:
                    student.kind.requires_grad_(False)
                    student.kind_context.requires_grad_(False)
                    return dict(accepted=True)
                raise RuntimeError("stop after preserved real baseline")
            with patch("prepare_neural_colloquial.reader_data", side_effect=lambda *_args: deepcopy(fit)), \
                 patch.object(reader, "sha", return_value=identity), \
                 patch.object(reader, "load_features", return_value=dict(identity=dict(backbone=identity, probe=identity), features={})), \
                 patch.object(reader, "extract", side_effect=extracted), \
                 patch.object(reader, "bind_curriculum_queries", side_effect=lambda g, *_args: (deepcopy(g), [])), \
                 patch.object(reader, "DynamicFeatures", NoEvaluationCache), \
                 patch.object(reader, "evaluate", return_value=metrics), \
                 patch.object(reader, "scope_evaluate", return_value=metrics), \
                 patch.object(reader, "kind_bank_diagnostics", side_effect=kind_metrics), \
                 patch.object(reader, "attention_bank_diagnostics", return_value={}), \
                 patch.object(reader, "prefit_temporal_kind", side_effect=stop_after_baseline), \
                 patch.object(reader, "temporal_training_loss", side_effect=AssertionError("vetoed time training ran")), \
                 patch.object(reader, "functional_replay_loss", side_effect=AssertionError("vetoed replay training ran")), \
                 patch.object(reader, "export", return_value="test-artifact-identity"), \
                 patch("builtins.print"):
                if veto:
                    reader.train(args)
                else:
                    with self.assertRaisesRegex(RuntimeError, "preserved real baseline"):
                        reader.train(args)
            self.assertIn("kind_before", captured)
            saved = torch.load(output / "selected.pt", weights_only=False, map_location="cpu")
            self.assertEqual(saved["reader_step"], 0)
            torch.testing.assert_close(saved["state_dict"]["kind_context.bias"], captured["kind_before"], atol=0, rtol=0)
            if veto:
                import json
                report = json.loads((output / "report.json").read_text())
                self.assertEqual(report["training_steps_executed"], 0)
                self.assertEqual(report["selected_step"], 0)
                self.assertFalse(report["kind_prefit"]["development_kind_guard_safe"])
                self.assertEqual(report["reader_functional_visits"]["steps"], 0)
                self.assertEqual(len(report["main_trainable_parameters"]), 6)
                self.assertTrue(report["reader_functional_teacher_all_parameters_unchanged"])
                self.assertFalse(report["reader_functional_replay_used_for_selection"])

    def test_train_preserves_real_step_zero_checkpoint_before_any_prefit(self):
        self.exercise_baseline_and_early_veto()

    def test_public_kind_regression_vetoes_frozen_kind_time_phase_without_fit_selection(self):
        self.exercise_baseline_and_early_veto(veto=True)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_actual_mps_prefit_and_frozen_kind_six_parameter_refinement(self):
        self.exercise_six_parameter_refinement("mps")


if __name__ == "__main__":
    unittest.main()
