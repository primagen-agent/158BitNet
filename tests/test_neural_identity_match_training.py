"""Matcher-only identity calibration preserves all frozen learned representations."""

from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import FORMAT, INPUT, KEYS, NeuralSlots
from neural_training_common import feature_texts_sha256, sha
import train_neural_dialogue_reader as reader


MATCH4 = {"identity_match.0.weight", "identity_match.0.bias",
          "identity_match.2.weight", "identity_match.2.bias"}
ROOT = Path(__file__).resolve().parents[1]


def arguments(**changes):
    args = reader.reader_argument_parser().parse_args([
        "--gguf", "unused.gguf", "--probe", "unused-probe", "--output", "unused-output",
        "--resume", "unused-resume.pt", "--steps", "1", "--train-heads", "identity-match",
        "--freeze-memory-encoder", "--semantic-topic-weight", "0",
    ])
    for key, value in changes.items():
        setattr(args, key, value)
    return args


def tiny_group():
    facts = [
        dict(text="Ada. I paint the feeder green.", reference="Ada", entity="Ada", prop="color", kind=0, status=0),
        dict(text="Ada. I store the brush in the shed.", reference="Ada", entity="Ada", prop="storage", kind=0, status=0),
        dict(text="Ben. I paint the feeder blue.", reference="Ben", entity="Ben", prop="color", kind=0, status=0),
        dict(text="Ben. I keep the ladder in the garage.", reference="Ben", entity="Ben", prop="storage", kind=0, status=0),
    ]
    original = "What color does Ada paint the feeder?"
    rows = [
        dict(text=original + "\nActivated evidence:\n" + facts[2]["text"] + "\n",
             reference="Ben", initial_text=original, initial_reference="Ada", entities=["Ada"],
             prop="color", kind=0, support_texts=[facts[0]["text"]]),
        # A wrong predicted reference must remain wrong; actor labels are targets only.
        dict(text="Where does Ben store his brush?", reference="Ada", entities=["Ben"],
             prop="storage", kind=0, support_texts=[]),
        dict(text="What brush does Ada use?", reference="Ada", entities=["Ada"],
             prop="", kind=0, support_texts=[]),
    ]
    for index, row in enumerate(rows):
        row["observation"] = dict(body=row["text"], context=row["text"], source_id=f"tiny-{index}")
        row.update(unresolved=False, binding_predicted=True)
    return dict(facts=facts, queries=rows)


def synthetic_features(groups):
    """Synthetic unit inputs only; these are not extracted backbone features."""
    texts = {row["text"] for group in groups for values in group.values() for row in values}
    texts |= {row[key] for group in groups for values in group.values() for row in values
              for key in ("reference", "initial_reference", "initial_text") if row.get(key)}
    result = {}
    for index, text in enumerate(sorted(texts)):
        length = 2 + index % 3
        size = len(text.encode())
        result[text] = dict(x=torch.randn(length, INPUT) * .2,
                            bounds=[(i * size // length, (i + 1) * size // length) for i in range(length)])
    return result


class IdentityMatchTrainingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, device="cpu"):
        torch.manual_seed(64153)
        model = NeuralSlots().train().to(device)
        # Keep all 33 matcher parameters active so a real step can test each tensor.
        with torch.no_grad():
            for parameter in model.identity_match.parameters():
                parameter.uniform_(-.3, .3)
        try:
            reader.configure_reader_heads(model, "identity-match", freeze_memory_encoder=True)
        except (KeyError, ValueError) as error:
            self.fail(f"identity-match must configure its four matcher tensors: {error}")
        group = tiny_group()
        cache = synthetic_features([group])
        cached = reader.DynamicFeatures(model, {"fit": group}, cache, device, freeze_memory_encoder=True)
        return model, group["facts"], group["queries"], cached

    def exercise_step(self, device):
        model, facts, rows, cached = self.fixture(device)
        parameters = dict(model.named_parameters())
        self.assertEqual(set(model.state_dict()), set(KEYS))
        self.assertEqual(len(parameters), 91)
        self.assertEqual({name for name, p in parameters.items() if p.requires_grad}, MATCH4)
        self.assertEqual(sum(parameters[name].numel() for name in MATCH4), 33)
        before = {name: p.detach().clone() for name, p in parameters.items()}
        with torch.no_grad():
            old_query, old_identity = reader.query(model, rows, cached)
            old_bank = cached.encode([fact["text"] for fact in facts])[0]
            old_references = cached.reference_ids(facts)
        loss = reader.identity_match_training_loss(model, rows, facts, cached)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        topic = torch.nn.Linear(64, 3).to(device)
        optimizer = reader.reader_optimizer(model, topic, "identity-match", .002, .01)
        self.assertEqual({id(p) for group in optimizer.param_groups for p in group["params"]},
                         {id(parameters[name]) for name in MATCH4})
        optimizer.step()
        frozen = 0
        for name, parameter in parameters.items():
            if name in MATCH4:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
                self.assertFalse(torch.equal(parameter, before[name]), name)
            else:
                frozen += 1
                self.assertIsNone(parameter.grad, name)
                self.assertEqual(parameter.detach().cpu().numpy().tobytes(),
                                 before[name].cpu().numpy().tobytes(), name)
        self.assertEqual(frozen, 87)
        self.assertTrue(all(p.grad is None for p in topic.parameters()))
        with torch.no_grad():
            new_query, new_identity = reader.query(model, rows, cached)
            torch.testing.assert_close(new_query, old_query, atol=0, rtol=0)
            torch.testing.assert_close(new_identity, old_identity, atol=0, rtol=0)
            torch.testing.assert_close(cached.encode([fact["text"] for fact in facts])[0], old_bank, atol=0, rtol=0)
            torch.testing.assert_close(cached.reference_ids(facts), old_references, atol=0, rtol=0)

    def test_cpu_step_changes_matcher33_only_and_preserves_other87_bitwise(self):
        self.exercise_step("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_actual_mps_step_changes_matcher33_only_and_preserves_other87_bitwise(self):
        self.exercise_step("mps")

    def test_loss_uses_predicted_references_and_actor_targets_without_role_or_semantic_losses(self):
        model, facts, rows, cached = self.fixture()
        before = deepcopy(rows)
        _, identity = reader.query(model, rows, cached)
        memories = cached.reference_ids(facts)
        target = identity.new_tensor([[1., 1., 0., 0.], [0., 0., 1., 1.], [1., 1., 0., 0.]])
        torch.manual_seed(15)
        expected = (reader.balanced(model.identity_logits(identity, memories), target)
                    + .25 * reader.identity_calibration(model, identity, memories, target))
        torch.manual_seed(15)
        with patch.object(reader, "role_supervision", side_effect=AssertionError("role objective used")), \
             patch.object(model, "score_bank", side_effect=AssertionError("semantic objective used")), \
             patch.object(reader, "semantic_relation_auxiliary", side_effect=AssertionError("relation objective used")):
            actual = reader.identity_match_training_loss(model, rows, facts, cached)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(rows, before)
        torch.testing.assert_close(identity[1:2], cached.reference_ids([dict(text="Ada", reference="Ada")]), atol=1e-6, rtol=1e-6)
        self.assertFalse(torch.equal(identity[1:2], cached.reference_ids([dict(text="Ben", reference="Ben")])))

    def test_loss_rejects_trainable_representations_dropout_and_evaluation_rows(self):
        model, facts, rows, cached = self.fixture()
        for name in ("role.0.weight", "identity.weight", "identity_update.weight", "query_encode.weight"):
            parameter = dict(model.named_parameters())[name]
            parameter.requires_grad_(True)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "matcher-only"):
                reader.identity_match_training_loss(model, rows, facts, cached)
            parameter.requires_grad_(False)
        for dropout in (.1, float("nan")):
            cached.feature_dropout = dropout
            with self.assertRaisesRegex(ValueError, "undropped"):
                reader.identity_match_training_loss(model, rows, facts, cached)
        cached.feature_dropout = 0.
        cached.freeze_memory_encoder = False
        with self.assertRaisesRegex(ValueError, "frozen"):
            reader.identity_match_training_loss(model, rows, facts, cached)
        cached.freeze_memory_encoder = True
        for marker in (dict(split="development"), dict(evaluation_only=True), dict(evidence_challenge_view=0)):
            with self.subTest(marker=marker), self.assertRaisesRegex(ValueError, "development or evaluation"):
                reader.identity_match_training_loss(model, [dict(rows[0], **marker)], facts, cached)

    def test_loss_rejects_conflicting_fit_provenance_in_rows_facts_and_original_observations(self):
        model, facts, rows, cached = self.fixture()
        for marker in (dict(query_binding_split="development"), dict(evaluation_role="evaluation-only")):
            for location in ("query", "fact", "observation"):
                invalid_rows, invalid_facts = deepcopy(rows), deepcopy(facts)
                target = (invalid_rows[0] if location == "query" else invalid_facts[0] if location == "fact"
                          else invalid_rows[0]["observation"])
                target.update(split="fit", evaluation_only=False, **marker)
                with self.subTest(location=location, marker=marker), \
                     self.assertRaisesRegex(ValueError, "development or evaluation"):
                    reader.identity_match_training_loss(model, invalid_rows, invalid_facts, cached)

    def test_parser_and_preflight_require_resume_frozen_undropped_identity_only_objective(self):
        try:
            valid = arguments()
        except SystemExit:
            self.fail("reader parser must accept identity-match")
        changes = (dict(resume=None), dict(freeze_memory_encoder=False), dict(feature_dropout=.1),
                   dict(feature_dropout=float("nan")), dict(semantic_topic_weight=.05),
                   dict(reader_anchor_weight=1.), dict(reader_functional_weight=1.),
                   dict(identity_diversity=True), dict(expansion=True), dict(evidence_cues=True))
        for change in changes:
            with self.subTest(change=change), \
                 patch.object(Path, "mkdir", side_effect=AssertionError("premature output")), \
                 self.assertRaises(ValueError):
                reader.train(arguments(**change))
        with patch.object(Path, "mkdir", side_effect=RuntimeError("validated output stage")), \
             self.assertRaisesRegex(RuntimeError, "validated output stage"):
            reader.train(valid)

    def run_tiny_train(self, *, different_planner=False, metric_scenario=None, raw_fit=None):
        """Run the real trainer with local synthetic features and a tiny data-loader seam."""
        torch.manual_seed(64153)
        group = tiny_group()
        from neural_activation_curriculum import evidence_challenge
        challenge = evidence_challenge(group, 7421, limit=24)
        scope_group = dict(facts=deepcopy(group["facts"]), scope_queries=deepcopy(group["queries"]))
        cache = synthetic_features([group, challenge])
        build = ROOT / "build"
        build.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="identity-match-test-", dir=build) as temporary:
            directory = Path(temporary)
            gguf, probe = directory / "synthetic.gguf", directory / "synthetic-probe"
            gguf.write_bytes(b"synthetic unit fixture; not a backbone")
            probe.write_bytes(b"synthetic unit fixture; not a probe")
            backbone = sha(gguf)
            model = NeuralSlots()
            checkpoint = dict(format=FORMAT, backbone_sha256=backbone, state_dict=model.state_dict(),
                              reader_training_mode="semantic", reader_heads=("read.", "encode."),
                              freeze_memory_encoder=False, semantic_topic_weight=.5,
                              reader_anchor_weight=2., reader_anchor_margin=.3, reader_anchor_teacher="parent",
                              reader_functional_weight=3., reader_functional_margin=.4,
                              reader_functional_teacher="parent", reader_functional_loss="parent",
                              reader_functional_intent_weight=.25, reader_functional_replay={"parent": True},
                              matcher_corrections=True)
            resume = directory / "resume.pt"
            torch.save(checkpoint, resume)
            planner = None
            if different_planner:
                planner = directory / "different-writer.pt"
                checkpoint = deepcopy(checkpoint)
                checkpoint["state_dict"]["query_context.weight"][0, 0] += .1
                torch.save(checkpoint, planner)
            features = directory / "features.pt"
            torch.save(dict(features=cache, identity=dict(backbone=backbone, probe=sha(probe),
                texts_sha256=feature_texts_sha256(cache))), features)
            curriculum = directory / "curriculum.json"
            curriculum.write_text(json.dumps(dict(fit=raw_fit or [], development=[])))
            output = directory / "output"
            args = arguments(gguf=str(gguf), probe=str(probe), output=str(output), resume=str(resume),
                             planner_checkpoint=str(planner) if planner else None,
                             features=str(features), curriculum=str(curriculum))
            baseline = dict(activation_f1=.8, support_set_exact=.8, positive_queries=10, positive_exact=8,
                            positive_support_set_exact=.8, null_false_activation=0., tp=8, fp=2, fn=2,
                            bank_size=4, queries=12)
            evidence_improved = dict(baseline, activation_f1=.95, support_set_exact=.9,
                                     positive_exact=9, positive_support_set_exact=.9, tp=9, fp=1, fn=1)
            degraded = dict(baseline, activation_f1=.7, support_set_exact=.7,
                            positive_exact=7, positive_support_set_exact=.7, tp=7, fn=3)

            def after_step(live):
                return not torch.equal(live.identity_match[2].bias.detach().cpu(), model.identity_match[2].bias)

            def controlled_evaluate(live, values, _cached):
                if not after_step(live):
                    return dict(baseline)
                if any("evidence_challenge_view" in row for row in values["queries"]):
                    return dict(evidence_improved)
                return dict(degraded if metric_scenario == "component" else baseline)

            def controlled_scope(live, _values, _cached):
                return dict(degraded if after_step(live) and metric_scenario == "scope" else baseline)

            component_metrics = (patch.object(reader, "evaluate", side_effect=controlled_evaluate)
                                 if metric_scenario else nullcontext())
            scope_metrics = (patch.object(reader, "scope_evaluate", side_effect=controlled_scope)
                             if metric_scenario else nullcontext())
            # The existing frozen-writer binder has independent native parity tests.
            # Supply already predicted references here to isolate main-loop dispatch.
            with patch("prepare_neural_colloquial.reader_data", side_effect=lambda *_: deepcopy(group)), \
                 patch.object(reader, "bind_curriculum_queries", side_effect=lambda value, *_: (deepcopy(value), [])), \
                 patch.object(reader, "scope_fit_group", return_value=deepcopy(scope_group)), \
                 patch.object(reader, "lifecycle_fit_group", return_value=deepcopy(group)), \
                 patch.object(reader, "scope_development_group", return_value=scope_group), \
                 patch.object(reader, "extract", side_effect=AssertionError("unexpected feature extraction")), \
                 patch.object(reader, "role_supervision", side_effect=AssertionError("role objective used")), \
                 patch.object(reader, "semantic_relation_auxiliary", side_effect=AssertionError("semantic objective used")), \
                 patch.object(reader, "export", return_value="synthetic-unit-export"), \
                 component_metrics, scope_metrics, redirect_stdout(io.StringIO()):
                reader.train(args)
            last = torch.load(output / "last-step.pt", weights_only=False)
            selected = torch.load(output / "selected.pt", weights_only=False)
            report = json.loads((output / "report.json").read_text())
            return model.state_dict(), last, selected, report

    def test_main_dispatch_saves_current91_and_changes_only_match4_from_resume(self):
        before, last, selected, report = self.run_tiny_train()
        self.assertEqual(last["reader_training_mode"], "identity-match")
        self.assertEqual(last["reader_step"], 1)
        self.assertEqual(set(last["state_dict"]), set(KEYS))
        self.assertEqual(set(report["main_trainable_parameters"]), MATCH4)
        self.assertEqual(report["training_steps_executed"], 1)
        changed = {name for name, value in last["state_dict"].items() if not torch.equal(value, before[name])}
        self.assertEqual(changed, MATCH4)
        self.assertEqual(selected["format"], FORMAT)
        self.assertFalse(report["production_qualified"])

    def test_main_rejects_writer_sync_that_would_change_frozen_representations(self):
        with self.assertRaisesRegex(ValueError, "identity-match.*planner.*resume"):
            self.run_tiny_train(different_planner=True)

    def test_saved_views_override_parent_heads_and_freeze_metadata_with_current_settings(self):
        _, last, selected, report = self.run_tiny_train()
        for checkpoint in (selected, last):
            self.assertEqual(checkpoint["reader_training_mode"], "identity-match")
            self.assertEqual(checkpoint["reader_heads"], ("identity_match.",))
            self.assertIs(checkpoint["freeze_memory_encoder"], True)
        self.assertEqual(report["reader_training_mode"], "identity-match")
        self.assertEqual(set(report["main_trainable_parameters"]), MATCH4)
        self.assertIs(report["freeze_memory_encoder"], True)
        for view in (selected, last, report):
            for key in ("semantic_topic_weight", "reader_anchor_weight", "reader_functional_weight",
                        "reader_functional_intent_weight"):
                self.assertEqual(view[key], 0., key)
            for key in ("reader_anchor_margin", "reader_functional_margin"):
                self.assertEqual(view[key], 1., key)
            for key in ("reader_anchor_teacher", "reader_functional_teacher", "reader_functional_loss",
                        "reader_functional_replay"):
                self.assertIsNone(view[key], key)
            self.assertIs(view["matcher_corrections"], False)

    def test_main_preserves_initial_component_and_scope_when_evidence_score_improves(self):
        for scenario in ("component", "scope"):
            with self.subTest(regression=scenario):
                before, last, selected, report = self.run_tiny_train(metric_scenario=scenario)
                self.assertTrue(report["scope_guard_enabled"])
                self.assertTrue(report["component_guard_enabled"])
                self.assertEqual(report["selected_step"], 0)
                self.assertEqual(selected["reader_step"], 0)
                self.assertEqual(selected["scope_development"], selected["initial_scope_development"])
                self.assertEqual(report["scope_development"], report["initial_scope_development"])
                self.assertEqual(last["initial_scope_development"], report["initial_scope_development"])
                self.assertEqual(last["last_step_scope_development"], report["last_step_scope_development"])
                self.assertEqual(report["initial_development_views"], selected["selection_metrics"])
                self.assertEqual(last["initial_development_views"], report["initial_development_views"])
                self.assertEqual(report["initial_scope_development"]["fn"], 2)
                if scenario == "component":
                    self.assertEqual(report["last_step_development_views"]["authored_development"]["fn"], 3)
                else:
                    self.assertEqual(report["last_step_scope_development"]["fn"], 3)
                for name, value in selected["state_dict"].items():
                    torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_main_can_select_better_evidence_when_both_independent_guards_are_preserved(self):
        _, _, selected, report = self.run_tiny_train(metric_scenario="safe")
        self.assertEqual(report["selected_step"], 1)
        self.assertEqual(selected["reader_step"], 1)
        self.assertEqual(report["scope_development"], report["initial_scope_development"])
        self.assertEqual(selected["selection_metrics"], report["initial_development_views"])

    def test_main_rejects_raw_fit_markers_before_reader_data_can_drop_fact_provenance(self):
        for marker in (dict(query_binding_split="development"), dict(evaluation_role="evaluation-only")):
            with self.subTest(marker=marker), self.assertRaisesRegex(ValueError, "development or evaluation"):
                self.run_tiny_train(raw_fit=[dict(tag="episode-fact", split="fit", **marker)])


if __name__ == "__main__":
    unittest.main()
