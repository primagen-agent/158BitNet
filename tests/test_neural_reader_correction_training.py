"""Finite matcher corrections over complete tiny banks and frozen 91-tensor parents."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import INPUT, LEXICAL, NeuralSlots, PLAN_KEYS, QUERY_KEYS
from neural_reader_correction import snapshot_reader_corrections, reader_correction_loss
import train_neural_dialogue_reader as reader


READ4 = {"read.0.weight", "read.0.bias", "read.2.weight", "read.2.bias"}


READ_CONTEXT_KEYS = {"read_context_memory.weight", "read_context_query.weight", "read_context_delta.weight"}


class ReaderCorrectionTrainingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, device="cpu", *, bias=-1.):
        torch.manual_seed(74731)
        model = NeuralSlots().eval()
        details = (
            ("Ada", "I check the dial gauge against a reference block.", "calibration"),
            ("Ada", "I keep the dial gauge in a padded case.", "storage"),
            ("Ada", "I brush grit out of the dial gauge after a job.", "care"),
            ("Ben", "I check the dial gauge using a measured rod.", "calibration"),
            ("Ben", "I put the dial gauge in a metal drawer.", "storage"),
            ("Ada", "I check the yard meter with a one-metre board.", "calibration"),
        )
        facts = [dict(text=name + ". " + body, entity=name, reference=name,
                      prop=prop, kind=0, status=0) for name, body, prop in details]
        rows = [
            dict(text="How does Ada check the dial gauge?", reference="Ada", entities=["Ada"],
                 prop="calibration", kind=0, support_texts=[facts[0]["text"]]),
            dict(text="Describe Ada's checking and cleaning of the dial gauge.", reference="Ada",
                 entities=["Ada"], prop=["calibration", "care"], kind=0,
                 support_texts=[facts[0]["text"], facts[2]["text"]]),
            dict(text="Where does Ben put the dial gauge?", reference="Ben", entities=["Ben"],
                 prop="storage", kind=0, support_texts=[facts[4]["text"]]),
            dict(text="What did Ada pay for the dial gauge?", reference="Ada", entities=["Ada"],
                 prop="", kind=0, support_texts=[]),
        ]
        original = rows[0]["text"]
        rows[0].update(text=original + "\nActivated evidence:\n" + facts[3]["text"] + "\n",
                       reference="Ben", initial_text=original, initial_reference="Ada", augmented=True)
        texts = sorted({row["text"] for row in facts + rows} | {original, "Ada", "Ben"})
        cache = {}
        for index, text in enumerate(texts):
            count = 1 if text in ("Ada", "Ben") else 2 + index % 4
            raw = torch.randn(count, INPUT) * .2
            if text in ("Ada", "Ben"):
                raw[:, LEXICAL:] = 0
                raw[:, LEXICAL + (text == "Ben")] = 1.
            cache[text] = dict(x=raw, bounds=[(i * len(text) // count, (i+1) * len(text) // count)
                                              for i in range(count)])
        with torch.no_grad():
            model.identity_match[0].weight.zero_()
            model.identity_match[0].bias.zero_()
            model.identity_match[0].weight[0, 0] = 3.
            model.identity_match[0].bias[0] = -.5
            model.identity_match[2].weight.zero_()
            model.identity_match[2].weight[0, 0] = 10.
            model.identity_match[2].bias.zero_()
            model.identity_update.weight.zero_()
            model.identity_update.bias.fill_(-12.)
            model.intent.weight.zero_()
            model.intent.bias[:3].fill_(5.)
            model.version[2].weight.zero_()
            model.version[2].bias.fill_(15.)
            model.read[0].weight.normal_(0, .015)
            model.read[0].bias.normal_(0, .03)
            model.read[2].weight.normal_(0, .04)
            model.read[2].bias.fill_(bias)
            model.kind_context.weight.normal_(0, .01)
            model.kind_context.bias.copy_(torch.tensor([.1, -.1, .03]))
        reader.configure_reader_heads(model, "matcher", freeze_memory_encoder=True)
        model.to(device)
        cached = reader.DynamicFeatures(model, {"fit": dict(facts=facts, queries=rows)}, cache,
                                        device, freeze_memory_encoder=True)
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        return model, parent, facts, rows, cached, parent_cached

    def correction_teacher(self, parent, rows, facts, parent_cached):
        with torch.no_grad():
            old = reader.functional_reader_trace(parent, rows, facts, parent_cached)
            gold = torch.tensor([[reader.supported(row, fact) for fact in facts] for row in rows],
                                dtype=old["activation"].dtype, device=old["activation"].device)
            teacher = snapshot_reader_corrections(old["semantic"], old["activation"], gold,
                                                 fixed_log_gate=old["fixed_log_gate"], split="fit", margin=1.)
        return old, gold, teacher

    def exercise_gradients(self, device):
        model, parent, facts, rows, cached, parent_cached = self.fixture(device)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertEqual(trainable, READ4)
        frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                  if not parameter.requires_grad}
        before_parent = {name: parameter.detach().clone() for name, parameter in parent.named_parameters()}
        self.assertEqual(len(frozen), 87)
        self.assertTrue(READ_CONTEXT_KEYS <= frozen.keys())
        self.assertEqual(len(before_parent), 91)
        self.assertTrue(set(PLAN_KEYS + QUERY_KEYS) <= frozen.keys())
        before_kind = {name: value.clone() for name, value in model.state_dict().items()
                       if name.startswith(("kind.", "kind_context."))}
        main = reader.matcher_correction_training_loss(model, rows, facts, cached, parent, parent_cached)
        replay, stats = reader.functional_replay_loss(model, parent, rows, facts, cached, parent_cached,
                                                     semantic_corrections=True)
        self.assertTrue(torch.isfinite(main + replay))
        self.assertEqual(float(stats["margin"]), 0.)
        self.assertEqual(float(stats["distillation"]), 0.)
        (main + replay).backward()
        optimizer = reader.reader_optimizer(model, torch.nn.Linear(64, 3).to(device), "matcher", .001, .01)
        self.assertEqual({id(p) for group in optimizer.param_groups for p in group["params"]},
                         {id(p) for p in model.parameters() if p.requires_grad})
        for name, parameter in model.named_parameters():
            if name in READ4:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
            else:
                self.assertIsNone(parameter.grad, name)
        optimizer.step()
        for name, parameter in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(parameter, frozen[name], atol=0, rtol=0)
                if name in READ_CONTEXT_KEYS:
                    self.assertFalse(parameter.requires_grad, name)
                    self.assertIsNone(parameter.grad, name)
                    self.assertEqual(parameter.detach().cpu().numpy().tobytes(),
                                     frozen[name].cpu().numpy().tobytes(), name)
        for name, value in before_kind.items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
        for name, parameter in parent.named_parameters():
            self.assertFalse(parameter.requires_grad, name)
            self.assertIsNone(parameter.grad, name)
            torch.testing.assert_close(parameter, before_parent[name], atol=0, rtol=0)

    def test_only_read4_learns_while_other87_and_all91_parent_parameters_stay_exact(self):
        self.exercise_gradients("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_actual_mps_read4_gradients_preserve_source_kind_and_independent91_parent(self):
        self.exercise_gradients("mps")

    def test_main_uses_full_bank_gold_and_independent_parent_even_after_student_read_changes(self):
        model, parent, facts, rows, cached, parent_cached = self.fixture()
        old, gold, teacher = self.correction_teacher(parent, rows, facts, parent_cached)
        self.assertEqual(tuple(gold.shape), (4, 6))
        self.assertEqual(gold.sum(-1).tolist(), [1., 2., 1., 0.])
        self.assertTrue(teacher.correct_pairs[0].any())
        self.assertTrue(teacher.wrong_positive_pairs[0].any())
        self.assertFalse(((old["activation"] > 0) == gold.bool()).all(-1)[0])
        with torch.no_grad():
            model.read[2].bias.add_(2.)
        live = reader.functional_reader_trace(model, rows, facts, cached)
        expected = reader_correction_loss(live["semantic"], teacher)[0]
        with patch.object(reader, "snapshot_reader_corrections", wraps=reader.snapshot_reader_corrections) as snapshot, \
             patch.object(reader, "balanced", side_effect=AssertionError("uncapped activation correction used")), \
             patch.object(reader, "semantic_relation_auxiliary", side_effect=AssertionError("uncapped semantic auxiliary used")):
            actual = reader.matcher_correction_training_loss(model, rows, facts, cached, parent, parent_cached)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(snapshot.call_args.args[0], old["semantic"], atol=0, rtol=0)
        torch.testing.assert_close(snapshot.call_args.args[1], old["activation"], atol=0, rtol=0)
        torch.testing.assert_close(snapshot.call_args.args[2], gold, atol=0, rtol=0)
        torch.testing.assert_close(snapshot.call_args.kwargs["fixed_log_gate"], old["fixed_log_gate"], atol=0, rtol=0)

    def test_replay_changes_supervised_term_and_keeps_full_parent_margin_and_distillation(self):
        model, parent, facts, rows, cached, parent_cached = self.fixture()
        old, gold, correction = self.correction_teacher(parent, rows, facts, parent_cached)
        full = reader.build_reader_functional_teacher(old["activation"], old["semantic"], old["identity"],
            old["intent"], old["temporal_gate"], gold, split="fit", margin=1.)
        with torch.no_grad():
            model.read[2].bias.add_(2.)
        live = reader.functional_reader_trace(model, rows, facts, cached)
        supervised = reader_correction_loss(live["semantic"], correction)[0]
        protected = reader.reader_functional_margin_loss(live["activation"], full)
        distilled = reader.reader_functional_distillation_loss(live["activation"], live["semantic"],
            live["identity"], live["intent"], live["temporal_gate"], full)
        self.assertGreater(float(protected.detach()), 0.)
        self.assertGreater(float(distilled.detach()), 0.)
        with patch.object(reader, "balanced", side_effect=AssertionError("old BCE used in correction replay")):
            loss, stats = reader.functional_replay_loss(model, parent, rows, facts, cached, parent_cached,
                                                        semantic_corrections=True)
        torch.testing.assert_close(loss, supervised + protected + .25 * distilled, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(stats["supervised"], supervised.detach(), atol=0, rtol=0)
        torch.testing.assert_close(stats["margin"], protected.detach(), atol=0, rtol=0)
        torch.testing.assert_close(stats["distillation"], distilled.detach(), atol=0, rtol=0)

    def test_default_replay_keeps_existing_activation_supervision(self):
        model, parent, facts, rows, cached, parent_cached = self.fixture()
        with patch.object(reader, "reader_correction_loss", side_effect=AssertionError("correction enabled by default")), \
             patch.object(reader, "balanced", wraps=reader.balanced) as original:
            loss, _ = reader.functional_replay_loss(model, parent, rows, facts, cached, parent_cached)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(original.call_count, 1)

    def test_unreachable_support_is_not_repaired_by_semantic_learning_or_reference_rewrite(self):
        model, parent, facts, rows, cached, parent_cached = self.fixture()
        rows = deepcopy(rows[:1])
        rows[0]["support_texts"] = [facts[3]["text"]]
        _, _, teacher = self.correction_teacher(parent, rows, facts, parent_cached)
        self.assertFalse(teacher.reachable_pairs[0, 3])
        self.assertFalse(teacher.wrong_positive_pairs.any())
        before = deepcopy(rows)
        loss = reader.matcher_correction_training_loss(model, rows, facts, cached, parent, parent_cached)
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        for name, parameter in model.named_parameters():
            if name in READ4:
                self.assertTrue(torch.equal(parameter.grad, torch.zeros_like(parameter.grad)), name)
        self.assertEqual(rows, before)

    def test_partial_pair_teacher_keeps_original_predicted_subject_across_evidence(self):
        model, parent, facts, rows, cached, parent_cached = self.fixture()
        before = deepcopy(rows)
        observed = []
        original = cached.reference_ids
        def record(items):
            observed.extend(deepcopy(items))
            return original(items)
        with patch.object(cached, "reference_ids", side_effect=record):
            reader.matcher_correction_training_loss(model, rows, facts, cached, parent, parent_cached)
        self.assertTrue(any(row.get("text") == rows[0]["initial_text"] and row.get("reference") == "Ada"
                            for row in observed))
        self.assertTrue(any(row.get("text") == rows[0]["text"] and row.get("reference") == "Ben"
                            for row in observed))
        self.assertEqual(rows, before)
        trace = reader.functional_reader_trace(model, rows, facts, cached)
        self.assertGreater(float(trace["identity"][0, 0]), float(trace["identity"][0, 3]))

    def test_main_and_replay_reject_evaluation_rows_and_facts(self):
        model, parent, facts, rows, cached, parent_cached = self.fixture()
        for target in ("rows", "facts"):
            for changes in (dict(evaluation_only=True), dict(split="development"), dict(evidence_challenge_view=0)):
                changed_rows, changed_facts = deepcopy(rows), deepcopy(facts)
                (changed_rows if target == "rows" else changed_facts)[0].update(changes)
                with self.subTest(target=target, changes=changes):
                    with self.assertRaises(ValueError):
                        reader.matcher_correction_training_loss(model, changed_rows, changed_facts, cached, parent, parent_cached)
                    with self.assertRaises(ValueError):
                        reader.functional_replay_loss(model, parent, changed_rows, changed_facts, cached, parent_cached,
                                                      semantic_corrections=True)

    def test_main_rejects_shared_parent_unfrozen_encoder_dropout_and_other_trainable_heads(self):
        model, parent, facts, rows, cached, parent_cached = self.fixture()
        with self.assertRaises(ValueError):
            reader.matcher_correction_training_loss(model, rows, facts, cached, model, cached)
        with self.assertRaises(ValueError):
            reader.matcher_correction_training_loss(model, rows, facts, cached, parent, cached)
        for owner, attribute, changed in ((cached, "freeze_memory_encoder", False),
                                         (parent_cached, "freeze_memory_encoder", False),
                                         (cached, "feature_dropout", .1), (parent_cached, "feature_dropout", .1)):
            previous = getattr(owner, attribute)
            setattr(owner, attribute, changed)
            with self.assertRaises(ValueError):
                reader.matcher_correction_training_loss(model, rows, facts, cached, parent, parent_cached)
            setattr(owner, attribute, previous)
        model.encode.bias.requires_grad_(True)
        with self.assertRaises(ValueError):
            reader.matcher_correction_training_loss(model, rows, facts, cached, parent, parent_cached)
        model.encode.bias.requires_grad_(False)
        parent.train()
        with self.assertRaises(ValueError):
            reader.matcher_correction_training_loss(model, rows, facts, cached, parent, parent_cached)

    def test_semantic_replay_keywords_and_margin_validation_fail_closed(self):
        model, parent, facts, rows, cached, parent_cached = self.fixture()
        for changes in (dict(semantic_corrections=1), dict(semantic_corrections=True, temporal_only=True),
                        dict(semantic_corrections=True, intent_weight=.25)):
            with self.assertRaises(ValueError):
                reader.functional_replay_loss(model, parent, rows, facts, cached, parent_cached, **changes)
        for margin in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                reader.matcher_correction_training_loss(model, rows, facts, cached, parent, parent_cached, margin=margin)

    def test_matcher_flag_requires_focused_frozen_full_parent_configuration(self):
        valid = dict(mode="matcher", resumed=True, frozen=True, feature_dropout=0., functional_weight=1., topic_weight=0.)
        self.assertIs(reader.validate_matcher_corrections(True, **valid), True)
        self.assertIs(reader.validate_matcher_corrections(False, **valid), False)
        for flag in (0, 1, None, "yes"):
            with self.assertRaises(ValueError):
                reader.validate_matcher_corrections(flag, **valid)
        for change in (dict(mode="semantic"), dict(resumed=False), dict(frozen=False),
                       dict(feature_dropout=.1), dict(functional_weight=0.), dict(functional_weight=-1.),
                       dict(functional_weight=float("nan")), dict(functional_weight=float("inf")), dict(topic_weight=.1)):
            options = dict(valid, **change)
            with self.assertRaises(ValueError):
                reader.validate_matcher_corrections(True, **options)


if __name__ == "__main__":
    unittest.main()
