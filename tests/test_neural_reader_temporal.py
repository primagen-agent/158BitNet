"""Temporal refinement preserves the learned source, relation and identity paths."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import INPUT, SEMANTIC, WIDTH, NeuralSlots
import train_neural_dialogue_reader as reader


TEMPORAL_PARAMETERS = {
    "kind.weight", "kind.bias", "kind_context.weight", "kind_context.bias", "intent.weight", "intent.bias",
    "version.0.weight", "version.0.bias", "version.2.weight", "version.2.bias",
}


class TemporalReaderContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        torch.manual_seed(73102)
        model = NeuralSlots().eval()
        facts = [
            dict(text="Ada. I keep chisels in a chest.", entity="Ada", reference="Ada",
                 prop="storage", kind=1, source_kind=0, status=1),
            dict(text="Ada. I intend to put chisels in a crate.", entity="Ada", reference="Ada",
                 prop="storage", kind=2, status=3),
            dict(text="Ada. I keep chisels in a cabinet.", entity="Ada", reference="Ada",
                 prop="storage", kind=0, status=0),
            dict(text="Ben. I keep measuring tools in a drawer.", entity="Ben", reference="Ben",
                 prop="storage", kind=0),
        ]
        rows = [
            dict(text="Where did Ada previously keep the chisels?", entities=["Ada"],
                 prop="storage", kind=1, allowed_kinds=[1], support_texts=[facts[0]["text"]]),
            dict(text="Where does Ada intend to put the chisels?", entities=["Ada"],
                 prop="storage", kind=2, allowed_kinds=[2], support_texts=[facts[1]["text"]]),
            dict(text="Where are Ada's chisels kept now?", entities=["Ada"],
                 prop="storage", kind=0, allowed_kinds=[0], support_texts=[facts[2]["text"]]),
            dict(text="List Ada's past, current and proposed chisel locations.", entities=["Ada"],
                 prop="storage", kind=0, allowed_kinds=[0, 1, 2],
                 support_texts=[fact["text"] for fact in facts[:3]]),
        ]
        texts = sorted({row["text"] for row in facts + rows} | {"Ada", "Ben"})
        cache = {
            text: dict(
                x=torch.randn(3, INPUT),
                bounds=[(i * len(text) // 3, (i + 1) * len(text) // 3) for i in range(3)],
            )
            for text in texts
        }
        group = dict(facts=facts, queries=rows)
        cached = reader.DynamicFeatures(model, {"fit": group}, cache, "cpu",
                                        freeze_memory_encoder=True)
        return model, facts, rows, cached

    def test_only_ten_time_tensors_learn_and_every_other_tensor_remains_exact(self):
        model, facts, rows, cached = self.fixture()
        self.assertEqual(reader.configure_reader_heads(model, "temporal", freeze_memory_encoder=True),
                         ("kind.", "kind_context.", "intent.", "version."))
        parameters = dict(model.named_parameters())
        trainable = {name for name, parameter in parameters.items() if parameter.requires_grad}
        self.assertEqual(trainable, TEMPORAL_PARAMETERS)
        self.assertEqual(sum(parameters[name].numel() for name in trainable), 541319)
        self.assertEqual(len(parameters) - len(trainable), 81)
        frozen = {name: parameter.detach().clone() for name, parameter in parameters.items()
                  if name not in trainable}
        optimizer = reader.reader_optimizer(model, torch.nn.Linear(WIDTH, 3), "temporal", 0.001, 0.01)
        self.assertEqual({id(parameter) for group in optimizer.param_groups for parameter in group["params"]},
                         {id(parameters[name]) for name in trainable})
        with patch.object(reader, "semantic_relation_auxiliary", side_effect=AssertionError("semantic teacher used")), \
             patch.object(reader, "identity_calibration", side_effect=AssertionError("identity teacher used")), \
             patch.object(reader, "role_supervision", side_effect=AssertionError("role teacher used")):
            loss = reader.temporal_training_loss(model, rows, facts, cached)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in parameters.items():
            if name in trainable:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
            else:
                self.assertIsNone(parameter.grad, name)
        optimizer.step()
        for name, expected in frozen.items():
            torch.testing.assert_close(parameters[name], expected, atol=0, rtol=0)

    def test_text_kind_and_multiple_allowed_physical_states_use_distinct_teachers(self):
        model, facts, rows, cached = self.fixture()
        reader.configure_reader_heads(model, "temporal", freeze_memory_encoder=True)
        with patch.object(F, "cross_entropy", wraps=F.cross_entropy) as source_ce, \
             patch.object(F, "binary_cross_entropy_with_logits",
                          wraps=F.binary_cross_entropy_with_logits) as binary:
            loss = reader.temporal_training_loss(model, rows, facts, cached)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(source_ce.call_count, 1)
        self.assertEqual(source_ce.call_args.args[1].tolist(), [0, 2, 0, 0])
        self.assertNotEqual(source_ce.call_args.args[1][0].item(), facts[0]["status"])
        intent_calls = [call for call in binary.call_args_list if call.args[0].shape == (4, 3)]
        self.assertEqual(len(intent_calls), 1)
        self.assertEqual(intent_calls[0].args[1].tolist(),
                         [[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1]])

    def test_prediction_uses_hard_kind_and_preserves_existing_physical_status(self):
        model, facts, _, cached = self.fixture()
        with torch.no_grad():
            model.kind.weight.zero_()
            model.kind.bias.copy_(torch.tensor([-2.0, -3.0, 4.0]))
            _, _, _, metadata = reader.make_bank(facts, cached)
        self.assertEqual(metadata[:, :3].tolist(),
                         [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]])
        self.assertEqual(reader.source_kind_targets(facts, "cpu").tolist(), [0, 2, 0, 0])
        self.assertEqual(facts[3]["kind"], 0)
        # The ordinary last slot uses its model prediction, never the source label.
        self.assertEqual(int(metadata[3, :3].argmax()), 2)

    def test_scope_logits_stay_exact_after_temporal_step_and_kind_diagnostics_honor_source_kind(self):
        model, facts, rows, cached = self.fixture()
        reader.configure_reader_heads(model, "temporal", freeze_memory_encoder=True)

        def scope_scores():
            bank, ids, mask, metadata = reader.make_bank(facts, cached)
            query, identity = reader.query(model, rows, cached, scope=True)
            return model.score_bank(query, identity, bank, ids, metadata, mask, scope=True, memory_context_latents=(query).new_zeros((len(mask), 16)))

        with torch.no_grad():
            before = scope_scores().clone()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        reader.temporal_training_loss(model, rows, facts, cached).backward()
        optimizer.step()
        with torch.no_grad():
            after = scope_scores()
            diagnostics = reader.kind_bank_diagnostics(model, dict(facts=facts), cached)
        torch.testing.assert_close(after, before, atol=0, rtol=0)
        self.assertEqual(diagnostics["teacher_counts"], [3, 0, 1])
        self.assertEqual(sum(diagnostics["bank_status_counts"]), 4)
        self.assertTrue(all(count >= 1 for count in diagnostics["bank_status_counts"]))

    def test_invalid_source_kind_and_evaluation_inputs_are_rejected(self):
        model, facts, rows, cached = self.fixture()
        for value in (-1, 3, True, "0"):
            with self.assertRaisesRegex(ValueError, "source-kind"):
                reader.source_kind_targets([dict(kind=0, source_kind=value)], "cpu")
        self.assertEqual(reader.source_kind_targets([dict(kind=1)], "cpu").tolist(), [1])
        with self.assertRaisesRegex(ValueError, "development"):
            reader.temporal_training_loss(model, [dict(rows[0], evaluation_only=True)], facts, cached)
        for bad_rows, bad_facts in (([], facts), (rows, [])):
            with self.assertRaisesRegex(ValueError, "fit facts and queries"):
                reader.temporal_training_loss(model, bad_rows, bad_facts, cached)

    def test_temporal_mode_requires_a_trained_parent_and_omits_topic_objective(self):
        settings = dict(steps=1, train_heads="temporal", semantic_topic_weight=0,
                        resume="parent.pt", output="uncreated-output", relation_paraphrases=False,
                        episode_relations=False)
        for change in (dict(resume=None), dict(semantic_topic_weight=0.05)):
            with patch.object(Path, "mkdir", side_effect=AssertionError("output was created")):
                with self.assertRaisesRegex(ValueError, "temporal"):
                    reader.train(SimpleNamespace(**dict(settings, **change)))
        self.assertIn("temporal", reader.GUARDED_READER_MODES)
        self.assertNotIn("temporal", reader.RELATION_MODES)


if __name__ == "__main__":
    unittest.main()
