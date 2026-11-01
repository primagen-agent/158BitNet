"""Matcher protection uses complete fit sets and the exact learned runtime gates."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import train_neural_dialogue_reader as reader
from neural_reader_anchor import MATCHER_KEYS
from neural_slot_model import SEMANTIC
import test_neural_reader_temporal as temporal_tests


class ReaderPreservationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def protected_fixture(self):
        model, facts, original_rows, cached = temporal_tests.TemporalReaderContract().fixture()
        # These controlled gates isolate classification protection, without
        # assuming random synthetic backbone features predict real identities.
        with torch.no_grad():
            model.identity_match[2].weight.zero_()
            model.identity_match[2].bias.fill_(20.)
            model.version[2].weight.zero_()
            model.version[2].bias.fill_(20.)
            model.intent.weight.zero_()
            model.intent.bias[:3].fill_(20.)
            model.read[2].weight.mul_(100.)
            model.read[2].bias.zero_()
        reader.configure_reader_heads(model, "matcher", freeze_memory_encoder=True)
        with torch.no_grad():
            bank, ids, mask, meta = reader.make_bank(facts, cached)
            query, identity = reader.query(model, original_rows, cached)
            semantic = model.semantic_bank_logits(query, bank, mask, memory_context_latents=(query).new_zeros((len(mask), 16)))
            maxima = semantic.max(-1).values
            positive_index, null_index = int(maxima.argmax()), int(maxima.argmin())
            self.assertNotEqual(positive_index, null_index)
            model.read[2].bias.copy_(-(maxima[positive_index] + maxima[null_index]) / 2)
            activation = model.score_bank(query, identity, bank, ids, meta, mask, memory_context_latents=(query).new_zeros((len(mask), 16)))
        rows = [deepcopy(original_rows[index]) for index in (positive_index, null_index)]
        for row, index in zip(rows, (positive_index, null_index)):
            row["support_texts"] = [fact["text"] for column, fact in enumerate(facts)
                                    if activation[index, column] > 0]
        self.assertTrue(rows[0]["support_texts"])
        self.assertEqual(rows[1]["support_texts"], [])
        return model, facts, rows, cached, reader.snapshot_reader_matcher(model)

    def test_parent_initial_loss_and_gradient_are_exactly_zero(self):
        model, facts, rows, cached, parent = self.protected_fixture()
        before = {name: value.clone() for name, value in parent.items()}
        loss = reader.matcher_preservation_loss(model, rows, facts, cached, parent,
                                                weight=2., margin=0.5)
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        for name, parameter in model.named_parameters():
            if name in MATCHER_KEYS:
                self.assertIsNotNone(parameter.grad, name)
                self.assertEqual(float(parameter.grad.abs().sum()), 0, name)
            else:
                self.assertIsNone(parameter.grad, name)
        for name, value in parent.items():
            self.assertFalse(value.requires_grad)
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)

    def test_lost_positive_or_null_margin_trains_only_read4_without_mutating_parent(self):
        for target in ("positive", "null"):
            with self.subTest(lost_parent_margin=target):
                model, facts, rows, cached, parent = self.protected_fixture()
                before_parent = {name: value.clone() for name, value in parent.items()}
                frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                          if name not in MATCHER_KEYS}
                with torch.no_grad():
                    bank, _, mask, _ = reader.make_bank(facts, cached)
                    query, _ = reader.query(model, rows, cached)
                    semantic = model.semantic_bank_logits(query, bank, mask, memory_context_latents=(query).new_zeros((len(mask), 16)))
                    shift = (-float(semantic[0].max()) - 1. if target == "positive"
                             else 1. - float(semantic[1].max()))
                    model.read[2].bias.add_(shift)
                loss = reader.matcher_preservation_loss(model, rows, facts, cached, parent,
                                                        weight=2., margin=0.5)
                self.assertGreater(float(loss.detach()), 0)
                loss.backward()
                parameters = dict(model.named_parameters())
                for name, parameter in parameters.items():
                    if name in MATCHER_KEYS:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                        self.assertGreater(float(parameter.grad.abs().sum()), 0, name)
                    else:
                        self.assertIsNone(parameter.grad, name)
                optimizer = torch.optim.SGD([parameters[name] for name in MATCHER_KEYS], lr=0.001)
                optimizer.step()
                for name, expected in frozen.items():
                    torch.testing.assert_close(parameters[name], expected, atol=0, rtol=0)
                for name, value in parent.items():
                    torch.testing.assert_close(value, before_parent[name], atol=0, rtol=0)

    def test_external_semantic_scores_match_batched_and_independent_single_slot_paths(self):
        model, facts, rows, cached = temporal_tests.TemporalReaderContract().fixture()
        with torch.no_grad():
            bank, ids, mask, meta = reader.make_bank(facts, cached)
            dynamic = meta[None].expand(len(rows), -1, -1).clone()
            dynamic[:, :, 3] *= torch.linspace(0.25, 1., len(rows))[:, None]
            for scope in (False, True):
                query, identity = reader.query(model, rows, cached, scope=scope)
                semantic = model.semantic_bank_logits(query, bank, mask, scope=scope, memory_context_latents=(query).new_zeros((len(mask), 16)))
                for metadata in (meta, dynamic):
                    external = model.score_bank_from_semantic(semantic, query, identity, ids,
                                                              metadata, scope=scope)
                    ordinary = model.score_bank(query, identity, bank, ids, metadata, mask, scope=scope, memory_context_latents=(query).new_zeros((len(mask), 16)))
                    torch.testing.assert_close(external, ordinary, atol=0, rtol=0)
                    single = torch.stack([
                        torch.stack([
                            model.score(query[index], identity[index], bank[column, mask[column]],
                                        ids[column], metadata[column] if metadata.ndim == 2
                                        else metadata[index, column], scope=scope, memory_context_latents=(bank[column, mask[column]]).new_zeros(16))
                            for column in range(len(facts))
                        ])
                        for index in range(len(rows))
                    ])
                    torch.testing.assert_close(external, single, atol=3e-6, rtol=3e-6)

    def test_preservation_rejects_evaluation_and_nonfit_teacher_labels(self):
        model, facts, rows, cached, parent = self.protected_fixture()
        with self.assertRaisesRegex(ValueError, "development"):
            reader.matcher_preservation_loss(model, [dict(rows[0], evaluation_only=True)],
                                            facts, cached, parent)
        with torch.no_grad():
            bank, ids, mask, meta = reader.make_bank(facts, cached)
            query, identity = reader.query(model, rows, cached)
            semantic = model.semantic_bank_logits(query, bank, mask, memory_context_latents=(query).new_zeros((len(mask), 16)))
            activation = model.score_bank_from_semantic(semantic, query, identity, ids, meta)
            gold = activation > 0
        for split in ("development", "blind", "runtime"):
            with self.assertRaisesRegex(ValueError, "fit"):
                reader.build_reader_anchor_teacher(semantic, activation, gold,
                                                   fixed_log_gate=reader.functional_reader_trace(model, rows, facts, cached)["fixed_log_gate"], split=split)

    def test_preservation_passes_actual_frozen_gate_without_reencoding_the_bank_or_query(self):
        model, facts, rows, cached, parent = self.protected_fixture()
        with torch.no_grad():
            expected = reader.functional_reader_trace(model, rows, facts, cached)["fixed_log_gate"].clone()
        teacher_builder = reader.build_reader_anchor_teacher
        with patch.object(reader, "make_bank", wraps=reader.make_bank) as make_bank, \
             patch.object(reader, "query", wraps=reader.query) as query, \
             patch.object(reader, "build_reader_anchor_teacher", wraps=teacher_builder) as build:
            loss = reader.matcher_preservation_loss(model, rows, facts, cached, parent)
        self.assertEqual(make_bank.call_count, 1)
        self.assertEqual(query.call_count, 1)
        self.assertEqual(build.call_count, 1)
        actual = build.call_args.kwargs["fixed_log_gate"]
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(float(loss.detach()), 0.)

    def settings(self, **changes):
        values = dict(steps=1, eval_every=10, train_heads="matcher", semantic_topic_weight=0,
                      resume="parent.pt", output="never-created", reader_anchor_weight=1.,
                      reader_anchor_margin=1., freeze_memory_encoder=True, feature_dropout=0.,
                      relation_paraphrases=False, episode_relations=False)
        values.update(changes)
        return SimpleNamespace(**values)

    def test_invalid_anchor_modes_rates_and_intervals_fail_before_output_creation(self):
        cases = [dict(train_heads="semantic"), dict(train_heads="temporal"), dict(resume=None),
                 dict(freeze_memory_encoder=False), dict(feature_dropout=0.1),
                 dict(reader_anchor_weight=-1.), dict(reader_anchor_margin=0.)]
        for value in (float("nan"), float("inf"), -float("inf")):
            cases += [dict(reader_anchor_weight=value), dict(reader_anchor_margin=value)]
        cases += [dict(eval_every=value) for value in (0, -1, 2.5, True)]
        for changes in cases:
            with self.subTest(changes=changes), \
                 patch.object(Path, "mkdir", side_effect=AssertionError("output was created")):
                with self.assertRaises(ValueError):
                    reader.train(self.settings(**changes))
        # Accepted intervals reach the output stage; no directory is created.
        for interval in (1, 10, 100):
            with self.subTest(valid_interval=interval), \
                 patch.object(Path, "mkdir", side_effect=RuntimeError("valid output stage")):
                with self.assertRaisesRegex(RuntimeError, "valid output stage"):
                    reader.train(self.settings(eval_every=interval))


if __name__ == "__main__":
    unittest.main()
