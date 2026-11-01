"""Explicit query-conditioned original-hidden sidecars and frozen reader paths."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from neural_slot_model import (NeuralSlots, KEYS, HIDDEN, WIDTH, SEMANTIC, QUERY,
                               READ_CONTEXT_WIDTH, require_parameters)
from neural_reader_anchor import matcher_pair_features
import train_neural_dialogue_reader as reader
import test_neural_reader_temporal as temporal_tests

CONTEXT_KEYS = {"read_context_memory.weight", "read_context_query.weight", "read_context_delta.weight"}


class ReaderContextContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(84117)

    def inputs(self, dtype=torch.float32, device="cpu"):
        model = NeuralSlots().to(device=device, dtype=dtype)
        queries = torch.randn(3, QUERY, device=device, dtype=dtype)
        bank = torch.randn(4, 5, WIDTH, device=device, dtype=dtype)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1],
                             [1, 0, 0, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool, device=device)
        contexts = F.normalize(torch.randn(4, 2 * HIDDEN, device=device, dtype=dtype), dim=-1)
        return model, queries, bank, mask, contexts

    def test_current_only_ninety_one_tensors_and_exact_context_geometry(self):
        model = NeuralSlots()
        self.assertEqual(tuple(KEYS[87:90]), tuple(name for name in model.state_dict() if name in CONTEXT_KEYS))
        self.assertEqual(len(KEYS), 91)
        self.assertEqual(sum(model.state_dict()[name].numel() for name in CONTEXT_KEYS), 70656)
        self.assertEqual(float(model.read_context_delta.weight.detach().abs().sum()), 0.)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        old = {name: value for name, value in before.items() if name not in CONTEXT_KEYS}
        with self.assertRaisesRegex(ValueError, "exactly the current"):
            require_parameters(model, old)
        bad = dict(before, **{"read_context_query.weight": torch.zeros(16, SEMANTIC + 1)})
        with self.assertRaisesRegex(ValueError, "geometry"):
            require_parameters(model, bad)
        self.assertTrue(all(torch.equal(model.state_dict()[name], value) for name, value in before.items()))

    def test_context_constructor_preserves_original_source_rng(self):
        original = torch.nn.Linear
        captured = []
        def linear(*args, **kwargs):
            if args == (2 * HIDDEN, READ_CONTEXT_WIDTH):
                captured.append(torch.get_rng_state().clone())
            return original(*args, **kwargs)
        with patch("neural_slot_model.nn.Linear", side_effect=linear):
            NeuralSlots()
        self.assertEqual(len(captured), 1)
        self.assertTrue(torch.equal(torch.get_rng_state(), captured[0]))

    def exercise_gradients(self, device):
        model, queries, bank, mask, contexts = self.inputs(device=device)
        model.requires_grad_(False)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in CONTEXT_KEYS)
        old = {name: value.clone() for name, value in model.state_dict().items() if name not in CONTEXT_KEYS}
        pairs = matcher_pair_features(queries, bank, mask, model.attention_scales())
        expected = model.read(pairs).squeeze(-1)
        latents = model.memory_context_latents(contexts)
        actual = model.semantic_bank_logits(queries, bank, mask, memory_context_latents=latents)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.square().sum().backward()
        self.assertGreater(float(model.read_context_delta.weight.grad.abs().sum()), 0.)
        for head in (model.read_context_memory, model.read_context_query):
            self.assertIsNotNone(head.weight.grad)
            self.assertEqual(float(head.weight.grad.abs().sum()), 0.)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.05)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.semantic_bank_logits(queries, bank, mask, memory_context_latents=model.memory_context_latents(contexts)).square().sum().backward()
        for name, parameter in model.named_parameters():
            if name in CONTEXT_KEYS:
                self.assertEqual(parameter.grad.device.type, device)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0.)
            else:
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, old[name], atol=0, rtol=0)
        if device == "mps":
            cpu = deepcopy(model).to("cpu")
            with torch.no_grad():
                expected_cpu = cpu.semantic_bank_logits(queries.cpu(), bank.cpu(), mask.cpu(),
                    memory_context_latents=cpu.memory_context_latents(contexts.cpu()))
                actual_mps = model.semantic_bank_logits(queries, bank, mask,
                    memory_context_latents=model.memory_context_latents(contexts)).cpu()
            torch.testing.assert_close(actual_mps, expected_cpu, atol=3e-5, rtol=3e-5)

    def test_zero_residual_exact_parent_and_all_three_cpu_gradients(self):
        self.exercise_gradients("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_actual_mps_nonzero_context_gradients_and_cpu_operator_parity(self):
        self.exercise_gradients("mps")

    def test_nonzero_context_formula_scalar_bank_padding_and_scope_independence(self):
        model, queries, bank, mask, contexts = self.inputs(dtype=torch.float64)
        model.requires_grad_(False)
        model.read_context_delta.weight.normal_(0, .2)
        latents = model.memory_context_latents(contexts)
        pairs = matcher_pair_features(queries, bank, mask, model.attention_scales())
        pre = model.read[0](pairs)
        product = latents[None] * model.read_context_query(queries[:, :SEMANTIC]).tanh()[:, None]
        expected = model.read[2](pre.tanh()).squeeze(-1) + F.linear(
            (pre + model.read_context_delta(product)).tanh() - pre.tanh(), model.read[2].weight).squeeze(-1)
        actual = model.semantic_bank_logits(queries, bank, mask, memory_context_latents=latents)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        changed = bank.clone(); changed[~mask] = 1e8
        torch.testing.assert_close(model.semantic_bank_logits(queries, changed, mask, memory_context_latents=latents), actual, atol=0, rtol=0)
        ids = F.normalize(torch.randn(3, 2112, dtype=torch.float64), dim=-1)
        memory_ids = F.normalize(torch.randn(4, 2112, dtype=torch.float64), dim=-1)
        meta = torch.tensor([[1., 0., 0., 0.]] * 4, dtype=torch.float64)
        for scope in (False, True):
            scored = model.score_bank(queries, ids, bank, memory_ids, meta, mask,
                                      memory_context_latents=latents, scope=scope)
            scalar = torch.stack([torch.stack([model.score(queries[i], ids[i], bank[j, mask[j]],
                memory_ids[j], meta[j], memory_context_latents=latents[j], scope=scope)
                for j in range(4)]) for i in range(3)])
            torch.testing.assert_close(scored, scalar, atol=2e-12, rtol=2e-12)
        scope_before = model.semantic_bank_logits(queries, bank, mask, memory_context_latents=latents, scope=True)
        for name, parameter in model.named_parameters():
            if name in CONTEXT_KEYS: parameter.add_(.4)
        scope_after = model.semantic_bank_logits(queries, bank, mask, memory_context_latents=-latents, scope=True)
        torch.testing.assert_close(scope_after, scope_before, atol=0, rtol=0)

    def test_context_representation_is_required_and_empty_banks_are_explicit(self):
        model, queries, bank, mask, contexts = self.inputs()
        with self.assertRaises(TypeError): model.semantic_bank_logits(queries, bank, mask)
        for value in (torch.zeros(4, 15), torch.zeros(4, 16, dtype=torch.float64),
                      torch.full((4, 16), float("nan")), torch.full((4, 16), 1.1), None):
            with self.assertRaisesRegex(ValueError, "memory context latents"):
                model.semantic_bank_logits(queries, bank, mask, memory_context_latents=value)
        empty = model.semantic_bank_logits(queries, bank[:0], mask[:0], memory_context_latents=queries.new_zeros(0, 16))
        self.assertEqual(tuple(empty.shape), (3, 0))
        empty_queries = model.semantic_bank_logits(queries[:0], bank, mask,
                                                   memory_context_latents=model.memory_context_latents(contexts))
        self.assertEqual(tuple(empty_queries.shape), (0, 4))

    def test_dynamic_parent_recomputes_own_context_and_retains_original_subject_gates(self):
        model, facts, rows, cached = temporal_tests.TemporalReaderContract().fixture()
        rows = deepcopy(rows)
        rows[0].update(initial_text=rows[1]["text"], initial_reference="Ada")
        with torch.no_grad(): model.read_context_delta.weight.normal_(0, .5)
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        texts = [fact["text"] for fact in facts]
        self.assertEqual(parent_cached.kind_contexts.data_ptr(), cached.kind_contexts.data_ptr())
        with torch.no_grad():
            original = reader.functional_reader_trace(parent, rows, facts, parent_cached)
            before = reader.functional_reader_trace(model, rows, facts, cached)
            old_sidecars = parent_cached.memory_context_latents(texts).clone()
            model.read_context_memory.weight.add_(.1)
            model.read_context_query.weight.add_(.3)
            model.read_context_delta.weight.add_(.2)
            after_parent = reader.functional_reader_trace(parent, rows, facts, parent_cached)
            live = reader.functional_reader_trace(model, rows, facts, cached)
            bank, ids, mask, meta = reader.make_bank(facts, cached)
            q, identity = reader.query(model, rows, cached)
            score = model.score_bank(q, identity, bank, ids, meta, mask,
                                     memory_context_latents=cached.memory_context_latents(texts))
        self.assertTrue(all(torch.equal(original[key], after_parent[key]) for key in original))
        self.assertFalse(torch.equal(old_sidecars, cached.memory_context_latents(texts)))
        self.assertFalse(torch.equal(before["semantic"], live["semantic"]))
        for key in ("identity", "intent", "temporal_gate", "fixed_log_gate"):
            torch.testing.assert_close(before[key], live[key], atol=0, rtol=0)
        torch.testing.assert_close(live["activation"], score, atol=0, rtol=0)
        with patch.object(model, "advance_identity", wraps=model.advance_identity) as update:
            reader.query(model, rows, cached)
        initial_rows = []
        for row in rows:
            initial = dict(text=row.get("initial_text", row["text"]))
            reference = row.get("initial_reference", row.get("reference"))
            if reference: initial["reference"] = reference
            initial_rows.append(initial)
        original_id = reader.raw_query(model, initial_rows, cached)[1]
        torch.testing.assert_close(update.call_args.args[2], original_id, atol=0, rtol=0)

    def test_matcher_parent_and_anchor_include_nonzero_frozen_context_branch(self):
        model, facts, rows, cached = temporal_tests.TemporalReaderContract().fixture()
        reader.configure_reader_heads(model, "matcher", freeze_memory_encoder=True)
        with torch.no_grad():
            model.read_context_delta.weight.normal_(0, .4)
            model.read[2].bias.fill_(3.)
            model.identity_match[2].weight.zero_(); model.identity_match[2].bias.fill_(20.)
            model.version[2].weight.zero_(); model.version[2].bias.fill_(20.)
            model.intent.weight.zero_(); model.intent.bias[:3].fill_(20.)
            activation = reader.functional_reader_trace(model, rows, facts, cached)["activation"]
        rows = deepcopy(rows)
        for i, row in enumerate(rows):
            row["support_texts"] = [fact["text"] for j, fact in enumerate(facts) if activation[i, j] > 0]
        parent = reader.snapshot_reader_matcher(model)
        loss = reader.matcher_preservation_loss(model, rows, facts, cached, parent)
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        for name, parameter in model.named_parameters():
            if name.startswith("read."):
                self.assertIsNotNone(parameter.grad)
                self.assertEqual(float(parameter.grad.abs().sum()), 0.)
            else: self.assertIsNone(parameter.grad)
        model.zero_grad(set_to_none=True)
        with torch.no_grad(): model.read[2].bias.sub_(20.)
        lost = reader.matcher_preservation_loss(model, rows, facts, cached, parent)
        self.assertGreater(float(lost.detach()), 0.)
        lost.backward()
        for name, parameter in model.named_parameters():
            if name.startswith("read."):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            else: self.assertIsNone(parameter.grad)

    def test_head_freezing_includes_new_context_only_for_semantic_training(self):
        for mode in ("all", "read", "semantic", "activation", "matcher", "query", "temporal", "scope", "identity"):
            model = NeuralSlots()
            reader.configure_reader_heads(model, mode, freeze_memory_encoder=mode in reader.GUARDED_READER_MODES)
            trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
            self.assertEqual(CONTEXT_KEYS <= trainable, mode in ("all", "read", "semantic", "activation"), mode)
            self.assertEqual(CONTEXT_KEYS.isdisjoint(trainable), mode not in ("all", "read", "semantic", "activation"), mode)
            self.assertTrue(set(reader.PLAN_KEYS + reader.QUERY_KEYS).isdisjoint(trainable), mode)


if __name__ == "__main__":
    unittest.main()
