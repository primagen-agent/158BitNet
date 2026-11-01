"""Current contextual kind head, raw-feature routing and fit-only calibration."""

import importlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
candidate = importlib.import_module("neural_slot_model")
CONTEXT_KEYS = {"kind_context.weight", "kind_context.bias"}
TIME_KEYS = {"intent.weight", "intent.bias", "version.0.weight", "version.0.bias", "version.2.weight", "version.2.bias"}

def candidate_reader():
    return importlib.import_module("train_neural_dialogue_reader")

class KindContextContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.backbone_sha = 'ab' * 32

    def setUp(self):
        torch.manual_seed(6181)

    def model(self, device='cpu'):
        return candidate.NeuralSlots().to(device)

    def test_current_geometry_has91_tensors_and_zero_context_residual(self):
        model = self.model()
        self.assertEqual(len(candidate.KEYS), 91)
        self.assertEqual(set(model.state_dict()), set(candidate.KEYS))
        self.assertEqual(set(candidate.KEYS[85:87]), CONTEXT_KEYS)
        for key in CONTEXT_KEYS:
            self.assertEqual(float(model.state_dict()[key].abs().sum()), 0.0)
        self.assertEqual(tuple(model.kind_context.weight.shape), (3, 4096))
        self.assertEqual(tuple(model.kind_context.bias.shape), (3,))

    def test_zero_context_preserves_kind64_logits_for_ragged_sources(self):
        model = self.model()
        for count in (1, 2, 7):
            raw = torch.randn(count, candidate.INPUT)
            with torch.no_grad():
                encoded = model.encode(model.augment(raw)).tanh()
                actual = model.kind_logits_from_mean(encoded.mean(0), model.kind_context_inputs(raw))
                torch.testing.assert_close(actual, model.kind(encoded.mean(0)), atol=0, rtol=0)

    def test_strict_checkpoint_requires_91_and_preserves_model_on_invalid_load(self):
        model = self.model()
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        bad_geometry = dict(before)
        bad_geometry['kind_context.weight'] = torch.zeros(3, 4095)
        bad_values = dict(before)
        bad_values['kind_context.bias'] = torch.full((3,), float('nan'))
        for state in ({key: value for key, value in before.items() if key not in CONTEXT_KEYS}, bad_geometry, bad_values):
            with self.assertRaises(ValueError):
                candidate.require_parameters(model, state)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_hidden_last_and_mean_have_one_joint_l2_scale(self):
        model = self.model()
        raw = torch.randn(5, candidate.INPUT, dtype=torch.float64)
        hidden = raw[:, :candidate.HIDDEN]
        expected = torch.cat((hidden[-1], hidden.mean(0)))
        expected /= expected.norm().clamp_min(1e-12)
        actual = model.kind_context_inputs(raw)
        self.assertEqual(tuple(actual.shape), (4096,))
        self.assertEqual(actual.dtype, raw.dtype)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual.norm(), torch.ones((), dtype=raw.dtype), atol=1e-14, rtol=1e-14)

    def test_summary_is_independent_of_token_embedding_values(self):
        model = self.model()
        raw = torch.randn(4, candidate.INPUT)
        altered = raw.clone()
        altered[:, candidate.HIDDEN:] = torch.randn_like(altered[:, candidate.HIDDEN:]) * 100000000.0
        torch.testing.assert_close(model.kind_context_inputs(raw), model.kind_context_inputs(altered), atol=0, rtol=0)
        zero_hidden = raw.clone()
        zero_hidden[:, :candidate.HIDDEN] = 0
        self.assertTrue(torch.equal(model.kind_context_inputs(zero_hidden), torch.zeros(4096)))

    def test_summary_hidden_gradients_exclude_embeddings(self):
        model = self.model()
        raw = torch.randn(4, candidate.INPUT, requires_grad=True)
        weights = torch.linspace(-1.0, 1.0, 4096)
        (model.kind_context_inputs(raw) * weights).sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(float(raw.grad[:, :candidate.HIDDEN].abs().sum()), 0.0)
        self.assertEqual(float(raw.grad[:, candidate.HIDDEN:].abs().sum()), 0.0)

    def test_ragged_batched_logits_exclude_padded_rows(self):
        model = self.model()
        with torch.no_grad():
            model.kind_context.weight.normal_(0, 0.1)
            model.kind_context.bias.copy_(torch.tensor([0.2, -0.1, 0.3]))
        sources = [torch.randn(n, candidate.INPUT) for n in (1, 3, 5)]
        padded = torch.nn.utils.rnn.pad_sequence(sources, batch_first=True, padding_value=10000000.0)
        lengths = torch.tensor([len(x) for x in sources])
        mask = torch.arange(padded.shape[1])[None] < lengths[:, None]
        contexts = torch.stack([model.kind_context_inputs(padded[i, mask[i]]) for i in range(3)])
        means = torch.randn(3, candidate.WIDTH)
        expected = torch.stack([model.kind_logits_from_mean(means[i], model.kind_context_inputs(sources[i])) for i in range(3)])
        torch.testing.assert_close(model.kind_logits_from_mean(means, contexts), expected, atol=3e-06, rtol=3e-06)
        with self.assertRaises(ValueError):
            model.kind_context_inputs(padded)

    def test_summary_rejects_empty_malformed_integer_and_nonfinite_inputs(self):
        model = self.model()
        bad = [None, [], torch.zeros(0, candidate.INPUT), torch.zeros(candidate.INPUT), torch.zeros(2, candidate.INPUT - 1), torch.zeros(2, candidate.INPUT, dtype=torch.int64), torch.full((2, candidate.INPUT), float('nan')), torch.full((2, candidate.INPUT), float('inf'))]
        for raw in bad:
            with self.subTest(shape=getattr(raw, 'shape', None)):
                with self.assertRaises(ValueError):
                    model.kind_context_inputs(raw)

    def test_context_token_limit_matches_native_maximum(self):
        self.assertEqual(candidate.TOKEN_LIMIT, 4096)
        row = torch.zeros(1, candidate.INPUT)
        row[0, 0], row[0, candidate.HIDDEN - 1] = .6, .8
        expected = candidate.NeuralSlots.kind_context_inputs(row)
        maximum = row.expand(candidate.TOKEN_LIMIT - 1, -1)
        torch.testing.assert_close(candidate.NeuralSlots.kind_context_inputs(maximum), expected,
                                   atol=1e-7, rtol=1e-7)
        for count in (candidate.TOKEN_LIMIT, candidate.TOKEN_LIMIT + 1):
            with self.subTest(count=count), self.assertRaises(ValueError):
                candidate.NeuralSlots.kind_context_inputs(row.expand(count, -1))

    def test_context_logits_reject_misaligned_geometry_dtype_and_nonfinite_values(self):
        model = self.model()
        means, contexts = (torch.randn(2, 64), torch.randn(2, 4096))
        bad = [(None, contexts), (means, None), (means[:, :63], contexts), (means, contexts[:, :4095]), (means, contexts[0]), (means[None], contexts[None]), (means, contexts.double()), (means.long(), contexts), (means, contexts.long()), (torch.full_like(means, float('nan')), contexts), (means, torch.full_like(contexts, float('inf')))]
        for memory, context in bad:
            with self.subTest(memory=getattr(memory, 'shape', None), context=getattr(context, 'shape', None)):
                with self.assertRaises(ValueError):
                    model.kind_logits_from_mean(memory, context)

    def test_raw_context_can_change_class_while_64_mean_is_identical(self):
        model = self.model()
        with torch.no_grad():
            model.kind.weight.zero_()
            model.kind.bias.copy_(torch.tensor([0.25, 0.0, -0.25]))
            model.kind_context.weight.zero_()
            model.kind_context.bias.zero_()
            for kind in (0, 1, 2):
                model.kind_context.weight[kind, kind] = 4.0
        means = torch.zeros(3, 64)
        raw = []
        for kind in (0, 1, 2):
            feature = torch.zeros(2, candidate.INPUT)
            feature[:, kind] = 1.0
            raw.append(feature)
        logits = model.kind_logits_from_mean(means, torch.stack([model.kind_context_inputs(x) for x in raw]))
        self.assertEqual(model.kind(means).argmax(-1).tolist(), [0, 0, 0])
        self.assertEqual(logits.argmax(-1).tolist(), [0, 1, 2])

    def routing_fixture(self):
        reader = candidate_reader()
        model = self.model()
        names = ('Ada', 'Ben', 'Cyra')
        facts = [dict(text=f'{name}. I check a gauge.', entity=name, reference=name, prop='gauge_check', kind=0) for name in names]
        texts = [fact['text'] for fact in facts] + list(names)
        cache = {}
        for index, text in enumerate(texts):
            count = (2, 4, 6)[index] if index < 3 else 1
            raw = torch.randn(count, candidate.INPUT)
            cache[text] = dict(x=raw, bounds=torch.tensor([(i * len(text.encode()) // count, (i + 1) * len(text.encode()) // count) for i in range(count)]))
        cached = reader.DynamicFeatures(model, {'fit': dict(facts=facts, queries=[])}, cache, 'cpu', freeze_memory_encoder=True)
        return (reader, model, facts, cache, cached)

    def test_dynamic_features_routes_original_raw_summaries_with_reordered_ragged_sources(self):
        _, model, facts, cache, cached = self.routing_fixture()
        with torch.no_grad():
            model.kind_context.weight.normal_(0, 0.1)
            model.kind_context.bias.copy_(torch.tensor([0.03, -0.02, 0.1]))
        texts = [facts[index]['text'] for index in (2, 0, 1)]
        bank, mask, _ = cached.encode(texts)
        means = (bank * mask[..., None]).sum(1) / mask.sum(1)[:, None]
        expected = torch.stack([model.kind_logits_from_mean(model.encode(model.augment(cache[text]['x'])).tanh().mean(0), model.kind_context_inputs(cache[text]['x'])) for text in texts])
        torch.testing.assert_close(cached.kind_logits(texts, means), expected, atol=3e-06, rtol=3e-06)
        contexts = torch.stack([model.kind_context_inputs(cache[text]['x']) for text in texts])
        torch.testing.assert_close(cached.kind_contexts[cached.indices(texts)], contexts, atol=0, rtol=0)
        self.assertFalse(cached.kind_contexts.requires_grad)

    def test_independent_parent_keeps_own_kind_heads_on_shared_original_contexts(self):
        reader, model, facts, _, cached = self.routing_fixture()
        with torch.no_grad():
            model.kind.weight.zero_()
            model.kind.bias.copy_(torch.tensor([2.0, 0.0, -2.0]))
        parent, parent_cached = reader.snapshot_functional_reader(model, cached)
        self.assertEqual(parent_cached.kind_contexts.data_ptr(), cached.kind_contexts.data_ptr())
        self.assertNotEqual(parent.kind_context.weight.data_ptr(), model.kind_context.weight.data_ptr())
        raw_context = cached.kind_contexts.clone()
        parent_weights = {name: value.detach().clone() for name, value in parent.named_parameters()}
        texts = [fact['text'] for fact in facts]
        with torch.no_grad():
            model.kind_context.bias.copy_(torch.tensor([-5.0, 0.0, 5.0]))
        for features, expected in ((parent_cached, [0, 0, 0]), (cached, [2, 2, 2])):
            bank, mask, _ = features.encode(texts)
            means = (bank * mask[..., None]).sum(1) / mask.sum(1)[:, None]
            self.assertEqual(features.kind_logits(texts, means).argmax(-1).tolist(), expected)
        bank, mask, _ = cached.encode(texts)
        means = (bank * mask[..., None]).sum(1) / mask.sum(1)[:, None]
        F.cross_entropy(cached.kind_logits(texts, means), torch.tensor([0, 1, 2])).backward()
        self.assertGreater(float(model.kind_context.bias.grad.abs().sum()), 0.0)
        for name, value in parent.named_parameters():
            self.assertIsNone(value.grad, name)
            torch.testing.assert_close(value, parent_weights[name], atol=0, rtol=0)
        torch.testing.assert_close(cached.kind_contexts, raw_context, atol=0, rtol=0)

    def test_bank_context_kind_changes_only_unspecified_physical_status(self):
        reader, model, facts, _, cached = self.routing_fixture()
        facts[1]['status'] = 1
        facts[2]['status'] = 3
        with torch.no_grad():
            model.kind.weight.zero_()
            model.kind.bias.copy_(torch.tensor([2.0, 0.0, -2.0]))
            model.kind_context.bias.copy_(torch.tensor([-5.0, 0.0, 5.0]))
        _, _, _, metadata = reader.make_bank(facts, cached)
        self.assertEqual(metadata[:, :3].tolist(), [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        gradient = torch.autograd.grad(metadata[0, 0], model.kind_context.bias, retain_graph=True)[0]
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        physical_gradient = torch.autograd.grad(metadata[1:, 0].sum(), model.kind_context.bias)[0]
        torch.testing.assert_close(physical_gradient, torch.zeros_like(physical_gradient), atol=0, rtol=0)

    def calibration_module(self):
        return importlib.import_module("neural_kind_context")

    def calibration_inputs(self, labels=(0, 1, 2)):
        labels = torch.tensor(labels, dtype=torch.long)
        means = torch.zeros(len(labels), candidate.WIDTH)
        contexts = torch.zeros(len(labels), 4096)
        contexts[torch.arange(len(labels)), labels] = 1.0
        return (means, contexts, labels)

    def test_context_ridge_calibration_changes_only_two_residual_tensors(self):
        model = self.model()
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        means, contexts, labels = self.calibration_inputs((0, 1, 2, 0, 1, 2))
        result = self.calibration_module().calibrate_kind_context(model, means, contexts, labels)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['final']['correct'], len(labels))
        self.assertGreater(result['final']['minimum_gold_margin'], 1e-05)
        self.assertEqual(result['ridge'], 0.001)
        self.assertEqual(result['target_scale'], 3.0)
        self.assertTrue(result['fit_only'])
        self.assertFalse(result['runtime_qualified'])
        self.assertGreater(float(model.kind_context.weight.detach().abs().sum()), 0.0)
        self.assertGreater(float(model.kind_context.bias.detach().abs().sum()), 0.0)
        for key, value in model.state_dict().items():
            if key not in CONTEXT_KEYS:
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_class_balancing_prevents_repeated_source_class_from_changing_ridge_solution(self):
        models = (self.model(), self.model())
        models[1].load_state_dict(models[0].state_dict(), strict=True)
        helper = self.calibration_module()
        for model, labels in zip(models, ((0, 1, 2), (0, 0, 0, 0, 0, 1, 1, 2))):
            means, contexts, labels = self.calibration_inputs(labels)
            self.assertTrue(helper.calibrate_kind_context(model, means, contexts, labels)['accepted'])
        torch.testing.assert_close(models[0].kind_context.weight, models[1].kind_context.weight, atol=1e-06, rtol=1e-06)
        torch.testing.assert_close(models[0].kind_context.bias, models[1].kind_context.bias, atol=1e-06, rtol=1e-06)

    def test_failed_ridge_acceptance_preserves_existing_residual_and_all_old_parameters(self):
        model = self.model()
        with torch.no_grad():
            model.kind_context.weight.normal_(0, 0.02)
            model.kind_context.bias.copy_(torch.tensor([2.0, -1.0, 3.0]))
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        means, contexts, labels = self.calibration_inputs()
        contexts.zero_()
        result = self.calibration_module().calibrate_kind_context(model, means, contexts, labels)
        self.assertFalse(result['accepted'])
        self.assertLess(result['final']['correct'], len(labels))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_ridge_rejects_evaluation_invalid_values_geometry_and_missing_class_without_mutation(self):
        model = self.model()
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        means, contexts, labels = self.calibration_inputs()
        invalid = [dict(split='development'), dict(split='sealed'), dict(means=torch.full_like(means, float('nan'))), dict(contexts=torch.full_like(contexts, float('inf'))), dict(contexts=contexts * 2.0), dict(contexts=contexts[:, :4095]), dict(labels=torch.tensor([0, 0, 1])), dict(labels=labels.float()), dict(labels=torch.tensor([-1, 1, 2])), dict(labels=torch.tensor([0, 1, 3]))]
        for overrides in invalid:
            args = dict(means=means, contexts=contexts, labels=labels, split='fit')
            args.update(overrides)
            with self.subTest(changes=list(overrides)):
                with self.assertRaises(ValueError):
                    self.calibration_module().calibrate_kind_context(model, **args)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_prefit_pipeline_freezes_kind_four_and_retains_only_time_six_trainable(self):
        reader, model, facts, _, cached = self.routing_fixture()
        reader.configure_reader_heads(model, 'temporal', freeze_memory_encoder=True)
        self.assertEqual({name for name, parameter in model.named_parameters() if parameter.requires_grad}, TIME_KEYS | CONTEXT_KEYS | {'kind.weight', 'kind.bias'})
        for index, fact in enumerate(facts):
            fact['kind'] = index
        prepared = dict(source_kind_facts={index: [fact] for index, fact in enumerate(facts)})
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        result = reader.prefit_temporal_kind(model, prepared, cached)
        self.assertTrue(result['accepted'])
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertEqual(trainable, TIME_KEYS)
        optimizer = reader.reader_optimizer(model, torch.nn.Linear(64, 3), 'temporal', 0.001, 0.01)
        self.assertEqual({id(parameter) for group in optimizer.param_groups for parameter in group['params']}, {id(parameter) for parameter in model.parameters() if parameter.requires_grad})
        for key, value in model.state_dict().items():
            if key not in CONTEXT_KEYS:
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_prefit_pipeline_rejects_evaluation_facts_before_calibration(self):
        reader, model, facts, _, cached = self.routing_fixture()
        for index, fact in enumerate(facts):
            fact['kind'] = index
        facts[1]['evaluation_only'] = True
        prepared = dict(source_kind_facts={index: [fact] for index, fact in enumerate(facts)})
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        with self.assertRaisesRegex(ValueError, 'development'):
            reader.prefit_temporal_kind(model, prepared, cached)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def _context_gradient_contract(self, device):
        model = self.model(device)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in CONTEXT_KEYS)
        frozen = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
        raw = [torch.randn(n, candidate.INPUT, device=device) for n in (2, 3, 5)]
        means = torch.stack([model.encode(model.augment(x)).tanh().mean(0).detach() for x in raw])
        contexts = torch.stack([model.kind_context_inputs(x) for x in raw])
        logits = model.kind_logits_from_mean(means, contexts)
        loss = F.cross_entropy(logits, torch.tensor([0, 1, 2], device=device))
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.03)
        loss.backward()
        for name, value in model.named_parameters():
            if name in CONTEXT_KEYS:
                self.assertIsNotNone(value.grad)
                self.assertTrue(torch.isfinite(value.grad).all())
                self.assertGreater(float(value.grad.abs().sum()), 0.0)
            else:
                self.assertIsNone(value.grad, name)
        optimizer.step()
        self.assertEqual(len(frozen), 89)
        self.assertTrue(set(candidate.PLAN_KEYS + candidate.QUERY_KEYS) <= frozen.keys())
        for name, value in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(value, frozen[name], atol=0, rtol=0)

    def test_context_stage_has_exactly_two_trainable_parameters_and_frozen_source46(self):
        self._context_gradient_contract('cpu')

    def _time_gradient_contract(self, device):
        model = self.model(device)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in TIME_KEYS)
        frozen = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
        raw = [torch.randn(n, candidate.INPUT, device=device) for n in (2, 3, 5)]
        intent = torch.stack([model.intent(model.summary(x)) for x in raw])
        metadata = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 1.0]], device=device)
        version = model.version(torch.cat((intent, metadata), -1)).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(version, torch.tensor([1.0, 0.0, 1.0], device=device))
        loss += F.binary_cross_entropy_with_logits(intent[:, :3], metadata[:, :3])
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        loss.backward()
        for name, value in model.named_parameters():
            if name in TIME_KEYS:
                self.assertIsNotNone(value.grad)
                self.assertTrue(torch.isfinite(value.grad).all())
                self.assertGreater(float(value.grad.abs().sum()), 0.0)
            else:
                self.assertIsNone(value.grad, name)
        optimizer.step()
        self.assertEqual(len(frozen), 85)
        self.assertTrue(CONTEXT_KEYS <= frozen.keys())
        self.assertTrue(set(candidate.PLAN_KEYS + candidate.QUERY_KEYS) <= frozen.keys())
        for name, value in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(value, frozen[name], atol=0, rtol=0)

    def test_time_stage_has_exactly_six_trainable_parameters_and_frozen_kind_context(self):
        self._time_gradient_contract('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_actual_mps_context_two_gradients_and_frozen_89(self):
        self._context_gradient_contract('mps')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_actual_mps_time_six_gradients_and_frozen_85(self):
        self._time_gradient_contract('mps')

    def test_strict_export91_geometry_crc_and_incomplete_state_rejection(self):
        model = self.model()
        with tempfile.TemporaryDirectory(dir=ROOT / 'build') as directory:
            artifact = Path(directory) / 'candidate.bnsmodel'
            candidate.export(model, self.backbone_sha, artifact)
            raw = artifact.read_bytes()
            self.assertEqual(raw[:8], b'BNSMODEL')
            self.assertEqual(struct.unpack('<4I', raw[8:24]), (1, candidate.INPUT, candidate.WIDTH, 91))
            self.assertEqual(raw[24:56].hex(), self.backbone_sha)
            offset = 56 + 91 * 8
            for index, key in enumerate(candidate.KEYS):
                size, crc = struct.unpack_from('<II', raw, 56 + index * 8)
                self.assertEqual(size, model.state_dict()[key].numel() * 4)
                self.assertEqual(zlib.crc32(raw[offset:offset + size]), crc)
                offset += size
            self.assertEqual(offset, len(raw))
            incomplete = self.model()
            del incomplete.kind_context
            with self.assertRaisesRegex(ValueError, 'tensors'):
                candidate.export(incomplete, self.backbone_sha, artifact)
            self.assertEqual(artifact.read_bytes(), raw)

    def test_export_rejects_wrong_context_geometry_and_nonfinite_parameters(self):
        for failure in ('geometry', 'nan'):
            model = self.model()
            if failure == 'geometry':
                model.kind_context = torch.nn.Linear(4095, 3)
            else:
                with torch.no_grad():
                    model.kind_context.weight[0, 0] = float('nan')
            with tempfile.TemporaryDirectory(dir=ROOT / "build") as directory:
                artifact = Path(directory) / 'invalid.bnsmodel'
                with self.assertRaisesRegex(ValueError, 'geometry'):
                    candidate.export(model, self.backbone_sha, artifact)
                self.assertFalse(artifact.exists())

    def test_context_normalization_is_invariant_to_positive_hidden_scale(self):
        model = self.model()
        raw = torch.randn(4, candidate.INPUT, dtype=torch.float64)
        scaled = raw.clone()
        scaled[:, :candidate.HIDDEN] *= 1000.0
        torch.testing.assert_close(model.kind_context_inputs(raw), model.kind_context_inputs(scaled), atol=1e-14, rtol=1e-14)

if __name__ == "__main__":
    unittest.main()
