"""Optimizer mechanics on tiny artificial tensors, never research training."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from availability_supervision import ReplyTargets
from eval_joint_baseline import advance, fixed_panel
from audit_joint_baseline import verify_input_bytes
from joint_memory_model import create_joint_model
from joint_optimizer import (_run_fixed_loop, save_checkpoint, load_checkpoint, tensor_digest,
                             validate_config, bind_schedule, train_candidate)
from joint_training_data import JointTrainingPackage, JointForwardFeatures
from memory_token_read import TokenReadInput
from native_memory_encoder import sha_file
from token_memory_supervision import TokenSupervision
from train_joint_memory import paired_forward, paired_loss

ROOT = Path(__file__).resolve().parents[1]


class StopAfterFirst(Exception): pass


class JointOptimizerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        self.model = self.fresh()
        q = torch.randn(4, 16); h = torch.randn(3, 8); b = torch.randn(3, 13)
        self.features = [JointForwardFeatures(TokenReadInput(q, torch.randn(4, 16),
            torch.tensor([4, 5, 6, 7]), torch.ones(4, dtype=torch.bool), 'fixture'), h, b) for _ in range(2)]
        self.targets = [TokenSupervision(ReplyTargets((1, 2), (3, 4, 2), i, 1., 'fixture'),
            (True, True), (None, (1,), None) if i == 1 else (None,) * 3) for i in (1, 2)]
        self.head = torch.randn(13, 8)
        self.schedule = [{'step': i + 1, 'pairs': [{'id': f'{i}-{j}', 'kind': 'role_swap',
            'left_id': 'fixture-a', 'right_id': 'fixture-b'} for j in range(4)]} for i in range(50)]

    @staticmethod
    def fresh(arm='joint_aux'):
        return create_joint_model('fixture', (0,), arm, hidden=8, vocab=13, layers=2, heads=2)

    def sample(self, pair): return self.features, self.targets

    def test_accumulation_matches_mean_of_four_pair_losses(self):
        expected = copy.deepcopy(self.model)
        params = [p for p in expected.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=.0001, weight_decay=.01, foreach=False)
        losses = [paired_loss(paired_forward(expected, self.features, self.head, 4.),
                              self.features, self.targets, 'role_swap')['total'] for _ in range(4)]
        (sum(losses) / 4).backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True); optimizer.step()
        rows = []
        def record(row):
            rows.append(row); raise StopAfterFirst()
        with self.assertRaises(StopAfterFirst):
            _run_fixed_loop(self.model, self.head, 4., self.schedule, self.sample,
                            on_checkpoint=lambda *args: None, on_step=record)
        for a, b in zip(self.model.parameters(), expected.parameters()):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)
            self.assertIsNone(a.grad)
        self.assertAlmostEqual(rows[0]['batch_mean_loss'], float((sum(losses) / 4).detach()), places=6)
        self.assertAlmostEqual(rows[0]['gradient_norm_before_clip'], float(norm), places=5)

    def test_full_tiny_loop_exact_budget_frozen_parameters_and_checkpoints(self):
        frozen = {n: p.clone() for n, p in self.model.named_parameters() if not p.requires_grad}
        head = self.head.clone(); checkpoints = []; rows = []; before = tensor_digest(self.model.state_dict())
        result = _run_fixed_loop(self.model, self.head, 4., self.schedule, self.sample,
            on_checkpoint=lambda step, *args: checkpoints.append(step), on_step=rows.append)
        self.assertEqual(checkpoints, [0, 50]); self.assertEqual(len(rows), 50)
        self.assertEqual(result['optimizer_steps'], 50); self.assertEqual(result['record_uses'], 400)
        self.assertNotEqual(before, tensor_digest(self.model.state_dict()))
        self.assertTrue(torch.equal(head, self.head)); self.assertIsNone(self.head.grad)
        self.assertTrue(all(torch.equal(p, frozen[n]) for n, p in self.model.named_parameters() if n in frozen))

    def test_product_loop_and_final_checkpoint_restore_optimizer_audit(self):
        model = self.fresh('joint_product'); initial = tensor_digest(model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'final.pt'; binding = {'fixture': True}; saved = []
            def checkpoint(step, candidate, optimizer, original):
                if step == 50: saved.append(save_checkpoint(path, candidate, optimizer, step, binding, original))
            _run_fixed_loop(model, self.head, 4., self.schedule, self.sample,
                            on_checkpoint=checkpoint, on_step=lambda row: None)
            restored = self.fresh('joint_product')
            value = load_checkpoint(path, expected_sha256=saved[0]['sha256'], model=restored,
                                    expected_binding=binding, expected_initial_digest=initial)
            self.assertEqual(tensor_digest(restored.state_dict()), tensor_digest(model.state_dict()))
            self.assertEqual(value['step'], 50)
            self.assertTrue(all(int(v['step']) == 50 for v in value['optimizer_state']['state'].values()))
            self.assertFalse(value['resume_allowed'])

    def test_invalid_budget_rejected_before_updates(self):
        before = tensor_digest(self.model.state_dict())
        for schedule in (self.schedule[:49], self.schedule + [self.schedule[0]],
                         [dict(self.schedule[0], step=2)] + self.schedule[1:]):
            with self.assertRaises(ValueError):
                _run_fixed_loop(self.model, self.head, 4., schedule, self.sample,
                                on_checkpoint=lambda *a: None, on_step=lambda r: None)
        self.assertEqual(before, tensor_digest(self.model.state_dict()))

    def test_optional_factor_gradients_do_not_reuse_previous_step(self):
        schedule = copy.deepcopy(self.schedule)
        for p in schedule[1]['pairs']: p['kind'] = 'ordinary_empty'
        normal = [TokenSupervision(replace(t.reply, state_index=0), (None, None), (None,) * 3) for t in self.targets]
        memory = self.features[1].memory
        empty = replace(self.features[1], memory=replace(memory, source=memory.source[:0],
            token_ids=memory.token_ids[:0], copy_allowed=memory.copy_allowed[:0]))
        after_first = {}
        def sample(pair):
            return ([self.features[0], empty], normal) if pair['kind'] == 'ordinary_empty' else self.sample(pair)
        def emit(row):
            if row['step'] == 1:
                after_first.update({n: p.clone() for n, p in self.model.named_parameters() if n.startswith('factor_heads.')})
            if row['step'] == 2:
                for n, p in self.model.named_parameters():
                    if n in after_first:
                        self.assertIsNone(p.grad)
                        self.assertTrue(torch.equal(p, after_first[n]))
                raise StopAfterFirst()
        with self.assertRaises(StopAfterFirst):
            _run_fixed_loop(self.model, self.head, 4., schedule, sample,
                            on_checkpoint=lambda *args: None, on_step=emit)

    def test_bad_fourth_pair_prevents_entire_update_and_clears_gradients(self):
        before = tensor_digest(self.model.state_dict()); calls = []
        def sample(pair):
            calls.append(pair)
            if len(calls) == 4: raise ValueError('bad sample')
            return self.sample(pair)
        with self.assertRaisesRegex(ValueError, 'bad sample'):
            _run_fixed_loop(self.model, self.head, 4., self.schedule, sample,
                            on_checkpoint=lambda *a: None, on_step=lambda r: self.fail('update occurred'))
        self.assertEqual(before, tensor_digest(self.model.state_dict()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_checkpoint_roundtrip_binding_and_exclusive_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'initial.pt'; binding = {'fixture': True}
            initial = tensor_digest(self.model.state_dict())
            opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=.0001)
            info = save_checkpoint(path, self.model, opt, 0, binding, initial)
            restored = self.fresh()
            data = load_checkpoint(path, expected_sha256=info['sha256'], model=restored,
                                   expected_binding=binding, expected_initial_digest=initial)
            self.assertEqual(tensor_digest(restored.state_dict()), initial)
            self.assertFalse(data['resume_allowed']); self.assertFalse(data['deployment_approved'])
            with self.assertRaises(FileExistsError): save_checkpoint(path, self.model, opt, 0, binding, initial)
            self.assertEqual(sha_file(path), info['sha256'])
            for m, b, sha in ((self.fresh('joint_product'), binding, info['sha256']),
                              (self.fresh(), {'fixture': False}, info['sha256']), (self.fresh(), binding, 'bad')):
                before = tensor_digest(m.state_dict())
                with self.assertRaises(ValueError):
                    load_checkpoint(path, expected_sha256=sha, model=m, expected_binding=b, expected_initial_digest=initial)
                self.assertEqual(before, tensor_digest(m.state_dict()))

    def test_checkpoint_rejects_nonfinite_or_changed_frozen_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=.0001)
            initial = tensor_digest(self.model.state_dict()); binding = {'fixture': True}
            with torch.no_grad(): self.model.roles.state_head.bias.add_(1)
            path = Path(directory) / 'bad.pt'
            info = save_checkpoint(path, self.model, opt, 50, binding, initial)
            with self.assertRaisesRegex(ValueError, 'frozen'):
                load_checkpoint(path, expected_sha256=info['sha256'], model=self.fresh(),
                                expected_binding=binding, expected_initial_digest=initial)
            with torch.no_grad(): self.model.reader.state_head.bias[0] = float('nan')
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                save_checkpoint(Path(directory) / 'nan.pt', self.model, opt, 50, binding, initial)
            self.assertFalse((Path(directory) / 'nan.pt').exists())

    def test_config_drift_and_production_launch_rejected(self):
        config = json.loads((ROOT / 'training/memory/neural-system/experiments/CG-003.json').read_text())
        validate_config(config)
        for key, value in (('initial_seed', 99), ('constraints', {}), ('optimizer', {})):
            bad = copy.deepcopy(config); bad[key] = value
            with self.assertRaises(ValueError): validate_config(bad)
        config['training_approved'] = config['prerequisites_complete'] = True
        with self.assertRaisesRegex(ValueError, 'Training blocked'):
            train_candidate(None, config, None, None, 'joint_aux', 'must-not-be-created')

    def test_baseline_exact_panel_and_actual_prefix_stop(self):
        audit = json.loads((ROOT / 'training/memory/neural-system/checks/CG-003-data.json').read_text())
        corpus = ROOT / 'training/memory/neural-system/data/JB-001'
        records, panel = fixed_panel(corpus, audit)
        self.assertEqual(len(records), 24); self.assertEqual(panel, audit['native_panel'])
        bad = copy.deepcopy(audit); bad['native_panel'].reverse()
        with self.assertRaises(ValueError): fixed_panel(corpus, bad)
        generated, visible = [], []
        prefix, done = advance((1, 2), generated, visible, 3, (9,))
        self.assertEqual(prefix, (1, 2, 3)); self.assertFalse(done)
        prefix, done = advance(prefix, generated, visible, 9, (9,))
        self.assertEqual(prefix, (1, 2, 3)); self.assertTrue(done)
        self.assertEqual(generated, [3, 9]); self.assertEqual(visible, [3])

    def test_raw_prefix_audit_rejects_teacher_prefix_and_trailing_bytes(self):
        import struct
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.bin'
            raw = b'BNPI0001' + struct.pack('<I', 1) + struct.pack('<III', 2, 1, 2)
            path.write_bytes(raw)
            verify_input_bytes(path, [(1, 2)])
            with self.assertRaises(ValueError): verify_input_bytes(path, [(1, 3)])
            path.write_bytes(raw + b'x')
            with self.assertRaises(ValueError): verify_input_bytes(path, [(1, 2)])


if __name__ == '__main__': unittest.main()
