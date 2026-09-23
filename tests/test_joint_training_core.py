"""CG-003 route/loss/packing guards. No optimizer and no accuracy claims."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from availability_supervision import ReplyTargets
from joint_memory_model import create_joint_model
from joint_training_data import JointForwardFeatures, JointTrainingPackage, allowed_splits, compile_records
from memory_token_read import TokenReadInput
from token_memory_supervision import TokenSupervision, supervised_token_loss
from train_joint_memory import paired_forward, paired_loss, require_joint_launch


class JointTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.models = [create_joint_model('test', (0,), arm, hidden=8, vocab=13, layers=2, heads=2)
                       for arm in ('joint_aux', 'joint_product')]
        q = torch.randn(4, 16); h = torch.randn(3, 8); z = torch.randn(3, 13)
        self.features = [JointForwardFeatures(TokenReadInput(q, torch.randn(4, 16), torch.tensor([4, 5, 6, 7]),
                            torch.ones(4, dtype=torch.bool), 'test'), h, z) for _ in range(2)]
        self.head = torch.randn(13, 8)

    def targets(self, value=False):
        return [TokenSupervision(ReplyTargets((1, 2), (3, 4, 2), 1, 1., 'a'), (True, True), (None, (1,), None)),
                TokenSupervision(ReplyTargets((1, 2), (3, 5, 2), 1 if value else 2, 1., 'b'),
                                 (True, True), (None, (2,), None) if value else (None,) * 3)]

    def test_identical_initial_weights_and_only_unused_classifier_frozen(self):
        a, b = self.models
        self.assertEqual(set(a.state_dict()), set(b.state_dict()))
        self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))
        frozen = {n for n, p in a.named_parameters() if not p.requires_grad}
        self.assertEqual(frozen, {'roles.state_encoder.weight', 'roles.state_encoder.bias', 'roles.state_head.weight', 'roles.state_head.bias'})
        self.assertEqual(int(torch.count_nonzero(a.roles.content.output.weight)), 0)
        self.assertGreater(float(a.roles.content.layer_gain[-1].detach()), 0)

    def test_route_is_only_forward_difference_between_arms(self):
        states = [m.prefill(self.features[0].memory) for m in self.models]
        torch.testing.assert_close(states[0].core.state_logits.exp(), states[0].core.pointer.state_logits.softmax(-1))
        self.assertLess(float(states[1].core.state_logits[1].detach()), float(states[0].core.state_logits[1].detach()))
        outputs = [m.branches(self.features[0].hidden, self.features[0].base, self.head, 4., s) for m, s in zip(self.models, states)]
        for field in ('supported', 'uncertainty', 'factor_logits', 'positions', 'copy_mass'):
            self.assertTrue(torch.equal(getattr(outputs[0], field), getattr(outputs[1], field)), field)

    def test_reply_policy_and_parameters_cannot_change(self):
        model = self.models[0]; f = self.features[0]; state = model.prefill(f.memory)
        model._route_policy = 'joint_product'
        with self.assertRaises(ValueError): model.read(f.hidden, f.base, self.head, 4., state)
        with self.assertRaises(AttributeError): model.route_policy = 'joint_aux'

    def test_role_pair_loss_matches_equation_and_gradients_are_finite(self):
        m = self.models[0]; targets = self.targets()
        out = paired_forward(m, self.features, self.head, 4.)
        loss = paired_loss(out, self.features, targets, 'role_swap')
        margins = [o.state_logits.double()[1] - o.state_logits.double()[2] for o in out]
        expected = sum(supervised_token_loss(o, t)['weighted_total'] for o, t in zip(out, targets)) / 2 + F.softplus(1-(margins[0]-margins[1]))
        torch.testing.assert_close(loss['total'], expected)
        params = [p for p in m.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss['total'], params, allow_unused=True)
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_value_pair_requires_same_c_prefix_and_real_value_target(self):
        m = self.models[0]; targets = self.targets(True)
        outputs = paired_forward(m, self.features, self.head, 4.)
        loss = paired_loss(outputs, self.features, targets, 'value_swap')
        self.assertTrue(torch.isfinite(loss['total']))
        bad = [self.features[0], replace(self.features[1], base=self.features[1].base + .1)]
        with self.assertRaises(ValueError): paired_loss(outputs, bad, targets, 'value_swap')
        bad_targets = [replace(targets[0], positions=(None,) * 3), targets[1]]
        with self.assertRaises(ValueError): paired_loss(outputs, self.features, bad_targets, 'value_swap')

    def test_query_and_prompt_mismatch_or_unknown_pair_rejected(self):
        bad = [self.features[0], replace(self.features[1], memory=replace(self.features[1].memory, query=self.features[1].memory.query + 1))]
        with self.assertRaises(ValueError): paired_forward(self.models[0], bad, self.head, 4.)
        out = paired_forward(self.models[0], self.features, self.head, 4.)
        with self.assertRaises(ValueError): paired_loss(out, self.features, self.targets(), 'unknown')
        targets = self.targets(); targets[1] = replace(targets[1], reply=replace(targets[1].reply, prompt_token_ids=(1, 9)))
        with self.assertRaises(ValueError): paired_loss(out, self.features, targets, 'role_swap')

    def test_sealed_split_rejected_before_reading_any_file(self):
        allowed_splits(('train', 'dev'))
        for splits in (('test',), ('train', 'test'), ('train', 'train'), ['train']):
            with self.assertRaises(ValueError): compile_records('absent', 'absent', None, splits=splits)

    def test_smoke_and_dev_cannot_feed_optimizer(self):
        package = object.__new__(JointTrainingPackage)
        for scope, split in (('smoke', 'train'), ('full', 'dev')):
            record = {'split': split}; package.records = [record]; package.manifest = {'scope': scope}
            with self.assertRaises(ValueError): package.sample(record, training=True)

    def test_json_approval_flags_cannot_bypass_unimplemented_launch(self):
        with self.assertRaisesRegex(ValueError, 'Training blocked'):
            require_joint_launch(training_approved=True, prerequisites_complete=True)


if __name__ == '__main__': unittest.main()
