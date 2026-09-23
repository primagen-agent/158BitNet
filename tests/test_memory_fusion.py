"""Architecture regressions: gradient flow, causality, and causal controls."""
from pathlib import Path
import copy
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from memory_fusion import GatedMemoryFusion
from prepare_memory_fusion_curriculum import make_rows
from torch_backbone import TorchBackbone
from train_memory_fusion import judge, prompt
from rescore_memory_fusion import rescore


def tiny_backbone():
    model = TorchBackbone.__new__(TorchBackbone)
    torch.nn.Module.__init__(model)
    model.cfg = SimpleNamespace(hidden=8, n_layers=3, n_heads=2, n_kv_heads=1,
                                q_dim=8, head_dim=4, rope_dim=4, rope_freq_base=10000.,
                                rope_factors=None, embedding_scale=1., residual_scale=1.,
                                rms_eps=1e-6, logit_scale=1.)
    model.device, model.dtype = "cpu", torch.float32
    model.token_embd = torch.randn(24, 8)
    model.out_proj = torch.randn(24, 8)
    model.out_norm = torch.ones(8)
    model.layers = [{"attn_norm": torch.ones(8), "ffn_norm": torch.ones(8),
                     **{key: torch.randn(rows, cols) * .1 for key, rows, cols in
                        (("q", 8, 8), ("k", 4, 8), ("v", 4, 8), ("o", 8, 8),
                         ("gate", 16, 8), ("up", 16, 8), ("down", 8, 16))}}
                    for _ in range(3)]
    return model


class MemoryFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.backbone = tiny_backbone()
        self.fusion = GatedMemoryFusion(8, 3, 2)
        self.memory = torch.randn(7, 16)
        self.ids = torch.tensor([1, 5, 3, 4])

    def test_zero_initialization_and_empty_memory_preserve_logits_exactly(self):
        baseline = self.backbone(self.ids, logits_all=True)
        self.assertTrue(torch.equal(baseline, self.backbone(self.ids, logits_all=True,
                        memory_fusion=self.fusion.bind(self.memory))))
        with torch.no_grad(): self.fusion.layer_gain.fill_(.7)
        self.assertTrue(torch.equal(baseline, self.backbone(self.ids, logits_all=True,
                        memory_fusion=self.fusion.bind(torch.empty(0, 16)))))

    def test_generation_loss_reaches_memory_through_frozen_layers(self):
        before = self.backbone.layers[2]["down"].clone()
        with torch.no_grad(): self.fusion.layer_gain.fill_(.1)
        logits = self.backbone(self.ids, logits_all=True, memory_fusion=self.fusion.bind(self.memory))
        torch.nn.functional.cross_entropy(logits, torch.tensor([5, 3, 4, 2])).backward()
        for parameter in (self.fusion.encoder.weight, self.fusion.query.weight,
                          self.fusion.value.weight, self.fusion.layer_gain):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertIsNone(self.backbone.layers[2]["down"].grad)
        self.assertTrue(torch.equal(before, self.backbone.layers[2]["down"]))

    def test_first_step_can_open_zero_initialized_gates(self):
        logits = self.backbone(self.ids, memory_fusion=self.fusion.bind(self.memory))
        logits.sum().backward()
        self.assertGreater(float(self.fusion.layer_gain.grad.abs().sum()), 0.)

    def test_no_future_response_tokens_influence_earlier_logits(self):
        with torch.no_grad(): self.fusion.layer_gain.fill_(.2)
        first = self.backbone(self.ids, logits_all=True, memory_fusion=self.fusion.bind(self.memory))
        longer = self.backbone(torch.cat((self.ids, torch.tensor([7, 8]))), logits_all=True,
                               memory_fusion=self.fusion.bind(self.memory))
        self.assertTrue(torch.allclose(first, longer[:len(self.ids)], atol=1e-5, rtol=1e-5))

    def test_memory_content_changes_logits_without_changing_prompt(self):
        with torch.no_grad(): self.fusion.layer_gain.fill_(.2)
        first = self.backbone(self.ids, memory_fusion=self.fusion.bind(self.memory))
        changed = self.backbone(self.ids, memory_fusion=self.fusion.bind(-self.memory))
        self.assertGreater(float((first - changed).detach().abs().max()), 1e-4)

    def test_memory_token_order_without_positions_does_not_change_readout(self):
        with torch.no_grad(): self.fusion.layer_gain.fill_(.2)
        first = self.backbone(self.ids, memory_fusion=self.fusion.bind(self.memory))
        changed = self.backbone(self.ids, memory_fusion=self.fusion.bind(self.memory.flip(0)))
        self.assertTrue(torch.allclose(first, changed, atol=1e-5, rtol=1e-5))

    def test_pairs_share_prompt_but_not_answers_and_never_inject_sources(self):
        train, valid = make_rows("train", 4, 7), make_rows("valid", 4, 7)
        self.assertFalse({r["group"] for r in train} & {r["group"] for r in valid})
        for i in range(0, len(train), 4):
            first, swapped = train[i:i+2]
            self.assertEqual(first["query"], swapped["query"])
            self.assertNotEqual(first["expected_value"], swapped["expected_value"])
            self.assertNotIn(first["memory"][0], prompt(first["query"]))
            self.assertNotIn(first["expected_value"], prompt(first["query"]))
            self.assertTrue(first["oracle_episode_boundaries"])
            self.assertFalse(first["automatic_memory"])

    def test_judge_rejects_value_lists_and_memory_meta_replies(self):
        row = make_rows("valid", 1, 7)[0]
        self.assertTrue(judge(row, row["answer"])["passed"])
        self.assertFalse(judge(row, " ".join(row["value_vocabulary"]))["passed"])
        self.assertFalse(judge(row, "I remember: " + row["answer"])["passed"])
        wrong_name = row["answer"].replace(row["answer"].split()[0], "WrongPerson", 1)
        self.assertTrue(judge(row, wrong_name)["value_grounded"])
        self.assertFalse(judge(row, wrong_name)["passed"])

    def test_entity_diversity_does_not_modify_development_cases(self):
        self.assertEqual(make_rows("valid", 8, 7), make_rows("valid", 8, 7, True))
        self.assertNotEqual(make_rows("train", 8, 7), make_rows("train", 8, 7, True))

    def test_evidence_gate_supervision_reaches_memory_before_gate_opens(self):
        model = GatedMemoryFusion(8, 3, 2, evidence_gate=True)
        with torch.no_grad(): model.use_gate.weight.normal_(0, .1)
        seen = []
        self.backbone(self.ids, memory_fusion=model.bind(self.memory, seen))
        torch.stack([x[-1, 0] for x in seen]).square().mean().backward()
        self.assertGreater(float(model.encoder.weight.grad.abs().sum()), 0.)
        self.assertEqual(len(seen), 3)

    def test_evidence_gate_decision_is_causal_at_query_boundary(self):
        model = GatedMemoryFusion(8, 3, 2, evidence_gate=True)
        with torch.no_grad():
            model.layer_gain.fill_(.2); model.use_gate.weight.normal_(0, .1)
        before, after = [], []
        self.backbone(self.ids, memory_fusion=model.bind(self.memory, before))
        self.backbone(torch.cat((self.ids, torch.tensor([9, 2]))), memory_fusion=model.bind(self.memory, after))
        for left, right in zip(before, after):
            self.assertTrue(torch.allclose(left[-1], right[len(self.ids) - 1], atol=1e-5))

    def test_saved_generation_audit_rejects_changed_evidence(self):
        rows = make_rows("valid", 1, 7)
        report = {"empty_memory_logits_exact": True, "cases": [
            {"id": r["id"], "group": r["group"], "condition": r["condition"],
             "query": r["query"], "memory": r["memory"], "expected": r["answer"],
             "actual": r["answer"]} for r in rows]}
        self.assertTrue(rescore(copy.deepcopy(report), rows)["gate_one_passed"])
        wrong = copy.deepcopy(report); wrong["cases"][0]["memory"] = ["different evidence"]
        with self.assertRaises(ValueError): rescore(wrong, rows)
        wrong = copy.deepcopy(report); wrong["cases"][0]["expected"] = "different target"
        with self.assertRaises(ValueError): rescore(wrong, rows)


if __name__ == "__main__": unittest.main()
