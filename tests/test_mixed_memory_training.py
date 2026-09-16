import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from prepare_mixed_memory_curriculum import combine
from train_typed_memory_writer import load_raw_worlds, retention_passes, writer_selection_key
from train_typed_span_tagger import field_selection_key, stratified_sample_indices
from train_context_memory_operation import evaluate, operation_selection_key


def writer_metrics(score):
    return dict.fromkeys(("create_accuracy", "update_accuracy", "entity_span_exact",
        "value_span_exact", "predicate_attention_hit_accuracy", "joint_span_exact", "operation_macro_accuracy"), score)


class MixedMemoryTrainingTest(unittest.TestCase):
    def test_pooled_improvement_cannot_hide_weak_domain(self):
        balanced = {**writer_metrics(.6), "domains": {"replay": writer_metrics(.6), "dialogue": writer_metrics(.6)}}
        uneven = {**writer_metrics(.9), "domains": {"replay": writer_metrics(.99), "dialogue": writer_metrics(.2)}}
        self.assertGreater(writer_selection_key(balanced), writer_selection_key(uneven))

    def test_retention_floor_is_relative_to_initial_old_domain(self):
        base = {"domains": {"replay": writer_metrics(.9), "dialogue": writer_metrics(.2)}}
        current = copy.deepcopy(base)
        current["domains"]["dialogue"] = writer_metrics(.8)
        current["domains"]["replay"]["value_span_exact"] = .8
        self.assertFalse(retention_passes(current, base, "replay", .02))
        self.assertTrue(retention_passes(base, base, "replay", .02))
        with self.assertRaises(ValueError):
            retention_passes(base, base, "missing", .02)

    def test_span_sampling_balances_domains_and_operations(self):
        examples = [{"domain": d, "operation": op} for d, n in (("replay", 100), ("dialogue", 2))
                    for op in (0, 1) for _ in range(n)]
        selected = stratified_sample_indices(examples, 32, .5, random.Random(4))
        for domain in ("replay", "dialogue"):
            for op in (0, 1):
                self.assertEqual(sum(examples[i] == {"domain": domain, "operation": op} for i in selected), 8)
        metrics = {"domains": {"replay": {"value_create_span_exact": 1, "value_update_span_exact": 1},
                              "dialogue": {"value_create_span_exact": .2, "value_update_span_exact": .4}}}
        self.assertEqual(field_selection_key(metrics, None, "value")[0], .2)

    def test_operation_selection_reports_each_domain(self):
        metrics = evaluate(torch.nn.Identity(), torch.tensor([[1., 0.], [0., 1.], [1., 0.], [1., 0.]]),
                           torch.tensor([0, 1, 0, 1]), ["replay", "replay", "dialogue", "dialogue"])
        self.assertEqual(metrics["domains"]["dialogue"]["update"], 0)
        self.assertEqual(operation_selection_key(metrics)[0], 0)

    def test_combiner_never_opens_sealed_test_and_preserves_domain(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for domain, prefix in (("replay", "natural"), ("dialogue", "dialogue")):
                path = root / domain
                path.mkdir()
                (path / "test.jsonl").write_text("MUST NOT BE READ")
                for split in ("train", "valid"):
                    identity = f"{prefix}-{split}-0"
                    row = {"sample_id": identity, "metadata": {"world_id": identity,
                           "typed_events": [{"episode": 0, "operation": "create"}]}}
                    (path / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
            combine(root / "replay", root / "dialogue", root / "mixed")
            worlds = load_raw_worlds(root / "mixed" / "train.jsonl")
            self.assertEqual({w["domain"] for w in worlds.values()}, {"replay", "dialogue"})
            bad = root / "dialogue" / "train.jsonl"
            row = json.loads(bad.read_text()); row["evaluation_only"] = True
            bad.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "evaluation-only"):
                combine(root / "replay", root / "dialogue", root / "bad")
            self.assertFalse((root / "bad").exists())


if __name__ == "__main__":
    unittest.main()
