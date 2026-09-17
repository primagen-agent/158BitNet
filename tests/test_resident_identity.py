import copy
import io
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from train_resident_identity import IdentityResidentMemorySet, entity_role_labels, identity_loss
from train_resident_memory_set import collate, compile_states, read_batch, select_sets, create_model
from prepare_memory_set_curriculum import make_world


class ByteTokenizer:
    def encode(self, text, add_bos=True): return ([256] if add_bos else []) + list(text.encode())
    def decode_pieces(self, ids): return [b"<bos>" if i == 256 else bytes([i]) for i in ids]


class ResidentIdentityTest(unittest.TestCase):
    def test_role_labels_use_full_runtime_bytes_and_word_boundaries(self):
        text = "Avery and Avery's notes do not name Averyson."
        labels, mentioned = entity_role_labels(text, ["Avery"], ByteTokenizer())
        self.assertEqual(mentioned, {"Avery"})
        self.assertEqual(int(labels.sum()), 10)
        self.assertEqual(int(labels[0]), 0)
        labels, _ = entity_role_labels("小李 and 李明", ["小李"], ByteTokenizer())
        self.assertEqual(int(labels.sum()), len("小李".encode()))
        with self.assertRaises(ValueError): entity_role_labels("Averyson", ["Avery"], ByteTokenizer())

    def test_identity_training_keeps_baseline_frozen_and_counts_identical(self):
        torch.manual_seed(12)
        w = make_world(0, "train", 1)
        f = {"events": [torch.randn(6, 16).half() for _ in w["events"]],
             "queries": [torch.randn(6, 16).half() for _ in w["queries"]]}
        roles = torch.tensor([0., 1., 1., 0., 0., 0.])
        labels = [{"events": [roles for _ in w["events"]], "queries": [roles for _ in w["queries"]],
                   "event_entities": ["a" if i % 2 else "b" for i in range(len(w["events"]))],
                   "query_entities": [{"a"} for _ in w["queries"]]}]
        indices = [(0, i) for i in range(len(w["queries"]))]
        data = collate([w], [f], indices, "cpu")
        model = IdentityResidentMemorySet(8, 16).train()
        baseline = {k: v.clone() for k, v in model.baseline.state_dict().items()}
        before = model.entity_role[0].weight.clone()
        loss = identity_loss(model, data, labels, indices)
        loss.backward()
        for parameter in model.baseline.parameters(): self.assertIsNone(parameter.grad)
        self.assertGreater(float(model.entity_role[0].weight.grad.abs().sum()), 0)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.entity_role[0].weight))
        for key, value in model.baseline.state_dict().items(): self.assertTrue(torch.equal(value, baseline[key]))
        scores, counts = model(*data[:-1])
        semantic_data = (data[0][..., :8], data[1], data[2][..., :8], *data[3:-1])
        _, old_counts = model.baseline(*semantic_data)
        self.assertTrue(torch.equal(counts, old_counts))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertFalse(model.baseline.training)

    def test_identity_state_roundtrip_needs_no_raw_text_or_gold_roles(self):
        torch.manual_seed(13)
        w = make_world(0, "train", 1)
        f = {"events": [torch.randn(5, 16).half() for _ in w["events"]],
             "queries": [torch.randn(6, 16).half() for _ in w["queries"]]}
        model = create_model(16, 16, .025, "identity").eval()
        indices = [(0, i) for i in range(3)]
        data = collate([w], [f], indices, "cpu")
        scores, counts = model(*data[:-1])
        changed = copy.deepcopy(w)
        for q in changed["queries"]: q["targets"] = []; q["answers"] = []
        again = model(*collate([changed], [f], indices, "cpu")[:-1])
        self.assertTrue(torch.equal(scores, again[0]))
        states = compile_states(model, [f], "cpu")
        buffer = io.BytesIO(); torch.save(states, buffer); buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)
        inputs = read_batch([{"queries": f["queries"]}], restored, indices, "cpu")
        actual = model.read(*inputs)
        self.assertTrue(torch.allclose(scores, actual[0], atol=1e-5))
        self.assertTrue(torch.equal(counts, actual[1]))
        permutation = torch.arange(inputs[2].shape[1] - 1, -1, -1)
        ps, pc = model.read(inputs[0], inputs[1], inputs[2][:, permutation], inputs[3][:, permutation], inputs[4][:, permutation])
        self.assertTrue(torch.allclose(scores[:, permutation], ps, atol=1e-5))
        self.assertTrue(torch.allclose(counts, pc, atol=1e-5))
        empty = read_batch([{"queries": f["queries"]}], [restored[0][:0]], indices, "cpu")
        es, ec = model.read(*empty)
        self.assertTrue(torch.isfinite(es).all() and torch.isfinite(ec).all())
        self.assertFalse(select_sets(es, ec, empty[-2]).any())


if __name__ == "__main__": unittest.main()
