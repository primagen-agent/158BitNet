import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from train_memory_span_gate import (PREFIX, SUFFIX, SpanGate, align_roles, annotate_cases,
                                    balance_quote_format, load_feature_cache, role_loss, source_segments)
from eval_memory_span_gate_format import add_material_quotes, remove_material_quotes


class ByteTokenizer:
    def bos(self): return 999
    def decode_pieces(self, ids): return [b"" if i == 999 else bytes([i]) for i in ids]


class SpanGateTest(unittest.TestCase):
    def test_synthetic_regions_exclude_chatter_and_questions(self):
        text = "[2024-03-01] Avery: Here is a little news. I work at Elm Studio. How has your week been?"
        spans = source_segments(text, "dialogue")
        pieces = [text[a:b] for a, b in spans]
        self.assertEqual(pieces, ["[2024-03-01] Avery", "I work at Elm Studio."])

    def test_same_content_opposite_roles_and_no_answers(self):
        text = "Avery: I work at Elm Studio."
        source = {"answer": "GOLD_NOT_USED", "metadata": {"training_domain": "dialogue", "raw_episodes": [text]}}
        rows = [{"text": text, "label": 1}, {"text": "Translate: " + text, "label": 0}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.jsonl"; path.write_text(json.dumps(source))
            result = annotate_cases(rows, path)
            self.assertEqual({s["role"] for s in result[0]["segments"]}, {1})
            self.assertEqual({s["role"] for s in result[1]["segments"]}, {2})
            self.assertNotIn("GOLD_NOT_USED", json.dumps(result))
            source["evaluation_only"] = True; path.write_text(json.dumps(source))
            with self.assertRaises(ValueError): annotate_cases(rows, path)

    def test_unicode_byte_alignment_and_prompt_mask(self):
        text = "I live in 北京. Any news?"
        end = len("I live in 北京.".encode())
        row = {"text": text, "label": 1, "segments": [{"start": 0, "end": end, "role": 1}]}
        ids = [999] + list((" " + PREFIX + text + SUFFIX).encode())
        mask, labels = align_roles(row, ids, ByteTokenizer())
        self.assertEqual(int(mask.sum()), len(text.encode()))
        self.assertEqual(int((labels == 1).sum()), end)
        self.assertTrue((labels[~mask] == -100).all())
        with self.assertRaises(ValueError): align_roles(row, ids[:-1], ByteTokenizer())

    def test_padding_invariant_safe_initialization_and_gradient(self):
        torch.manual_seed(10)
        model = SpanGate(8, 16).eval()
        x = torch.randn(2, 5, 8); valid = torch.ones(2, 5, dtype=torch.bool)
        mask = valid.clone(); mask[:, 0] = False
        margin, logits = model(x, valid, mask)
        self.assertTrue((margin < 0).all())
        labels = torch.tensor([[-100, 1, 1, 0, 0], [-100, 2, 2, 0, 0]])
        role_loss(logits, labels).backward()
        self.assertTrue(torch.isfinite(model.roles.weight.grad).all())
        with torch.no_grad(): model.roles.weight.normal_(std=.1)
        a, _ = model(x, valid, mask)
        b, _ = model(torch.nn.functional.pad(x, (0, 0, 0, 3)),
                     torch.nn.functional.pad(valid, (0, 3)), torch.nn.functional.pad(mask, (0, 3)))
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_no_gold_roles_are_model_inputs(self):
        import inspect
        self.assertEqual(list(inspect.signature(SpanGate.forward).parameters), ["self", "x", "valid", "mask"])

    def test_format_probe_preserves_payload_bytes_and_label(self):
        prefix = "Proofread this excerpt: "
        content = "I work in 北京."
        start = len(prefix.encode())
        row = {"text": prefix + content, "label": 0, "kind": "non_assertion",
               "segments": [{"start": start, "end": start + len(content.encode()), "role": 2}]}
        quoted = add_material_quotes(row)
        s = quoted["segments"][0]
        self.assertEqual(quoted["text"].encode()[s["start"]:s["end"]], content.encode())
        self.assertEqual(remove_material_quotes(quoted), row)
        with self.assertRaises(ValueError): add_material_quotes({**row, "label": 1})

    def test_balanced_format_preserves_intent_and_does_not_augment_dev_templates(self):
        content = "I work in 北京."
        rows = []
        for prefix, label in (("Translate this sentence: ", 0),
                              ("Keep the following information for future conversations: ", 1),
                              ("Proofread this excerpt: ", 0), ("", 1)):
            start = len(prefix.encode())
            rows.append({"text": prefix + content, "label": label, "segments": [
                {"start": start, "end": start + len(content.encode()), "role": 1 if label else 2}]})
        result = balance_quote_format(rows)
        self.assertEqual(len(result), 6)
        self.assertEqual(balance_quote_format(result), result)
        for r in result:
            s = r["segments"][0]
            self.assertEqual(r["text"].encode()[s["start"]:s["end"]], content.encode())

    def test_cache_rejects_backbone_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "features.pt"
            torch.save({"backbone_sha256": "a" * 64}, path)
            with self.assertRaises(ValueError): load_feature_cache(path, "b" * 64)


if __name__ == "__main__": unittest.main()
