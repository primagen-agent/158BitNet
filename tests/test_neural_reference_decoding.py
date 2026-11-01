"""Reference byte decoding and reader binding, without backbone or model weights."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import INPUT, NeuralSlots, observation_context
from train_neural_dialogue import decode_references
import train_neural_dialogue_reader as reader


class BindingPlanner:
    """Predetermined learned outputs keep this test limited to binding control flow."""

    def __init__(self, need, labels):
        self.need = need
        self.labels = labels

    def plan_encoding(self, body, context, body_words):
        return body_words.new_zeros((len(body_words), 1))

    query_binding_context = staticmethod(NeuralSlots.query_binding_context)

    def query_reference(self, candidate, words, types, *, raw_context):
        if raw_context.shape != (4 * INPUT,):
            raise AssertionError("missing original query context")
        if len(words) != len(self.labels) or types.shape != (len(words), 4):
            raise AssertionError("unaligned literal source units")
        logits = words.new_full((len(words), 3), -4.)
        logits[torch.arange(len(words)), self.labels] = 4.
        return words.new_tensor(self.need), logits


class ReferenceDecodingTests(unittest.TestCase):
    @staticmethod
    def decode(context, body, context_units, body_units, labels, prefix=None):
        return decode_references(
            dict(context=context, body=body), labels,
            dict(context_units=context_units, body_units=body_units),
            len(body_units) if prefix is None else prefix,
        )

    def test_only_punctuation_spans_decode_to_no_references(self):
        references = self.decode("?", ".!", [(0, 1)], [(0, 1), (1, 2)], [1, 1, 1])
        self.assertEqual(references, [])
        self.assertEqual("; ".join(references), "")

    def test_empty_spans_do_not_pollute_utf8_references(self):
        references = self.decode(
            "!?陈瑜?!", ".", [(0, 1), (1, 2), (2, 5), (5, 8), (8, 9), (9, 10)],
            [(0, 1)], [1, 0, 1, 2, 0, 1, 1],
        )
        self.assertEqual(references, ["陈瑜"])
        self.assertEqual("; ".join(references), "陈瑜")

    def test_span_cannot_cross_context_and_body_documents(self):
        references = self.decode("Ada", "玲玲", [(0, 3)], [(0, 3), (3, 6)], [1, 2, 2])
        self.assertEqual(references, ["Ada", "玲玲"])
        self.assertEqual("; ".join(references), "Ada; 玲玲")

    def test_causal_body_prefix_excludes_later_sources(self):
        references = self.decode(
            "", "陈瑜; Dana", [], [(0, 3), (3, 6), (6, 7), (8, 12)],
            [1, 2, 1, 1], prefix=2,
        )
        self.assertEqual(references, ["陈瑜"])

    def binding_fixture(self, past, selected_sources, need=1.):
        body = "Where does she work?"
        context = observation_context("User", "", past)
        context_units, labels = [], []
        raw = context.encode()
        for source, tag in selected_sources:
            start = raw.index(source.encode())
            context_units.append((start, start + len(source.encode())))
            labels.append(tag)
        body_units = [(0, len(body.encode()))]
        labels.append(0)
        row = dict(body=body, context=context, speaker="User", timestamp="",
                   source_id="synthetic-reference-decoding", whole_need=need > 0,
                   whole_refs=["GoldReferenceMustNotBeUsed"] if need > 0 else [])
        group = dict(facts=[], queries=[dict(
            text=body, reference="GoldReferenceMustNotBeUsed", observation=row,
        )])
        item = dict(body=torch.zeros(1, INPUT), context=torch.zeros(len(context_units), INPUT),
                    body_words=torch.zeros(1, INPUT), context_words=torch.zeros(len(context_units), INPUT),
                    body_units=body_units, context_units=context_units)
        prepared = {(body, context): item}
        planner = BindingPlanner(need, labels)
        return group, planner, {body: None, context: None}, prepared

    def bind(self, fixture):
        group, planner, cache, prepared = fixture
        with patch.object(reader, "prepare", return_value=prepared):
            return reader.bind_curriculum_queries(group, planner, cache)

    def test_reader_binding_empty_spans_remains_unresolved(self):
        fixture = self.binding_fixture("?.!", [("?", 1), (".", 0), ("!", 1)])
        group, records = self.bind(fixture)
        query = group["queries"][0]
        self.assertTrue(query["unresolved"])
        self.assertEqual(query["text"], fixture[0]["queries"][0]["observation"]["body"])
        self.assertNotIn("reference", query)
        self.assertEqual(records[0]["predicted_reference"], "")
        self.assertFalse(records[0]["binding_exact"])

    def test_reader_binding_retains_only_utf8_source(self):
        fixture = self.binding_fixture("? 陈瑜 !", [("?", 1), ("陈瑜", 1), ("!", 1)])
        group, records = self.bind(fixture)
        query = group["queries"][0]
        self.assertFalse(query["unresolved"])
        self.assertEqual(query["reference"], "陈瑜")
        self.assertEqual(query["text"], "陈瑜. Where does she work?")
        self.assertEqual(records[0]["predicted_reference"], "陈瑜")
        self.assertNotIn("GoldReferenceMustNotBeUsed", query["text"])

    def test_reader_binding_need_false_skips_source_decoding(self):
        for need in (-1., 0.):
            with self.subTest(need=need):
                fixture = self.binding_fixture("陈瑜", [("陈瑜", 1)], need=need)
                with patch.object(reader, "decode_references", side_effect=AssertionError("unused source decoder")):
                    group, records = self.bind(fixture)
                query = group["queries"][0]
                self.assertFalse(query["unresolved"])
                self.assertNotIn("reference", query)
                self.assertEqual(query["text"], "Where does she work?")
                self.assertEqual(records[0]["predicted_reference"], "")
                self.assertTrue(records[0]["binding_exact"])


if __name__ == "__main__":
    unittest.main()
