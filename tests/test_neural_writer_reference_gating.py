"""Unused write-reference outputs must not erase usable identity bindings."""

from copy import deepcopy
from pathlib import Path
import re
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_slot_model import INPUT, NeuralSlots, observation_context
from train_neural_dialogue import observation_loss, writer_train_keys


REFERENCE_PREFIXES = ("plan_ref_query.", "plan_ref_encode.", "plan_ref_pair.", "plan_origin.")


def fixture(model, *, need, resolved=False, action=1):
    speaker, timestamp = "Marith Reedvale", "2063-02-18 08:15 UTC"
    body = "I own a brass cup." if resolved else (
        "They own a brass cup." if need else "Velric Amberlane owns a brass cup.")
    past = "assistant: Velric Amberlane and Darven Elmreach are both under discussion.\n" if need and not resolved else ""
    context = observation_context(speaker, timestamp, past)

    def units(text):
        return [(len(text[:m.start()].encode()), len(text[:m.end()].encode()))
                for m in re.finditer(r"\S+", text)]

    body_units, context_units = units(body), units(context)
    device = next(model.parameters()).device
    body_words = torch.randn(len(body_units), INPUT, device=device) / 20
    context_words = torch.randn(len(context_units), INPUT, device=device) / 20
    item = dict(body=body_words, body_words=body_words, body_units=body_units,
                context=context_words, context_words=context_words, context_units=context_units)
    refs = [speaker] if resolved else []
    ranges = [dict(document="context", start=9, end=9 + len(speaker.encode()))] if resolved else []
    row = dict(body=body, speaker=speaker, timestamp=timestamp, past=past,
               context=context, source_id="reference-gating-unit-fixture",
               spans=[dict(start=0, end=len(body.encode()), action=action,
                           need=need, references=refs, ref_ranges=ranges)],
               whole_need=need, whole_refs=refs, whole_ref_ranges=ranges)
    return row, item


class WriterReferenceGating(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(78241)

    def assert_reference_gradients_absent(self, model):
        for name, parameter in model.named_parameters():
            if name.startswith(REFERENCE_PREFIXES):
                self.assertTrue(parameter.grad is None, name)

    def assert_ambiguous_reference_gradients(self, model):
        # NEED=True plus no gold source must still teach abstention: require
        # an identity, keep all candidate tokens outside BIO, reject origins.
        self.assertLess(float(model.plan_need.bias.grad), 0.)
        self.assertLess(float(model.plan_ref_pair[2].bias.grad[0]), 0.)
        self.assertGreater(float(model.plan_origin.bias.grad.sum()), 0.)
        for head in (model.plan_ref_query, model.plan_ref_encode):
            self.assertGreater(float(head.weight.grad.abs().sum()), 0.)

    def test_self_contained_write_trains_need_without_unused_reference_gradients(self):
        for mode in ("all", "references"):
            for action in (1, 2, 3):
                with self.subTest(mode=mode, action=action):
                    model = NeuralSlots()
                    model.writer_mode = mode
                    allowed = writer_train_keys(mode)
                    for name, parameter in model.named_parameters():
                        parameter.requires_grad_(name in allowed)
                    row, item = fixture(model, need=False, action=action)
                    observation_loss(model, row, item).backward()
                    self.assertGreater(float(model.plan_need.bias.grad), 0.)
                    self.assert_reference_gradients_absent(model)
                    # Source planning and the separate whole-query NEED gate
                    # remain active; the unused whole-query BIO stays idle.
                    if mode == "all":
                        self.assertGreater(float(model.plan_tags.bias.grad.abs().sum()), 0.)
                        self.assertGreater(float(model.plan_encode.weight.grad.abs().sum()), 0.)
                    self.assertGreater(float(model.query_need.bias.grad), 0.)
                    self.assertIsNone(model.query_ref_query.weight.grad)
                    self.assertIsNone(model.query_origin.bias.grad)

    def test_ambiguous_write_keeps_need_true_empty_reference_supervision(self):
        model = NeuralSlots()
        row, item = fixture(model, need=True)
        observation_loss(model, row, item).backward()
        self.assert_ambiguous_reference_gradients(model)

    def test_resolved_write_still_trains_reference_bio_and_origin(self):
        model = NeuralSlots()
        row, item = fixture(model, need=True, resolved=True)
        observation_loss(model, row, item).backward()
        self.assertLess(float(model.plan_need.bias.grad), 0.)
        self.assertLess(float(model.plan_origin.bias.grad[0]), 0.)
        for name, parameter in model.named_parameters():
            if name.startswith(REFERENCE_PREFIXES):
                self.assertIsNotNone(parameter.grad, name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0., name)

    def test_self_contained_span_does_not_dilute_another_spans_reference_gradients(self):
        model = NeuralSlots()
        row, item = fixture(model, need=True, resolved=True)
        first = "Velric Amberlane owns a brass cup. "
        row["body"] = first + row["body"]
        offset = len(first.encode())
        row["spans"][0]["start"] += offset
        row["spans"][0]["end"] += offset
        prefix_units = [(m.start(), m.end()) for m in re.finditer(r"\S+", first)]
        item["body_units"] = prefix_units + [(a + offset, b + offset) for a, b in item["body_units"]]
        prefix_words = torch.randn(len(prefix_units), INPUT) / 20
        item["body"] = item["body_words"] = torch.cat((prefix_words, item["body_words"]))
        with_both = deepcopy(row)
        with_both["spans"].insert(0, dict(start=0, end=offset - 1, action=1,
                                          need=False, references=[], ref_ranges=[]))
        gradients = []
        for observation in (row, with_both):
            model.zero_grad(set_to_none=True)
            observation_loss(model, observation, item).backward()
            gradients.append({name: parameter.grad.clone()
                              for name, parameter in model.named_parameters()
                              if name.startswith(REFERENCE_PREFIXES)})
        for name in gradients[0]:
            self.assertTrue(torch.equal(gradients[0][name], gradients[1][name]), name)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_optional_and_ambiguous_reference_gradient_gates_on_mps(self):
        for need in (False, True):
            with self.subTest(need=need):
                model = NeuralSlots().to("mps")
                row, item = fixture(model, need=need)
                observation_loss(model, row, item).backward()
                if need:
                    self.assert_ambiguous_reference_gradients(model)
                else:
                    self.assertGreater(float(model.plan_need.bias.grad), 0.)
                    self.assert_reference_gradients_absent(model)


if __name__ == "__main__":
    unittest.main()
