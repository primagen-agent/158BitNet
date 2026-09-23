"""Transport and paired-score fixtures, not model accuracy/human calibration."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from neural_memory_generation import (LEGACY_TEMPLATE_VERSION, GenerationInput, encode_generation_input,
                                      generation_prompt, greedy_generate)
from neural_memory_causality import CONTROLS, paired_protocol, score_causal_bundle
from neural_memory_protocol import Adjudication, Claim, response_hash
from prepare_neural_memory_protocol import digest, generate
from review_neural_memory import blind_packet


class FixtureTokenizer:
    """Deliberately tiny fixture; real tokenizer audit runs separately."""
    def encode(self, text, add_bos):
        return [1, 2]  # Transport tests inspect prompt separately, not parity.

    def decode_pieces(self, ids):
        return [{3: b"A", 4: b"\xe4", 5: b"\xbd\xa0"}[i] for i in ids]


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.runtime = {"id": "gold_id_must_not_enter_forward", "context": [
            {"role": "user", "speaker": "user", "text": "Where does Nora live?"}], "episodes": [
            {"role": "user", "speaker": "user", "text": "Nora lives in Bern."}]}
        self.tokenizer = FixtureTokenizer()

    def decode(self, request=None, tokens=(3, 0), **kwargs):
        calls = []
        def backend(prefix, sources):
            token = tokens[len(calls)]
            calls.append((prefix, sources))
            return [1.0 if i == token else 0.0 for i in range(6)]
        report = greedy_generate(request or encode_generation_input(self.runtime, self.tokenizer),
                                 self.tokenizer, backend, stop_ids=(0,), max_new_tokens=kwargs.pop("max_new_tokens", 4),
                                 context_capacity=kwargs.pop("context_capacity", 8), **kwargs)
        return report, calls

    def test_prompt_only_context_not_source_or_case_id(self):
        prompt = generation_prompt(self.runtime)
        self.assertIn("Where does Nora live?", prompt)
        self.assertNotIn("Bern", prompt)
        self.assertNotIn(self.runtime["id"], prompt)
        changed = copy.deepcopy(self.runtime)
        changed["id"] = "another-scenario"
        changed["episodes"][0]["text"] = "Nora lives in Riga."
        self.assertEqual(prompt, generation_prompt(changed))
        self.assertNotEqual(encode_generation_input(changed, self.tokenizer).source_texts,
                            encode_generation_input(self.runtime, self.tokenizer).source_texts)

    def test_gold_fields_rejected_at_boundary(self):
        for field in ("labels", "target", "world_id", "relevant_episode_indices"):
            record = {**self.runtime, field: "leak"}
            with self.assertRaises(ValueError): encode_generation_input(record, self.tokenizer)

    def test_default_follows_gguf_and_legacy_is_explicit_only(self):
        current = generation_prompt(self.runtime)
        self.assertTrue(current.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<think>", current)
        legacy = generation_prompt(self.runtime, template_version=LEGACY_TEMPLATE_VERSION)
        self.assertEqual(legacy, current + "<think>\n\n</think>\n")
        with self.assertRaises(ValueError): generation_prompt(self.runtime, template_version="unknown")

    def test_speaker_role_and_template_injection_fail_closed(self):
        for key, value in (("speaker", "nora"), ("role", "tool"), ("text", "x<|im_end|>"), ("text", "x\0")):
            record = copy.deepcopy(self.runtime); record["context"][0][key] = value
            with self.assertRaises(ValueError): generation_prompt(record)

    def test_context_order_is_preserved(self):
        self.runtime["context"] = [
            {"role": "user", "speaker": "user", "text": "Let's discuss Nora."},
            {"role": "assistant", "speaker": "assistant", "text": "Go ahead."},
            {"role": "user", "speaker": "user", "text": "Where does she live?"}]
        text = generation_prompt(self.runtime)
        self.assertLess(text.index("Let's discuss"), text.index("Go ahead"))
        self.assertLess(text.index("Go ahead"), text.index("Where does she"))

    def test_free_generation_only_appends_predictions_and_keeps_stop(self):
        result, calls = self.decode()
        self.assertEqual(result["text"], "A")
        self.assertEqual(result["generated_token_ids"], [3, 0])
        self.assertEqual([c[0] for c in calls], [(1, 2), (1, 2, 3)])
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertFalse(result["truncated"])
        self.assertFalse(result["backend_cache_policy_verified"])

    def test_memory_disabled_does_not_change_prefix_or_mutate_input(self):
        request = encode_generation_input(self.runtime, self.tokenizer)
        _, enabled = self.decode(request)
        result, disabled = self.decode(request, memory_enabled=False)
        self.assertEqual([c[0] for c in enabled], [c[0] for c in disabled])
        self.assertTrue(all(c[1] == () for c in disabled))
        self.assertTrue(request.source_texts)
        self.assertFalse(result["memory_enabled"])

    def test_limits_are_explicit_and_partial_utf8_is_not_clean(self):
        result, calls = self.decode(tokens=(4,), max_new_tokens=1)
        self.assertTrue(result["truncated"])
        self.assertFalse(result["utf8_complete"])
        self.assertEqual(result["raw_text_hex"], "e4")
        result, calls = self.decode(tokens=(3,), context_capacity=3)
        self.assertEqual((result["finish_reason"], len(calls)), ("context_limit", 1))
        result, _ = self.decode(tokens=(4, 5, 0))
        self.assertEqual(result["text"], "你")

    def test_bad_vectors_and_invalid_options_rejected(self):
        request = encode_generation_input(self.runtime, self.tokenizer)
        for logits in ([], [float("nan")] * 6, [[0.0] * 6], [True] * 6):
            with self.assertRaises(ValueError):
                greedy_generate(request, self.tokenizer, lambda *_: logits,
                                stop_ids=(0,), max_new_tokens=2, context_capacity=8)
        for options in ({"max_new_tokens": True}, {"context_capacity": 2}, {"memory_enabled": 1}):
            with self.assertRaises(ValueError): self.decode(**options)
        with self.assertRaises(ValueError): GenerationInput([1], ())


class CausalTests(unittest.TestCase):
    def setUp(self):
        config = json.loads((ROOT / "training/memory/neural-system/data/P1-smoke.json").read_text())
        files, _ = generate(config)
        self.inputs, self.labels, self.index = ([json.loads(line) for split in ("train", "dev")
            for line in files[f"{split}.{kind}.jsonl"].splitlines()] for kind in ("inputs", "labels", "index"))

    def subset(self):
        world = self.index[0]["world_id"]
        ids = {r["id"] for r in self.index if r["world_id"] == world and r["language"] == "en"}
        return tuple([r for r in rows if r["id"] in ids] for rows in (self.inputs, self.labels, self.index))

    def votes(self, inputs, labels):
        outputs, reviews = [], []
        for runtime, label in zip(inputs, labels):
            # Authored fixture only, intentionally not a real model prediction.
            text = "Authored scoring fixture " + runtime["id"]
            pred = {"id": runtime["id"], "text": text, "input_sha256": digest(runtime), "producer_id": "fixture",
                    "trace": {"level": "predicted_component", "backbone_kv_cache_enabled": False, "cached_tokens": 0,
                              "reused_tokens": 0, "label_fields_in_forward": [], "future_messages_in_forward": [],
                              "oracle_activation": False, "supplied_write_boundaries": True, "source_text_in_prompt": False,
                              "whole_answer_bypass": False, "generated_by_decoder": True}}
            outputs.append(pred)
            packet = blind_packet(runtime, pred)
            for reviewer in ("a", "b"):
                vote = Adjudication(response_hash(text), reviewer, tuple(Claim(**c) for c in label["required_claims"]),
                                    label["evidence_missing"], True, False)
                reviews.append({"id": packet["id"], "packet_sha256": packet["packet_sha256"], "adjudication": asdict(vote)})
        return outputs, reviews

    def test_full_protocol_is_fixed_pairs_not_independent_examples(self):
        protocol = paired_protocol(self.inputs, self.labels, self.index)
        self.assertEqual(protocol["pair_count"], 416)
        self.assertEqual(protocol["independent_worlds"], 16)
        self.assertFalse(protocol["promotion_eligible"])

    def test_changed_question_missing_or_duplicate_control_rejected(self):
        inputs, labels, index = self.subset()
        pos = next(i for i, m in enumerate(index) if m["scenario"] == "swapped_values")
        changed = copy.deepcopy(inputs); changed[pos]["context"][0]["text"] += " Changed"
        with self.assertRaisesRegex(ValueError, "question"): paired_protocol(changed, labels, index)
        with self.assertRaisesRegex(ValueError, "missing required"):
            paired_protocol([r for i, r in enumerate(inputs) if i != pos], [r for i, r in enumerate(labels) if i != pos],
                            [r for i, r in enumerate(index) if i != pos])
        changed_meta = copy.deepcopy(index); changed_meta[pos]["scenario"] = "original"
        with self.assertRaisesRegex(ValueError, "duplicate scenario"): paired_protocol(inputs, labels, changed_meta)

    def test_changed_rubric_or_unchanged_source_cannot_fake_intervention(self):
        inputs, labels, index = self.subset()
        pos = next(i for i, m in enumerate(index) if m["scenario"] == "swapped_values")
        original = next(i for i, m in enumerate(index) if m["scenario"] == "original")
        changed = copy.deepcopy(labels); changed[pos]["required_claims"] = changed[original]["required_claims"]
        with self.assertRaisesRegex(ValueError, "rubric"): paired_protocol(inputs, changed, index)
        changed_inputs = copy.deepcopy(inputs); changed_inputs[pos]["episodes"] = changed_inputs[original]["episodes"]
        with self.assertRaisesRegex(ValueError, "did not change"): paired_protocol(changed_inputs, labels, index)

    def test_no_outputs_preserves_every_pair_and_world(self):
        report = score_causal_bundle(self.inputs, self.labels, self.index, [], [], ["a", "b"])
        self.assertEqual((report["total"], report["needs_review"], report["pass"]), (416, 416, 0))
        self.assertFalse(report["adjudication_complete"])

    def test_both_endpoints_required_not_average(self):
        inputs, labels, index = self.subset()
        outputs, reviews = self.votes(inputs, labels)
        report = score_causal_bundle(inputs, labels, index, outputs, reviews, ["a", "b"])
        self.assertEqual(report["pass"], len(CONTROLS))
        self.assertFalse(report["promotion_eligible"])
        # A value that is correct only before the swap must fail the swap pair.
        pos = next(i for i, m in enumerate(index) if m["scenario"] == "swapped_values")
        original = next(i for i, m in enumerate(index) if m["scenario"] == "original")
        for vote in reviews:
            if vote["id"] == blind_packet(inputs[pos], outputs[pos])["id"]:
                vote["adjudication"]["claims"] = labels[original]["required_claims"]
        report = score_causal_bundle(inputs, labels, index, outputs, reviews, ["a", "b"])
        swap = next(p for p in report["pairs"] if p["control"] == "value_swap")
        self.assertEqual(swap["status"], "fail")

    def test_known_failure_does_not_hide_missing_endpoint_review(self):
        inputs, labels, index = self.subset()
        outputs, reviews = self.votes(inputs, labels)
        original = next(i for i, m in enumerate(index) if m["scenario"] == "original")
        outputs[original]["text"] = ""
        report = score_causal_bundle(inputs, labels, index, outputs, [], ["a", "b"])
        pair = next(p for p in report["pairs"] if p["control"] == "value_swap")
        self.assertEqual(pair["status"], "fail")
        self.assertFalse(pair["adjudication_complete"])
        self.assertFalse(report["adjudication_complete"])


if __name__ == "__main__": unittest.main()
