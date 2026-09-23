"""Test authored review fixtures, not actual independent human calibration."""
import copy
from dataclasses import asdict
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from neural_memory_protocol import Adjudication, Claim, response_hash
from prepare_neural_memory_protocol import digest
from review_neural_memory import blind_packet, score_bundle


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.input = {"id": "w01-hypothetical-en", "context": [{"role": "user", "speaker": "user", "text": "Where does Mara live?"}],
                      "episodes": [{"role": "user", "speaker": "user", "text": "Mara lives in Lima."}]}
        self.claim = Claim("Mara", "home_city", "Lima", "current", "actual")
        self.label = {"id": self.input["id"], "required_claims": [asdict(self.claim)], "allowed_claims": [asdict(self.claim)], "evidence_missing": False}
        self.meta = {"id": self.input["id"], "world_id": "w01", "scenario": "original", "language": "en"}
        self.output = {"id": self.input["id"], "text": "Mara lives in Lima.", "input_sha256": digest(self.input), "producer_id": "tested-model",
                       "trace": {"level": "predicted_component", "backbone_kv_cache_enabled": False, "cached_tokens": 0, "reused_tokens": 0,
                                 "label_fields_in_forward": [], "future_messages_in_forward": [], "oracle_activation": False,
                                 "supplied_write_boundaries": True, "source_text_in_prompt": False, "whole_answer_bypass": False, "generated_by_decoder": True}}

    def vote(self, name, claims=None):
        packet = blind_packet(self.input, self.output)
        review = Adjudication(response_hash(self.output["text"]), name, (self.claim,) if claims is None else claims, False, True, False)
        return {"id": packet["id"], "packet_sha256": packet["packet_sha256"], "adjudication": asdict(review)}

    def score(self, votes=(), outputs=None):
        return score_bundle([self.input], [self.label], [self.meta], [self.output] if outputs is None else outputs,
                            list(votes), ["reviewer-a", "reviewer-b"])

    def test_blind_packet_hides_model_gold_and_scenario_encoded_id(self):
        packet = blind_packet(self.input, self.output)
        self.assertEqual(set(packet), {"id", "context", "episodes", "response", "packet_sha256"})
        self.assertNotEqual(packet["id"], self.input["id"])
        self.assertNotIn("hypothetical", str(packet))
        self.assertNotIn("tested-model", str(packet))

    def test_two_agreeing_reviews_score_claims_and_keep_denominator(self):
        report = self.score([self.vote("reviewer-a"), self.vote("reviewer-b")])
        self.assertEqual((report["pass"], report["total"], report["independent_worlds"]), (1, 1, 1))
        self.assertTrue(report["adjudication_complete"])
        self.assertFalse(report["promotion_eligible"])

    def test_missing_output_and_missing_reviews_never_disappear(self):
        for output, votes in (([], []), ([self.output], []), ([self.output], [self.vote("reviewer-a")])):
            report = self.score(votes, output)
            self.assertEqual((report["total"], report["needs_review"], report["pass"]), (1, 1, 0))
            self.assertFalse(report["adjudication_complete"])

    def test_disagreement_is_not_majority_or_score_agreement(self):
        # Even if both sets would fail, their conflicting claim extraction needs resolution.
        report = self.score([self.vote("reviewer-a", ()), self.vote("reviewer-b", (Claim("Bob", "home_city", "Lima", "current", "actual"),))])
        self.assertEqual(report["cases"][0]["reasons"], ["reviewer_disagreement"])

    def test_wrong_subject_fails_after_agreed_extraction(self):
        wrong = (Claim("Bob", "home_city", "Lima", "current", "actual"),)
        report = self.score([self.vote("reviewer-a", wrong), self.vote("reviewer-b", wrong)])
        self.assertEqual(report["fail"], 1)

    def test_duplicate_unregistered_and_self_review_are_rejected(self):
        vote = self.vote("reviewer-a")
        for votes in ((vote, vote), (self.vote("outsider"),), (self.vote("tested-model"),)):
            with self.assertRaises(ValueError): self.score(votes)
        self.output["producer_id"] = "reviewer-a"
        with self.assertRaisesRegex(ValueError, "self-review"): self.score([self.vote("reviewer-a")])

    def test_review_is_bound_to_context_not_only_identical_response(self):
        votes = [self.vote("reviewer-a"), self.vote("reviewer-b")]
        self.input["context"][0]["text"] = "Where does Bob live?"
        self.output["input_sha256"] = digest(self.input)
        with self.assertRaises(ValueError): self.score(votes)

    def test_invalid_trace_hash_unknown_or_duplicate_output_fail_closed(self):
        variants = []
        wrong = copy.deepcopy(self.output); wrong["input_sha256"] = "0" * 64; variants.append([wrong])
        wrong = copy.deepcopy(self.output); wrong["trace"]["cached_tokens"] = 1; variants.append([wrong])
        wrong = copy.deepcopy(self.output); wrong["id"] = "unknown"; variants.append([wrong])
        variants.append([self.output, self.output])
        for outputs in variants:
            with self.assertRaises(ValueError): self.score(outputs=outputs)
        other_input, other_label, other_meta, other_output = (copy.deepcopy(x) for x in (self.input, self.label, self.meta, self.output))
        for row in (other_input, other_label, other_meta, other_output): row["id"] = "another-case"
        other_output["input_sha256"] = digest(other_input)
        other_output["trace"]["level"] = "oracle_component"
        with self.assertRaisesRegex(ValueError, "mix evaluation"):
            score_bundle([self.input, other_input], [self.label, other_label], [self.meta, other_meta],
                         [self.output, other_output], [], ["reviewer-a", "reviewer-b"])

    def test_empty_reply_is_failure_not_uncertainty(self):
        self.output["text"] = " "
        self.assertEqual(self.score()["fail"], 1)


if __name__ == "__main__": unittest.main()
