"""Reject incomplete feature-evaluation evidence before native qualification."""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_feature_trace import summarize_feature_trace


def event(**changes):
    value = dict(pid=317, requested_tokens=7, context_created=True, eval_attempted=True,
                 kv_before=0, kv_after=7, eval_status=0, hidden_rows=7, encode_status=0)
    value.update(changes)
    return "[ns-feature-eval] " + json.dumps(value) + "\n"


class NeuralFeatureTraceTests(unittest.TestCase):
    def summarize(self, stderr, **kwargs):
        return summarize_feature_trace(stderr, expected_pid=317, **kwargs)

    def test_counts_runtime_records_and_ignores_unrelated_diagnostics(self):
        summary = self.summarize("model loaded\n" + event() + "ordinary diagnostic\n" +
                                 event(requested_tokens=3, kv_after=3, hidden_rows=3),
                                 expected_calls=2)
        self.assertEqual(summary["native_pid"], 317)
        self.assertEqual(summary["actual_new_context_calls"], 2)
        self.assertEqual(summary["fresh_context_calls"], 2)
        self.assertEqual(summary["requested_tokens"], 10)
        self.assertEqual(summary["evaluated_tokens"], 10)
        self.assertEqual(summary["reused_kv_tokens"], 0)
        self.assertEqual(len(summary["records"]), 2)

    def test_concurrent_completion_order_does_not_change_totals(self):
        first = event()
        second = event(requested_tokens=4, kv_after=4, hidden_rows=4)
        forward = self.summarize(first + second)
        reverse = self.summarize(second + first)
        self.assertEqual({k: v for k, v in forward.items() if k != "records"},
                         {k: v for k, v in reverse.items() if k != "records"})

    def test_missing_events_and_known_incomplete_counts_are_rejected(self):
        for text, count in (("", None), ("ordinary diagnostic\n", None), (event(), 2)):
            with self.subTest(text=text, count=count), self.assertRaises(ValueError):
                self.summarize(text, expected_calls=count)

    def test_truncated_malformed_and_duplicate_key_events_are_rejected(self):
        malformed = ("[ns-feature-eval]\n", "[ns-feature-eval] {\n",
                     "[ns-feature-eval] []\n", "[ns-feature-eval] null\n",
                     event().rstrip("\n"), " " + event(),
                     event().replace('"pid": 317', '"pid": 1, "pid": 317'))
        for text in malformed:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.summarize(text)

    def test_wrong_process_is_rejected(self):
        with self.assertRaises(ValueError):
            self.summarize(event(pid=318))

    def test_malformed_trace_cannot_hide_after_an_unrelated_prefix(self):
        with self.assertRaises(ValueError):
            self.summarize(event() + "another diagnostic " + event())

    def test_prior_kv_is_rejected_even_when_all_requested_tokens_were_evaluated(self):
        with self.assertRaises(ValueError):
            self.summarize(event(kv_before=4, kv_after=11))

    def test_failed_evaluation_or_final_encoding_cannot_qualify(self):
        for change in (dict(eval_status=-2), dict(encode_status=-1),
                       dict(context_created=False), dict(eval_attempted=False)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.summarize(event(**change))

    def test_unmeasured_values_cannot_be_assumed_zero(self):
        fields = ("requested_tokens", "kv_before", "kv_after", "eval_status", "hidden_rows",
                  "encode_status")
        for key in fields:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.summarize(event(**{key: None}))

    def test_inconsistent_counts_and_wrong_types_are_rejected(self):
        changes = (dict(requested_tokens=1), dict(hidden_rows=6), dict(kv_after=6),
                   dict(kv_before=-1), dict(pid=True), dict(requested_tokens=7.0),
                   dict(hidden_rows="7"), dict(eval_status=False),
                   dict(context_created=1), dict(eval_attempted=1))
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.summarize(event(**change))

    def test_missing_fields_are_rejected(self):
        record = json.loads(event().split(" ", 1)[1])
        for key in record:
            partial = {k: v for k, v in record.items() if k != key}
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.summarize("[ns-feature-eval] " + json.dumps(partial) + "\n")

    def test_invalid_expected_process_or_call_count_is_rejected(self):
        for pid, count in ((0, None), (True, None), (317, 0), (317, True), (317, -1)):
            with self.subTest(pid=pid, count=count), self.assertRaises(ValueError):
                summarize_feature_trace(event(), expected_pid=pid, expected_calls=count)


if __name__ == "__main__":
    unittest.main()
