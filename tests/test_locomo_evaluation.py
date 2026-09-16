import unittest

from eval_locomo import conversation_turns, official_f1, query_payload, score_response, summarize


class LoCoMoEvaluationTest(unittest.TestCase):
    def test_scoring_matches_category_rules(self):
        self.assertEqual(official_f1("running", "runs", 4), 1)
        self.assertEqual(official_f1("dogs", "dog; alternate answer", 3), 1)
        self.assertEqual(official_f1("red", "red, blue", 1), 0.5)
        self.assertEqual(official_f1("", "blue", 2), 0)
        with self.assertRaises(ValueError):
            official_f1("", "", 5)

    def test_no_qa_or_gold_fields_in_requests(self):
        sample = {"qa": [{"answer": "SECRET"}], "observation": "SECRET",
            "conversation": {"session_2_date_time": "later", "session_1_date_time": "earlier",
                "session_2": [{"speaker": "A", "text": "two", "dia_id": "D2:1"}],
                "session_1": [{"speaker": "B", "text": "one", "dia_id": "D1:1", "blip_caption": "cat"}]}}
        rows = list(conversation_turns(sample))
        self.assertEqual(rows[0]["text"], "[earlier] B: one\n[Image caption: cat]")
        self.assertNotIn("SECRET", str(rows))
        self.assertEqual(query_payload("x", "q")["memory_action"], "ignore")

    def test_generation_does_not_count_as_memory(self):
        response = {"choices": [{"message": {"content": "correct"}}],
                    "bitnet_session": {"cached_tokens": 0, "reused_tokens": 0}}
        row = score_response({"question": "q", "category": 4, "answer": "correct"}, response)
        self.assertEqual(row["f1"], 0)
        null = score_response({"question": "q", "category": 5}, response)
        self.assertTrue(null["null_rejected"])
        self.assertEqual(summarize([row, null])["categories_1_4"]["total"], 1)
        response["bitnet_session"]["cached_tokens"] = 1
        with self.assertRaises(AssertionError):
            score_response({"question": "q", "category": 5}, response)


if __name__ == "__main__":
    unittest.main()
