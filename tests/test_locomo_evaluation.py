"""External LoCoMo scoring rules, separate from training and activation."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from locomo_evaluation import official_f1, normalize


class ScoringRules(unittest.TestCase):
    def test_categories_use_stemming_and_multi_answer_mean(self):
        self.assertEqual(official_f1("running", "runs", 4), 1)
        self.assertEqual(official_f1("dogs", "dog; alternate answer", 3), 1)
        self.assertEqual(official_f1("red", "red, blue", 1), 0.5)

    def test_empty_answer_and_rejection_are_distinct(self):
        self.assertEqual(official_f1("", "blue", 2), 0)
        with self.assertRaises(ValueError):
            official_f1("", "", 5)

    def test_normalization_matches_official_articles_and_punctuation(self):
        self.assertEqual(normalize("The red, blue and green!"), "red blue green")


if __name__ == "__main__":
    unittest.main()
