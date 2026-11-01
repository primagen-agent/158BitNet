"""Checkpoint selection must expose evidence regressions and reserve native QA."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from neural_feedback_evaluation import (
    qualification_observation_ids, exclude_qualification,
    evidence_view_groups, feedback_selection_score,
    component_nonregression_safe,
)


def metrics(exact, null=0, f1=0.8, fp=0):
    return dict(positive_queries=12, positive_support_set_exact=exact,
                null_false_activation=null, activation_f1=f1, fp=fp)


class FeedbackEvaluation(unittest.TestCase):
    def test_ordinary_improvement_cannot_hide_four_cue_regression(self):
        unstable = feedback_selection_score(
            {"ordinary": metrics(1, f1=1)},
            {"0": metrics(1), "1": metrics(1), "4": metrics(0.1)},
        )
        stable = feedback_selection_score(
            {"ordinary": metrics(0.6, f1=0.6)},
            {str(c): metrics(0.7) for c in (0, 1, 4)},
        )
        self.assertGreater(stable, unstable)

    def test_empty_questions_do_not_inflate_positive_quality(self):
        clean = feedback_selection_score({"ordinary": metrics(0.7)}, {"4": metrics(0.7)})
        noisy = feedback_selection_score({"ordinary": metrics(1)}, {"4": metrics(1, null=0.5)})
        self.assertGreater(clean, noisy)
        for invalid in (dict(metrics(1), positive_queries=0), metrics(float("nan"))):
            with self.assertRaises(ValueError):
                feedback_selection_score({"ordinary": metrics(1)}, {"4": invalid})

    def test_qualification_ids_are_reserved_without_reading_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.json"
            path.write_text(json.dumps(dict(schema="observation curriculum", seed=13119, development=[
                dict(tag="episode-query", source_id="reserved", arbitrary_answer={"opaque": 1}),
                dict(tag="episode-fact", source_id="fact"),
            ])))
            ids = qualification_observation_ids(path)
        self.assertEqual(ids, {"reserved"})
        group = dict(facts=[dict(text="ordinary evidence")], queries=[
            dict(text="held out", observation_id="reserved"),
            dict(text="selection", observation_id="selection"),
        ])
        filtered = exclude_qualification(group, ids)
        self.assertEqual([q["text"] for q in filtered["queries"]], ["selection"])
        self.assertEqual(len(group["queries"]), 2)

    def test_fixed_views_keep_support_labels_separate(self):
        group = dict(facts=[dict(text="fact")], queries=[
            dict(text=str(c), evidence_challenge_view=c, support_texts=["fact"])
            for c in (0, 1, 4)
        ])
        views = evidence_view_groups(group)
        self.assertEqual(set(views), {"0", "1", "4"})
        self.assertTrue(all(v["queries"][0]["support_texts"] == ["fact"] for v in views.values()))

    def test_component_guard_keeps_baseline_and_each_view_independent(self):
        baseline = {"ordinary": metrics(.8, null=.1, fp=4), "episode": metrics(.6, fp=8)}
        self.assertTrue(component_nonregression_safe(baseline, baseline))
        improvement = {"ordinary": metrics(.9, null=0, fp=3), "episode": metrics(.6, fp=8)}
        self.assertTrue(component_nonregression_safe(improvement, baseline))
        for field, value in (("positive_support_set_exact", .79), ("fp", 5), ("null_false_activation", .11)):
            candidate = {name: dict(row) for name, row in improvement.items()}
            candidate["ordinary"][field] = value
            self.assertFalse(component_nonregression_safe(candidate, baseline), field)

    def test_feedback_improvement_cannot_pass_an_ordinary_regression_guard(self):
        baseline = {"ordinary": metrics(.8, f1=.85, fp=4)}
        regressed = {"ordinary": metrics(.5, f1=.55, fp=3)}
        initial_score = feedback_selection_score(baseline, {str(c): metrics(.6) for c in (0, 1, 4)})
        candidate_score = feedback_selection_score(regressed, {str(c): metrics(.65) for c in (0, 1, 4)})
        self.assertGreater(candidate_score, initial_score)
        self.assertFalse(component_nonregression_safe(regressed, baseline))

    def test_component_guard_fails_closed_for_missing_or_invalid_metrics(self):
        baseline = {"ordinary": metrics(.8, fp=2)}
        self.assertFalse(component_nonregression_safe({}, baseline))
        self.assertFalse(component_nonregression_safe({"other": metrics(.8, fp=2)}, baseline))
        for field, value in (("positive_queries", 0), ("positive_queries", None),
                             ("positive_queries", float("nan")), ("positive_support_set_exact", float("nan")),
                             ("fp", -1), ("fp", float("inf")), ("null_false_activation", None),
                             ("null_false_activation", float("nan"))):
            self.assertFalse(component_nonregression_safe({"ordinary": dict(metrics(.8, fp=2), **{field: value})}, baseline))
        missing = metrics(.8, fp=2)
        del missing["null_false_activation"]
        self.assertFalse(component_nonregression_safe({"ordinary": missing}, baseline))
        no_null = {"ordinary": metrics(.8, null=None, fp=2)}
        self.assertTrue(component_nonregression_safe(no_null, no_null))


if __name__ == "__main__":
    unittest.main()
