"""Source-kind checkpoint guards protect each independent development class."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from neural_feedback_evaluation import kind_component_nonregression_safe


def bank(matrix):
    return dict(facts=sum(map(sum, matrix)),
                teacher_counts=[sum(row) for row in matrix],
                prediction_counts=[sum(row[column] for row in matrix) for column in range(3)],
                confusion=deepcopy(matrix))


def initial():
    return dict(episode=bank([[32, 0, 0], [4, 8, 0], [4, 0, 0]]),
                authored=bank([[18, 0, 0], [0, 0, 0], [0, 0, 0]]))


class NeuralKindGuardTests(unittest.TestCase):
    def test_equal_overall_accuracy_cannot_exchange_current_for_future_errors(self):
        parent = initial()
        changed = deepcopy(parent)
        changed["episode"] = bank([[28, 0, 4], [0, 8, 4], [0, 0, 4]])
        self.assertEqual(sum(changed["episode"]["confusion"][i][i] for i in range(3)), 40)
        self.assertEqual(sum(parent["episode"]["confusion"][i][i] for i in range(3)), 40)
        self.assertFalse(kind_component_nonregression_safe(changed, parent))

    def test_initial_and_per_class_improvements_are_eligible_including_zero_baseline(self):
        parent = initial()
        self.assertTrue(kind_component_nonregression_safe(parent, parent))
        better = deepcopy(parent)
        better["episode"] = bank([[32, 0, 0], [2, 10, 0], [0, 0, 4]])
        self.assertTrue(kind_component_nonregression_safe(better, parent))
        self.assertEqual(parent, initial())

    def test_higher_overall_accuracy_still_fails_a_single_class_regression(self):
        changed = initial()
        changed["episode"] = bank([[31, 0, 1], [2, 10, 0], [2, 0, 2]])
        self.assertGreater(sum(changed["episode"]["confusion"][i][i] for i in range(3)), 40)
        changed["episode"]["source_kind_accuracy"] = 1.0
        self.assertFalse(kind_component_nonregression_safe(changed, initial()))

    def test_zero_count_classes_and_empty_fact_banks_are_valid(self):
        single = dict(single=bank([[3, 0, 0], [0, 0, 0], [0, 0, 0]]))
        self.assertTrue(kind_component_nonregression_safe(single, single))
        empty = dict(empty=bank([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
        self.assertTrue(kind_component_nonregression_safe(empty, empty))

    def test_missing_bank_or_valid_label_count_drift_raises(self):
        parent = initial()
        for change in ({"episode": parent["episode"]},
                       dict(parent, another=parent["authored"]),
                       dict(parent, authored=bank([[19, 0, 0], [0, 0, 0], [0, 0, 0]])),
                       dict(parent, episode=bank([[31, 0, 0], [5, 8, 0], [4, 0, 0]]))):
            with self.assertRaises(ValueError):
                kind_component_nonregression_safe(change, parent)

    def test_malformed_matrices_counts_and_booleans_are_rejected_on_either_side(self):
        changes = [dict(facts=True), dict(facts=-1), dict(teacher_counts=[32, 12]),
                   dict(teacher_counts=[32, 12, True]), dict(prediction_counts=[40, 8, -1]),
                   dict(confusion=[[32, 0], [4, 8, 0], [4, 0, 0]]),
                   dict(confusion=[[32, 0, 0], [4, 8, 0], [4, 0, -1]]),
                   dict(confusion=[[32, 0, 0], [4, 8, 0], [4, 0, False]]),
                   dict(confusion=[[32., 0, 0], [4, 8, 0], [4, 0, 0]]),
                   dict(prediction_counts=[39, 9, 0]), dict(teacher_counts=[31, 13, 4])]
        for fields in changes:
            malformed = initial();malformed["episode"].update(fields)
            for first, second in ((malformed, initial()), (initial(), malformed)):
                with self.subTest(fields=fields):
                    with self.assertRaises(ValueError):
                        kind_component_nonregression_safe(first, second)

    def test_missing_required_fields_or_named_banks_are_rejected(self):
        for value in (None, [], {}, {"": bank([[0, 0, 0]] * 3)}, {"episode": None}):
            with self.assertRaises(ValueError):
                kind_component_nonregression_safe(value, initial())
        for key in ("facts", "teacher_counts", "prediction_counts", "confusion"):
            incomplete = initial();incomplete["episode"].pop(key)
            with self.assertRaises(ValueError):
                kind_component_nonregression_safe(incomplete, initial())


if __name__ == "__main__":
    unittest.main()
