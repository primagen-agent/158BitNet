"""Independent evidence-conditioned development views for checkpoint selection.

These fixed challenges measure response to feedback inputs. They are not a
three-round native rollout or a qualification result.
"""

from copy import deepcopy
import json
import math
from pathlib import Path


def qualification_observation_ids(path):
    """Read observation identifiers only; qualification answers stay unused."""
    saved = json.loads(Path(path).read_text())
    groups = [rows for rows in saved.values() if isinstance(rows, list)]
    if not groups or any(not isinstance(row, dict) for rows in groups for row in rows):
        raise ValueError("qualification curriculum needs observation-row lists")
    return {
        row["source_id"]
        for rows in groups
        for row in rows
        if row.get("tag") == "episode-query"
    }


def exclude_qualification(group, excluded):
    result = deepcopy(group)
    result["queries"] = [
        row for row in result["queries"]
        if (row.get("observation_id") or row.get("observation", {}).get("source_id"))
        not in excluded
    ]
    if not result["queries"]:
        raise ValueError("checkpoint selection lacks independent development questions")
    return result


def evidence_view_groups(group):
    result = {}
    for count in (0, 1, 4):
        rows = [q for q in group["queries"] if q["evidence_challenge_view"] == count]
        if not rows:
            raise ValueError("all fixed evidence views must contain questions")
        result[str(count)] = dict(facts=group["facts"], queries=rows)
    return result


def feedback_selection_score(component, feedback):
    """Prioritize worst-view strict accuracy, then mean accuracy and evidence F1.

    Empty/ambiguous questions cannot inflate the positive metric. An improved
    ordinary component score cannot hide regression under four distracting cues.
    """
    if not component or not feedback:
        raise ValueError("component and feedback development views are required")
    accuracy, f1 = [], []
    for metrics in feedback.values():
        if metrics.get("positive_queries", 0) < 1:
            raise ValueError("each feedback view needs positive questions")
        values = [metrics[k] for k in
                  ("positive_support_set_exact", "null_false_activation", "activation_f1")]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("invalid feedback metrics")
        accuracy.append(values[0] - values[1])
        f1.append(values[2])
    component_f1 = [m["activation_f1"] for m in component.values()]
    return (min(accuracy), sum(accuracy) / len(accuracy),
            sum(f1 + component_f1) / (len(f1) + len(component_f1)))


def component_nonregression_safe(component, initial):
    """Require every ordinary development view to retain its starting quality.

    Inputs are named metric mappings, as in ``feedback_selection_score``. Strict
    positive support sets, false slot matches and null rejection each have an
    independent floor; better feedback cannot purchase an ordinary regression.
    Missing views or invalid metrics fail closed. The initial point is eligible.
    """
    if not component or component.keys() != initial.keys():
        return False
    for name, baseline in initial.items():
        metrics = component[name]
        for values in (metrics, baseline):
            if not {"positive_queries", "positive_support_set_exact", "fp", "null_false_activation"} <= values.keys():
                return False
            count = values["positive_queries"]
            if type(count) is not int or count < 1:
                return False
            exact, fp, null = (values.get(key) for key in
                               ("positive_support_set_exact", "fp", "null_false_activation"))
            if (not isinstance(exact, (int, float)) or not math.isfinite(exact)
                    or not 0 <= exact <= 1 or not isinstance(fp, (int, float))
                    or not math.isfinite(fp) or fp < 0
                    or (null is not None and (not isinstance(null, (int, float))
                        or not math.isfinite(null) or not 0 <= null <= 1))):
                return False
        if metrics["positive_support_set_exact"] < baseline["positive_support_set_exact"]:
            return False
        if metrics["fp"] > baseline["fp"]:
            return False
        null = metrics["null_false_activation"]
        initial_null = baseline["null_false_activation"]
        if initial_null is None:
            if null is not None:
                return False
        elif null is None or null > initial_null:
            return False
    return True


def kind_component_nonregression_safe(metrics, initial):
    """Require every source-kind class to retain its development true positives.

    These counts describe independently authored development fact banks, never
    fit teachers. Aggregate accuracy cannot purchase a regression in another
    class. Malformed diagnostics or changed bank geometry raise ValueError;
    aligned valid diagnostics return False only for a class regression.
    """
    def validate_banks(banks):
        if (not isinstance(banks, dict) or not banks
                or any(not isinstance(name, str) or not name for name in banks)):
            raise ValueError("source-kind guard needs named development banks")
        required = {"facts", "teacher_counts", "prediction_counts", "confusion"}
        for bank in banks.values():
            if not isinstance(bank, dict) or not required <= bank.keys():
                raise ValueError("source-kind guard requires complete count diagnostics")
            facts = bank["facts"]
            if type(facts) is not int or facts < 0:
                raise ValueError("source-kind fact count must be a nonnegative integer")
            for key in ("teacher_counts", "prediction_counts"):
                counts = bank[key]
                if (not isinstance(counts, (list, tuple)) or len(counts) != 3
                        or any(type(value) is not int or value < 0 for value in counts)
                        or sum(counts) != facts):
                    raise ValueError("source-kind class counts must match the fact count")
            matrix = bank["confusion"]
            if (not isinstance(matrix, (list, tuple)) or len(matrix) != 3
                    or any(not isinstance(row, (list, tuple)) or len(row) != 3
                           or any(type(value) is not int or value < 0 for value in row)
                           for row in matrix)):
                raise ValueError("source-kind confusion must be a nonnegative integer 3x3 matrix")
            if [sum(row) for row in matrix] != list(bank["teacher_counts"]):
                raise ValueError("source-kind confusion rows must match teacher counts")
            if [sum(row[column] for row in matrix) for column in range(3)] != list(bank["prediction_counts"]):
                raise ValueError("source-kind confusion columns must match prediction counts")

    validate_banks(metrics)
    validate_banks(initial)
    if metrics.keys() != initial.keys():
        raise ValueError("source-kind guard requires the same development banks")
    for name, baseline in initial.items():
        bank = metrics[name]
        if (bank["facts"] != baseline["facts"]
                or list(bank["teacher_counts"]) != list(baseline["teacher_counts"])):
            raise ValueError("source-kind guard requires unchanged development fact labels")
    return all(bank["confusion"][kind][kind] >= initial[name]["confusion"][kind][kind]
               for name, bank in metrics.items() for kind in range(3))
