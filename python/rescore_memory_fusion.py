"""Audit saved generation with subject-aware scoring; never regenerate answers."""
import argparse
import collections
import json
from pathlib import Path

from train_memory_fusion import judge, load_rows
from typed_memory_training import file_fingerprint


def rescore(report, rows):
    by_id = {row["id"]: row for row in rows}
    if len(report["cases"]) != len(rows) or {row["id"] for row in report["cases"]} != set(by_id):
        raise ValueError("generation/corpus IDs do not match")
    groups = collections.defaultdict(list); pairs = collections.defaultdict(list)
    for case in report["cases"]:
        row = by_id[case["id"]]
        if case["query"] != row["query"] or case["memory"] != row["memory"] or case["expected"] != row["answer"]:
            raise ValueError("generation inputs/targets differ from frozen corpus")
        case.update(judge(row, case["actual"]))
        groups[case["condition"]].append(case)
        if case["condition"] in ("original", "swapped"):
            pairs[case["group"]].append(case)
    report["groups"] = {kind: {"total": len(items), **{key: sum(x[key] for x in items) for key in
                         ("passed", "value_grounded", "natural_surface", "subject_exact", "exact_sentence")}}
                        for kind, items in groups.items()}
    report["paired_swaps"] = {"passed": sum(len(items) == 2 and all(x["passed"] for x in items) for items in pairs.values()),
                              "total": len(pairs)}
    report["gate_one_passed"] = (all(x["passed"] / x["total"] >= .9 for x in report["groups"].values())
                                  and report["paired_swaps"]["passed"] / len(pairs) >= .9
                                  and report["empty_memory_logits_exact"])
    report["grader_version"] = "value_and_subject_v2"
    report["gate_threshold"] = "at least 90% value-and-subject correctness per group and in paired swaps; empty-memory logits identical"
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report"); p.add_argument("valid_jsonl"); p.add_argument("output")
    a = p.parse_args(); report = json.loads(Path(a.report).read_text())
    if report["valid_sha256"] != file_fingerprint(a.valid_jsonl):
        raise ValueError("validation corpus fingerprint mismatch")
    result = rescore(report, load_rows(a.valid_jsonl, "valid"))
    result["original_report_sha256"] = file_fingerprint(a.report)
    result["grader_source_sha256"] = file_fingerprint(Path(__file__).with_name("train_memory_fusion.py"))
    with Path(a.output).open("x") as target:
        target.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("groups", "paired_swaps", "gate_one_passed")}))


if __name__ == "__main__": main()
