#!/usr/bin/env python3
"""Aggregate build-strict/perf-logs/<device>/summary.tsv into a markdown report skeleton."""
import csv
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG_ROOT = REPO / "build-strict" / "perf-logs"
OUT = REPO / "docs" / "superpowers" / "reports" / "2026-07-03-three-device-perf.md"

DEVICES = ["android", "mac", "x86-linux"]

def load(device):
    f = LOG_ROOT / device / "summary.tsv"
    if not f.exists():
        return []
    with f.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))

def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Three-Device Adaptation + Perf Sweep Report",
             "",
             "**Date**: 2026-07-03  **Branch**: feature/x86-optimization",
             "",
             "## Summary",
             ""]

    # Per-device summary
    summary_rows = []
    for d in DEVICES:
        rows = load(d)
        decode = [r for r in rows if r["test_bin"] == "test_profile_decode"]
        by_model = defaultdict(list)
        for r in decode:
            if r["tok_per_s"]:
                by_model[r["model"]].append(float(r["tok_per_s"]))
        best = {m: (max(v) if v else "—") for m, v in by_model.items()}
        ctest_fail = sum(1 for r in rows if r["test_bin"] == "ctest" and r["rc"] != "0")
        ctest_total = sum(1 for r in rows if r["test_bin"] == "ctest")
        summary_rows.append([
            d, f"{ctest_total - ctest_fail}/{ctest_total}",
            best.get("bitcpm4-0.5b-tq2_0.gguf", "—"),
            best.get("bitcpm4-1b-tq2_0.gguf", "—"),
            best.get("bitcpm4-3b-tq2_0.gguf", "—"),
            best.get("bitcpm4-8b-tq2_0.gguf", "—"),
        ])
    lines.append(md_table(
        ["Device", "ctest", "best 0.5B tok/s", "best 1B tok/s",
         "best 3B tok/s", "best 8B tok/s"], summary_rows))
    lines.append("")

    # Per-device detail
    for d in DEVICES:
        rows = load(d)
        lines += [f"## {d}", ""]
        ctest = [r for r in rows if r["test_bin"] == "ctest"]
        decode = [r for r in rows if r["test_bin"] == "test_profile_decode"]
        lines.append("### ctest")
        if ctest:
            lines.append(md_table(
                ["model", "threads", "tier", "rc", "elapsed_s"],
                [[r["model"], r["threads"], r["tier"], r["rc"], r["elapsed_s"]]
                 for r in ctest]))
        else:
            lines.append("_(no ctest runs recorded)_")
        lines += ["", "### decode (test_profile_decode)"]
        if decode:
            lines.append(md_table(
                ["model", "threads", "tier", "tok/s", "elapsed_s"],
                [[r["model"], r["threads"], r["tier"],
                  r["tok_per_s"] or "—", r["elapsed_s"]] for r in decode]))
        else:
            lines.append("_(no decode runs recorded)_")
        lines.append("")

    # Tier sweep
    tier_rows = [r for r in load("x86-linux")
                 if r["test_bin"] == "test_profile_decode" and r["tier"] != "default"]
    lines += ["## x86 Linux tier sweep", ""]
    if tier_rows:
        lines.append(md_table(
            ["model", "tier", "threads", "tok/s", "elapsed_s"],
            [[r["model"], r["tier"], r["threads"],
              r["tok_per_s"] or "—", r["elapsed_s"]] for r in tier_rows]))
    else:
        lines.append("_(no tier sweep runs recorded)_")
    lines += ["", "## Issues found", "", "_TBD — fill in after run._",
              "", "## Recommendations", "", "_TBD — fill in after run._"]

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
