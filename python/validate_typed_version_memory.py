#!/usr/bin/env python3
"""Validate typed immutable events and deterministic version semantics."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from prepare_typed_memory_features import (
    compile_episode_native_row,
)


def load_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("typed-version dataset is empty")
    return rows


def validate_event_graph(events):
    by_episode = {}
    chains = collections.defaultdict(list)
    for event in events:
        episode = int(event["episode"])
        if episode in by_episode:
            raise ValueError(f"duplicate episode: {episode}")
        by_episode[episode] = event
        chains[
            (str(event["entity"]),
             str(event["predicate"]))
        ].append(event)
    if sorted(by_episode) != list(range(len(events))):
        raise ValueError("episodes must be contiguous and ordered")
    for key, chain in chains.items():
        chain.sort(key=lambda event: int(event["version"]))
        active = []
        for version, event in enumerate(chain):
            expected_previous = (
                int(chain[version - 1]["episode"])
                if version else None)
            if (
                int(event["version"]) != version
                or event.get("previous_episode")
                != expected_previous
                or event["operation"]
                != ("create" if version == 0 else "update")
            ):
                raise ValueError(
                    f"broken version chain for {key}")
            if bool(event["active"]):
                active.append(event)
        if len(active) != 1 or active[0] is not chain[-1]:
            raise ValueError(
                f"chain does not have one latest active event: {key}")
    return by_episode, chains


def resolve_target(target, by_episode, chains):
    key = (
        str(target["entity"]),
        str(target["predicate"]))
    version = target["version"]
    if version == "active":
        matches = [
            event for event in chains.get(key, ())
            if event["active"]]
        if len(matches) != 1:
            return None
        return int(matches[0]["episode"])
    if version == "exact":
        matches = [
            event for event in chains.get(key, ())
            if (
                event["value"] == target.get("value")
                and event["time"] == target.get("time")
            )
        ]
        if len(matches) != 1:
            return None
        return int(matches[0]["episode"])
    if version == "previous":
        anchor = by_episode.get(
            int(target["anchor_episode"]))
        if anchor is None or (
            str(anchor["entity"]),
            str(anchor["predicate"])
        ) != key:
            return None
        previous = anchor.get("previous_episode")
        return None if previous is None else int(previous)
    raise ValueError(f"unknown version selector: {version}")


def resolve_query(row):
    metadata = row.get("metadata") or {}
    events = metadata.get("typed_events") or []
    plan = metadata.get("query_plan") or {}
    by_episode, chains = validate_event_graph(events)
    return [
        episode
        for target in plan.get("targets", [])
        for episode in [
            resolve_target(target, by_episode, chains)]
        if episode is not None
    ]


def validate_rows(rows):
    counters = collections.Counter()
    world_events = {}
    for row in rows:
        metadata = row.get("metadata") or {}
        world_id = str(metadata.get("world_id", ""))
        events = metadata.get("typed_events") or []
        encoded = json.dumps(
            events, sort_keys=True, separators=(",", ":"))
        previous = world_events.setdefault(world_id, encoded)
        if previous != encoded:
            raise ValueError(
                f"typed events changed within world {world_id}")
        compiled = compile_episode_native_row(row)
        if compiled is None:
            raise ValueError(
                f"row does not compile: {row.get('sample_id')}")
        expected = sorted(
            compiled["gold_episode_indices"])
        actual = sorted(resolve_query(row))
        family = str(metadata.get("family", "unknown"))
        counters["rows"] += 1
        counters[f"family_{family}"] += 1
        counters["exact"] += int(actual == expected)
        if actual != expected:
            raise ValueError(
                f"typed resolver mismatch for "
                f"{row.get('sample_id')}: "
                f"expected={expected}, actual={actual}")
    counters["worlds"] = len(world_events)
    counters["accuracy_ppm"] = (
        counters["exact"] * 1_000_000
        // max(counters["rows"], 1))
    return counters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="+")
    args = parser.parse_args()
    total = collections.Counter()
    for path in args.dataset:
        metrics = validate_rows(load_rows(path))
        total.update(metrics)
        print(json.dumps({
            "phase": "typed_version_validation",
            "path": path,
            **metrics,
            "accuracy": (
                metrics["exact"]
                / max(metrics["rows"], 1)),
            "locomo_used": False,
        }, separators=(",", ":")))
    print(json.dumps({
        "phase": "typed_version_validation_done",
        "files": len(args.dataset),
        "rows": total["rows"],
        "exact": total["exact"],
        "accuracy": (
            total["exact"] / max(total["rows"], 1)),
        "locomo_used": False,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
