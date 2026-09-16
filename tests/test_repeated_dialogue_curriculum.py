#!/usr/bin/env python3
import sys
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from prepare_repeated_dialogue_curriculum import (  # noqa: E402
    EXPANDED_HELDOUT_PREDICATE_ALIASES,
    EXPANDED_PREDICATE_ALIASES,
    EXPANDED_PREDICATES,
    EXPANDED_TRAIN_PREDICATE_ALIASES,
    LARGE_HELDOUT_PREDICATE_ALIASES,
    LARGE_PREDICATE_ALIASES,
    LARGE_PREDICATES,
    LARGE_TRAIN_PREDICATE_ALIASES,
    HELDOUT_PREDICATE_ALIASES,
    PREDICATE_ALIASES,
    PREDICATES,
    ROLE_CHALLENGE_STATEMENTS,
    ROLE_CHALLENGE_UPDATES,
    ROLE_TRAIN_STATEMENTS,
    ROLE_TRAIN_UPDATES,
    ROLE_VALIDATION_STATEMENTS,
    ROLE_VALIDATION_UPDATES,
    SPEAKERS,
    TEST_PREDICATES,
    TEST_PREDICATE_ALIASES,
    TEST_SPEAKERS,
    TEST_STATEMENTS,
    TEST_UPDATES,
    TRAIN_STATEMENTS,
    TRAIN_PREDICATE_ALIASES,
    TRAIN_UPDATES,
    VALIDATION_PREDICATES,
    VALIDATION_PREDICATE_ALIASES,
    VALIDATION_STATEMENTS,
    VALIDATION_SPEAKERS,
    VALIDATION_UPDATES,
    build_world,
    write_split,
)
from prepare_typed_memory_features import (  # noqa: E402
    compile_episode_native_row,
)


class RepeatedDialogueCurriculumTest(unittest.TestCase):
    def test_role_layout_templates_are_strictly_disjoint(self):
        self.assertTrue(
            set(ROLE_TRAIN_STATEMENTS).isdisjoint(
                ROLE_VALIDATION_STATEMENTS
            )
        )
        self.assertTrue(
            set(ROLE_TRAIN_UPDATES).isdisjoint(
                ROLE_VALIDATION_UPDATES
            )
        )
        self.assertTrue(
            set(ROLE_TRAIN_STATEMENTS).isdisjoint(
                TEST_STATEMENTS
            )
        )
        self.assertTrue(
            set(ROLE_TRAIN_UPDATES).isdisjoint(
                TEST_UPDATES
            )
        )
        self.assertTrue(
            set(ROLE_TRAIN_STATEMENTS).isdisjoint(
                ROLE_CHALLENGE_STATEMENTS
            )
        )
        self.assertTrue(
            set(ROLE_TRAIN_UPDATES).isdisjoint(
                ROLE_CHALLENGE_UPDATES
            )
        )
        self.assertTrue(
            set(ROLE_VALIDATION_STATEMENTS).isdisjoint(
                ROLE_CHALLENGE_STATEMENTS
            )
        )
        self.assertTrue(
            set(ROLE_VALIDATION_UPDATES).isdisjoint(
                ROLE_CHALLENGE_UPDATES
            )
        )

    def test_relation_test_catalog_is_fully_disjoint(self):
        train_names = {
            predicate["name"]
            for predicate in LARGE_PREDICATES}
        validation_names = {
            predicate["name"]
            for predicate in VALIDATION_PREDICATES}
        test_names = {
            predicate["name"]
            for predicate in TEST_PREDICATES}
        self.assertEqual(len(test_names), 8)
        self.assertEqual(
            test_names, set(TEST_PREDICATE_ALIASES))
        self.assertTrue(
            train_names.isdisjoint(test_names))
        self.assertTrue(
            validation_names.isdisjoint(test_names))
        self.assertTrue(
            set(SPEAKERS).isdisjoint(TEST_SPEAKERS))
        self.assertTrue(
            set(VALIDATION_SPEAKERS).isdisjoint(
                TEST_SPEAKERS))
        self.assertTrue(all(
            len(aliases) == 3
            for aliases
            in TEST_PREDICATE_ALIASES.values()))

    def test_expanded_relation_catalog_is_complete(self):
        names = {
            predicate["name"]
            for predicate in EXPANDED_PREDICATES}
        self.assertEqual(len(names), 32)
        self.assertEqual(
            names, set(EXPANDED_PREDICATE_ALIASES))
        self.assertTrue(all(
            len(aliases) == 3
            for aliases
            in EXPANDED_PREDICATE_ALIASES.values()))
        self.assertEqual(
            names,
            set(EXPANDED_TRAIN_PREDICATE_ALIASES))
        self.assertEqual(
            names,
            set(EXPANDED_HELDOUT_PREDICATE_ALIASES))

    def test_large_relation_catalog_is_complete(self):
        names = {
            predicate["name"]
            for predicate in LARGE_PREDICATES}
        self.assertEqual(len(names), 64)
        self.assertEqual(
            names, set(LARGE_PREDICATE_ALIASES))
        self.assertTrue(all(
            len(aliases) == 3
            for aliases
            in LARGE_PREDICATE_ALIASES.values()))
        self.assertEqual(
            names,
            set(LARGE_TRAIN_PREDICATE_ALIASES))
        self.assertEqual(
            names,
            set(LARGE_HELDOUT_PREDICATE_ALIASES))
        all_aliases = []
        for name in names:
            train_aliases = set(
                LARGE_TRAIN_PREDICATE_ALIASES[name])
            heldout_aliases = set(
                LARGE_HELDOUT_PREDICATE_ALIASES[name])
            self.assertEqual(len(train_aliases), 2)
            self.assertEqual(len(heldout_aliases), 1)
            self.assertFalse(
                train_aliases & heldout_aliases)
            self.assertEqual(
                train_aliases | heldout_aliases,
                set(LARGE_PREDICATE_ALIASES[name]))
            all_aliases.extend(
                LARGE_PREDICATE_ALIASES[name])
        self.assertEqual(
            len(all_aliases), len(set(all_aliases)))

    def test_world_reuses_episode_bank_and_entities(self):
        import random

        rows = build_world(7, 64, 12, random.Random(19))
        self.assertEqual(len(rows), 12)
        evidence = rows[0]["evidence"]
        self.assertTrue(all(row["evidence"] == evidence for row in rows))
        self.assertTrue(all(
            row["metadata"]["world_id"]
            == rows[0]["metadata"]["world_id"]
            for row in rows))
        speakers = [
            line.split("] ", 1)[1].split(":", 1)[0]
            for line in evidence.splitlines()
        ]
        self.assertLessEqual(len(set(speakers)), 4)
        self.assertGreater(
            min(speakers.count(name) for name in set(speakers)),
            10)

    def test_rows_compile_to_full_bank_targets(self):
        import random

        rows = build_world(3, 64, 12, random.Random(23))
        compiled = [
            compile_episode_native_row(row) for row in rows]
        self.assertTrue(all(item is not None for item in compiled))
        self.assertTrue(all(
            len(item["episodes"]) == 64 for item in compiled))
        self.assertTrue(any(
            item["null_target"] for item in compiled))
        self.assertTrue(any(
            len(item["gold_episode_indices"]) > 1
            for item in compiled
            if not item["null_target"]))
        self.assertTrue(all(
            item["world_id"] == rows[0]["metadata"]["world_id"]
            for item in compiled))

    def test_validation_split_is_world_and_vocabulary_ood(self):
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.jsonl"
            valid_path = Path(directory) / "valid.jsonl"
            write_split(
                train_path, 2, 64, 6, 101,
                speakers_catalog=SPEAKERS,
                predicates=PREDICATES,
                statement_templates=TRAIN_STATEMENTS,
                update_templates=TRAIN_UPDATES)
            write_split(
                valid_path, 2, 64, 6, 307,
                world_offset=1_000_000,
                speakers_catalog=VALIDATION_SPEAKERS,
                predicates=VALIDATION_PREDICATES,
                statement_templates=VALIDATION_STATEMENTS,
                update_templates=VALIDATION_UPDATES)
            train = [
                json.loads(line)
                for line in train_path.read_text().splitlines()]
            valid = [
                json.loads(line)
                for line in valid_path.read_text().splitlines()]
        train_worlds = {
            row["metadata"]["world_id"] for row in train}
        valid_worlds = {
            row["metadata"]["world_id"] for row in valid}
        self.assertTrue(train_worlds.isdisjoint(valid_worlds))
        train_entities = {
            event["entity"] for row in train[:1]
            for event in row["metadata"]["typed_events"]}
        valid_entities = {
            event["entity"] for row in valid[:1]
            for event in row["metadata"]["typed_events"]}
        train_predicates = {
            event["predicate"] for row in train[:1]
            for event in row["metadata"]["typed_events"]}
        valid_predicates = {
            event["predicate"] for row in valid[:1]
            for event in row["metadata"]["typed_events"]}
        self.assertTrue(train_entities.isdisjoint(valid_entities))
        self.assertTrue(
            train_predicates.isdisjoint(valid_predicates))
        self.assertFalse(
            any(
                template.split("{", 1)[0]
                in valid[0]["evidence"]
                for template in (
                    "Lately my {predicate}",
                    "I wanted to mention",
                    "For now, record",
                    "I have settled on",
                )
            ))

    def test_relation_test_split_is_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid.jsonl"
            test_path = root / "test.jsonl"
            write_split(
                valid_path, 1, 64, 6, 307,
                world_offset=1_000_000,
                speakers_catalog=VALIDATION_SPEAKERS,
                predicates=VALIDATION_PREDICATES,
                predicate_aliases=(
                    VALIDATION_PREDICATE_ALIASES),
                statement_templates=VALIDATION_STATEMENTS,
                update_templates=VALIDATION_UPDATES)
            write_split(
                test_path, 1, 64, 6, 709,
                world_offset=3_000_000,
                speakers_catalog=TEST_SPEAKERS,
                predicates=TEST_PREDICATES,
                predicate_aliases=TEST_PREDICATE_ALIASES,
                statement_templates=TEST_STATEMENTS,
                update_templates=TEST_UPDATES)
            valid = json.loads(
                valid_path.read_text().splitlines()[0])
            test = json.loads(
                test_path.read_text().splitlines()[0])
        valid_events = valid["metadata"]["typed_events"]
        test_events = test["metadata"]["typed_events"]
        self.assertTrue({
            event["entity"] for event in valid_events
        }.isdisjoint({
            event["entity"] for event in test_events
        }))
        self.assertTrue({
            event["predicate"] for event in valid_events
        }.isdisjoint({
            event["predicate"] for event in test_events
        }))
        self.assertNotEqual(
            valid["metadata"]["world_id"],
            test["metadata"]["world_id"])
        lines = test["evidence"].splitlines()
        for event in test_events:
            self.assertEqual(
                lines[event["episode"]].count(
                    event["predicate_surface"]),
                1)

    def test_training_layout_views_share_semantics_not_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            write_split(
                path, 1, 64, 6, 101,
                statement_templates=TRAIN_STATEMENTS,
                update_templates=TRAIN_UPDATES,
                layout_views=2)
            rows = [
                json.loads(line)
                for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 12)
        views = {}
        for row in rows:
            metadata = row["metadata"]
            views.setdefault(
                metadata["layout_view"], row)
        self.assertEqual(set(views), {0, 1})
        self.assertEqual(
            views[0]["metadata"]["semantic_world_id"],
            views[1]["metadata"]["semantic_world_id"])
        self.assertNotEqual(
            views[0]["metadata"]["world_id"],
            views[1]["metadata"]["world_id"])
        self.assertEqual(
            views[0]["metadata"]["typed_events"],
            views[1]["metadata"]["typed_events"])
        self.assertNotEqual(
            views[0]["evidence"],
            views[1]["evidence"])

    def test_semantic_reference_variation_preserves_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            valid_path = root / "valid.jsonl"
            write_split(
                train_path, 1, 128, 6, 211,
                predicate_aliases=PREDICATE_ALIASES,
                statement_templates=TRAIN_STATEMENTS,
                update_templates=TRAIN_UPDATES,
                layout_views=2)
            write_split(
                valid_path, 1, 128, 6, 307,
                world_offset=1_000_000,
                speakers_catalog=VALIDATION_SPEAKERS,
                predicates=VALIDATION_PREDICATES,
                predicate_aliases=VALIDATION_PREDICATE_ALIASES,
                statement_templates=VALIDATION_STATEMENTS,
                update_templates=VALIDATION_UPDATES)
            train = [
                json.loads(line)
                for line in train_path.read_text().splitlines()]
            valid = [
                json.loads(line)
                for line in valid_path.read_text().splitlines()]
        train_events = train[0]["metadata"]["typed_events"]
        valid_events = valid[0]["metadata"]["typed_events"]
        train_views = {
            row["metadata"]["layout_view"]:
                row["metadata"]["typed_events"]
            for row in train
        }
        canonical_keys = (
            "episode", "entity", "predicate", "value",
            "time", "operation", "version",
            "previous_episode", "active",
        )
        self.assertEqual(
            [
                tuple(event[key] for key in canonical_keys)
                for event in train_views[0]
            ],
            [
                tuple(event[key] for key in canonical_keys)
                for event in train_views[1]
            ])
        self.assertTrue(any(
            left["predicate_surface"]
            != right["predicate_surface"]
            for left, right in zip(
                train_views[0], train_views[1])))
        surfaces_by_predicate = {}
        for event in train_events:
            surfaces_by_predicate.setdefault(
                event["predicate"], set()).add(
                    event["predicate_surface"])
            self.assertIn(
                event["predicate_surface"],
                train[0]["evidence"])
            self.assertEqual(
                event["entity"], event["entity_surface"])
        self.assertTrue(
            all(
                len(surfaces) > 1
                for surfaces in surfaces_by_predicate.values()))
        train_surfaces = {
            event["predicate_surface"]
            for event in train_events}
        valid_surfaces = {
            event["predicate_surface"]
            for event in valid_events}
        self.assertTrue(
            train_surfaces.isdisjoint(valid_surfaces))

    def test_factorized_alias_split_uses_heldout_surfaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            alias_path = root / "valid_alias.jsonl"
            write_split(
                train_path, 1, 128, 6, 401,
                predicates=PREDICATES,
                predicate_aliases=TRAIN_PREDICATE_ALIASES,
                statement_templates=TRAIN_STATEMENTS,
                update_templates=TRAIN_UPDATES)
            write_split(
                alias_path, 1, 128, 6, 503,
                world_offset=2_000_000,
                speakers_catalog=VALIDATION_SPEAKERS,
                predicates=PREDICATES,
                predicate_aliases=HELDOUT_PREDICATE_ALIASES,
                include_canonical_predicate_surface=False,
                statement_templates=VALIDATION_STATEMENTS,
                update_templates=VALIDATION_UPDATES)
            train = json.loads(
                train_path.read_text().splitlines()[0])
            alias = json.loads(
                alias_path.read_text().splitlines()[0])
        train_events = train["metadata"]["typed_events"]
        alias_events = alias["metadata"]["typed_events"]
        self.assertEqual(
            {event["predicate"] for event in train_events},
            {event["predicate"] for event in alias_events})
        train_surfaces = {
            event["predicate_surface"]
            for event in train_events}
        alias_surfaces = {
            event["predicate_surface"]
            for event in alias_events}
        self.assertTrue(
            train_surfaces.isdisjoint(alias_surfaces))
        self.assertTrue(all(
            event["predicate_surface"]
            in HELDOUT_PREDICATE_ALIASES[
                event["predicate"]]
            for event in alias_events))


if __name__ == "__main__":
    unittest.main()
