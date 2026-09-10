#!/usr/bin/env python3
"""Regression tests for the strict-OOD V91 action curriculum."""
from __future__ import annotations

import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from prepare_memory_v91_action_router import (  # noqa: E402
    ACTION_IDS,
    OOD_DELETE_TEMPLATES,
    OOD_IGNORE_TEMPLATES,
    OOD_IGNORE_TOPICS,
    OOD_IMPLICIT_WRITE_TEMPLATES,
    OOD_UPDATE_TEMPLATES,
    TRAIN_DELETE_TEMPLATES,
    TRAIN_IGNORE_TEMPLATES,
    TRAIN_IGNORE_TOPICS,
    TRAIN_IMPLICIT_WRITE_TEMPLATES,
    TRAIN_UPDATE_TEMPLATES,
    action_templates,
    render_action,
)
from memory_text_encoding import action_text  # noqa: E402


def test_action_ids_match_c_runtime():
    assert ACTION_IDS == {
        "ignore": 0, "write": 1, "update": 2, "delete": 3,
    }


def test_ood_templates_are_held_out():
    pairs = (
        (TRAIN_IMPLICIT_WRITE_TEMPLATES, OOD_IMPLICIT_WRITE_TEMPLATES),
        (TRAIN_UPDATE_TEMPLATES, OOD_UPDATE_TEMPLATES),
        (TRAIN_DELETE_TEMPLATES, OOD_DELETE_TEMPLATES),
        (TRAIN_IGNORE_TEMPLATES, OOD_IGNORE_TEMPLATES),
    )
    for train, valid in pairs:
        assert set(train).isdisjoint(valid)
    for action in ACTION_IDS:
        assert action_templates("train", action)
        assert action_templates("valid", action)
    assert set(TRAIN_IGNORE_TOPICS).isdisjoint(OOD_IGNORE_TOPICS)


def test_rendered_rows_are_balanced_and_labelled():
    rows = [
        render_action(
            "valid", action, 3,
            random.Random(4), random.Random(5))
        for action in ACTION_IDS
    ]
    assert [row["label"] for row in rows] == [0, 1, 2, 3]
    assert all(row["metadata"]["strict_ood"] for row in rows)
    assert all(not row["metadata"]["locomo_used"] for row in rows)
    assert len({row["text"] for row in rows}) == 4


def test_templates_and_fields_use_independent_random_streams():
    template_rng = random.Random(10)
    field_rng = random.Random(20)
    rows = [
        render_action(
            "valid", "ignore", index, template_rng, field_rng)
        for index in range(120)
    ]
    overview = [
        row["text"] for row in rows
        if row["text"].startswith("Give me a concise overview")]
    assert len(overview) >= 10
    value_words = {
        text.rsplit(" ", 1)[-1].split("-", 1)[0]
        for text in overview}
    assert len(value_words) >= 5


def test_action_prompt_ends_at_shared_classification_position():
    rendered = action_text("Remember blue.")
    assert rendered.startswith("Classify the request")
    assert "Request: Remember blue." in rendered
    assert rendered.endswith("Memory operation:")


if __name__ == "__main__":
    test_action_ids_match_c_runtime()
    test_ood_templates_are_held_out()
    test_rendered_rows_are_balanced_and_labelled()
    test_templates_and_fields_use_independent_random_streams()
    test_action_prompt_ends_at_shared_classification_position()
    print("memory V91 action-router tests: PASS")
