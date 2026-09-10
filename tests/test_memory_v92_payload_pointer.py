#!/usr/bin/env python3
"""Regression tests for the V92 write/update payload curriculum."""
from __future__ import annotations

import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from prepare_memory_v92_payload_pointer import (  # noqa: E402
    CompoundPayloadFactory,
    OPERATIONS,
    OOD_COMPOUND_PATTERNS,
    OOD_COMPOUND_WORDS,
    TRAIN_COMPOUND_PATTERNS,
    TRAIN_COMPOUND_WORDS,
    TRAIN_IMPLICIT_POINTER_TEMPLATES,
    TRAIN_UPDATE_POINTER_TEMPLATES,
    make_sample,
    templates_for,
)


class FakeTokenizer:
    def encode(self, text, add_bos=False):
        del add_bos
        return list(range(len(text.split("|"))))


def test_operations_and_templates():
    assert OPERATIONS == (
        "explicit_write", "implicit_write", "update")
    for operation in OPERATIONS:
        assert templates_for("train", operation)
        assert templates_for("valid_id", operation)
        assert templates_for("valid_ood", operation)
        assert set(templates_for("train", operation)).isdisjoint(
            templates_for("valid_ood", operation))
    assert len(TRAIN_IMPLICIT_POINTER_TEMPLATES) >= 200
    assert len(TRAIN_UPDATE_POINTER_TEMPLATES) >= 800


def test_compound_curriculum_has_disjoint_ood_components():
    assert set(TRAIN_COMPOUND_WORDS).isdisjoint(OOD_COMPOUND_WORDS)
    assert TRAIN_COMPOUND_PATTERNS
    assert OOD_COMPOUND_PATTERNS
    assert ("hyphen", "{a}-{b}") in OOD_COMPOUND_PATTERNS
    assert "cobalt" in OOD_COMPOUND_WORDS
    assert "cedar" in OOD_COMPOUND_WORDS


def test_reserved_compound_requires_exact_token_length():
    factory = CompoundPayloadFactory(
        FakeTokenizer(), ("a", "b"), (("test", "{a}-{b}"),), 7)
    payload, token_ids, family = factory.register(
        "cobalt-cedar", 1)
    assert payload == "cobalt-cedar"
    assert token_ids == [0]
    assert family == "compound_reserved"
    try:
        factory.register("alpha|beta", 1)
    except ValueError as error:
        assert "has 2 tokens" in str(error)
    else:
        raise AssertionError("wrong token length was accepted")


def test_payload_is_exact_substring_for_every_operation():
    payload = "amber-orchid"
    for index, operation in enumerate(OPERATIONS):
        sample = make_sample(
            "valid_ood", index, 2, payload, [7, 9],
            random.Random(100 + index), "legacy-value")
        assert sample["metadata"]["operation"] == operation
        record = sample["messages"][0][0]["content"]
        answer = sample["messages"][1][-1]["content"]
        assert record.count(payload) == 1
        assert answer == payload
        assert not sample["metadata"]["locomo_used"]
        if operation == "update":
            assert "legacy-value" in record


def test_ambiguous_payloads_are_detectable_as_label_noise():
    sample = make_sample(
        "valid_ood", 1, 1, "is", [3],
        random.Random(7), "this-is-old")
    record = sample["messages"][0][0]["content"].encode()
    assert record.count(b"is") > 1


if __name__ == "__main__":
    test_operations_and_templates()
    test_compound_curriculum_has_disjoint_ood_components()
    test_reserved_compound_requires_exact_token_length()
    test_payload_is_exact_substring_for_every_operation()
    test_ambiguous_payloads_are_detectable_as_label_noise()
    print("memory V92 payload-pointer tests: PASS")
