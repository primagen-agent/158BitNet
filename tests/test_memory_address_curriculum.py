#!/usr/bin/env python3
"""Regression tests for addressed-memory attribute coverage."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from train_addressed_memory_controller import (  # noqa: E402
    ATTRS,
    NAMES,
    OOD_IMPLICIT_UPDATE_TEMPLATES,
    OOD_IMPLICIT_WRITE_TEMPLATES,
    TRAIN_IMPLICIT_UPDATE_TEMPLATES,
    TRAIN_IMPLICIT_WRITE_TEMPLATES,
    make_examples,
)


def test_runtime_attributes_are_covered():
    attributes = {name for name, _values in ATTRS}
    assert {
        "project codename", "home city", "response language",
        "editor theme", "default device", "meeting preference",
    } <= attributes
    assert len(NAMES) * len(ATTRS) >= 340


def test_address_templates_have_strict_ood_split():
    assert set(TRAIN_IMPLICIT_WRITE_TEMPLATES).isdisjoint(
        OOD_IMPLICIT_WRITE_TEMPLATES)
    assert set(TRAIN_IMPLICIT_UPDATE_TEMPLATES).isdisjoint(
        OOD_IMPLICIT_UPDATE_TEMPLATES)
    train = make_examples(7, 10)
    valid = make_examples(7, 10, offset=10, strict_ood=True)
    assert {row.address for row in train}.isdisjoint(
        row.address for row in valid)


if __name__ == "__main__":
    test_runtime_attributes_are_covered()
    test_address_templates_have_strict_ood_split()
    print("memory address curriculum tests: PASS")
