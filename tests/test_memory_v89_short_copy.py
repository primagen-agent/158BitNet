#!/usr/bin/env python3
"""Regression tests for the exact-token V89 short-copy curriculum."""
from __future__ import annotations

import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from prepare_memory_v89_short_copy import (  # noqa: E402
    OOD_QUERY_TEMPLATES,
    OOD_STORE_TEMPLATES,
    TRAIN_QUERY_TEMPLATES,
    TRAIN_STORE_TEMPLATES,
    PayloadFactory,
    make_sample,
)


def test_id_and_ood_templates_are_separate():
    assert len(TRAIN_STORE_TEMPLATES) == 524
    assert len(TRAIN_QUERY_TEMPLATES) == 108
    assert set(TRAIN_STORE_TEMPLATES).isdisjoint(OOD_STORE_TEMPLATES)
    assert set(TRAIN_QUERY_TEMPLATES).isdisjoint(OOD_QUERY_TEMPLATES)
    brace_templates = [
        template for template in TRAIN_STORE_TEMPLATES
        if "{{{payload}}}" in template]
    assert brace_templates
    assert brace_templates[0].format(payload="violet").count(
        "{violet}") == 1
    payload = "violet"
    token_ids = [123]
    id_sample = make_sample(
        "valid_id", 0, 1, payload, token_ids, random.Random(1))
    ood_sample = make_sample(
        "valid_ood", 0, 1, payload, token_ids, random.Random(1))

    assert id_sample["metadata"]["template_family"] == "id"
    assert ood_sample["metadata"]["template_family"] == "ood"
    id_store = id_sample["messages"][0][0]["content"]
    id_query = id_sample["messages"][1][0]["content"]
    ood_store = ood_sample["messages"][0][0]["content"]
    ood_query = ood_sample["messages"][1][0]["content"]
    assert id_store in {
        template.format(payload=payload)
        for template in TRAIN_STORE_TEMPLATES}
    assert id_query in set(TRAIN_QUERY_TEMPLATES)
    assert ood_store in {
        template.format(payload=payload)
        for template in OOD_STORE_TEMPLATES}
    assert ood_query in set(OOD_QUERY_TEMPLATES)


class FakeTokenizer:
    def encode(self, text, add_bos=False):
        del add_bos
        return text.split("+")


def test_payload_factory_honors_cross_split_reservations():
    heldout = set()
    valid = PayloadFactory(
        FakeTokenizer(), ["alpha", "beta"], ["+x"], 1,
        forbidden=heldout, shared_used=heldout)
    first, _ = valid.make(2)
    train = PayloadFactory(
        FakeTokenizer(), ["alpha", "beta"], ["+x", "+y"], 2,
        forbidden=heldout)
    second, _ = train.make(2)
    assert first in heldout
    assert second not in heldout


if __name__ == "__main__":
    test_id_and_ood_templates_are_separate()
    test_payload_factory_honors_cross_split_reservations()
    print("memory V89 short-copy tests: PASS")
