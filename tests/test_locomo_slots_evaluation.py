"""Integrity guards for the new LoCoMo slot-memory evaluation protocol."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from eval_locomo_slots import (
    write_payload,
    question_payload,
    verify_response,
    score_question,
    summarize,
    structured_turns,
    pack_writes,
    questions_for_prefix,
    quality_stop_reason,
)
import tempfile, struct, hashlib, zlib
from memory_test_utils import validate_memory_network, read_slots


def response(answer="Paris", ids=None):
    return dict(
        choices=[dict(message=dict(content=answer))],
        bitnet_session=dict(cached_tokens=0, reused_tokens=0),
        memory=dict(
            mode="neural_slots_prompt",
            persistent_kv_capacity=0,
            read_only=1,
            written=0,
            updated=0,
            deleted=0,
            records=2,
            activated_ids=ids or [],
            complete=1,
        ),
    )


class LoCoMoSlotsEvaluation(unittest.TestCase):
    def test_current_1b_header_and_backbone_binding_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gguf = root / "backbone.gguf"
            gguf.write_bytes(b"independent header-binding test fixture")
            identity = hashlib.sha256(gguf.read_bytes()).digest()
            network = root / "network.bnsmodel"

            def write_header(input_width=4096, width=64, count=91, version=1,
                             raw_outputs=2, raw_context=3, scope_width=64,
                             membership_outputs=1, membership_context=64,
                             source_context_hidden_width=2176,
                             source_context_hidden_outputs=32,
                             source_context_hidden_bias=32,
                             source_context_output_width=32,
                             source_context_output_outputs=1,
                             source_context_output_bias=1,
                             kind_context_width=4096,
                             kind_context_outputs=3,
                             kind_context_bias=3,
                             read_context_width=16, query_context_input=16384):
                # This preflight fixture declares the raw head; other tensor
                # sizes and payload integrity are the native loader's job.
                declarations = bytearray(91 * 8)
                struct.pack_into("<I", declarations, 75 * 8, raw_outputs * raw_context * 4096 * 4)
                for index, elements in ((76, scope_width * 4098), (77, scope_width),
                                        (78, 4 * scope_width), (79, 4)):
                    struct.pack_into("<I", declarations, index * 8, elements * 4)
                struct.pack_into("<I", declarations, 80 * 8,
                                 membership_outputs * (membership_context + raw_context * 4096) * 4)
                struct.pack_into("<I", declarations, 81 * 8,
                                 source_context_hidden_outputs * source_context_hidden_width * 4)
                struct.pack_into("<I", declarations, 82 * 8, source_context_hidden_bias * 4)
                struct.pack_into("<I", declarations, 83 * 8,
                                 source_context_output_outputs * source_context_output_width * 4)
                struct.pack_into("<I", declarations, 84 * 8, source_context_output_bias * 4)
                struct.pack_into("<I", declarations, 85 * 8,
                                 kind_context_outputs * kind_context_width * 4)
                struct.pack_into("<I", declarations, 86 * 8, kind_context_bias * 4)
                for index, elements in ((87, read_context_width * 4096),
                                        (88, read_context_width * 256),
                                        (89, 64 * read_context_width),
                                        (90, 64 * query_context_input)):
                    struct.pack_into("<I", declarations, index * 8, elements * 4)
                network.write_bytes(
                    b"BNSMODEL"
                    + struct.pack("<4I", version, input_width, width, count)
                    + identity
                    + declarations
                )

            write_header()
            geometry = validate_memory_network(network, gguf)
            self.assertEqual(geometry["input"], 4096)
            self.assertEqual(geometry["identity"], 2112)
            self.assertEqual(geometry["query"], 320)
            self.assertEqual(geometry["semantic"], 256)
            self.assertEqual(geometry["query_heads"], 4)
            self.assertEqual(geometry["tensors"], 91)
            self.assertEqual(geometry["query_context_shape"], (64, 16384))
            self.assertEqual(geometry["raw_boundary_shape"], (2, 12288))
            self.assertEqual(geometry["raw_membership_shape"], (1, 12352))
            self.assertEqual(geometry["source_context_hidden_shape"], (32, 2176))
            self.assertEqual(geometry["source_context_output_shape"], (1, 32))
            self.assertEqual(geometry["scope_query_encoder_shape"], (64, 4098))
            self.assertEqual(geometry["backbone_sha256"], identity.hex())
            for values in (
                dict(input_width=2048),
                dict(width=128),
                dict(count=76),
                dict(count=80),
                dict(count=81),
                dict(count=82),
                dict(count=83),
                dict(count=84),
                dict(count=85),
                dict(count=86), dict(count=87), dict(count=89), dict(count=90),
                dict(read_context_width=15), dict(query_context_input=8192),
                dict(version=2),
                dict(raw_outputs=64),
                dict(raw_context=1),
                dict(scope_width=65),
                dict(membership_outputs=2),
                dict(membership_context=0),
                dict(source_context_hidden_width=64),
                dict(source_context_hidden_width=128),
                dict(source_context_hidden_outputs=31),
                dict(source_context_hidden_bias=0),
                dict(source_context_hidden_bias=31),
                dict(source_context_output_width=31),
                dict(source_context_output_outputs=2),
                dict(source_context_output_bias=0),
                dict(source_context_output_bias=2),
                dict(kind_context_width=4095),
                dict(kind_context_outputs=2),
                dict(kind_context_bias=2),
            ):
                write_header(**values)
                with self.assertRaises(ValueError):
                    validate_memory_network(network, gguf)
            write_header()
            gguf.write_bytes(b"different backbone")
            with self.assertRaises(ValueError):
                validate_memory_network(network, gguf)
            network.write_bytes(network.read_bytes()[:24])
            with self.assertRaises(ValueError):
                validate_memory_network(network)

    def test_snapshot_identity_width_preserves_following_record_alignment(self):
        def string(value):
            encoded = value.encode()
            return struct.pack("<I", len(encoded)) + encoded

        def slot(identifier, identity_width):
            content = f"I keep tool set {identifier} in a drawer."
            return (
                struct.pack("<QQQI", identifier, 0, identifier, 0)
                + b"".join(
                    string(value)
                    for value in (
                        content, content, "Aileen", "Aileen",
                        "2034-02-17 10:30 UTC", f"independent-{identifier}",
                    )
                )
                + struct.pack("<QQI", 0, len(content.encode()), 1)
                + bytes((identity_width + 64 + 16) * 4)
            )

        def snapshot(identity_width):
            body = struct.pack("<IQ", 2, 3) + slot(1, identity_width) + slot(2, identity_width)
            return (
                b"BNSSTATE" + struct.pack("<I", 2) + bytes(32)
                + struct.pack("<II", len(body), zlib.crc32(body)) + body
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.bnslots"
            path.write_bytes(snapshot(2112))
            slots = read_slots(path)
            self.assertEqual([s["id"] for s in slots], [1, 2])
            self.assertEqual(slots[1]["source_id"], "independent-2")
            path.write_bytes(snapshot(1088))
            with self.assertRaises(AssertionError):
                read_slots(path)

    def test_structured_body_and_literal_metadata_are_not_gold_inputs(self):
        sample = dict(
            conversation=dict(
                session_1_date_time="20 January 2023",
                session_1=[
                    dict(
                        speaker="Jon",
                        dia_id="D1:1",
                        text="I lost my job.",
                        blip_caption="A studio.",
                    )
                ],
            ),
            qa=[
                dict(question="When?", category=2, answer="SECRET", evidence=["D1:1"]),
                dict(question="Later?", category=2, answer="LATER", evidence=["D2:1"]),
            ],
        )
        turns = list(structured_turns(sample))
        row = turns[0]
        payload = write_payload("x", row, [])
        self.assertEqual(
            payload["messages"][0]["content"],
            "I lost my job.\n[Image caption: A studio.]",
        )
        self.assertEqual(
            payload["memory_observation"],
            dict(speaker="Jon", timestamp="20 January 2023", source_id="D1:1"),
        )
        self.assertNotIn("SECRET", str(payload))
        self.assertEqual(
            [q["original_qa_index"] for q in questions_for_prefix(sample, turns, True)],
            [0],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "in"
            pack_writes(path, turns, 4)
            with path.open("rb") as f:
                self.assertEqual(f.read(8), b"BNSOBS01")
                self.assertEqual(struct.unpack("<I", f.read(4))[0], 1)
                fields = []
                for _ in range(5):
                    n = struct.unpack("<I", f.read(4))[0]
                    fields.append(f.read(n).decode())
                self.assertEqual(
                    fields, [row["body"], "", "Jon", "20 January 2023", "D1:1"]
                )
                self.assertEqual(f.read(), b"")

    def test_quality_guard_waits_for_sample_then_stops_bad_results(self):
        qa = dict(category=2, question="Where?", answer="Paris", evidence=["D1:1"])
        good = score_question(qa, response(ids=["1"]), {"1": "D1:1"})
        bad = score_question(qa, response(answer="unknown"), {})
        self.assertIsNone(quality_stop_reason([bad] * 7))
        self.assertIn("answer F1", quality_stop_reason([bad] * 8))
        self.assertIsNone(quality_stop_reason([good] * 8))
        ungrounded = score_question(qa, response(), {})
        self.assertIn("evidence hit", quality_stop_reason([ungrounded] * 8))
        failed = score_question(qa, None, {}, "HTTP 500")
        self.assertIn("failure", quality_stop_reason([failed]))

    def test_raw_inputs_and_readonly_queries_never_contain_gold(self):
        turn = dict(
            text="[yesterday] A: hello",
            body="hello",
            speaker="A",
            timestamp="yesterday",
            dia_id="D1:3",
            answer="SECRET",
            evidence=["GOLD"],
        )
        write = write_payload("x", turn, [dict(text="[yesterday] B: hi")])
        self.assertEqual(
            [r["content"] for r in write["messages"]],
            ["[yesterday] B: hi", turn["body"]],
        )
        self.assertNotIn("SECRET", str(write))
        self.assertEqual(write["memory_observation"]["source_id"], "D1:3")
        self.assertTrue(question_payload("x", "Where?", 96)["memory_read_only"])
        self.assertNotIn("memory_action", question_payload("x", "Where?", 96))

    def test_cache_writes_missing_protocol_and_record_changes_are_rejected(self):
        verify_response(response(), readonly=True, expected_records=2)
        for bucket, key, value in [
            ("bitnet_session", "cached_tokens", 1),
            ("memory", "written", 1),
            ("memory", "persistent_kv_capacity", 8192),
            ("memory", "read_only", 0),
            ("memory", "records", 3),
            ("memory", "mode", "old_pointer"),
        ]:
            r = response()
            r[bucket][key] = value
            with self.assertRaises(AssertionError):
                verify_response(r, readonly=True, expected_records=2)

    def test_generation_evidence_and_activation_metrics_are_distinct(self):
        qa = dict(
            question="Where?", answer="Paris", category=2, evidence=["D1:3", "D2:1"]
        )
        ungrounded = score_question(qa, response(), {})
        self.assertEqual(ungrounded["f1"], 1)
        self.assertEqual(ungrounded["activation_conditioned_f1"], 0)
        row = score_question(qa, response(ids=["1"]), {"1": "D1:3"})
        self.assertEqual(row["annotated_evidence_recall"], 0.5)
        self.assertTrue(row["annotated_evidence_any"])
        self.assertFalse(row["annotated_evidence_all"])
        with self.assertRaises(AssertionError):
            score_question(qa, response(ids=["9"]), {"1": "D1:3"})

    def test_failed_requests_do_not_count_as_successful_null_rejection(self):
        failed = score_question(
            dict(category=5, question="Unknown?"), None, {}, "HTTP 500"
        )
        rejected = score_question(dict(category=5, question="Unknown?"), response(), {})
        self.assertFalse(failed["null_activation_rejected"])
        summary = summarize([failed, rejected])["by_category"]["5"]
        self.assertEqual(summary["request_errors"], 1)
        self.assertEqual(summary["null_activation_rejection"], 0.5)

    def test_annotated_turn_precision_and_exact_set_deduplicate_slots_and_penalize_extras(self):
        qa = dict(category=2, question="Where?", answer="Paris",
                  evidence=["D1:3", "D2:1", "D1:3"])
        provenance = {"1": "D1:3", "2": "D1:3", "3": "D2:1", "4": "D3:1"}
        extra = score_question(qa, response(ids=["1", "2", "3", "4"]), provenance)
        exact = score_question(qa, response(ids=["1", "2", "3"]), provenance)
        self.assertEqual(extra["activated_turns"], ["D1:3", "D2:1", "D3:1"])
        self.assertEqual(extra["annotated_evidence_recall"], 1.)
        self.assertTrue(extra["annotated_evidence_all"])
        self.assertAlmostEqual(extra["annotated_evidence_precision"], 2 / 3)
        self.assertAlmostEqual(extra["annotated_evidence_f1"], 4 / 5)
        self.assertFalse(extra["annotated_support_set_exact"])
        self.assertEqual(exact["annotated_evidence_precision"], 1.)
        self.assertEqual(exact["annotated_evidence_f1"], 1.)
        self.assertTrue(exact["annotated_support_set_exact"])
        summary = summarize([extra, exact])["categories_1_4"]
        self.assertEqual(summary["known_evidence_annotated_questions"], 2)
        self.assertAlmostEqual(summary["annotated_evidence_precision"], 5 / 6)
        self.assertAlmostEqual(summary["annotated_evidence_f1"], .9)
        self.assertEqual(summary["annotated_support_set_exact"], .5)

    def test_new_annotated_metrics_exclude_missing_annotations_and_category_five(self):
        fields = ("annotated_evidence_precision", "annotated_evidence_f1",
                  "annotated_support_set_exact")
        qa = dict(category=2, question="Where?", answer="Paris", evidence=["D1:3"])
        empty = score_question(qa, response(), {})
        for field in fields:
            self.assertEqual(empty[field], 0., field)
        excluded = []
        for question in (
            dict(category=2, question="Where?", answer="Paris"),
            dict(category=4, question="Where?", answer="Paris", evidence=[]),
            dict(category=5, question="Unknown?", evidence=["D1:3"]),
        ):
            row = score_question(question, response(ids=["1"]), {"1": "D1:3"})
            excluded.append(row)
            for field in fields:
                self.assertIsNone(row[field], field)
        summary = summarize(excluded)
        for group in (summary["categories_1_4"], *summary["by_category"].values()):
            self.assertEqual(group["known_evidence_annotated_questions"], 0)
            for field in fields:
                self.assertIsNone(group[field], field)
        valid = score_question(qa, response(ids=["1"]), {"1": "D1:3"})
        mixed = summarize([valid, empty, *excluded])["categories_1_4"]
        self.assertEqual(mixed["known_evidence_annotated_questions"], 2)
        for field in fields:
            self.assertEqual(mixed[field], .5, field)

    def test_new_annotated_metrics_never_change_existing_early_stop_policy(self):
        qa = dict(category=2, question="Where?", answer="Paris", evidence=["D1:3"])
        good = score_question(qa, response(ids=["1"]), {"1": "D1:3"})
        bad_answer = score_question(qa, response(answer="unknown"), {})
        no_evidence = score_question(qa, response(), {})
        fields = ("annotated_evidence_precision", "annotated_evidence_f1",
                  "annotated_support_set_exact")
        for row in (good, bad_answer, no_evidence):
            initial = quality_stop_reason([row] * 8)
            for value in (None, 0., 1.):
                changed = dict(row, **{field: value for field in fields})
                self.assertEqual(quality_stop_reason([changed] * 8), initial)
                self.assertIsNone(quality_stop_reason([changed] * 7))


if __name__ == "__main__":
    unittest.main()
