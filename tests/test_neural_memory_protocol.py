"""P1 contract tests only: passing these does not establish memory accuracy."""
from dataclasses import FrozenInstanceError, asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from neural_memory_contract import (
    Activation, Availability, ConversationContext, Episode, FactStatus, MemoryState,
    Message, ModelBinding, Operation, ReadContext, SourceSpan, WriteDelta, WriteOperation,
)
from neural_memory_launch import require_episode_training_ready
from neural_memory_protocol import (
    Adjudication, AnswerRubric, Claim, EvaluationLevel, InferenceTrace, RuntimeInput,
    response_hash, score_response,
)


def message(text="Alice lives in Lima.", **changes):
    return replace(Message("user-1", "chat-1", "m1", "user-1", "user", 1, 10, text), **changes)


def episode(event_id="e1", source=None, **changes):
    source = source or message()
    return replace(Episode(event_id, source.scope, (SourceSpan(source, 0, len(source.text.encode())),), 10), **changes)


def state(*episodes):
    return MemoryState("user-1", 1, ModelBinding(*("a" * 64 for _ in range(4))), tuple(episodes))


def context(*messages):
    return ConversationContext("user-1", "chat-1", "request-1", 20, tuple(messages) or (message(),))


def trace(**changes):
    return replace(InferenceTrace(EvaluationLevel.END_TO_END, False, 0, 0, (), (),
                                  False, False, False, False, True), **changes)


class ContractTests(unittest.TestCase):
    def test_runtime_input_excludes_supervision_and_future_messages(self):
        runtime = RuntimeInput(context(), state(episode()))
        before = asdict(runtime)
        labels = {"answer": "Alice lives in Lima.", "gold_targets": ["e1"]}
        labels.update(answer="Mallory lives in Rome.", gold_targets=[])
        self.assertEqual(before, asdict(runtime))
        with self.assertRaises(TypeError): RuntimeInput(runtime.context, runtime.state, answer="leak")
        with self.assertRaises(TypeError): RuntimeInput(**{**asdict(runtime), "gold_targets": ["e1"]})
        with self.assertRaises(ValueError): context(message(observed_at_ms=21))
        with self.assertRaises(ValueError): context(message(role="assistant"))
        with self.assertRaises(ValueError): RuntimeInput(context(), state(episode(learned_at_ms=21)))

    def test_snapshots_are_immutable_and_preserve_episode_boundaries(self):
        snapshot = state(episode(), episode("e2", message("Bob lives in Rome.", message_id="m2")))
        self.assertEqual(len(snapshot.episodes), 2)
        with self.assertRaises(FrozenInstanceError): snapshot.revision = 2
        with self.assertRaises(ValueError): replace(snapshot, episodes=list(snapshot.episodes))
        with self.assertRaises(ValueError): state(episode(), episode())

    def test_source_spans_are_exact_utf8_not_character_offsets(self):
        source = message("我住在杭州。")
        self.assertEqual(SourceSpan(source, 9, 15).text, "杭州")
        for start, end in ((1, 3), (0, 1), (0, 100), (3, 3), (-1, 3)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                SourceSpan(source, start, end)

    def test_scope_is_not_conversation_id(self):
        previous_chat = message(conversation_id="old-chat")
        RuntimeInput(context(), state(episode(source=previous_chat)))
        with self.assertRaises(ValueError): context(message(scope="user-2"))
        with self.assertRaises(ValueError): state(episode(source=message(scope="user-2")))
        with self.assertRaises(ValueError): context(message(), message(message_id="m2", sequence=1))
        with self.assertRaises(ValueError): context(message(), message(sequence=2))

    def test_unknown_fact_status_and_semantic_time_are_not_forced_actual(self):
        self.assertEqual(episode().fact_status, FactStatus.UNKNOWN)
        self.assertIsNone(episode().event_time)
        for status in FactStatus:
            item = episode(fact_status=status, event_time="2020-01")
            self.assertEqual(item.event_time, "2020-01")
        with self.assertRaises(ValueError): episode(fact_status="actual")

    def test_write_accepts_zero_or_multiple_units_and_read_is_independent(self):
        snapshot, current = state(), context()
        WriteDelta("user-1", 1, "request-1", ()).validate_against(snapshot, current)
        writes = tuple(WriteOperation(Operation.APPEND, episode(name, learned_at_ms=20)) for name in ("a", "b"))
        WriteDelta("user-1", 1, "request-1", writes).validate_against(snapshot, current)
        ReadContext("user-1", 1, snapshot.binding, Availability.MISSING, ()).validate_against(snapshot)
        self.assertEqual(snapshot.episodes, ())  # Validation is not a commit.

    def test_write_rejects_stale_duplicate_foreign_and_fabricated_sources(self):
        snapshot, current = state(episode()), context()
        append = WriteOperation(Operation.APPEND, episode("new", learned_at_ms=20))
        delta = WriteDelta("user-1", 1, "request-1", (append,))
        delta.validate_against(snapshot, current)
        bad = (replace(delta, scope="other"), replace(delta, base_revision=2),
               replace(delta, request_id="invented"), replace(delta, operations=(append, append)),
               replace(delta, operations=(replace(append, candidate=episode(learned_at_ms=20)),)),
               replace(delta, operations=(replace(append, candidate=episode("new", message("fake"), learned_at_ms=20)),)),
               replace(delta, operations=(replace(append, candidate=episode("new", learned_at_ms=19)),)))
        for item in bad:
            with self.subTest(item=item), self.assertRaises(ValueError): item.validate_against(snapshot, current)

    def test_assistant_generation_cannot_be_automatically_self_written(self):
        assistant = message("I think you live in Rome.", role="assistant")
        current = context(assistant, message(message_id="m2", sequence=2))
        write = WriteOperation(Operation.APPEND, episode("new", assistant, learned_at_ms=20))
        with self.assertRaises(ValueError): WriteDelta("user-1", 1, "request-1", (write,)).validate_against(state(), current)

    def test_directed_updates_and_retractions_have_existing_targets(self):
        snapshot, current = state(episode()), context()
        for operation in (Operation.REPLACE, Operation.COEXIST):
            write = WriteOperation(operation, episode("new", learned_at_ms=20), ("e1",))
            WriteDelta("user-1", 1, "request-1", (write,)).validate_against(snapshot, current)
        retract = WriteOperation(Operation.RETRACT, None, ("e1",))
        WriteDelta("user-1", 1, "request-1", (retract,)).validate_against(snapshot, current)
        for writes in ((retract, retract), (replace(retract, target_ids=("missing",)),)):
            with self.assertRaises(ValueError): WriteDelta("user-1", 1, "request-1", writes).validate_against(snapshot, current)
        with self.assertRaises(ValueError): WriteOperation(Operation.RETRACT, episode(), ("e1",))
        with self.assertRaises(ValueError): WriteOperation(Operation.APPEND, episode(), ("e1",))

    def test_null_disabled_and_conflict_are_distinct(self):
        snapshot = state(episode())
        for availability in (Availability.DISABLED, Availability.NOT_NEEDED, Availability.MISSING):
            ReadContext("user-1", 1, snapshot.binding, availability, ()).validate_against(snapshot)
            with self.assertRaises(ValueError):
                ReadContext("user-1", 1, snapshot.binding, availability, (Activation("e1", .8),))
        for availability in (Availability.AVAILABLE, Availability.CONFLICT):
            with self.assertRaises(ValueError): ReadContext("user-1", 1, snapshot.binding, availability, ())

    def test_payload_copy_only_from_activated_live_snapshot(self):
        first, other = episode(), episode("e2", message("Bob lives in Rome.", message_id="m2"))
        snapshot = state(first, other)
        read = ReadContext("user-1", 1, snapshot.binding, Availability.AVAILABLE,
                           (Activation("e1", .8, (SourceSpan(first.sources[0].message, 15, 19),)),))
        read.validate_against(snapshot)
        for bad in (replace(read, revision=0), replace(read, binding=replace(snapshot.binding, encoder_sha256="b" * 64)),
                    replace(read, activations=(Activation("e1", .8, other.sources),)),
                    replace(read, activations=(Activation("missing", .8),))):
            with self.subTest(bad=bad), self.assertRaises(ValueError): bad.validate_against(snapshot)
        with self.assertRaises(ValueError): read.validate_against(state(replace(first, retracted=True), other))
        for weight in (0, -1, float("nan"), float("inf"), 1.1, True):
            with self.assertRaises(ValueError): Activation("e1", weight)


class MeasurementTests(unittest.TestCase):
    def test_trace_rejects_all_known_shortcuts_and_missing_observations(self):
        trace().validate()
        for key, value in (("backbone_kv_cache_enabled", True), ("cached_tokens", 1), ("reused_tokens", 1),
                           ("cached_tokens", None), ("label_fields_in_forward", ("answer",)),
                           ("future_messages_in_forward", ("next-reply",)), ("oracle_activation", True),
                           ("supplied_write_boundaries", True), ("source_text_in_prompt", True),
                           ("whole_answer_bypass", True), ("generated_by_decoder", False)):
            with self.subTest(key=key), self.assertRaises(ValueError): trace(**{key: value}).validate()

    def test_oracle_component_is_explicit_and_never_called_end_to_end(self):
        trace(level=EvaluationLevel.ORACLE_COMPONENT, oracle_activation=True, supplied_write_boundaries=True).validate()
        trace(level=EvaluationLevel.PREDICTED_COMPONENT, supplied_write_boundaries=True,
              generated_by_decoder=False).validate(generation_required=False)
        with self.assertRaises(ValueError):
            trace(level=EvaluationLevel.PREDICTED_COMPONENT, oracle_activation=True).validate()

    def test_subject_relation_value_time_and_status_are_jointly_scored(self):
        correct = Claim("Alice", "home_city", "Lima", "current", "actual")
        rubric = AnswerRubric((correct,), (correct,), False)
        # Each text/claim pair is independently authored test supervision, not a model verdict.
        fixtures = [
            ("Alice lives in Lima.", correct, "pass"),
            ("She lives in Lima.", correct, "pass"),  # Pronoun resolved in the supplied conversation.
            ("Bob lives in Lima.", replace(correct, subject="Bob"), "fail"),
            ("Alice was born in Lima.", replace(correct, relation="birth_city"), "fail"),
            ("Alice lives in Rome.", replace(correct, value="Rome"), "fail"),
            ("Alice used to live in Lima.", replace(correct, time="past"), "fail"),
            ("Alice does not live in Lima.", replace(correct, status="negated"), "fail"),
            ("If Alice lived in Lima...", replace(correct, status="hypothetical"), "fail"),
        ]
        for text, claim, expected in fixtures:
            review = Adjudication(response_hash(text), "fixture-author", (claim,), False, True, False)
            with self.subTest(text=text): self.assertEqual(score_response(text, rubric, review)["status"], expected)
        text = "Alice lives in Lima and Bob lives in Rome."
        review = Adjudication(response_hash(text), "fixture-author", (correct, replace(correct, subject="Bob", value="Rome")), False, True, False)
        self.assertEqual(score_response(text, rubric, review)["status"], "fail")

    def test_role_reversal_and_multiple_required_facts(self):
        original = Claim("Alice", "lent_book_to", "Bob", "yesterday", "actual")
        reversed_role = Claim("Bob", "lent_book_to", "Alice", "yesterday", "actual")
        rubric = AnswerRubric((original,), (original,), False)
        text = "Bob lent a book to Alice yesterday."
        review = Adjudication(response_hash(text), "fixture-author", (reversed_role,), False, True, False)
        self.assertEqual(score_response(text, rubric, review)["status"], "fail")
        rubric = AnswerRubric((original, reversed_role), (original, reversed_role), False)
        self.assertEqual(score_response(text, rubric, review)["status"], "fail")

    def test_missing_evidence_empty_reply_and_unsolicited_narration(self):
        rubric = AnswerRubric((), (), True)
        text = "我还不知道她住在哪里。"
        review = Adjudication(response_hash(text), "fixture-author", (), True, True, False)
        self.assertEqual(score_response(text, rubric, review)["status"], "pass")
        for bad in (replace(review, acknowledges_missing_evidence=False), replace(review, natural_reply=False),
                    replace(review, unsolicited_memory_narration=True)):
            self.assertEqual(score_response(text, rubric, bad)["status"], "fail")
        self.assertEqual(score_response("  ", rubric)["status"], "fail")
        self.assertEqual(score_response("Lima", rubric)["status"], "needs_review")
        with self.assertRaises(ValueError): score_response("我住在杭州。", rubric, review)


class LaunchInterlockTests(unittest.TestCase):
    def test_current_registration_blocks_training(self):
        with self.assertRaisesRegex(ValueError, "P1.1-P1.7"):
            require_episode_training_ready()

    def test_changing_json_flag_does_not_approve_unreviewed_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            path.write_text(json.dumps({"prerequisites_complete": True}))
            with self.assertRaisesRegex(ValueError, "unreviewed draft"):
                require_episode_training_ready(path)
            path.write_text("broken")
            with self.assertRaisesRegex(ValueError, "invalid experiment"):
                require_episode_training_ready(path)
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "invalid experiment"):
                require_episode_training_ready(path)

    def test_trainer_exits_before_reading_model_or_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            result = subprocess.run([sys.executable, str(ROOT / "python/train_episode_memory.py"),
                                     "missing.gguf", "missing-data", str(output),
                                     "--lib", "missing.so", "--tok-probe", "missing-probe"],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("Training blocked: P1.1-P1.7", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
