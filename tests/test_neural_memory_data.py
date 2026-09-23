"""Synthetic protocol and real reader-call boundary tests; no training/accuracy claim."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from c_tokenizer import CTokenizer
from episode_memory import EpisodeBindingReader
from episode_memory_inputs import ReaderFeatures, encode_reader_input, encoder_texts, reader_forward
from prepare_neural_memory_protocol import generate, materialize, semantic_key, validate_records, SCENARIOS
from train_episode_memory import BindingExample, BindingSupervision, evaluate, loss_for, token_rows

CONFIG = ROOT / "training/memory/neural-system/data/P1-smoke.json"


def records(files, kind):
    return [json.loads(line) for split in ("train", "dev") for line in files[f"{split}.{kind}.jsonl"].decode().splitlines()]


def fake_features(text):
    """Synthetic feature fixture, NOT a substitute tokenizer or backbone."""
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    return torch.randn(5, 16, generator=torch.Generator().manual_seed(seed))


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG.read_text())
        cls.files, cls.report = generate(cls.config)

    def rows(self):
        return tuple(records(self.files, kind) for kind in ("inputs", "labels", "index"))

    def test_reproducible_complete_smoke_not_large_sample_acceptance(self):
        files, report = generate(self.config)
        self.assertEqual(files, self.files)
        self.assertEqual(report["cases"], 704)
        self.assertEqual(report["worlds"], 16)
        self.assertEqual(len(report["groups"]), len(SCENARIOS) * 4)
        self.assertEqual(set(report["groups"].values()), {8})

    def test_bilingual_paraphrases_and_counterfactuals_stay_in_family(self):
        inputs, labels, index = self.rows()
        families = {}
        for meta in index:
            families.setdefault(meta["world_id"], set()).add(meta["split"])
        self.assertTrue(all(len(splits) == 1 for splits in families.values()))
        self.assertEqual(validate_records(inputs, labels, index)["cases"], 704)
        train_people, dev_people = set(), set()
        for meta in index:
            group = train_people if meta["split"] == "train" else dev_people
            group.update(f["subject"] for f in meta["family_facts"])
        self.assertFalse(train_people & dev_people)

    def test_paraphrase_new_ids_and_reordering_do_not_evade_leak_check(self):
        for mode in ("world", "family", "events", "fact"):
            inputs, labels, index = self.rows()
            pos = next(i for i, m in enumerate(index) if m["split"] == "train" and m["scenario"] == "original")
            new_input, new_label, new_meta = (copy.deepcopy(rows[pos]) for rows in (inputs, labels, index))
            for row in (new_input, new_label, new_meta): row["id"] = "disguised-copy"
            new_meta["split"] = "dev"
            new_input["episodes"][0]["text"] = "A paraphrase with the same underlying facts."
            if mode != "world": new_meta["world_id"] = "fake-new-world"
            if mode in ("events", "fact"):
                new_meta["family_facts"] = [{"invented": "different-family-label"}]
            if mode == "events": new_meta["episode_facts"].reverse()
            if mode == "fact": new_meta["episode_facts"][1] = {"invented": "additional-fact"}
            for rows, row in zip((inputs, labels, index), (new_input, new_label, new_meta)): rows.append(row)
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "cross-split"):
                validate_records(inputs, labels, index)

    def test_gold_is_separate_and_targets_have_sources(self):
        inputs, labels, index = self.rows()
        self.assertEqual(set(inputs[0]), {"id", "context", "episodes"})
        labels[0]["relevant_episode_indices"] = [99]
        with self.assertRaisesRegex(ValueError, "no source"): validate_records(inputs, labels, index)
        inputs, labels, index = self.rows()
        labels.pop()
        with self.assertRaisesRegex(ValueError, "ID mismatch"): validate_records(inputs, labels, index)

    def test_relevance_does_not_assert_hypothetical_negative_or_quoted_fact(self):
        _, labels, index = self.rows()
        for label, meta in zip(labels, index):
            if meta["scenario"] in ("hypothetical", "negated", "quoted"):
                self.assertEqual(label["relevant_episode_indices"], [0])
                self.assertEqual(label["required_claims"], [])
                self.assertTrue(label["evidence_missing"])
                self.assertNotEqual(label["allowed_claims"][0]["status"], "actual")

    def test_semantic_key_ignores_surface_order_not_roles_status_or_time(self):
        base = [{"subject": "A", "relation": "lent", "value": "B", "time": "2020", "status": "actual"},
                {"subject": "C", "relation": "home", "value": "Lima", "time": "now", "status": "actual"}]
        self.assertEqual(semantic_key(base), semantic_key(list(reversed(base))))
        for field, value in (("subject", "B"), ("time", "2024"), ("status", "hypothetical")):
            changed = copy.deepcopy(base); changed[0][field] = value
            self.assertNotEqual(semantic_key(base), semantic_key(changed))

    def test_manifest_verification_regenerates_all_files_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "corpus"
            manifest = materialize(CONFIG, output)
            self.assertEqual(manifest, json.loads((CONFIG.parent / "P1-smoke.manifest.json").read_text()))
            self.assertEqual(manifest, materialize(CONFIG, output, verify=True))
            self.assertFalse(manifest["automatic_memory"])
            self.assertEqual(manifest["purpose"], "protocol_smoke_not_promotion")
            with self.assertRaises(FileExistsError): materialize(CONFIG, output)
            extra = output / "extra-training.jsonl"
            extra.write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "unexpected"): materialize(CONFIG, output, verify=True)
            extra.unlink()
            (output / "dev.labels.jsonl").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "mismatch"): materialize(CONFIG, output, verify=True)


class ForwardBoundaryTests(unittest.TestCase):
    def setUp(self):
        files, _ = generate(json.loads(CONFIG.read_text()))
        self.runtime = records(files, "inputs")[0]
        self.label = records(files, "labels")[0]
        torch.manual_seed(19)
        self.reader = EpisodeBindingReader(8).eval()

    def test_annotation_changes_do_not_change_encoder_input_features_or_logits(self):
        calls = []
        def encoder(text):
            calls.append(text)
            return fake_features(text)
        original = encode_reader_input(self.runtime, encoder)
        first_calls = list(calls); calls.clear()
        self.label.update(required_claims=[{"answer": "SECRET_FUTURE_ANSWER"}], relevant_episode_indices=[1])
        changed = encode_reader_input(self.runtime, encoder)
        self.assertEqual(first_calls, calls)
        self.assertNotIn("SECRET_FUTURE_ANSWER", "".join(calls))
        self.assertTrue(torch.equal(original.query, changed.query))
        for a, b in zip(original.episodes, changed.episodes): self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(reader_forward(self.reader, original)[0], reader_forward(self.reader, changed)[0]))

    def test_forward_only_receives_tensors_not_ids_text_answers_or_scenario(self):
        observed = []
        hook = self.reader.register_forward_pre_hook(lambda model, args: observed.append(args))
        try:
            features = encode_reader_input(self.runtime, fake_features)
            reader_forward(self.reader, features)
        finally: hook.remove()
        args = observed[0]
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], features.query)
        self.assertIs(args[1], features.episodes)
        with self.assertRaises(ValueError): reader_forward(self.reader, {"features": features, "label": self.label})

    def test_true_source_changes_can_change_logits_without_injecting_query(self):
        before_query, _ = encoder_texts(self.runtime)
        original = encode_reader_input(self.runtime, fake_features)
        changed = copy.deepcopy(self.runtime)
        changed["episodes"][0]["text"] = "Different source content."
        self.assertEqual(encoder_texts(changed)[0], before_query)
        result = encode_reader_input(changed, fake_features)
        self.assertFalse(torch.equal(reader_forward(self.reader, original)[0], reader_forward(self.reader, result)[0]))

    def test_unknown_fields_future_reply_and_nonfinite_features_are_rejected(self):
        for field in ("answer", "gold_targets", "scenario", "future_messages"):
            with self.assertRaises(ValueError): encode_reader_input({**self.runtime, field: "leak"}, fake_features)
        changed = copy.deepcopy(self.runtime); changed["context"].append({"role": "assistant", "speaker": "assistant", "text": "future answer"})
        with self.assertRaises(ValueError): encode_reader_input(changed, fake_features)
        changed = copy.deepcopy(self.runtime); changed["episodes"][0]["gold_status"] = "actual"
        with self.assertRaises(ValueError): encode_reader_input(changed, fake_features)
        with self.assertRaises(ValueError): ReaderFeatures(torch.full((2, 16), float("nan")), ())

    def test_ids_do_not_leak_and_feature_bank_storage_is_not_shared(self):
        calls = {}
        def encoder(text):
            calls.setdefault(text, fake_features(text))
            return calls[text]
        original = encode_reader_input(self.runtime, encoder)
        changed = copy.deepcopy(self.runtime); changed["id"] = "gold-target-is-zero"
        other = encode_reader_input(changed, encoder)
        self.assertTrue(torch.equal(original.query, other.query))
        original.query.zero_()
        self.assertFalse(torch.equal(original.query, other.query))

    def test_loss_and_evaluation_call_same_label_free_boundary(self):
        features = encode_reader_input(self.runtime, fake_features)
        role = torch.tensor([1., 0., 0., 0., 0.])
        supervision = BindingSupervision(role, (role, role), torch.tensor([1., 0.]), torch.ones(2))
        example = BindingExample("one", "original", features, supervision)
        altered = replace(example, case_id="changed", supervision=replace(supervision, entity=1 - supervision.entity))
        calls = []
        hook = self.reader.register_forward_hook(lambda model, args, result: calls.append(result[0].detach().clone()))
        try:
            loss_a, loss_b = loss_for(self.reader, example), loss_for(self.reader, altered)
            eval_a, eval_b = evaluate(self.reader, [example]), evaluate(self.reader, [altered])
        finally: hook.remove()
        self.assertTrue(all(torch.equal(calls[0], output) for output in calls[1:]))
        self.assertNotEqual(float(loss_a.detach()), float(loss_b.detach()))
        self.assertEqual(eval_a["cases"][0]["actual"], eval_b["cases"][0]["actual"])
        self.assertNotEqual(eval_a["cases"][0]["expected"], eval_b["cases"][0]["expected"])
        loss_a.backward()
        self.assertIsNotNone(self.reader.owner_role[0].weight.grad)

    @unittest.skipUnless((ROOT / "build/tok_probe").exists() and (ROOT / "models/bitcpm4-0.5b-tq2_0.gguf").exists(), "local C tokenizer and exact 0.5B fixture required")
    def test_legacy_training_adapter_with_actual_c_tokenizer_keeps_answer_out(self):
        tokenizer = CTokenizer(str(ROOT / "build/tok_probe"), str(ROOT / "models/bitcpm4-0.5b-tq2_0.gguf"))
        try:
            row = {"id": "legacy", "condition": "original", "query": "Where do Alice and Bob live?",
                   "memory": ["Alice's home city is Lima.", "Bob's home city is Rome."],
                   "answer": "Alice lives in Lima.", "expected_value": "Lima"}
            texts = [row["query"]] + row["memory"]
            bank = {text: torch.randn(len(tokenizer.encode(text, True)) - 1, 16) for text in texts}
            original = token_rows([row], bank, tokenizer, "cpu")[0]
            changed = token_rows([{**row, "answer": "Bob lives in Rome.", "expected_value": "Rome"}], bank, tokenizer, "cpu")[0]
            self.assertFalse(torch.equal(original.supervision.entity, changed.supervision.entity))
            self.assertTrue(torch.equal(original.features.query, changed.features.query))
            self.assertTrue(torch.equal(reader_forward(self.reader, original.features)[0], reader_forward(self.reader, changed.features)[0]))
        finally:
            tokenizer._proc.terminate(); tokenizer._proc.wait()
            tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


if __name__ == "__main__": unittest.main()
