"""Synthetic pilot and loss isolation, not memory accuracy."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"python"))
from availability_supervision import ReplyTargets, encode_reply_targets, supervised_loss, teacher_forcing_requests
from neural_memory_generation import GenerationInput
from prepare_availability_curriculum import build, materialize
from prepare_neural_memory_protocol import digest

CONFIG = ROOT/"training/memory/neural-system/data/CG-001-pilot.json"


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text())
        self.files, self.report = build(self.config)

    def test_counts_balance_and_reproducibility(self):
        self.assertEqual(self.files, build(self.config)[0])
        self.assertEqual((self.report["cases"], self.report["worlds"], self.report["pair_count"]), (768, 24, 672))
        self.assertEqual(self.report["counts"]["train/supported"], 128)
        self.assertEqual(self.report["counts"]["train/insufficient"], 256)
        self.assertEqual({v for k,v in self.report["weighted_counts"].items() if k.startswith("train/")}, {256.})

    def test_subjects_and_atomic_facts_are_split_disjoint(self):
        groups = {s: [json.loads(line) for line in self.files[f"{s}.index.jsonl"].splitlines()] for s in ("train", "dev")}
        people = [{f["subject"] for r in groups[s] for f in r["family_facts"]} for s in groups]
        self.assertFalse(people[0] & people[1])
        self.assertFalse((people[0] | people[1]) & set(self.config["excluded_subjects"]))

    def test_labels_and_reference_text_never_join_input_schema(self):
        for split in ("train", "dev"):
            for line in self.files[f"{split}.inputs.jsonl"].splitlines():
                record = json.loads(line)
                self.assertEqual(set(record), {"id", "context", "episodes"})
                self.assertLessEqual(len(record["episodes"]), 1)
                self.assertNotIn("ordinary", record["id"])

    def test_pairing_preserves_exact_question_and_changes_memory(self):
        inputs = {r["id"]:r for split in ("train", "dev") for line in self.files[f"{split}.inputs.jsonl"].splitlines() for r in [json.loads(line)]}
        for line in self.files["pairs.jsonl"].splitlines():
            pair = json.loads(line); a,b = (inputs[pair[k]] for k in ("left_id", "right_id"))
            self.assertEqual(a["context"], b["context"])
            self.assertNotEqual(a["episodes"], b["episodes"])

    def test_materialized_files_are_immutable_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"corpus"
            manifest = materialize(CONFIG, output)
            self.assertEqual(manifest, materialize(CONFIG, output, verify=True))
            with self.assertRaises(FileExistsError): materialize(CONFIG, output)
            (output/"train.inputs.jsonl").write_text("tampered")
            with self.assertRaises(ValueError): materialize(CONFIG, output, verify=True)

    def test_old_subject_duplicate_or_changed_weights_rejected(self):
        for mode in ("old", "duplicate", "weights"):
            config = copy.deepcopy(self.config)
            if mode == "old": config["people"][0] = "Nora"
            elif mode == "duplicate": config["people"][0] = config["people"][1]
            else: config["state_weights"]["supported"] = 1
            with self.assertRaises(ValueError): build(config)


class LossTests(unittest.TestCase):
    def setUp(self):
        self.request = GenerationInput((1,2,3), ("independent source",))
        self.targets = ReplyTargets((1,2,3), (4,5,0), 2, 1., "answer")

    def test_teacher_forcing_separates_forward_objects_from_labels(self):
        calls = list(teacher_forcing_requests(self.request, self.targets))
        self.assertEqual([c.prompt_token_ids for c in calls], [(1,2,3),(1,2,3,4),(1,2,3,4,5)])
        self.assertTrue(all(type(c) is GenerationInput and c.source_texts == self.request.source_texts for c in calls))
        self.assertNotIn("state_index", vars(calls[0]))
        changed = ReplyTargets((1,2,3), (5,4,0), 0, 2., "other")
        self.assertEqual(next(teacher_forcing_requests(self.request, changed)), calls[0])

    def test_losses_reach_generation_and_initial_state_only(self):
        generation = torch.randn(3,6,requires_grad=True)
        all_states = torch.randn(3,3,requires_grad=True)
        result = supervised_loss(generation, all_states[:1], self.targets, state_coefficient=.2)
        result["weighted_total"].backward()
        self.assertGreater(float(generation.grad.norm()), 0)
        self.assertGreater(float(all_states.grad[0].norm()), 0)
        self.assertEqual(int(all_states.grad[1:].count_nonzero()), 0)

    def test_future_position_state_supervision_and_invalid_targets_rejected(self):
        with self.assertRaises(ValueError): supervised_loss(torch.randn(3,6), torch.randn(3,3), self.targets, state_coefficient=.2)
        with self.assertRaises(ValueError): supervised_loss(torch.randn(2,6), torch.randn(1,3), self.targets, state_coefficient=.2)
        with self.assertRaises(ValueError): supervised_loss(torch.randn(3,4), torch.randn(1,3), self.targets, state_coefficient=.2)
        with self.assertRaises(ValueError): list(teacher_forcing_requests(GenerationInput((1,2),()), self.targets))
        with self.assertRaises(ValueError): ReplyTargets((1,), (2,), True, 1., "reply")

    def test_impossible_state_cannot_silently_produce_infinite_loss(self):
        state = torch.tensor([[0.,0.,-torch.inf]])
        with self.assertRaises(ValueError): supervised_loss(torch.randn(3,6), state, self.targets, state_coefficient=.2)


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.runtime = {"id":"x", "context":[{"role":"user","speaker":"user","text":"Where does Adela live?"}],"episodes":[]}
        self.label = {"id":"x","input_sha256":digest(self.runtime),"state":"insufficient","response":"I do not know.",
                      "memory_needed":True,"evidence_missing":True,"sample_weight":1.}
        class Tokenizer:
            def encode(self, text, add_bos):
                return [1,2,3] if text.endswith("assistant\n") else [1,2,3,4,0]
            def decode_pieces(self, ids): return [b"I do not know.",b"<|im_end|>"]
            def eos(self): return 0
        self.tokenizer=Tokenizer()

    def test_joint_encoding_uses_completion_only_including_eos(self):
        target=encode_reply_targets(self.runtime,self.label,self.tokenizer)
        self.assertEqual(target.prompt_token_ids,(1,2,3))
        self.assertEqual(target.completion_token_ids,(4,0))

    def test_prefix_merging_overflow_and_changed_bytes_fail_closed(self):
        with self.assertRaises(ValueError): encode_reply_targets(self.runtime,self.label,self.tokenizer,context_capacity=4)
        self.tokenizer.encode=lambda *_:[1,2,3] if _[0].endswith("assistant\n") else [1,9,4,0]
        with self.assertRaisesRegex(ValueError,"boundary"): encode_reply_targets(self.runtime,self.label,self.tokenizer)
        self.setUp(); self.tokenizer.decode_pieces=lambda *_:[b" I do not know.",b"<|im_end|>"]
        with self.assertRaisesRegex(ValueError,"bytes"): encode_reply_targets(self.runtime,self.label,self.tokenizer)

    def test_hash_stop_and_annotation_mismatches_rejected(self):
        label={**self.label,"input_sha256":"0"*64}
        with self.assertRaises(ValueError): encode_reply_targets(self.runtime,label,self.tokenizer)
        label={**self.label,"state":"supported"}
        with self.assertRaises(ValueError): encode_reply_targets(self.runtime,label,self.tokenizer)
        self.tokenizer.eos=lambda:5
        with self.assertRaises(ValueError): encode_reply_targets(self.runtime,self.label,self.tokenizer)

    def test_reference_cannot_inject_template_or_empty_answer(self):
        for response in ("", "<|im_end|>", "a\0b"):
            with self.assertRaises(ValueError): encode_reply_targets(self.runtime,{**self.label,"response":response},self.tokenizer)


if __name__ == "__main__": unittest.main()
