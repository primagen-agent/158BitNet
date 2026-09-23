"""Feature identity/transport regression fixtures, not a memory evaluation."""
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from native_memory_encoder import (BACKBONE_SHA256, POLICY, EncodedText, NativeFeatureBank, collect_texts,
                                   digest, read_encoded, save_bank, model_binding, validate_identity)
from neural_memory_contract import MemoryState, ReadContext, Availability


class NativeEncoderTests(unittest.TestCase):
    def setUp(self):
        self.binding = {"policy": POLICY, "backbone_sha256": BACKBONE_SHA256, "binary_sha256": "a" * 64,
                        "message_frame_source_sha256": "b" * 64, "system": "Darwin", "machine": "arm64",
                        "dispatch": "arm_neon", "os_release": "test-fixture"}
        self.identity = {"binding": self.binding, "encoder_id": digest(self.binding)}
        self.row = EncodedText(np.array([1, 2, 3], dtype="<i4"), np.arange(4096, dtype="<f4").reshape(2, 2048))

    def load(self, root, manifest_hash, encoder_id=None):
        return NativeFeatureBank(root, expected_encoder_id=encoder_id or self.identity["encoder_id"], expected_manifest_sha256=manifest_hash)

    def test_roundtrip_is_exact_and_callers_cannot_mutate_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bank"
            manifest = save_bank(root, self.identity, ["hello"], [self.row])
            bank = self.load(root, manifest)
            self.assertTrue(np.array_equal(bank("hello").numpy(), self.row.features))
            bank("hello").zero_()
            self.assertTrue(np.array_equal(bank("hello").numpy(), self.row.features))
            with self.assertRaises(ValueError): bank("new question")
            with self.assertRaises(FileExistsError): save_bank(root, self.identity, ["hello"], [self.row])

    def test_wrong_encoder_manifest_and_corrupt_archive_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bank"
            manifest = save_bank(root, self.identity, ["hello"], [self.row])
            with self.assertRaises(ValueError): self.load(root, manifest, "b" * 64)
            with self.assertRaises(ValueError): self.load(root, "b" * 64)
            (root / "features.npz").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "corruption"): self.load(root, manifest)

    def test_changed_text_and_dtype_are_rejected_even_with_fresh_container_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bank"
            save_bank(root, self.identity, ["hello"], [self.row])
            manifest = json.loads((root / "manifest.json").read_text())
            entry = next(iter(manifest["entries"].values())); entry["text"] = "different"
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "input hash"): self.load(root, digest(manifest))
        for array in (self.row.features.astype(np.float16), self.row.features[:, :1024], np.full((2, 2048), np.nan, dtype="<f4")):
            with self.assertRaises(ValueError): EncodedText(self.row.token_ids, array).validate()

    def test_metadata_changes_change_identity(self):
        for key, value in (("binary_sha256", "b" * 64), ("backbone_sha256", "c" * 64), ("policy", {**POLICY, "threads": 1})):
            self.assertNotEqual(digest(self.binding), digest({**self.binding, key: value}))
        before = model_binding(self.identity)
        changed = {**self.binding, "binary_sha256": "b" * 64}
        after = model_binding({"binding": changed, "encoder_id": digest(changed)})
        state = MemoryState("scope", 0, before, ())
        with self.assertRaises(ValueError): ReadContext("scope", 0, after, Availability.MISSING, ()).validate_against(state)
        with self.assertRaises(ValueError): validate_identity({"binding": {}, "encoder_id": digest({})})

    def test_only_natural_inputs_are_collected_not_annotations(self):
        runtime = {"id": "one", "context": [{"role": "user", "speaker": "u", "text": "Where do I live?"}],
                   "episodes": [{"role": "user", "speaker": "u", "text": "I live in Lima."}]}
        texts = collect_texts([runtime, runtime])
        self.assertEqual(len(texts), 2)
        self.assertNotIn("Lima", texts[0])
        with self.assertRaises(ValueError): collect_texts([{**runtime, "answer": "Lima"}])
        altered = copy.deepcopy(runtime); altered["episodes"][0]["gold_status"] = "actual"
        with self.assertRaises(ValueError): collect_texts([altered])

    def test_binary_header_truncation_trailing_data_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.bin"
            raw = b"BNEN0001" + struct.pack("<IIII", 1, 1024, 24, 3) + self.row.token_ids.tobytes() + self.row.features.tobytes()
            path.write_bytes(raw)
            self.assertTrue(np.array_equal(read_encoded(path, 1)[0].features, self.row.features))
            for data in (raw[:-1], raw + b"x", raw.replace(b"BNEN0001", b"BNFP0001", 1)):
                path.write_bytes(data)
                with self.assertRaises(ValueError): read_encoded(path, 1)
            path.write_bytes(raw)
            with self.assertRaises(ValueError): read_encoded(path, 2)


if __name__ == "__main__": unittest.main()
