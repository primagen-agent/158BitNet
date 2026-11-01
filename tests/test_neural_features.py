"""Feature provenance prevents silently mixing stale and subset caches."""

from pathlib import Path
from types import SimpleNamespace
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from neural_slot_model import INPUT, TOKEN_LIMIT
from neural_training_common import extract, feature_texts_sha256, load_features, subset_features


class FeatureIdentityTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(
            features={text: dict(x=torch.zeros(1, INPUT), bounds=[[0, 1]]) for text in ("alpha", "beta")},
            identity=dict(backbone="a" * 64, probe="b" * 64, texts_sha256=feature_texts_sha256(("alpha", "beta"))),
        )

    def load(self, saved, **expected):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.pt"
            torch.save(saved, path)
            return load_features(path, **expected)

    def test_subset_records_exact_identity_without_mutating_parent(self):
        small = subset_features(self.saved, ["beta"])
        self.assertEqual(small["identity"]["parent_texts_sha256"], self.saved["identity"]["texts_sha256"])
        self.assertEqual(small["identity"]["texts_sha256"], feature_texts_sha256(["beta"]))
        self.assertNotIn("parent_texts_sha256", self.saved["identity"])
        self.assertEqual(set(self.load(small)["features"]), {"beta"})

    def test_parent_identity_cannot_describe_a_subset(self):
        self.saved["features"].pop("alpha")
        with self.assertRaisesRegex(ValueError, "text identity mismatch"):
            self.load(self.saved)

    def test_changed_backbone_and_probe_are_rejected(self):
        for field in ("backbone", "probe"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field + " mismatch"):
                self.load(self.saved, **{field: "c" * 64})
        self.load(self.saved, backbone="a" * 64, probe="b" * 64)

    def test_missing_identity_and_missing_subset_text_are_rejected(self):
        self.saved["identity"].pop("probe")
        with self.assertRaisesRegex(ValueError, "cache identity"):
            self.load(self.saved)
        with self.assertRaisesRegex(ValueError, "texts are missing"):
            subset_features(self.saved, ["missing"])

    def test_cache_and_extracted_headers_enforce_native_token_limit(self):
        row = torch.zeros(1, INPUT)
        bounds = torch.tensor([[0, 1]])
        identity = dict(self.saved["identity"], texts_sha256=feature_texts_sha256(["alpha"]))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "build") as directory:
            cache_path = Path(directory) / "boundary-features.pt"
            for count in (TOKEN_LIMIT - 1, 0, TOKEN_LIMIT):
                saved = dict(features={"alpha": dict(x=row.expand(count, -1),
                                                    bounds=bounds.expand(count, -1))},
                             identity=identity)
                torch.save(saved, cache_path)
                with self.subTest(stage="cache", count=count):
                    if count == TOKEN_LIMIT - 1:
                        actual = load_features(cache_path, backbone=identity["backbone"],
                                               probe=identity["probe"])
                        self.assertEqual(len(actual["features"]["alpha"]["x"]), count)
                        self.assertEqual(actual["identity"], identity)
                    else:
                        with self.assertRaisesRegex(ValueError, "geometry"):
                            load_features(cache_path)
            args = SimpleNamespace(output=directory, probe="unused-probe", gguf="unused.gguf")
            for count in (0, TOKEN_LIMIT):
                def invalid_probe(command, *, check):
                    Path(command[3]).write_bytes(b"BNSFEAT1" + struct.pack("<II", 1, count))
                with self.subTest(stage="extract", count=count), \
                        patch("neural_training_common.subprocess.run", side_effect=invalid_probe), \
                        patch("neural_training_common.np.frombuffer", side_effect=AssertionError("read before rejecting count")), \
                        self.assertRaisesRegex(ValueError, "token count"):
                    extract(args, {"fit": [dict(body="alpha", context="alpha")]})
                self.assertFalse((Path(directory) / "features.pt").exists())


if __name__ == "__main__":
    unittest.main()
