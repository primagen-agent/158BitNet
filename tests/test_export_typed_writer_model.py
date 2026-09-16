#!/usr/bin/env python3
"""Tests for the autonomous typed-writer binary exporter."""

import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from export_typed_writer_model import (  # noqa: E402
    MAGIC,
    export_typed_writer_binary,
)
from train_typed_anchor_keys import (  # noqa: E402
    FORMAT as ANCHOR_FORMAT,
)
from train_typed_memory_writer import (  # noqa: E402
    SLOT_NAMES,
)
from train_typed_span_tagger import (  # noqa: E402
    FORMAT as TAGGER_FORMAT,
)


class ExportTypedWriterModelTests(unittest.TestCase):
    def test_export_header_and_component_count(self):
        hidden = 5
        rank = 3
        bands = [[0], [1]]
        max_span = 4
        writer_state = {
            "band_logits": torch.zeros(4, 2),
            "localizer_band_logits": torch.zeros(2, 2),
            "token_keys": torch.zeros(4, rank),
            "span_boundary_keys": torch.zeros(4, 2, rank),
            "operation_head.0.weight":
                torch.zeros(rank, rank * 4),
            "operation_head.0.bias": torch.zeros(rank),
            "operation_head.2.weight": torch.zeros(2, rank),
            "operation_head.2.bias": torch.zeros(2),
        }
        for name in SLOT_NAMES:
            writer_state[
                f"projections.{name}.weight"
            ] = torch.zeros(rank, hidden)
        for name in ("entity", "predicate"):
            writer_state[
                f"localizer_projections.{name}.weight"
            ] = torch.zeros(rank, hidden)
        tagger_state = {}
        for name in SLOT_NAMES:
            prefix = f"fields.{name}."
            tagger_state[prefix + "context.weight"] = (
                torch.zeros(rank, rank, 3)
            )
            tagger_state[prefix + "context.bias"] = (
                torch.zeros(rank)
            )
            tagger_state[prefix + "output.weight"] = (
                torch.zeros(1, rank)
            )
            tagger_state[prefix + "output.bias"] = (
                torch.zeros(1)
            )
            tagger_state[prefix + "length_logits"] = (
                torch.zeros(max_span)
            )
            tagger_state[prefix + "residual_scale"] = (
                torch.tensor(0.0)
            )
        sha = "11" * 32
        writer = {
            "format": "TYPED_MEMORY_WRITER_V1",
            "backbone_sha256": sha,
            "hidden": hidden,
            "rank": rank,
            "hidden_layer_bands": bands,
            "state_dict": writer_state,
            "separate_address_localizer": True,
            "predict_span_boundaries": True,
            "use_token_embeddings": True,
            "contiguous_span_pooling": False,
        }
        tagger = {
            "format": TAGGER_FORMAT,
            "backbone_sha256": sha,
            "writer_checkpoint_fingerprint": "writer-sha",
            "rank": rank,
            "max_span": max_span,
            "state_dict": tagger_state,
        }
        anchors = {
            "format": ANCHOR_FORMAT,
            "backbone_sha256": sha,
            "writer_checkpoint_fingerprint": "writer-sha",
            "state_dict": {
                "keys.predicate": torch.zeros(rank),
                "keys.value": torch.zeros(rank),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                root / "writer.pt",
                root / "tagger.pt",
                root / "anchors.pt",
            ]
            for path, value in zip(
                paths, (writer, tagger, anchors)
            ):
                torch.save(value, path)
            output = root / "model.bntwrite"
            with mock.patch(
                "export_typed_writer_model.file_fingerprint",
                return_value="writer-sha",
            ):
                export_typed_writer_binary(
                    *paths, output
                )
            payload = output.read_bytes()
        self.assertEqual(payload[:8], MAGIC)
        values = struct.unpack_from("<IIIIIIIff", payload, 8)
        self.assertEqual(values[:7], (
            1, hidden, rank, 2, 2, max_span, 39
        ))
        self.assertAlmostEqual(values[7], 1.0)
        self.assertAlmostEqual(values[8], 0.25)


if __name__ == "__main__":
    unittest.main()
