#!/usr/bin/env python3
"""C/PyTorch learned-head and source-byte decoding parity, using frozen C features."""

import argparse
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from neural_slot_model import identity_word_features, INPUT, WIDTH, IDENTITY, QUERY, SEMANTIC, KEYS, LEXICAL
from neural_training_common import load_features
from train_neural_dialogue_reader import load_model
from memory_test_utils import validate_memory_network


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checker")
    p.add_argument("gguf")
    p.add_argument("artifact")
    p.add_argument("checkpoint")
    p.add_argument("features")
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    geometry = validate_memory_network(args.artifact, args.gguf)
    assert geometry["input"] == INPUT and geometry["identity"] == IDENTITY
    assert geometry["query"] == QUERY and geometry["semantic"] == SEMANTIC
    assert geometry["tensors"] == len(KEYS)
    assert geometry["backbone_sha256"] == checkpoint["backbone_sha256"]
    model = load_model(checkpoint)
    model.eval()

    saved = load_features(args.features)
    assert saved["identity"]["backbone"] == checkpoint["backbone_sha256"]
    cache = saved["features"]
    texts = (
        sorted(cache)[:4] + sorted(t for t in cache if "Activated evidence:" in t)[:2]
    )
    texts = list(dict.fromkeys(texts))
    if len(texts) < 6:
        texts = list(dict.fromkeys(texts + sorted(cache)))[:6]
    assert texts
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        inputs = root / "sources"
        features = root / "features"
        output = root / "output"
        with inputs.open("wb") as f, features.open("wb") as g:
            f.write(struct.pack("<I", len(texts)))
            g.write(b"BNSFEAT1" + struct.pack("<I", len(texts)))
            for text in texts:
                b = text.encode()
                f.write(struct.pack("<I", len(b)) + b)
                item = cache[text]
                n = len(item["x"])
                assert item["x"].shape == (n, INPUT) and n > 0
                assert item["bounds"].shape == (n, 2)
                assert torch.isfinite(item["x"]).all()
                g.write(struct.pack("<I", n))
                g.write(np.asarray(item["bounds"], dtype="<u4").tobytes())
                g.write(np.asarray(item["x"], dtype="<f4").tobytes())
        subprocess.run(
            [
                args.checker,
                args.artifact,
                args.gguf,
                str(features),
                str(output),
                str(inputs),
            ],
            check=True,
        )
        with output.open("rb") as f, torch.no_grad():
            assert f.read(8) == b"BNSCHECK" and struct.unpack("<I", f.read(4))[
                0
            ] == len(texts)

            def array(n):
                return np.frombuffer(f.read(4 * n), dtype="<f4").copy()

            for text in texts:
                x = cache[text]["x"]
                packed = (x, identity_word_features(text, cache[text])[0])
                n = struct.unpack("<I", f.read(4))[0]
                assert n == len(x)
                encoded, identity = model.prepare(packed)
                q, qid = model.query(packed)
                scope_q, scope_id = model.query(packed, scope=True)
                assert encoded.shape == (n, WIDTH)
                assert identity.numel() == qid.numel() == IDENTITY
                assert q.numel() == QUERY
                kind_logits = model.kind_logits_from_mean(encoded.mean(0), model.kind_context_inputs(x))
                memory_context = model.memory_context_latents(model.kind_context_inputs(x))
                assert struct.unpack("<i", f.read(4))[0] == int(kind_logits.argmax())
                for expected in (encoded.flatten(), identity, q, qid):
                    np.testing.assert_allclose(
                        array(expected.numel()), expected.numpy(), atol=2e-4, rtol=2e-4
                    )
                meta = torch.tensor([1.0, 0.0, 0.0, 0.0])
                scores = [model.score(q, qid, encoded, identity, meta, memory_context_latents=memory_context).item(),
                          model.score(scope_q, scope_id, encoded, identity, meta, memory_context_latents=memory_context, scope=True).item()]
                np.testing.assert_allclose(array(2), scores, atol=3e-4, rtol=3e-4)
                initial = q * 0.5 + 0.03 * (torch.arange(QUERY) % 5)
                advanced = model.advance_query(initial, q)
                np.testing.assert_allclose(
                    array(QUERY), advanced.numpy(), atol=3e-4, rtol=3e-4
                )
                original_id = qid * 0.5 + 0.01 * ((torch.arange(IDENTITY) % 7) - 3)
                original_id = torch.cat(
                    (F.normalize(original_id[:LEXICAL], dim=0),
                     F.normalize(original_id[LEXICAL:], dim=0))
                )
                advanced_id = model.advance_identity(initial, q, original_id, qid)
                np.testing.assert_allclose(
                    array(IDENTITY), advanced_id.numpy(), atol=3e-4, rtol=3e-4
                )
                boundary = model.plan_boundary_scores(encoded, x)
                for column in range(2):
                    np.testing.assert_allclose(
                        array(n), boundary[:, column].numpy(), atol=2e-4, rtol=2e-4
                    )
                np.testing.assert_allclose(
                    array(n), model.plan_membership_scores(encoded, x).numpy(), atol=2e-4, rtol=2e-4
                )
                body = encoded * 0.5 + 0.03 * (torch.arange(n * WIDTH).reshape(n, WIDTH) % 7)
                np.testing.assert_allclose(
                    array(n), model.plan_context_residual(encoded, body, x).numpy(), atol=2e-4, rtol=2e-4
                )
                for expected in (scope_q, scope_id):
                    np.testing.assert_allclose(
                        array(expected.numel()), expected.numpy(), atol=2e-4, rtol=2e-4
                    )
                np.testing.assert_allclose(array(3), kind_logits.numpy(), atol=2e-4, rtol=2e-4)
                np.testing.assert_allclose(array(16), memory_context.numpy(), atol=2e-4, rtol=2e-4)
            assert not f.read()
    print(
        f"PASS: {len(texts)} C/PyTorch read/scope query, score, identity, kind, update and source-boundary/membership/context head parity cases"
    )


if __name__ == "__main__":
    main()
