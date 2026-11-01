"""Frozen C features and checkpoint identities for current memory training."""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import numpy as np
import torch
from neural_slot_model import INPUT, TOKEN_LIMIT


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def feature_texts_sha256(texts):
    return hashlib.sha256(json.dumps(sorted(texts)).encode()).hexdigest()


def subset_features(saved, texts):
    """Keep the exact selected feature identity, with its parent recorded."""
    selected = sorted(set(texts))
    if not set(selected) <= saved["features"].keys():
        raise ValueError("requested feature texts are missing")
    identity = dict(saved["identity"])
    identity["parent_texts_sha256"] = identity["texts_sha256"]
    identity["texts_sha256"] = feature_texts_sha256(selected)
    return dict(features={t: saved["features"][t] for t in selected}, identity=identity)


def load_features(path, *, backbone=None, probe=None):
    saved = torch.load(path, weights_only=False, map_location="cpu")
    if set(saved) != {"features", "identity"}:
        raise ValueError("feature cache must be a single current cache")
    identity = saved["identity"]
    if not isinstance(saved["features"], dict) or not isinstance(identity, dict):
        raise ValueError("invalid feature cache identity")
    for key in ("backbone", "probe", "texts_sha256"):
        value = identity.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid feature cache identity")
    if identity["texts_sha256"] != feature_texts_sha256(saved["features"]):
        raise ValueError("feature text identity mismatch")
    if backbone is not None and identity["backbone"] != backbone:
        raise ValueError("frozen feature backbone mismatch")
    if probe is not None and identity["probe"] != probe:
        raise ValueError("frozen feature probe mismatch")
    for text, item in saved["features"].items():
        values = item["x"]
        bounds = torch.as_tensor(item["bounds"]).to(torch.int64)
        if (
            values.ndim != 2
            or values.shape[1] != INPUT
            or not len(values)
            or len(values) >= TOKEN_LIMIT
            or bounds.shape != (len(values), 2)
            or not torch.isfinite(values).all()
            or (bounds < 0).any()
            or (bounds[:, 0] > bounds[:, 1]).any()
            or (bounds > len(text.encode())).any()
        ):
            raise ValueError("invalid frozen C feature geometry or values")
    return saved


def extract(args, rows):
    root = Path(args.output)
    cache = {}
    texts = sorted(
        {r[k] for group in rows.values() for r in group for k in ("body", "context")}
    )
    with tempfile.TemporaryDirectory(prefix="features-", dir=root) as temporary:
        source = Path(temporary) / "features.in"
        out = Path(temporary) / "features.bin"
        with source.open("wb") as f:
            f.write(struct.pack("<I", len(texts)))
            for text in texts:
                b = text.encode()
                f.write(struct.pack("<I", len(b)) + b)
        subprocess.run([args.probe, args.gguf, str(source), str(out)], check=True)
        with out.open("rb") as f:
            assert f.read(8) == b"BNSFEAT1" and struct.unpack("<I", f.read(4))[0] == len(
                texts
            )
            for text in texts:
                n = struct.unpack("<I", f.read(4))[0]
                if not n or n >= TOKEN_LIMIT:
                    raise ValueError("invalid C feature token count")
                bounds = np.frombuffer(f.read(n * 8), dtype="<u4").copy().reshape(n, 2)
                x = torch.from_numpy(
                    np.frombuffer(f.read(n * INPUT * 4), dtype="<f4").copy().reshape(n, INPUT)
                )
                if not torch.isfinite(x).all():
                    raise ValueError("invalid C features")
                cache[text] = dict(x=x, bounds=bounds)
            assert not f.read()
    identity = dict(
        backbone=sha(args.gguf),
        probe=sha(args.probe),
        texts_sha256=feature_texts_sha256(texts),
        training_data="independent synthetic generator",
    )
    torch.save(dict(features=cache, identity=identity), root / "features.pt")
    return cache, identity
