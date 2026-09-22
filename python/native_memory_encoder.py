"""Exact C-produced reader inputs. Research-only, not a generation backend.

No trainable parameters or labels are handled here. Feature archives are offline
training artifacts; serving must encode new natural input, not retrieve answers.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import struct
import subprocess

import numpy as np
import torch

from episode_memory_inputs import encoder_texts
from neural_memory_contract import ModelBinding

BACKBONE_SHA256 = "44cb4e0db8374d4247bba391b3d3b7c0b3ee815c36cf2e20d0d1a3570677bd95"
HIDDEN, LAYERS, VOCAB = 1024, 24, 73448
POLICY = {"format": "native-reader-f32-v1", "wire_format": "BNEN0001", "hidden": HIDDEN, "layers": LAYERS,
          "dtype": "float32", "bos": "encoded_then_drop_feature_row", "normalization": "c_ordered_f32_l2_eps1e-12",
          "message_frame": "json-role-speaker-text-v1", "threads": 4, "metal": False,
          "context_capacity": 512, "maximum_tokens_with_bos": 511,
          "cross_input_kv_reuse": False, "internal_prefill_kv_buffers": True}


def sha_file(path):
    # GPU host uses Python 3.10, which lacks hashlib.file_digest.
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): checksum.update(block)
    return checksum.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def text_key(text):
    if not isinstance(text, str) or not text or "\0" in text or len(text.encode()) > 65536:
        raise ValueError("invalid encoder text")
    return hashlib.sha256(text.encode()).hexdigest()


def encoder_identity(gguf, probe):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("only macOS arm64 native extraction has been qualified for this pilot")
    if sha_file(gguf) != BACKBONE_SHA256: raise ValueError("exact 0.5B GGUF required")
    frame_source = Path(__file__).with_name("episode_memory_inputs.py")
    binding = {"policy": POLICY, "backbone_sha256": BACKBONE_SHA256, "binary_sha256": sha_file(probe),
               "message_frame_source_sha256": sha_file(frame_source), "system": platform.system(),
               "machine": platform.machine(), "os_release": platform.release(), "dispatch": "arm_neon"}
    result = {"encoder_id": digest(binding), "binding": binding}
    validate_identity(result)
    return result


def validate_identity(identity):
    if not isinstance(identity, dict) or set(identity) != {"encoder_id", "binding"}:
        raise ValueError("incomplete encoder identity")
    binding = identity["binding"]
    keys = {"policy", "backbone_sha256", "binary_sha256", "message_frame_source_sha256", "system", "machine", "os_release", "dispatch"}
    if not isinstance(binding, dict) or set(binding) != keys: raise ValueError("incomplete numeric domain")
    if binding["policy"] != POLICY or binding["backbone_sha256"] != BACKBONE_SHA256:
        raise ValueError("unsupported feature policy/backbone")
    for field in ("binary_sha256", "message_frame_source_sha256"):
        if not isinstance(binding[field], str) or not re.fullmatch(r"[0-9a-f]{64}", binding[field]):
            raise ValueError("invalid implementation fingerprint")
    if (binding["system"], binding["machine"], binding["dispatch"]) != ("Darwin", "arm64", "arm_neon") or not isinstance(binding["os_release"], str) or not binding["os_release"]:
        raise ValueError("unqualified extraction backend")
    if identity["encoder_id"] != digest(binding): raise ValueError("encoder binding hash mismatch")


def model_binding(identity):
    """Bridge to research MemoryState/ReadContext identity checks, not a model export."""
    validate_identity(identity)
    binding = identity["binding"]
    return ModelBinding(binding["backbone_sha256"],
                        digest({"gguf": binding["backbone_sha256"], "tokenizer_binary": binding["binary_sha256"]}),
                        identity["encoder_id"], digest({"policy": binding["policy"], "framing": binding["message_frame_source_sha256"]}))


@dataclass(frozen=True)
class EncodedText:
    token_ids: np.ndarray
    features: np.ndarray

    def validate(self):
        if self.token_ids.dtype != np.dtype("<i4") or self.token_ids.ndim != 1 or not 2 <= len(self.token_ids) < 512:
            raise ValueError("invalid token geometry/dtype")
        if (self.token_ids < 0).any() or (self.token_ids >= VOCAB).any(): raise ValueError("token outside bound vocabulary")
        if self.features.dtype != np.dtype("<f4") or self.features.shape != (len(self.token_ids) - 1, 2 * HIDDEN):
            raise ValueError("invalid feature geometry/dtype")
        if not np.isfinite(self.features).all(): raise ValueError("nonfinite features")


def read_encoded(path, count):
    def read_exact(stream, length):
        data = stream.read(length)
        if len(data) != length: raise ValueError("truncated encoder output")
        return data
    with Path(path).open("rb") as stream:
        if read_exact(stream, 8) != b"BNEN0001": raise ValueError("wrong encoder output version")
        if struct.unpack("<III", read_exact(stream, 12)) != (count, HIDDEN, LAYERS):
            raise ValueError("encoder output geometry mismatch")
        rows = []
        for _ in range(count):
            n = struct.unpack("<I", read_exact(stream, 4))[0]
            if not 2 <= n < 512: raise ValueError("invalid encoded length")
            ids = np.frombuffer(read_exact(stream, 4 * n), dtype="<i4").copy()
            features = np.frombuffer(read_exact(stream, 4 * (n - 1) * HIDDEN * 2), dtype="<f4").copy().reshape(n - 1, HIDDEN * 2)
            row = EncodedText(ids, features); row.validate(); rows.append(row)
        if stream.read(): raise ValueError("trailing encoder data")
    return rows


class NativeMemoryEncoder:
    def __init__(self, gguf, probe, expected_encoder_id=None):
        self.gguf, self.probe = str(Path(gguf).resolve()), str(Path(probe).resolve())
        self.identity = encoder_identity(self.gguf, self.probe)
        if expected_encoder_id is not None and self.identity["encoder_id"] != expected_encoder_id:
            raise ValueError("encoder identity mismatch")

    def encode(self, texts, audit_dir, batch_size=32):
        if not texts or not 1 <= batch_size <= 1000: raise ValueError("invalid extraction batch")
        for text in texts: text_key(text)
        if encoder_identity(self.gguf, self.probe) != self.identity: raise ValueError("encoder changed before extraction")
        root = Path(audit_dir); root.mkdir(parents=True, exist_ok=False)
        # Ambient overrides must not silently alter the declared numeric backend.
        env = {k: v for k, v in os.environ.items() if not k.startswith("BITNET_")}
        env.update(BITNET_NUM_THREADS="4", BITNET_QUIET="0", BITNET_METAL_I2S="0", BITNET_METAL_OUTPUT="0")
        rows, files = [], {}
        for batch, start in enumerate(range(0, len(texts), batch_size)):
            group = texts[start:start + batch_size]
            source, target, log_path = (root / f"batch-{batch}.{suffix}" for suffix in ("input", "bin", "log"))
            with source.open("xb") as stream:
                stream.write(struct.pack("<I", len(group)))
                for text in group:
                    raw = text.encode(); stream.write(struct.pack("<I", len(raw))); stream.write(raw)
            with log_path.open("x") as log:
                subprocess.run([self.probe, self.gguf, str(source), str(target), str(LAYERS), "reader-v1"],
                               check=True, stdout=log, stderr=log, timeout=180, env=env)
            if "[bitnet] cpu tier: arm_neon" not in log_path.read_text(): raise ValueError("unverified CPU dispatch")
            rows.extend(read_encoded(target, len(group)))
            for path in (source, target, log_path): files[path.name] = sha_file(path)
        if encoder_identity(self.gguf, self.probe) != self.identity: raise ValueError("encoder changed during extraction")
        manifest = {"identity": self.identity, "input_sha256": [text_key(t) for t in texts], "file_sha256": files,
                    "count": len(rows), "batch_size": batch_size}
        with (root / "manifest.json").open("x") as stream: json.dump(manifest, stream, indent=2); stream.write("\n")
        return rows


def collect_texts(records):
    texts = []
    seen = set()
    for record in records:
        query, episodes = encoder_texts(record)
        for text in (query,) + episodes:
            key = text_key(text)
            if key not in seen: texts.append(text); seen.add(key)
    return texts


def save_bank(path, identity, texts, rows):
    if len(texts) != len(rows) or not rows or len({text_key(t) for t in texts}) != len(texts):
        raise ValueError("feature bank requires unique complete inputs")
    validate_identity(identity)
    arrays, entries = {}, {}
    for i, (text, row) in enumerate(zip(texts, rows)):
        row.validate()
        arrays[f"ids_{i}"] = row.token_ids; arrays[f"features_{i}"] = row.features
        entries[text_key(text)] = {"index": i, "text": text}
    root = Path(path); root.mkdir(parents=True, exist_ok=False)
    with (root / "features.npz").open("xb") as stream: np.savez(stream, **arrays)
    manifest = {"format": "native-reader-bank-v1", "identity": identity, "entries": entries,
                "features_sha256": sha_file(root / "features.npz")}
    with (root / "manifest.json").open("x") as stream: json.dump(manifest, stream, ensure_ascii=False, indent=2); stream.write("\n")
    return digest(manifest)


class NativeFeatureBank:
    """Offline artifact reader. No implicit extraction, answer lookup or fallbacks."""
    def __init__(self, path, *, expected_encoder_id, expected_manifest_sha256):
        root = Path(path)
        if {p.name for p in root.iterdir()} != {"manifest.json", "features.npz"}: raise ValueError("incomplete or mixed feature artifact")
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["format"] != "native-reader-bank-v1" or digest(manifest) != expected_manifest_sha256:
            raise ValueError("feature manifest mismatch")
        identity = manifest["identity"]
        validate_identity(identity)
        if identity["encoder_id"] != expected_encoder_id or digest(identity["binding"]) != expected_encoder_id:
            raise ValueError("feature encoder identity mismatch")
        if sha_file(root / "features.npz") != manifest["features_sha256"]: raise ValueError("feature archive corruption")
        self.rows = {}
        with np.load(root / "features.npz", allow_pickle=False) as arrays:
            expected_keys = set()
            for key, entry in manifest["entries"].items():
                if text_key(entry["text"]) != key: raise ValueError("input hash mismatch")
                i = entry["index"]
                if type(i) is not int or i < 0: raise ValueError("invalid array index")
                names = {f"ids_{i}", f"features_{i}"}
                if names & expected_keys: raise ValueError("duplicate array index")
                expected_keys.update(names)
                row = EncodedText(arrays[f"ids_{i}"].copy(), arrays[f"features_{i}"].copy())
                row.validate(); self.rows[key] = row
            if set(arrays.files) != expected_keys or not self.rows: raise ValueError("unexpected feature arrays")
        self.encoder_id = expected_encoder_id
        self.model_binding = model_binding(identity)

    def __call__(self, text):
        key = text_key(text)
        if key not in self.rows: raise ValueError("input absent from offline feature artifact; encode new input explicitly")
        return torch.from_numpy(self.rows[key].features.copy())
