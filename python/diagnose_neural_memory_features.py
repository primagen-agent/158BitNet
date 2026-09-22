"""Fresh-input, exact-backbone feature-domain audit. No memory accuracy claim."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import struct
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from c_tokenizer import CTokenizer
from ggw import GGUFWeights
from torch_backbone import TorchBackbone
import torch_backbone as backbone_module

BACKBONE_SHA256 = "44cb4e0db8374d4247bba391b3d3b7c0b3ee815c36cf2e20d0d1a3570677bd95"


def c_q8_activation_reference(value):
    """Diagnostic NEON round-to-nearest-even reference, not a training STE."""
    maximum = value.abs().amax(dim=-1, keepdim=True)
    safe = maximum.clamp_min(torch.finfo(value.dtype).tiny)
    quantized = (value * (127.0 / safe)).clamp(-127, 127).round()
    return torch.where(maximum > 0, quantized * (maximum / 127.0), torch.zeros_like(value))


def file_hash(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def compare(native, reference, atol, rtol):
    if native.shape != reference.shape or not np.isfinite(native).all() or not np.isfinite(reference).all():
        raise ValueError("invalid feature geometry or nonfinite data")
    error = np.abs(native.astype(np.float64) - reference)
    cosine = np.sum(native.astype(np.float64) * reference, axis=-1) / np.maximum(
        np.linalg.norm(native.astype(np.float64), axis=-1) * np.linalg.norm(reference.astype(np.float64), axis=-1), 1e-30)
    return {"max_abs": float(error.max()), "mean_abs": float(error.mean()),
            "relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(reference.astype(np.float64)), 1e-30)),
            "minimum_row_cosine": float(cosine.min()),
            "outside_tolerance": int((error > atol + rtol * np.abs(reference)).sum()),
            "elements": int(error.size), "allclose": bool(np.allclose(native, reference, atol=atol, rtol=rtol))}


def read_native(path, expected_count, hidden, layers):
    with Path(path).open("rb") as stream:
        if stream.read(8) != b"BNFP0001": raise ValueError("wrong probe format")
        if struct.unpack("<III", stream.read(12)) != (expected_count, hidden, layers): raise ValueError("probe geometry mismatch")
        rows = []
        for _ in range(expected_count):
            n = struct.unpack("<I", stream.read(4))[0]
            if not 1 < n < 512: raise ValueError("invalid token count")
            ids = np.frombuffer(stream.read(4 * n), dtype="<i4").copy()
            matrices = []
            for width in (hidden, hidden, hidden * layers):
                matrices.append(np.frombuffer(stream.read(4 * n * width), dtype="<f4").copy().reshape(n, width))
            rows.append((ids, *matrices))
        if stream.read(): raise ValueError("trailing probe bytes")
    return rows


def run(args):
    if file_hash(args.gguf) != BACKBONE_SHA256: raise ValueError("exact 0.5B GGUF required")
    config = json.loads(Path(args.config).read_text())
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    weights = GGUFWeights(args.gguf, args.lib)
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    original_rope = backbone_module.rope_tables
    cases, torch_arrays = [], {}
    try:
        hidden, layers = weights.hidden, weights.n_layers
        with (root / "inputs.bin").open("xb") as stream:
            stream.write(struct.pack("<I", len(config["texts"])))
            for row in config["texts"]:
                data = row["text"].encode(); stream.write(struct.pack("<I", len(data))); stream.write(data)
        with (root / "native.log").open("w") as log:
            subprocess.run([args.probe, args.gguf, str(root / "inputs.bin"), str(root / "native.bin"), str(layers)],
                           env={**os.environ, "BITNET_NUM_THREADS": "4"}, stdout=log, stderr=log, check=True, timeout=180)
        native = read_native(root / "native.bin", len(config["texts"]), hidden, layers)
        backbone = TorchBackbone(weights, device="cpu", dtype=torch.float32).eval()
        if args.control in ("fp32-rope", "q8-and-fp32-rope"):
            backbone_module.rope_tables = lambda cfg, positions, device, dtype: original_rope(cfg, positions, device, torch.float32)
        if args.control in ("q8-activation", "q8-and-fp32-rope"):
            backbone.linear = lambda h, w: F.linear(c_q8_activation_reference(h), w)
        weights.close()
        for row, (ids, lexical, output, bands) in zip(config["texts"], native):
            encoded = tokenizer.encode(row["text"], True)
            if encoded != ids.tolist(): raise ValueError("C/Torch token ID mismatch")
            tokens = torch.tensor(encoded)
            with torch.inference_mode():
                h = backbone(tokens, return_hidden=True)
                residuals = backbone(tokens, return_hidden_layers=tuple(range(layers))).reshape(len(ids), layers, hidden)
                normalized = residuals / residuals.float().pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
                lex = backbone.token_embd[tokens]
                features = torch.cat((F.normalize(h, dim=-1)[1:], F.normalize(lex, dim=-1)[1:]), -1)
            torch_arrays[row["id"] + "_hidden"] = h.numpy()
            torch_arrays[row["id"] + "_lexical"] = lex.numpy()
            torch_arrays[row["id"] + "_bands"] = normalized.numpy()
            cfeatures = np.concatenate((output[1:] / np.maximum(np.linalg.norm(output[1:], axis=-1, keepdims=True), 1e-12),
                                        lexical[1:] / np.maximum(np.linalg.norm(lexical[1:], axis=-1, keepdims=True), 1e-12)), axis=-1)
            cmp = lambda x, y: compare(x, y, config["atol"], config["rtol"])
            record = {"id": row["id"], "text": row["text"], "token_ids": encoded, "bos_included_in_probe": True,
                      "lexical": cmp(lexical, lex.numpy()), "output_hidden": cmp(output, h.numpy()),
                      "memory_features_without_bos": cmp(cfeatures, features.numpy()),
                      "layers": [cmp(bands.reshape(len(ids), layers, hidden)[:, i], normalized.numpy()[:, i]) for i in range(layers)]}
            cases.append(record)
            print(json.dumps({"id": row["id"], "tokens": len(ids), "hidden": record["output_hidden"]}), flush=True)
        repeat_equal = all(np.array_equal(a, b) for a, b in zip(native[0], native[-1])) if config["texts"][0]["text"] == config["texts"][-1]["text"] else None
        repo = Path(__file__).resolve().parents[1]
        source_paths = ["python/diagnose_neural_memory_features.py", "python/torch_backbone.py", "python/ggw.py", "python/ggwshim.c",
                        "python/c_tokenizer.py", "tools/memory_feature_probe.c", "src/bitnet.c", "src/ops.c", "src/tokenizer.c", "src/quant_tq2_0.c", "CMakeLists.txt"]
        report = {"purpose": config["purpose"], "backbone_sha256": BACKBONE_SHA256, "torch_version": torch.__version__,
                  "diagnostic_control": args.control, "platform": platform.platform(),
                  "device": "cpu", "dtype": "float32", "atol": config["atol"], "rtol": config["rtol"],
                  "torch_kv_cache": False, "c_cross_input_kv_reuse": False, "c_internal_prefill_kv_buffers": True,
                  "c_repeated_input_exact": repeat_equal, "token_ids_exact": True, "cases": cases,
                  "all_features_within_tolerance": all(c["lexical"]["allclose"] and c["output_hidden"]["allclose"] and all(l["allclose"] for l in c["layers"]) for c in cases),
                  "source_sha256": {p: file_hash(repo / p) for p in source_paths},
                  "artifact_sha256": {p: file_hash(p) for p in (args.probe, args.tok_probe, args.lib, args.config)},
                  "cmake_cache_sha256": file_hash(Path(args.probe).parent / "CMakeCache.txt"),
                  "memory_accuracy_measured": False}
        np.savez(root / "torch_features.npz", **torch_arrays)
        report["raw_sha256"] = {p: file_hash(root / p) for p in ("inputs.bin", "native.bin", "torch_features.npz")}
        (root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"all_features_within_tolerance": report["all_features_within_tolerance"], "c_repeated_input_exact": repeat_equal}), flush=True)
    finally:
        backbone_module.rope_tables = original_rope
        weights.close(); tokenizer._proc.terminate(); tokenizer._proc.wait()
        tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gguf", "config", "output", "probe", "tok-probe", "lib"): parser.add_argument("--" + name, required=True)
    parser.add_argument("--control", choices=("none", "fp32-rope", "q8-activation", "q8-and-fp32-rope"), default="none")
    run(parser.parse_args())


if __name__ == "__main__": main()
