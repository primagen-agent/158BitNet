"""Localize first-layer differences by replaying actual C inputs per operator."""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from diagnose_neural_memory_features import BACKBONE_SHA256, c_q8_activation_reference, compare, file_hash, read_native
from ggw import GGUFWeights
from torch_backbone import TorchBackbone, rms_norm, rope_tables, apply_rope_hf

STAGES = {1: "input", 2: "attn_norm", 3: "q", 4: "k", 5: "v", 6: "q_rope", 7: "k_rope", 8: "attention",
          9: "attn_out", 10: "attn_residual", 11: "ffn_norm", 12: "gate", 13: "up", 14: "activation",
          15: "down", 16: "output", 17: "attn_quant", 18: "ffn_quant"}


def read_trace(path, counts):
    result = [{} for _ in counts]
    with Path(path).open("rb") as stream:
        if stream.read(8) != b"BNTR0001": raise ValueError("wrong trace format")
        while header := stream.read(16):
            if len(header) != 16: raise ValueError("truncated trace header")
            case, stage, token, size = struct.unpack("<IIII", header)
            if case >= len(counts) or stage not in STAGES or token >= counts[case] or not 0 < size <= 65536:
                raise ValueError("invalid trace coordinates")
            data = stream.read(size * 4)
            if len(data) != size * 4: raise ValueError("truncated trace data")
            values = np.frombuffer(data, dtype="<f4").copy()
            name = STAGES[stage]
            slots = result[case].setdefault(name, {})
            if token in slots: raise ValueError("duplicate trace record")
            slots[token] = values
    for case, count in zip(result, counts):
        if set(case) != set(STAGES.values()): raise ValueError("incomplete operator trace")
        for name, slots in case.items():
            if set(slots) != set(range(count)): raise ValueError("missing trace token")
            case[name] = torch.from_numpy(np.stack([slots[i] for i in range(count)]))
            if not torch.isfinite(case[name]).all(): raise ValueError("nonfinite trace")
    return result


def to_interleaved(x, head_dim):
    shaped = x.reshape(len(x), -1, 2, head_dim // 2)
    return shaped.transpose(-2, -1).reshape_as(x)


def to_split(x, head_dim):
    shaped = x.reshape(len(x), -1, head_dim)
    return torch.cat((shaped[..., ::2], shaped[..., 1::2]), -1).reshape_as(x)


def replay(backbone, c, quantized=False):
    cfg, L = backbone.cfg, backbone.layers[0]
    T, hd = len(c["input"]), cfg.head_dim
    cos, sin = rope_tables(cfg, torch.arange(T), "cpu", torch.float32)
    matmul = lambda x, w: F.linear(c_q8_activation_reference(x) if quantized else x, w)
    rotate = lambda x: to_interleaved(apply_rope_hf(to_split(x, hd).reshape(T, -1, hd).transpose(0, 1), cos, sin).transpose(0, 1).reshape_as(x), hd)
    q = to_split(c["q_rope"], hd).reshape(T, cfg.n_heads, hd).transpose(0, 1)
    k = to_split(c["k_rope"], hd).reshape(T, cfg.n_kv_heads, hd).transpose(0, 1)
    v = c["v"].reshape(T, cfg.n_kv_heads, hd).transpose(0, 1)
    repeat = cfg.n_heads // cfg.n_kv_heads
    attention = F.scaled_dot_product_attention(q, k.repeat_interleave(repeat, 0), v.repeat_interleave(repeat, 0), is_causal=True,
                                               scale=1 / math.sqrt(hd)).transpose(0, 1).reshape(T, cfg.q_dim)
    return {"attn_norm": rms_norm(c["input"], L["attn_norm"], cfg.rms_eps),
            "attn_quant": c_q8_activation_reference(c["attn_norm"]), "ffn_quant": c_q8_activation_reference(c["ffn_norm"]),
            "q": to_interleaved(matmul(c["attn_norm"], L["q"]), hd),
            "k": to_interleaved(matmul(c["attn_norm"], L["k"]), hd), "v": matmul(c["attn_norm"], L["v"]),
            "q_rope": rotate(c["q"]), "k_rope": rotate(c["k"]), "attention": attention,
            "attn_out": matmul(c["attention"], L["o"]),
            "attn_residual": c["input"] + c["attn_out"] * cfg.residual_scale,
            "ffn_norm": rms_norm(c["attn_residual"], L["ffn_norm"], cfg.rms_eps),
            "gate": matmul(c["ffn_norm"], L["gate"]), "up": matmul(c["ffn_norm"], L["up"]),
            "activation": F.silu(c["gate"]) * c["up"], "down": matmul(c["activation"], L["down"]),
            "output": c["attn_residual"] + c["down"] * cfg.residual_scale}


def first_layer_chain(backbone, inputs):
    """Expose accumulated q8-reference differences, not just local C-input replay."""
    cfg, L = backbone.cfg, backbone.layers[0]
    T, hd = len(inputs), cfg.head_dim
    c = {"input": inputs}
    linear = lambda x, w: F.linear(c_q8_activation_reference(x), w)
    c["attn_norm"] = rms_norm(inputs, L["attn_norm"], cfg.rms_eps)
    c["attn_quant"] = c_q8_activation_reference(c["attn_norm"])
    for key in ("q", "k", "v"):
        value = linear(c["attn_norm"], L[key])
        c[key] = to_interleaved(value, hd) if key != "v" else value
    cos, sin = rope_tables(cfg, torch.arange(T), "cpu", torch.float32)
    for key in ("q", "k"):
        value = to_split(c[key], hd).reshape(T, -1, hd).transpose(0, 1)
        value = apply_rope_hf(value, cos, sin).transpose(0, 1).reshape(T, -1)
        c[key + "_rope"] = to_interleaved(value, hd)
    q, k = (to_split(c[key], hd).reshape(T, -1, hd).transpose(0, 1) for key in ("q_rope", "k_rope"))
    v = c["v"].reshape(T, -1, hd).transpose(0, 1)
    repeat = cfg.n_heads // cfg.n_kv_heads
    c["attention"] = F.scaled_dot_product_attention(q, k.repeat_interleave(repeat, 0), v.repeat_interleave(repeat, 0),
        is_causal=True, scale=1 / math.sqrt(hd)).transpose(0, 1).reshape(T, cfg.q_dim)
    c["attn_out"] = linear(c["attention"], L["o"])
    c["attn_residual"] = inputs + c["attn_out"] * cfg.residual_scale
    c["ffn_norm"] = rms_norm(c["attn_residual"], L["ffn_norm"], cfg.rms_eps)
    c["ffn_quant"] = c_q8_activation_reference(c["ffn_norm"])
    c["gate"], c["up"] = (linear(c["ffn_norm"], L[key]) for key in ("gate", "up"))
    c["activation"] = F.silu(c["gate"]) * c["up"]
    c["down"] = linear(c["activation"], L["down"])
    c["output"] = c["attn_residual"] + c["down"] * cfg.residual_scale
    return c


def quantization_decision_changes(native, reference):
    def codes(x):
        maximum = x.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
        return (x * (127 / maximum)).clamp(-127, 127).round()
    changes = codes(native) != codes(reference)
    return {"changed_codes": int(changes.sum()), "total_codes": changes.numel()}


def run(args):
    if file_hash(args.gguf) != BACKBONE_SHA256: raise ValueError("exact backbone required")
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    with (output / "inputs.bin").open("xb") as stream:
        stream.write(struct.pack("<I", len(config["texts"])))
        for row in config["texts"]:
            data = row["text"].encode(); stream.write(struct.pack("<I", len(data))); stream.write(data)
    weights = GGUFWeights(args.gguf, args.lib)
    try:
        hidden, layers = weights.hidden, weights.n_layers
        for binary, name, extra in ((args.probe, "observed.bin", [str(output / "trace.bin")]), (args.reference_probe, "reference.bin", [])):
            with (output / (name + ".log")).open("w") as log:
                subprocess.run([binary, args.gguf, str(output / "inputs.bin"), str(output / name), str(layers)] + extra,
                               check=True, stdout=log, stderr=log, timeout=180, env={**os.environ, "BITNET_NUM_THREADS": "4"})
        if (output / "observed.bin").read_bytes() != (output / "reference.bin").read_bytes():
            raise ValueError("instrumentation changed C output")
        native = read_native(output / "reference.bin", len(config["texts"]), hidden, layers)
        traces = read_trace(output / "trace.bin", [len(row[0]) for row in native])
        backbone = TorchBackbone(weights, device="cpu", dtype=torch.float32).eval()
        weights.close()
        cases = []
        for text, c, native_row in zip(config["texts"], traces, native):
            with torch.inference_mode():
                dense, q8 = replay(backbone, c, False), replay(backbone, c, True)
                chain = first_layer_chain(backbone, c["input"])
                original_linear = backbone.linear
                try:
                    backbone.linear = lambda h, w: F.linear(c_q8_activation_reference(h), w)
                    actual = backbone(torch.tensor(native_row[0].astype(np.int64)), return_hidden_layer=0)
                finally: backbone.linear = original_linear
                if not torch.equal(actual, chain["output"]): raise ValueError("diagnostic chain differs from actual Torch forward")
            compare_case = lambda result: {key: compare(c[key].numpy(), value.numpy(), config["atol"], config["rtol"]) for key, value in result.items()}
            cases.append({"id": text["id"], "dense_on_c_inputs": compare_case(dense), "q8_on_c_inputs": compare_case(q8),
                          "q8_accumulated_chain": compare_case(chain),
                          "quantization_decision_changes": {k: quantization_decision_changes(c[k], chain[k])
                                                             for k in ("attn_norm", "attention", "ffn_norm", "activation")},
                          "diagnostic_chain_matches_torch_forward": True})
            print(json.dumps({"id": text["id"], "q8_relative_l2": {k: v["relative_l2"] for k, v in cases[-1]["q8_on_c_inputs"].items()}}), flush=True)
        repo = Path(__file__).resolve().parents[1]
        sources = ["python/diagnose_neural_memory_operators.py", "python/diagnose_neural_memory_features.py", "python/torch_backbone.py",
                   "python/ggw.py", "python/ggwshim.c", "tools/memory_operator_probe.c", "tools/memory_feature_probe.c",
                   "src/bitnet.c", "src/ops.c", "src/quant_tq2_0.c", "CMakeLists.txt"]
        report = {"purpose": "first_layer_operator_diagnostic", "backbone_sha256": BACKBONE_SHA256,
                  "instrumentation_output_exact": True, "memory_accuracy_measured": False, "input_count": len(cases),
                  "atol": config["atol"], "rtol": config["rtol"], "cases": cases,
                  "source_sha256": {p: file_hash(repo / p) for p in sources},
                  "artifact_sha256": {p: file_hash(p) for p in (args.probe, args.reference_probe, args.lib, args.config)},
                  "raw_sha256": {p: file_hash(output / p) for p in ("inputs.bin", "trace.bin", "reference.bin", "observed.bin")}}
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    finally: weights.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gguf", "config", "lib", "probe", "reference-probe", "output"): parser.add_argument("--" + name, required=True)
    run(parser.parse_args())


if __name__ == "__main__": main()
