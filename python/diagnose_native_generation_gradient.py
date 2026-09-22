"""DG-001: last-position native forward / surrogate-gradient numeric controls."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import torch
from torch.nn import functional as F

from c_tokenizer import CTokenizer
from ggw import GGUFWeights
from torch_backbone import TorchBackbone
from train_memory_fusion import prompt
from memory_fusion import GatedMemoryFusion
from native_memory_encoder import NativeMemoryEncoder, BACKBONE_SHA256, sha_file
from native_generation_bridge import native_value_with_surrogate_gradient, suffix_surrogate


def exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def native_forward(probe, gguf, ids, delta, output_dir, name, enabled=True, cell_trace=False):
    if delta.shape != (1024,) or delta.dtype != np.dtype("<f4") or not np.isfinite(delta).all(): raise ValueError("invalid diagnostic residual")
    root = Path(output_dir)
    source, target, log = (root / (name + suffix) for suffix in (".input", ".bin", ".log"))
    with source.open("xb") as stream:
        stream.write(b"BNGI0001" + struct.pack("<I", len(ids)) + np.asarray(ids, dtype="<i4").tobytes())
        stream.write(struct.pack("<III", 23, len(ids) - 1, int(enabled)) + delta.tobytes())
    env = {k: v for k, v in os.environ.items() if not k.startswith("BITNET_")}
    env.update(BITNET_NUM_THREADS="4", BITNET_METAL_I2S="0", BITNET_METAL_OUTPUT="0", BITNET_QUIET="0")
    with log.open("x") as stream:
        trace_args = [str(root / (name + ".cell"))] if cell_trace else []
        subprocess.run([probe, gguf, str(source), str(target)] + trace_args, check=True, stdout=stream, stderr=stream, timeout=120, env=env)
    data = target.read_bytes()
    if data[:8] != b"BNGO0001" or len(data) != 28 + 4 * (3 * 1024 + 73448): raise ValueError("invalid gradient probe output")
    if struct.unpack_from("<IIIII", data, 8) != (len(ids), 1024, 73448, 23, len(ids) - 1): raise ValueError("gradient probe geometry mismatch")
    values = np.frombuffer(data, dtype="<f4", offset=28).copy()
    if not np.isfinite(values).all(): raise ValueError("nonfinite native forward")
    return {"before": values[:1024], "after": values[1024:2048], "hidden": values[2048:3072], "logits": values[3072:]}


def nll(logits, target):
    return float(F.cross_entropy(torch.from_numpy(logits).double()[None, :], torch.tensor([target])))


def run(args):
    experiment = json.loads(Path(args.experiment).read_text())
    if experiment["id"] != "DG-001" or sha_file(args.gguf) != BACKBONE_SHA256: raise ValueError("wrong experiment/backbone")
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(experiment["seed"])
    encoder = NativeMemoryEncoder(args.gguf, args.encoder_probe)
    sources = encoder.encode([case["source"] for case in experiment["cases"]], root / "sources")
    weights = GGUFWeights(args.gguf, args.lib)
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    try:
        backbone = TorchBackbone(weights, device="cpu", dtype=torch.float32).eval()
        weights.close()
        results = []
        for ci, case in enumerate(experiment["cases"]):
            case_dir = root / case["id"]; case_dir.mkdir()
            ids = tokenizer.encode(prompt(case["query"]), True)
            target_tokens = tokenizer.encode(case["next_token_text"], False)
            pieces = tokenizer.decode_pieces(target_tokens)
            target = next(token for token, piece in zip(target_tokens, pieces) if piece.strip())
            zero = np.zeros(1024, dtype="<f4")
            disabled = native_forward(args.probe, args.gguf, ids, zero, case_dir, "disabled", False)
            no_memory = native_forward(args.probe, args.gguf, ids, zero, case_dir, "zero")
            reference = native_forward(args.reference_probe, args.gguf, ids, zero, case_dir, "reference", False)
            zero_exact = all(exact(disabled[k], no_memory[k]) and exact(no_memory[k], reference[k]) for k in ("hidden", "logits"))
            if not zero_exact: raise ValueError("disabled/zero instrumentation changed ordinary native forward")
            torch.manual_seed(experiment["seed"] + ci)
            fusion = GatedMemoryFusion(1024, 24, 8)
            with torch.no_grad(): fusion.layer_gain[23] = experiment["initial_fusion_gain"]
            memory = torch.from_numpy(sources[ci].features.copy())
            before = torch.from_numpy(disabled["before"].copy())[None, :]
            def residual():
                return fusion(23, before, fusion.prepare(memory)) - before
            base_delta = residual()
            native = native_forward(args.probe, args.gguf, ids, base_delta.detach().numpy()[0], case_dir, "base")
            if not exact(native["before"], disabled["before"]): raise ValueError("prefix changed without memory")
            changed = not exact(native["logits"], no_memory["logits"])
            base_loss = nll(native["logits"], target)
            variants = []
            for variant in experiment["backward_variants"]:
                fusion.zero_grad(set_to_none=True)
                delta = residual()
                proxy = suffix_surrogate(backbone, before + delta, variant)[0]
                bridged = native_value_with_surrogate_gradient(torch.from_numpy(native["logits"]), proxy)
                if not exact(bridged.detach().numpy(), native["logits"]): raise ValueError("bridge changed native logit bits")
                loss = F.cross_entropy(bridged.double()[None, :], torch.tensor([target])); loss.backward()
                gradient = float(fusion.layer_gain.grad[23])
                gradients = {name: float(p.grad.abs().sum()) if p.grad is not None else 0. for name, p in fusion.named_parameters()}
                finite_nonzero = bool(np.isfinite(gradient) and gradient != 0 and all(np.isfinite(v) for v in gradients.values()) and gradients["encoder.weight"] > 0)
                comparisons = []
                for pi, step in enumerate(experiment["gain_perturbations"]):
                    values = {}
                    try:
                        for sign, label in ((-1, "descent"), (1, "opposed")):
                            with torch.no_grad(): fusion.layer_gain[23] = experiment["initial_fusion_gain"] + sign * np.sign(gradient) * step
                            value = native_forward(args.probe, args.gguf, ids, residual().detach().numpy()[0], case_dir, f"{variant}-{pi}-{label}")
                            values[label] = nll(value["logits"], target)
                    finally:
                        with torch.no_grad(): fusion.layer_gain[23] = experiment["initial_fusion_gain"]
                    comparisons.append({"step": step, **values, "passes_direction_check": values["descent"] < base_loss - 1e-6 and values["opposed"] > base_loss + 1e-6,
                                        "central_difference_along_gradient_sign": (values["opposed"] - values["descent"]) / (2 * step)})
                passed = sum(c["passes_direction_check"] for c in comparisons)
                variants.append({"variant": variant, "native_value_exact": True, "gain_surrogate_gradient": gradient,
                                 "parameter_gradient_l1": gradients, "finite_nonzero_gradient": finite_nonzero,
                                 "directions_passed": passed, "directions_total": len(comparisons), "comparisons": comparisons,
                                 "gradient_gate_passed": finite_nonzero and passed >= 3})
                print(json.dumps({"case": case["id"], "variant": variant, "surrogate_gradient": gradient, "directions_passed": passed, "total": len(comparisons)}), flush=True)
            results.append({"id": case["id"], "prompt": prompt(case["query"]), "target_id": target, "prefix_ids": ids,
                            "zero_disabled_reference_exact": zero_exact, "nonzero_memory_changes_logits": changed, "base_nll": base_loss,
                            "variants": variants})
        repo = Path(__file__).resolve().parents[1]
        source_files = ["python/native_generation_bridge.py", "python/diagnose_native_generation_gradient.py", "python/torch_backbone.py", "python/memory_fusion.py",
                        "tools/memory_gradient_probe.c", "src/bitnet.c", "src/ops.c", "CMakeLists.txt"]
        report = {"experiment": experiment, "cases": results, "memory_accuracy_measured": False, "optimizer_steps": 0,
                  "forward_gate_passed": all(r["zero_disabled_reference_exact"] and r["nonzero_memory_changes_logits"] for r in results),
                  "gradient_gate_passed": all(v["gradient_gate_passed"] for r in results for v in r["variants"]),
                  "encoder_identity": encoder.identity,
                  "artifact_sha256": {p: sha_file(p) for p in (args.probe, args.reference_probe, args.tok_probe, args.lib, args.experiment)},
                  "source_sha256": {p: sha_file(repo / p) for p in source_files},
                  "raw_sha256": {str(p.relative_to(root)): sha_file(p) for p in root.rglob("*") if p.is_file()}}
        with (root / "summary.json").open("x") as stream: json.dump(report, stream, ensure_ascii=False, indent=2); stream.write("\n")
        print(json.dumps({k: report[k] for k in ("forward_gate_passed", "gradient_gate_passed", "optimizer_steps")}), flush=True)
    finally:
        weights.close(); tokenizer._proc.terminate(); tokenizer._proc.wait()
        tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "gguf", "probe", "reference-probe", "encoder-probe", "lib", "tok-probe", "output"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())


if __name__ == "__main__": main()
