"""DG-003: native continuous output correction and ordinary gradient checks."""
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
from memory_fusion import GatedMemoryFusion
from continuous_memory import continuous_logits
from native_memory_encoder import NativeMemoryEncoder, BACKBONE_SHA256, sha_file
from diagnose_native_generation_gradient import exact, native_forward, nll
from diagnose_neural_memory_features import compare
from train_memory_fusion import prompt


def forward(probe, gguf, ids, delta, directory, name, enabled=True):
    if delta.dtype != np.dtype("<f4") or delta.shape != (1024,) or not np.isfinite(delta).all():
        raise ValueError("invalid memory residual")
    root = Path(directory)
    source, target, log = (root / (name + suffix) for suffix in (".input", ".bin", ".log"))
    with source.open("xb") as stream:
        stream.write(b"BNCI0001" + struct.pack("<I", len(ids)) + np.asarray(ids, dtype="<i4").tobytes())
        stream.write(struct.pack("<I", int(enabled)) + delta.tobytes())
    env = {k: v for k, v in os.environ.items() if not k.startswith("BITNET_")}
    env.update(BITNET_NUM_THREADS="4", BITNET_METAL_I2S="0", BITNET_METAL_OUTPUT="0")
    with log.open("x") as stream:
        subprocess.run([probe, gguf, str(source), str(target)], check=True, stdout=stream, stderr=stream, env=env, timeout=120)
    data = target.read_bytes()
    if len(data) != 20 + (1024 + 3 * 73448) * 4 or data[:8] != b"BNCO0001": raise ValueError("invalid continuous output")
    if struct.unpack_from("<III", data, 8) != (len(ids), 1024, 73448): raise ValueError("continuous output geometry mismatch")
    values = np.frombuffer(data, dtype="<f4", offset=20).copy()
    if not np.isfinite(values).all(): raise ValueError("nonfinite continuous output")
    return {"hidden": values[:1024], "base": values[1024:1024+73448],
            "correction": values[1024+73448:1024+2*73448], "logits": values[1024+2*73448:]}


def token_rank(logits, target):
    return int((logits > logits[target]).sum()) + 1


def run(args):
    config_path = Path(args.experiment); config = json.loads(config_path.read_text())
    parent_path = config_path.with_name("DG-001.json"); parent = json.loads(parent_path.read_text())
    if config["id"] != "DG-003" or sha_file(args.gguf) != BACKBONE_SHA256: raise ValueError("wrong diagnostic/backbone")
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    encoder = NativeMemoryEncoder(args.gguf, args.encoder_probe)
    sources = encoder.encode([c["source"] for c in parent["cases"]], root / "sources")
    weights = GGUFWeights(args.gguf, args.lib); tokenizer = CTokenizer(args.tok_probe, args.gguf)
    try:
        output_tensor = "output.weight"
        try:
            output_array = weights.get_f32(output_tensor, (73448, 1024))
        except RuntimeError as error:
            if "rc=-2" not in str(error): raise
            output_tensor = "token_embd.weight"
            output_array = weights.get_f32(output_tensor, (73448, 1024))
        output_weight = torch.from_numpy(output_array.copy())
        logit_scale = weights.logit_scale; weights.close()
        cases = []
        for ci, case in enumerate(parent["cases"]):
            directory = root / case["id"]; directory.mkdir()
            ids = tokenizer.encode(prompt(case["query"]), True)
            targets = tokenizer.encode(case["next_token_text"], False)
            target = next(t for t, piece in zip(targets, tokenizer.decode_pieces(targets)) if piece.strip())
            zero = np.zeros(1024, dtype="<f4")
            reference = native_forward(args.reference_probe, args.gguf, ids, zero, directory, "reference", False)
            disabled = forward(args.probe, args.gguf, ids, np.ones(1024, dtype="<f4"), directory, "disabled", False)
            empty = forward(args.probe, args.gguf, ids, zero, directory, "zero")
            zero_exact = all(exact(r["logits"], reference["logits"]) and exact(r["hidden"], reference["hidden"]) for r in (disabled, empty))
            if not zero_exact: raise ValueError("disabled/zero differs from ordinary native C")
            torch.manual_seed(config["seed"] + ci)
            fusion = GatedMemoryFusion(1024, 24, 8)
            with torch.no_grad(): fusion.layer_gain[23] = config["initial_fusion_gain"]
            hidden = torch.from_numpy(empty["hidden"].copy())[None, :]
            base = torch.from_numpy(empty["base"].copy())[None, :]
            memory = torch.from_numpy(sources[ci].features.copy())
            def delta(): return fusion.residual(23, hidden, fusion.prepare(memory))
            parity = []
            def evaluate_delta(value, name):
                actual = forward(args.probe, args.gguf, ids, value.detach().numpy()[0], directory, name)
                if not exact(actual["base"], empty["base"]): raise ValueError("base changed with memory branch")
                with torch.no_grad(): expected, adjustment = continuous_logits(base, value, output_weight, logit_scale)
                tol = config["forward_tolerance"]
                checks = {key: compare(actual[key], ref.detach().numpy()[0], **tol) for key, ref in (("logits", expected), ("correction", adjustment))}
                parity.append({"name": name, **checks})
                return actual
            initial = delta()
            native = evaluate_delta(initial, "base")
            predicted, _ = continuous_logits(base, initial, output_weight, logit_scale)
            F.cross_entropy(predicted.double(), torch.tensor([target])).backward()
            gain_gradient = float(fusion.layer_gain.grad[23])
            projection_gradient = fusion.output.weight.grad.detach().clone()
            gradient_norm = float(projection_gradient.norm())
            if not np.isfinite(gain_gradient) or gain_gradient == 0 or not np.isfinite(gradient_norm) or gradient_norm <= 0:
                raise ValueError("invalid continuous gradient")
            base_loss, base_rank = nll(native["logits"], target), token_rank(native["logits"], target)
            gain_checks = []
            for pi, step in enumerate(config["gain_perturbations"]):
                values = {}
                try:
                    for sign, label in ((-1, "minus"), (1, "plus")):
                        with torch.no_grad(): fusion.layer_gain[23] = config["initial_fusion_gain"] + sign * step
                        values[label] = nll(evaluate_delta(delta(), f"gain-{pi}-{label}")["logits"], target)
                finally:
                    with torch.no_grad(): fusion.layer_gain[23] = config["initial_fusion_gain"]
                difference = (values["plus"] - values["minus"]) / (2 * step)
                tol = config["gain_derivative_tolerance"]
                matched = abs(difference - gain_gradient) <= tol["atol"] + tol["rtol"] * abs(gain_gradient) and np.sign(difference) == np.sign(gain_gradient)
                gain_checks.append({"step": step, **values, "finite_difference": difference, "passed": bool(matched)})
            original = fusion.output.weight.detach().clone()
            direction = projection_gradient / gradient_norm
            steps = []
            for pi, step in enumerate(config["projection_gradient_steps"]):
                values = {}
                try:
                    for sign, label in ((-1, "descent"), (1, "opposed")):
                        with torch.no_grad(): fusion.output.weight.copy_(original + sign * step * direction)
                        actual = evaluate_delta(delta(), f"weight-{pi}-{label}")
                        values[label] = {"nll": nll(actual["logits"], target), "target_rank": token_rank(actual["logits"], target),
                                         "top1": int(actual["logits"].argmax())}
                finally:
                    with torch.no_grad(): fusion.output.weight.copy_(original)
                steps.append({"step": step, **values, "direction_passed": values["descent"]["nll"] < base_loss - 1e-6 and values["opposed"]["nll"] > base_loss + 1e-6,
                              "rank_improved": values["descent"]["target_rank"] < base_rank and values["descent"]["nll"] < base_loss - 1e-6})
            result = {"id": case["id"], "target_id": target, "zero_disabled_exact": zero_exact, "base_nll": base_loss, "base_target_rank": base_rank,
                      "gain_gradient": gain_gradient, "projection_gradient_norm": gradient_norm, "gain_checks": gain_checks, "projection_steps": steps,
                      "forward_parity": parity, "all_forward_checks_passed": all(v[k]["allclose"] for v in parity for k in ("logits", "correction")),
                      "all_gain_checks_passed": all(g["passed"] for g in gain_checks),
                      "direction_gate_passed": sum(s["direction_passed"] for s in steps) >= 3, "rank_gate_passed": any(s["rank_improved"] for s in steps)}
            cases.append(result)
            print(json.dumps({k: result[k] for k in ("id", "base_target_rank", "all_forward_checks_passed", "all_gain_checks_passed", "direction_gate_passed", "rank_gate_passed")}), flush=True)
        repo = Path(__file__).resolve().parents[1]
        files = ["python/continuous_memory.py", "python/diagnose_continuous_memory.py", "python/memory_fusion.py", "tools/memory_continuous_probe.c",
                 "src/bitnet.c", "src/ops.c", "src/quant_q6k.c", "CMakeLists.txt"]
        report = {"experiment": config, "parent_experiment_sha256": sha_file(parent_path), "cases": cases,
                  "frozen_output_tensor": output_tensor, "logit_scale": logit_scale,
                  "numeric_gate_passed": all(all(c[k] for k in ("zero_disabled_exact", "all_forward_checks_passed", "all_gain_checks_passed", "direction_gate_passed", "rank_gate_passed")) for c in cases),
                  "memory_accuracy_measured": False, "optimizer_steps": 0, "training_approved": False, "encoder_identity": encoder.identity,
                  "source_sha256": {p: sha_file(repo / p) for p in files},
                  "artifact_sha256": {p: sha_file(p) for p in (args.probe, args.reference_probe, args.lib, args.tok_probe, args.experiment)},
                  "raw_sha256": {str(p.relative_to(root)): sha_file(p) for p in root.rglob("*") if p.is_file()}}
        with (root / "summary.json").open("x") as stream: json.dump(report, stream, ensure_ascii=False, indent=2); stream.write("\n")
        print(json.dumps({"numeric_gate_passed": report["numeric_gate_passed"], "training_approved": False}), flush=True)
    finally:
        weights.close(); tokenizer._proc.terminate(); tokenizer._proc.wait()
        tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "gguf", "probe", "reference-probe", "encoder-probe", "lib", "tok-probe", "output"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())


if __name__ == "__main__": main()
