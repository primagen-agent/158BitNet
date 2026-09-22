"""DG-002: separate fixed-code local derivatives from native code transitions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from c_tokenizer import CTokenizer
from ggw import GGUFWeights
from torch_backbone import TorchBackbone
from train_memory_fusion import prompt
from memory_fusion import GatedMemoryFusion
from native_memory_encoder import NativeMemoryEncoder, BACKBONE_SHA256, sha_file
from diagnose_native_generation_gradient import native_forward, nll, exact
from native_cell_gradient import read_cell, cell_changes, cell_local_suffix


def rank(logits, target):
    return {"target_rank": int((logits > logits[target]).sum()) + 1, "top1_token": int(logits.argmax())}


def run(args):
    experiment_path = Path(args.experiment)
    experiment = json.loads(experiment_path.read_text())
    parent_path = experiment_path.with_name("DG-001.json")
    parent = json.loads(parent_path.read_text())
    if experiment["id"] != "DG-002" or parent["id"] != "DG-001" or sha_file(args.gguf) != BACKBONE_SHA256:
        raise ValueError("wrong registered experiment/backbone")
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    encoder = NativeMemoryEncoder(args.gguf, args.encoder_probe)
    sources = encoder.encode([case["source"] for case in parent["cases"]], root / "sources")
    weights = GGUFWeights(args.gguf, args.lib); tokenizer = CTokenizer(args.tok_probe, args.gguf)
    try:
        backbone = TorchBackbone(weights, device="cpu", dtype=torch.float32).eval(); weights.close()
        results = []
        for ci, case in enumerate(parent["cases"]):
            directory = root / case["id"]; directory.mkdir()
            ids = tokenizer.encode(prompt(case["query"]), True)
            tokens = tokenizer.encode(case["next_token_text"], False)
            target = next(t for t, p in zip(tokens, tokenizer.decode_pieces(tokens)) if p.strip())
            zero = np.zeros(1024, dtype="<f4")
            disabled = native_forward(args.probe, args.gguf, ids, zero, directory, "disabled", False)
            reference = native_forward(args.reference_probe, args.gguf, ids, zero, directory, "reference", False)
            if not all(exact(disabled[k], reference[k]) for k in ("hidden", "logits")):
                raise ValueError("probe no longer matches ordinary C")
            torch.manual_seed(experiment["seed"] + ci)
            fusion = GatedMemoryFusion(1024, 24, 8)
            with torch.no_grad(): fusion.layer_gain[23] = experiment["initial_fusion_gain"]
            before = torch.from_numpy(disabled["before"].copy())[None, :]
            memory = torch.from_numpy(sources[ci].features.copy())
            def delta(): return fusion(23, before, fusion.prepare(memory)) - before
            base_delta = delta()
            plain = native_forward(args.probe, args.gguf, ids, base_delta.detach().numpy()[0], directory, "plain")
            native = native_forward(args.probe, args.gguf, ids, base_delta.detach().numpy()[0], directory, "base", cell_trace=True)
            if not all(exact(native[k], plain[k]) for k in native): raise ValueError("cell observation changed native values")
            cell = read_cell(directory / "base.cell")
            logits = cell_local_suffix(backbone, before + base_delta, native, cell)
            if not exact(logits.detach().numpy(), native["logits"]): raise ValueError("local bridge changed native logits")
            F.cross_entropy(logits.double()[None, :], torch.tensor([target])).backward()
            gradient = float(fusion.layer_gain.grad[23]); gradient_finite = bool(np.isfinite(gradient) and gradient != 0)
            if not gradient_finite: raise ValueError("invalid local gradient")
            base_loss = nll(native["logits"], target)
            comparisons, accepted = [], None
            for pi, step in enumerate(experiment["gain_perturbations"]):
                observations = {}
                try:
                    for sign, label in ((-1, "minus"), (1, "plus")):
                        with torch.no_grad(): fusion.layer_gain[23] = experiment["initial_fusion_gain"] + sign * step
                        name = f"step-{pi}-{label}"
                        value = native_forward(args.probe, args.gguf, ids, delta().detach().numpy()[0], directory, name, cell_trace=True)
                        trace = read_cell(directory / (name + ".cell"))
                        status = cell_changes(cell, trace, native["hidden"], value["hidden"])
                        scale_ratio = trace["quantizers"]["output_input"]["scale"] / cell["quantizers"]["output_input"]["scale"]
                        proportional = float(np.max(np.abs(value["logits"].astype(np.float64) - native["logits"].astype(np.float64) * scale_ratio)))
                        observations[label] = {"nll": nll(value["logits"], target), **status, **rank(value["logits"], target),
                                               "output_scale_ratio": scale_ratio, "logit_proportionality_max_abs": proportional}
                finally:
                    with torch.no_grad(): fusion.layer_gain[23] = experiment["initial_fusion_gain"]
                stable = observations["minus"]["same_cell"] and observations["plus"]["same_cell"]
                difference = (observations["plus"]["nll"] - observations["minus"]["nll"]) / (2 * step)
                tolerance = experiment["local_derivative_tolerance"]
                match = abs(difference - gradient) <= tolerance["atol"] + tolerance["rtol"] * abs(gradient) and np.sign(difference) == np.sign(gradient)
                direction = "minus" if gradient > 0 else "plus"
                improves = observations[direction]["nll"] < base_loss - 1e-6
                if accepted is None and improves:
                    accepted = {"step": step, "direction": direction, "same_cell": observations[direction]["same_cell"],
                                **rank(native["logits"], target), "candidate": observations[direction]}
                comparisons.append({"step": step, "same_cell_pair": stable, "finite_difference": difference,
                                    "local_derivative_matches": bool(match) if stable else None,
                                    "proposed_direction": direction, "actual_loss_improves": improves, **observations})
                print(json.dumps({"case": case["id"], "step": step, "same_cell": stable, "local_gradient": gradient,
                                  "finite_difference": difference, "improves": improves}), flush=True)
            stable = [c for c in comparisons if c["same_cell_pair"]]
            sufficient = len(stable) >= 2
            passed = sufficient and all(c["local_derivative_matches"] for c in stable)
            results.append({"id": case["id"], "target_id": target, "base_nll": base_loss, **rank(native["logits"], target),
                            "local_gradient": gradient, "finite_nonzero_gradient": gradient_finite,
                            "same_cell_pairs": len(stable), "all_pairs": len(comparisons), "sufficient_local_pairs": sufficient,
                            "local_gate_passed": passed, "accepted_diagnostic_step": accepted, "comparisons": comparisons})
        repo = Path(__file__).resolve().parents[1]
        files = ["python/native_cell_gradient.py", "python/diagnose_native_cell_gradient.py", "python/diagnose_native_generation_gradient.py",
                 "python/native_generation_bridge.py", "python/memory_fusion.py", "src/bitnet.c", "src/ops.c", "tools/memory_gradient_probe.c"]
        report = {"experiment": experiment, "parent_experiment_sha256": sha_file(parent_path), "cases": results,
                  "local_gate_passed": all(c["local_gate_passed"] for c in results), "native_forward_exact": True,
                  "memory_accuracy_measured": False, "optimizer_steps": 0, "training_approved": False,
                  "encoder_identity": encoder.identity, "source_sha256": {p: sha_file(repo / p) for p in files},
                  "artifact_sha256": {p: sha_file(p) for p in (args.probe, args.reference_probe, args.lib, args.tok_probe, args.experiment)},
                  "raw_sha256": {str(p.relative_to(root)): sha_file(p) for p in root.rglob("*") if p.is_file()}}
        with (root / "summary.json").open("x") as stream: json.dump(report, stream, ensure_ascii=False, indent=2); stream.write("\n")
        print(json.dumps({"local_gate_passed": report["local_gate_passed"], "training_approved": False}), flush=True)
    finally:
        weights.close(); tokenizer._proc.terminate(); tokenizer._proc.wait()
        tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "gguf", "probe", "reference-probe", "encoder-probe", "lib", "tok-probe", "output"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())


if __name__ == "__main__": main()
