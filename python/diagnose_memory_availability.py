"""DG-006 numeric and fresh C v2 generation checks. No optimizer steps."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from c_tokenizer import CTokenizer
from continuous_memory import continuous_logits
from diagnose_continuous_memory import forward, token_rank
from diagnose_native_generation_gradient import exact, native_forward, nll
from diagnose_neural_memory_features import compare
from ggw import GGUFWeights
from memory_availability import AvailabilityMemoryFusion, STATES
from native_continuous_generation import NativeContinuousBackend, check_trace, tokenizer_stop_ids
from native_memory_encoder import NativeMemoryEncoder, BACKBONE_SHA256, sha_file
from neural_memory_generation import TEMPLATE_VERSION, encode_generation_input, greedy_generate


def run(args):
    config = json.loads(Path(args.experiment).read_text())
    if config["id"] != "DG-006" or config["template_version"] != TEMPLATE_VERSION or sha_file(args.gguf) != BACKBONE_SHA256:
        raise ValueError("wrong experiment/template/backbone")
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    encoder = NativeMemoryEncoder(args.gguf, args.encoder_probe)
    weights = GGUFWeights(args.gguf, args.lib)
    torch.set_num_threads(4)
    cases = []
    try:
        # This exact GGUF has a tied F16 output head; no guessed cross-model head.
        output_weight = torch.from_numpy(weights.get_f32("token_embd.weight", (73448, 1024)).copy())
        logit_scale = weights.logit_scale; weights.close()
        stops = tokenizer_stop_ids(tokenizer)
        for ci, case in enumerate(config["cases"]):
            directory = root / case["id"]; directory.mkdir()
            runtime = {"id": case["id"], "context": [{"role": "user", "speaker": "user", "text": case["query"]}],
                       "episodes": [{"role": "user", "speaker": "user", "text": s} for s in case["sources"]]}
            request = encode_generation_input(runtime, tokenizer)
            rows = encoder.encode(list(request.source_texts), directory / "sources") if request.source_texts else []
            features = torch.from_numpy(rows[0].features.copy()) if rows else torch.empty(0, 2048)
            zero = np.zeros(1024, dtype="<f4")
            reference = native_forward(args.reference_probe, args.gguf, request.prompt_token_ids, zero, directory, "reference", False)
            disabled = forward(args.probe, args.gguf, request.prompt_token_ids, np.ones_like(zero), directory, "disabled", False)
            check_trace(directory / "disabled.log", len(request.prompt_token_ids))
            disabled_exact = exact(reference["logits"], disabled["logits"]) and exact(reference["hidden"], disabled["hidden"])
            torch.manual_seed(config["seed"] + ci)
            module = AvailabilityMemoryFusion(1024, 24, 8)
            hidden = torch.from_numpy(reference["hidden"].copy())[None, :]
            base = torch.from_numpy(reference["logits"].copy())[None, :]
            # Targets only enter a diagnostic loss after the label-free forward.
            target_tokens = tokenizer.encode(case["next_token_text"], False)
            target = next(t for t, piece in zip(target_tokens, tokenizer.decode_pieces(target_tokens)) if piece.strip())
            def read(): return module.inspect(23, hidden, module.prepare(features))
            initial = read()
            initial_native = forward(args.probe, args.gguf, request.prompt_token_ids, initial.residual.detach().numpy()[0], directory, "zero-init")
            check_trace(directory / "zero-init.log", len(request.prompt_token_ids))
            zero_exact = exact(initial_native["logits"], reference["logits"])
            logits, _ = continuous_logits(base, initial.residual, output_weight, logit_scale)
            F.cross_entropy(logits.double(), torch.tensor([target])).backward()
            initial_gradient = float(module.uncertainty_output.weight.grad.norm())
            module.zero_grad(set_to_none=True)
            with torch.no_grad():
                module.content.layer_gain[23] = config["content_gain"]
                module.uncertainty_output.weight.normal_(std=config["diagnostic_uncertainty_weight_std"])
            torch.save(module.state_dict(), directory / "untrained-module.pt")
            parity = []
            def evaluate(value, name):
                native = forward(args.probe, args.gguf, request.prompt_token_ids, value.detach().numpy()[0], directory, name)
                check_trace(directory / (name + ".log"), len(request.prompt_token_ids))
                if not exact(native["base"], reference["logits"]): raise ValueError("native baseline changed")
                expected, correction = continuous_logits(base, value, output_weight, logit_scale)
                parity.append({"name": name,
                    "logits": compare(native["logits"], expected.detach().numpy()[0], **config["forward_tolerance"]),
                    "correction": compare(native["correction"], correction.detach().numpy()[0], **config["forward_tolerance"])})
                return {"nll": nll(native["logits"], target), "target_rank": token_rank(native["logits"], target)}
            actual = read()
            baseline = evaluate(actual.residual, "active")
            predicted, _ = continuous_logits(base, actual.residual, output_weight, logit_scale)
            F.cross_entropy(predicted.double(), torch.tensor([target])).backward()
            parameter = module.uncertainty_output.weight
            gradient = parameter.grad.detach().clone()
            norm = float(gradient.norm())
            if not np.isfinite(norm) or norm <= 0: raise ValueError("missing uncertainty generation gradient")
            direction, original = gradient / norm, parameter.detach().clone()
            comparisons = []
            for index, step in enumerate([config["finite_difference_step"]] + config["projection_gradient_steps"]):
                values = {}
                try:
                    for sign, name in ((-1, "descent"), (1, "opposed")):
                        with torch.no_grad(): parameter.copy_(original + sign * step * direction)
                        values[name] = evaluate(read().residual, f"perturb-{index}-{name}")
                finally:
                    with torch.no_grad(): parameter.copy_(original)
                derivative = (values["opposed"]["nll"] - values["descent"]["nll"]) / (2 * step)
                tol = config["derivative_tolerance"]
                comparisons.append({"step": step, **values, "finite_difference": derivative,
                    "derivative_matches": abs(derivative - norm) <= tol["atol"] + tol["rtol"] * norm,
                    "direction_passed": values["descent"]["nll"] < baseline["nll"] - 1e-6 and values["opposed"]["nll"] > baseline["nll"] + 1e-6})
            if not torch.equal(parameter.detach(), original): raise ValueError("parameter perturbation was not restored")
            backend = NativeContinuousBackend(gguf=args.gguf, probe=args.probe, reference_probe=args.reference_probe,
                encoder_probe=args.encoder_probe, fusion=module, output=directory / "generation",
                condition="availability_active" if rows else "availability_empty")
            generation = greedy_generate(request, tokenizer, backend, stop_ids=stops,
                         max_new_tokens=config["max_new_tokens"], context_capacity=config["context_capacity"])
            backend_report = backend.finish()
            generation["backend_cache_policy_verified"] = True
            with (directory / "generation.json").open("x") as stream: json.dump(generation, stream, ensure_ascii=False, indent=2); stream.write("\n")
            passed = (disabled_exact and zero_exact and np.isfinite(initial_gradient) and initial_gradient > 0
                      and all(c[k]["allclose"] for c in parity for k in ("logits", "correction"))
                      and comparisons[0]["derivative_matches"] and all(c["direction_passed"] for c in comparisons[1:])
                      and all(s["base_reference_checked"] for s in backend_report["steps"]))
            content_zero = not bool(actual.content_residual.detach().count_nonzero())
            if not rows: passed = passed and content_zero
            result = {"id": case["id"], "template_version": TEMPLATE_VERSION, "supplied_episode_count": len(rows),
                      "target_token_id": target, "disabled_exact": disabled_exact, "zero_initialization_exact": zero_exact,
                      "content_residual_zero": content_zero, "initial_uncertainty_gradient_norm": initial_gradient,
                      "state_probabilities": dict(zip(STATES, actual.state_probabilities.detach().numpy()[0].tolist())),
                      "gradient_norm": norm, "baseline": baseline, "comparisons": comparisons, "parity": parity,
                      "generation": generation, "diagnostic_passed": bool(passed)}
            cases.append(result)
            print(json.dumps({k: result[k] for k in ("id", "diagnostic_passed", "state_probabilities", "generation")}, ensure_ascii=False), flush=True)
        repo = Path(__file__).resolve().parents[1]
        paths = [p for base_dir in ("src", "include") for p in (repo / base_dir).rglob("*") if p.suffix in (".c", ".h", ".mm")]
        paths += [repo / p for p in ("CMakeLists.txt", "tools/memory_continuous_probe.c", "tools/memory_gradient_probe.c", "tools/memory_feature_probe.c",
            "python/memory_availability.py", "python/diagnose_memory_availability.py", "python/native_continuous_generation.py", "python/native_memory_encoder.py",
            "python/neural_memory_generation.py", "python/memory_fusion.py", "python/continuous_memory.py", "python/diagnose_continuous_memory.py",
            "python/diagnose_native_generation_gradient.py", "python/c_tokenizer.py", "python/ggw.py")]
        report = {"experiment": config, "cases": cases, "diagnostic_passed": all(c["diagnostic_passed"] for c in cases),
                  "optimizer_steps": 0, "training_approved": False, "memory_accuracy_measured": False,
                  "source_sha256": {str(p.relative_to(repo)): sha_file(p) for p in paths},
                  "artifact_sha256": {p: sha_file(p) for p in (args.gguf, args.probe, args.reference_probe, args.encoder_probe, args.tok_probe, args.lib, args.experiment)},
                  "raw_sha256": {str(p.relative_to(root)): sha_file(p) for p in root.rglob("*") if p.is_file()}}
        with (root / "summary.json").open("x") as stream: json.dump(report, stream, ensure_ascii=False, indent=2); stream.write("\n")
        print(json.dumps({"diagnostic_passed": report["diagnostic_passed"], "training_approved": False}), flush=True)
    finally:
        weights.close(); tokenizer._proc.terminate(); tokenizer._proc.wait()
        tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "gguf", "probe", "reference-probe", "encoder-probe", "tok-probe", "lib", "output"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())


if __name__ == "__main__": main()
