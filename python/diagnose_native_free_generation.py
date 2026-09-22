"""DG-004 fresh-prefix generation certification, not memory accuracy."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from c_tokenizer import CTokenizer
from memory_fusion import GatedMemoryFusion
from native_continuous_generation import NativeContinuousBackend, tokenizer_stop_ids
from native_memory_encoder import BACKBONE_SHA256, sha_file
from neural_memory_generation import LEGACY_TEMPLATE_VERSION, GenerationInput, encode_generation_input, greedy_generate


def byte_replays(tokenizer, stop_ids):
    """Actual tokenizer bytes with scripted logits; NOT natural model outputs."""
    tokens = tuple(tokenizer.encode("你", False))
    if not tokens or any(t in stop_ids for t in tokens): raise ValueError("invalid real UTF-8 fixture")
    pieces = tokenizer.decode_pieces(list(range(73448)))
    partial = next((i for i, piece in enumerate(pieces) if piece == b"\xe4" and i not in stop_ids), None)
    if partial is None: raise ValueError("missing real partial UTF-8 byte token")
    request = GenerationInput((tokenizer.bos(), tokens[0]), ())
    def replay(sequence, limit):
        count = 0
        def backend(prefix, sources):
            nonlocal count
            logits = np.zeros(73448, dtype=np.float32)
            logits[sequence[count]] = 1
            count += 1
            return logits
        return greedy_generate(request, tokenizer, backend, stop_ids=stop_ids,
                               max_new_tokens=limit, context_capacity=128)
    stops = [replay(tokens + (stop,), len(tokens) + 2) for stop in stop_ids]
    clipped = replay((partial,), 1)
    passed = (all(r["generated_token_ids"] == list(tokens) + [s] and r["finish_reason"] == "stop_token"
                  and r["utf8_complete"] and r["raw_text_hex"] == b"".join(pieces[t] for t in tokens).hex()
                  for r, s in zip(stops, stop_ids)) and clipped["truncated"] and not clipped["utf8_complete"])
    return {"mode": "scripted_logits_real_token_bytes_not_model_generation", "stops": stops,
            "partial_utf8": clipped, "passed": passed}


def run(args):
    config = json.loads(Path(args.experiment).read_text())
    if config["id"] != "DG-004" or config["backbone_sha256"] != BACKBONE_SHA256:
        raise ValueError("wrong diagnostic registration")
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    torch.set_num_threads(4)
    results = []
    try:
        stops = tokenizer_stop_ids(tokenizer)
        replays = byte_replays(tokenizer, stops)
        for ci, case in enumerate(config["cases"]):
            torch.manual_seed(config["seed"] + ci)
            fusion = GatedMemoryFusion(1024, 24, 8)
            case_dir = root / case["id"]; case_dir.mkdir()
            torch.save(fusion.state_dict(), case_dir / "untrained-fusion.pt")
            runtime = {"id": case["id"], "context": [{"role": "user", "speaker": "user", "text": case["query"]}],
                       "episodes": [{"role": "user", "speaker": "user", "text": case["source"]}]}
            # DG-004 was registered before the DG-005 template discovery.
            request = encode_generation_input(runtime, tokenizer, template_version=LEGACY_TEMPLATE_VERSION)
            records = {}
            for condition in config["conditions"]:
                with torch.no_grad(): fusion.layer_gain[23] = 0 if condition == "zero_gain" else config["initial_gain"]
                current = (GenerationInput(request.prompt_token_ids, ()) if condition == "empty" else request)
                backend = NativeContinuousBackend(gguf=args.gguf, probe=args.probe, reference_probe=args.reference_probe,
                            encoder_probe=args.encoder_probe, fusion=fusion, output=case_dir / condition, condition=condition)
                output = greedy_generate(current, tokenizer, backend, stop_ids=stops,
                            max_new_tokens=config["max_new_tokens"], context_capacity=config["context_capacity"],
                            memory_enabled=condition not in ("reference", "disabled"))
                trace = backend.finish()
                # Only this audited caller may qualify its concrete backend run;
                # the generic transport deliberately does not trust arbitrary callables.
                output["backend_cache_policy_verified"] = True
                output["backend_evidence"] = str((case_dir / condition / "backend.json").relative_to(root))
                output["internal_prefill_kv_buffers"] = True
                with (case_dir / condition / "generation.json").open("x") as stream:
                    json.dump(output, stream, ensure_ascii=False, indent=2); stream.write("\n")
                records[condition] = {"generation": output, "steps": len(trace["steps"]),
                                      "every_base_reference_checked": all(s["base_reference_checked"] for s in trace["steps"]),
                                      "every_output_equals_base": all(s["output_equals_base"] for s in trace["steps"]),
                                      "every_residual_nonzero": all(s["neural_residual_nonzero"] for s in trace["steps"])}
                print(json.dumps({"case": case["id"], "condition": condition, "tokens": len(output["generated_token_ids"]),
                                  "finish_reason": output["finish_reason"], "text": output["text"]}, ensure_ascii=False), flush=True)
            reference = records["reference"]["generation"]
            equality_fields = ("text", "generated_token_ids", "visible_token_ids", "raw_text_hex", "utf8_complete", "finish_reason", "truncated")
            identical = all(all(records[c]["generation"][k] == reference[k] for k in equality_fields)
                            and records[c]["every_output_equals_base"] for c in ("disabled", "empty", "zero_gain"))
            active = records["active_untrained"]
            passed = (identical and all(r["every_base_reference_checked"] for r in records.values())
                      and active["every_residual_nonzero"])
            results.append({"id": case["id"], "conditions": records, "disabled_empty_zero_identical": identical,
                            "backend_gate_passed": passed})
        repo = Path(__file__).resolve().parents[1]
        paths = [p for base in ("src", "include") for p in (repo / base).rglob("*") if p.suffix in (".c", ".h", ".mm")]
        paths += [repo / p for p in ("CMakeLists.txt", "tools/memory_continuous_probe.c", "tools/memory_gradient_probe.c",
                  "tools/memory_feature_probe.c", "python/native_continuous_generation.py", "python/diagnose_native_free_generation.py",
                  "python/neural_memory_generation.py", "python/memory_fusion.py", "python/native_memory_encoder.py",
                  "python/diagnose_continuous_memory.py", "python/diagnose_native_generation_gradient.py", "python/c_tokenizer.py")]
        report = {"experiment": config, "cases": results, "byte_replays": replays, "stop_token_ids": stops,
                  "backend_gate_passed": replays["passed"] and all(c["backend_gate_passed"] for c in results),
                  "native_generation_forward_calls": sum((1 if name == "reference" else 3) * item["steps"]
                                                for case in results for name, item in case["conditions"].items()),
                  "source_encoder_calls": 2 * len(results),
                  "internal_prefill_kv_buffers": True, "cross_step_kv_reuse": False,
                  "optimizer_steps": 0, "training_approved": False, "memory_accuracy_measured": False,
                  "source_sha256": {str(p.relative_to(repo)): sha_file(p) for p in paths},
                  "artifact_sha256": {str(p): sha_file(p) for p in (args.gguf, args.probe, args.reference_probe,
                                                                  args.encoder_probe, args.tok_probe, args.experiment)},
                  "raw_sha256": {str(p.relative_to(root)): sha_file(p) for p in root.rglob("*") if p.is_file()}}
        with (root / "summary.json").open("x") as stream: json.dump(report, stream, ensure_ascii=False, indent=2); stream.write("\n")
        print(json.dumps({"backend_gate_passed": report["backend_gate_passed"], "training_approved": False}), flush=True)
    finally:
        tokenizer._proc.terminate(); tokenizer._proc.wait()
        tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "gguf", "probe", "reference-probe", "encoder-probe", "tok-probe", "output"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())


if __name__ == "__main__": main()
