"""Freeze causal controls and audit real C-tokenizer generation boundaries.

This command never loads a neural memory model or generates answers.
"""
import argparse
import json
from pathlib import Path

from c_tokenizer import CTokenizer
from native_memory_encoder import BACKBONE_SHA256, sha_file
from neural_memory_causality import paired_protocol
from neural_memory_generation import TEMPLATE_VERSION, encode_generation_input, generation_prompt
from prepare_neural_memory_protocol import digest, materialize
from review_neural_memory import read_rows


def run(args):
    manifest = materialize(args.config, args.corpus, verify=True)
    if sha_file(args.gguf) != BACKBONE_SHA256:
        raise ValueError("exact 0.5B backbone required")
    if type(args.max_new_tokens) is not int or not 0 < args.max_new_tokens < 128:
        raise ValueError("diagnostic capacity is 128 tokens")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    inputs, labels, index = ([row for split in ("train", "dev")
                             for row in read_rows(Path(args.corpus) / f"{split}.{kind}.jsonl")]
                            for kind in ("inputs", "labels", "index"))
    protocol = paired_protocol(inputs, labels, index)
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    requests, records = {}, []
    try:
        for runtime in inputs:
            request = encode_generation_input(runtime, tokenizer)
            # This checks capacity; it must not silently truncate a prompt or source.
            if len(request.prompt_token_ids) + args.max_new_tokens > 128:
                raise ValueError("free-generation prefix exceeds diagnostic capacity")
            if any(len(tokenizer.encode(s, True)) > 511 for s in request.source_texts):
                raise ValueError("source exceeds native encoder capacity")
            requests[runtime["id"]] = request
            records.append({"id": runtime["id"], "input_sha256": digest(runtime),
                            "prompt_sha256": digest(generation_prompt(runtime)),
                            "prompt_token_ids": request.prompt_token_ids,
                            "source_text_sha256": [digest(s) for s in request.source_texts]})
        for pair in protocol["pairs"]:
            a, b = (requests[pair[k]] for k in ("left_id", "right_id"))
            if a.prompt_token_ids != b.prompt_token_ids or a.source_texts == b.source_texts:
                raise ValueError("memory intervention leaked into generation prefix")
        source_paths = ("python/neural_memory_generation.py", "python/neural_memory_causality.py",
                        "python/audit_neural_memory_generation.py", "python/c_tokenizer.py",
                        "python/episode_memory_inputs.py", "python/review_neural_memory.py",
                        "python/neural_memory_protocol.py", "python/prepare_neural_memory_protocol.py")
        repo = Path(__file__).resolve().parents[1]
        report = {"format": "neural-memory-generation-audit-v2", "template_version": TEMPLATE_VERSION, "input_count": len(inputs),
                  "causal_pair_count": protocol["pair_count"], "independent_worlds": protocol["independent_worlds"],
                  "pairs_by_split": {s: sum(p["split"] == s for p in protocol["pairs"]) for s in ("train", "dev")},
                  "max_prompt_tokens": max(len(r.prompt_token_ids) for r in requests.values()),
                  "reserved_generation_tokens": args.max_new_tokens,
                  "all_paired_prefix_tokens_identical": True, "all_paired_sources_differ": True,
                  "corpus_manifest_sha256": digest(manifest), "protocol_sha256": digest(protocol),
                  "token_record_sha256": digest(records), "backbone_sha256": BACKBONE_SHA256,
                  "tokenizer_binary_sha256": sha_file(args.tok_probe),
                  "source_sha256": {p: sha_file(repo / p) for p in source_paths},
                  "memory_accuracy_measured": False, "model_inference_calls": 0, "optimizer_steps": 0,
                  "training_approved": False, "promotion_eligible": False}
        for name, value in (("pairs.json", protocol), ("token-records.json", records), ("summary.json", report)):
            with (root / name).open("x") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2); stream.write("\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        tokenizer._proc.terminate(); tokenizer._proc.wait()
        tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "corpus", "gguf", "tok-probe", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    run(parser.parse_args())


if __name__ == "__main__": main()
