"""Verify frozen pilot data and C token boundaries, without model inference."""
import argparse
import json
from pathlib import Path

from availability_supervision import encode_reply_targets, teacher_forcing_requests
from c_tokenizer import CTokenizer
from native_memory_encoder import BACKBONE_SHA256, sha_file
from neural_memory_generation import TEMPLATE_VERSION, encode_generation_input
from prepare_availability_curriculum import materialize
from prepare_neural_memory_protocol import digest
from review_neural_memory import read_rows


def run(args):
    corpus = Path(args.corpus)
    manifest = materialize(args.config, corpus, verify=True)
    if sha_file(args.gguf) != BACKBONE_SHA256: raise ValueError("bound 0.5B GGUF required")
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    records, requests = [], {}
    try:
        for split in ("train", "dev"):
            labels = {r["id"]: r for r in read_rows(corpus/f"{split}.labels.jsonl")}
            for runtime in read_rows(corpus/f"{split}.inputs.jsonl"):
                request = encode_generation_input(runtime, tokenizer)
                target = encode_reply_targets(runtime, labels[runtime["id"]], tokenizer)
                calls = list(teacher_forcing_requests(request, target))
                if calls[0] != request or any(r.source_texts != request.source_texts for r in calls):
                    raise ValueError("training wrapper changed source/input boundary")
                if any(len(tokenizer.encode(source, True)) > 511 for source in request.source_texts):
                    raise ValueError("source encoder capacity exceeded")
                requests[runtime["id"]] = request
                isolated = tuple(tokenizer.encode(target.response, False))
                records.append({"id": runtime["id"], "split": split, "input_sha256": digest(runtime),
                                "prompt_ids": target.prompt_token_ids, "completion_ids": target.completion_token_ids,
                                "state_index": target.state_index, "sample_weight": target.sample_weight,
                                "isolated_reply_tokens_differ": isolated != target.completion_token_ids[:-1]})
        pairs = read_rows(corpus/"pairs.jsonl")
        for pair in pairs:
            a, b = (requests[pair[k]] for k in ("left_id", "right_id"))
            if a.prompt_token_ids != b.prompt_token_ids or a.source_texts == b.source_texts:
                raise ValueError("causal pair changed prompt or failed to change source")
        source_files = ("python/availability_supervision.py", "python/audit_availability_supervision.py",
                        "python/neural_memory_generation.py", "python/prepare_availability_curriculum.py", "python/c_tokenizer.py")
        repo = Path(__file__).resolve().parents[1]
        report = {"format": "availability-supervision-audit-v1", "template_version": TEMPLATE_VERSION,
                  "cases": len(records), "pair_count": len(pairs), "max_prompt_tokens": max(len(r["prompt_ids"]) for r in records),
                  "max_completion_tokens": max(len(r["completion_ids"]) for r in records),
                  "max_joint_tokens": max(len(r["prompt_ids"])+len(r["completion_ids"]) for r in records),
                  "isolated_reply_tokenization_differs_count": sum(r["isolated_reply_tokens_differ"] for r in records),
                  "joint_prefix_and_reply_bytes_verified": True, "teacher_forcing_for_training_only": True,
                  "corpus_manifest_sha256": digest(manifest), "records_sha256": digest(records),
                  "backbone_sha256": BACKBONE_SHA256, "tokenizer_binary_sha256": sha_file(args.tok_probe),
                  "source_sha256": {p: sha_file(repo/p) for p in source_files},
                  "model_inference_calls": 0, "optimizer_steps": 0, "training_approved": False}
        for name, data in (("records.json", records), ("summary.json", report)):
            with (root/name).open("x") as stream: json.dump(data, stream, ensure_ascii=False, indent=2); stream.write("\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        tokenizer._proc.terminate(); tokenizer._proc.wait()
        tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "corpus", "gguf", "tok-probe", "output"): parser.add_argument("--"+name, required=True)
    run(parser.parse_args())


if __name__ == "__main__": main()
