"""Qualify native reader input transport, not reader accuracy or generation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from episode_memory import EpisodeBindingReader
from episode_memory_inputs import encode_reader_input, reader_forward
from native_memory_encoder import NativeMemoryEncoder, NativeFeatureBank, collect_texts, save_bank, sha_file, text_key
from prepare_neural_memory_protocol import materialize


def run(args):
    materialize(args.config, args.corpus, verify=True)
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    corpus = Path(args.corpus)
    inputs = [json.loads(line) for line in (corpus / "dev.inputs.jsonl").read_text().splitlines()]
    index = [json.loads(line) for line in (corpus / "dev.index.jsonl").read_text().splitlines()]
    # Predeclared smoke selection: every scenario/language from first dev world.
    world = index[0]["world_id"]
    ids = {r["id"] for r in index if r["world_id"] == world}
    selected = [r for r in inputs if r["id"] in ids]
    texts = collect_texts(selected)
    encoder = NativeMemoryEncoder(args.gguf, args.probe)
    print(json.dumps({"phase": "native_encode", "cases": len(selected), "unique_texts": len(texts), "encoder_id": encoder.identity["encoder_id"]}), flush=True)
    original = encoder.encode(texts, root / "first", batch_size=16)
    fresh = NativeMemoryEncoder(args.gguf, args.probe, encoder.identity["encoder_id"])
    reversed_rows = fresh.encode(list(reversed(texts)), root / "fresh-reversed", batch_size=7)
    exact = all(np.array_equal(a.token_ids, b.token_ids) and np.array_equal(a.features, b.features)
                for a, b in zip(original, reversed(reversed_rows)))
    if not exact: raise ValueError("fresh process/order/batch changed native encoding")
    manifest_id = save_bank(root / "bank", encoder.identity, texts, original)
    bank = NativeFeatureBank(root / "bank", expected_encoder_id=encoder.identity["encoder_id"], expected_manifest_sha256=manifest_id)
    direct = {text_key(t): r for t, r in zip(texts, original)}
    if not all(np.array_equal(bank(t).numpy(), r.features) for t, r in zip(texts, original)):
        raise ValueError("archive changed float32 values")
    torch.set_num_threads(4); torch.manual_seed(3180921)
    reader = EpisodeBindingReader(1024).eval()
    checks = []
    for record in selected:
        left = encode_reader_input(record, lambda t: torch.from_numpy(direct[text_key(t)].features.copy()))
        right = encode_reader_input(record, bank)
        with torch.no_grad():
            a, _ = reader_forward(reader, left); b, _ = reader_forward(reader, right)
        if not torch.equal(a, b): raise ValueError("archive changed reader logits")
        checks.append({"id": record["id"], "reader_logits_exact": True})
    example = encode_reader_input(selected[0], bank)
    score, _ = reader_forward(reader, example)
    score.sum().backward()
    gradient = float(reader.query_relation.weight.grad.abs().sum())
    if not np.isfinite(gradient) or gradient <= 0: raise ValueError("reader learning path lost")
    new_text = '[{"role":"user","speaker":"user","text":"Nadia changed her project code to ZX-908."}]'
    if text_key(new_text) in direct: raise ValueError("novel input control is not novel")
    novel = fresh.encode([new_text], root / "novel")
    novel[0].validate()
    report = {"purpose": "native_reader_input_transport_only", "identity": encoder.identity,
              "world": world, "independent_worlds": 1, "cases": checks, "unique_encoded_texts": len(texts),
              "fresh_process_order_batch_exact": exact, "archive_roundtrip_exact": True,
              "reader_forward_exact_cases": len(checks), "reader_gradient_l1": gradient,
              "novel_input_encoded": True, "memory_accuracy_measured": False, "generation_alignment_passed": False,
              "training_started": False, "feature_manifest_sha256": manifest_id,
              "source_sha256": {name: sha_file(Path(__file__).with_name(name)) for name in
                                ("native_memory_encoder.py", "diagnose_native_memory_encoder.py", "episode_memory_inputs.py", "episode_memory.py")},
              "input_manifest_sha256": sha_file(corpus / "manifest.json"),
              "extraction_manifest_sha256": {name: sha_file(root / name / "manifest.json") for name in ("first", "fresh-reversed", "novel")}}
    with (root / "summary.json").open("x") as stream: json.dump(report, stream, indent=2); stream.write("\n")
    print(json.dumps({k: report[k] for k in ("fresh_process_order_batch_exact", "archive_roundtrip_exact", "reader_forward_exact_cases", "novel_input_encoded", "reader_gradient_l1")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "corpus", "gguf", "probe", "output"): parser.add_argument("--" + name, required=True)
    run(parser.parse_args())


if __name__ == "__main__": main()
