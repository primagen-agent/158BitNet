"""Registered ordinary C prompt ablation; no memory, training, or answer grading."""
import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np

from c_tokenizer import CTokenizer
from diagnose_native_generation_gradient import native_forward
from native_continuous_generation import tokenizer_stop_ids
from native_memory_encoder import BACKBONE_SHA256, sha_file
from neural_memory_generation import ASSISTANT_SUFFIX, LEGACY_TEMPLATE_VERSION, GenerationInput, generation_prompt, greedy_generate


GGUF_TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
GGUF_TEMPLATE = GGUF_TEMPLATE.replace("\\n", "\n")


def declared_template(metadata):
    # gguf_inspect prints literal newlines inside the Jinja string. A line-only
    # parser silently truncates this metadata value.
    prefix = "Metadata: tokenizer.chat_template type=string value="
    values = re.findall(re.escape(prefix) + r"(.*?)(?=^(?:Metadata:|Tensor:)|\Z)", metadata, re.S | re.M)
    if len(values) != 1 or values[0].rstrip("\n") != GGUF_TEMPLATE:
        raise ValueError("unexpected GGUF template; do not guess rendering")
    return values[0].rstrip("\n")


def run(args):
    config = json.loads(Path(args.experiment).read_text())
    if config["id"] != "DG-005" or sha_file(args.gguf) != BACKBONE_SHA256:
        raise ValueError("wrong experiment/backbone")
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    metadata = subprocess.check_output([args.inspect, args.gguf], text=True)
    template = declared_template(metadata)
    with (root / "gguf-metadata.txt").open("x") as stream: stream.write(metadata)
    tokenizer = CTokenizer(args.tok_probe, args.gguf)
    rows = []
    try:
        stops = tokenizer_stop_ids(tokenizer)
        for case in config["cases"]:
            runtime = {"id": case["id"], "context": [{"role": "user", "speaker": "user", "text": case["query"]}], "episodes": []}
            original = generation_prompt(runtime, template_version=LEGACY_TEMPLATE_VERSION)
            if not original.endswith(ASSISTANT_SUFFIX): raise ValueError("research template changed")
            variants = {"research_no_think": original,
                        "gguf_declared": original[:-len(ASSISTANT_SUFFIX)] + "<|im_start|>assistant\n"}
            for name in config["variants"]:
                directory = root / (case["id"] + "-" + name); directory.mkdir()
                ids = tuple(tokenizer.encode(variants[name], True))
                steps = []
                def backend(tokens, sources):
                    if sources: raise ValueError("baseline must not receive memory")
                    if steps and tokens != tuple(steps[-1]["prefix_ids"]) + (steps[-1]["top1"],):
                        raise ValueError("not actual prediction continuation")
                    value = native_forward(args.reference_probe, args.gguf, tokens, np.zeros(1024, dtype="<f4"),
                                           directory, f"step-{len(steps):03d}", False)
                    steps.append({"prefix_ids": list(tokens), "top1": int(value["logits"].argmax())})
                    return value["logits"]
                output = greedy_generate(GenerationInput(ids, ()), tokenizer, backend, stop_ids=stops,
                            max_new_tokens=config["max_new_tokens"], context_capacity=128, memory_enabled=False)
                row = {"id": case["id"], "variant": name, "prompt": variants[name], "generation": output, "steps": steps}
                rows.append(row)
                with (directory / "generation.json").open("x") as stream:
                    json.dump(row, stream, ensure_ascii=False, indent=2); stream.write("\n")
                print(json.dumps({"case": case["id"], "variant": name, "text": output["text"],
                                  "finish_reason": output["finish_reason"]}, ensure_ascii=False), flush=True)
        if sha_file(args.gguf) != BACKBONE_SHA256: raise ValueError("backbone changed during baseline check")
        repo = Path(__file__).resolve().parents[1]
        report = {"experiment": config, "actual_gguf_template": template, "cases": rows,
                  "memory_accuracy_measured": False, "optimizer_steps": 0, "training_approved": False,
                  "source_sha256": {p: sha_file(repo / p) for p in ("python/diagnose_native_prompt.py",
                      "python/neural_memory_generation.py", "python/native_continuous_generation.py",
                      "python/diagnose_native_generation_gradient.py", "tools/memory_gradient_probe.c", "src/bitnet.c", "src/tokenizer.c")},
                  "artifact_sha256": {p: sha_file(p) for p in (args.gguf, args.reference_probe, args.tok_probe, args.inspect, args.experiment)},
                  "raw_sha256": {str(p.relative_to(root)): sha_file(p) for p in root.rglob("*") if p.is_file()}}
        with (root / "summary.json").open("x") as stream: json.dump(report, stream, ensure_ascii=False, indent=2); stream.write("\n")
    finally:
        tokenizer._proc.terminate(); tokenizer._proc.wait()
        tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("experiment", "gguf", "reference-probe", "tok-probe", "inspect", "output"):
        parser.add_argument("--" + name, required=True)
    run(parser.parse_args())


if __name__ == "__main__": main()
