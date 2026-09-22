"""Fixed 16-row step-zero C baseline. No memory, labels, or KV reuse in generation."""
import argparse
import json
from pathlib import Path

from c_tokenizer import CTokenizer
from native_continuous_generation import tokenizer_stop_ids
from native_memory_encoder import BACKBONE_SHA256, sha_file
from native_prefix_bank import extract_prefix_batch
from neural_memory_generation import TEMPLATE_VERSION, encode_generation_input
from prepare_availability_curriculum import materialize
from prepare_neural_memory_protocol import digest
from review_neural_memory import read_rows


def run(args):
    root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
    manifest=materialize(args.config,args.corpus,verify=True)
    if sha_file(args.gguf)!=BACKBONE_SHA256: raise ValueError("wrong baseline backbone")
    meta=read_rows(Path(args.corpus)/"dev.index.jsonl")
    worlds=sorted({r["world_id"] for r in meta})[:2]
    ids={r["id"] for r in meta if r["world_id"] in worlds and r["scenario"] in ("original","swapped_values","empty","ordinary_original")}
    records=[r for r in read_rows(Path(args.corpus)/"dev.inputs.jsonl") if r["id"] in ids]
    if len(records)!=16: raise ValueError("fixed panel inventory mismatch")
    tokenizer=CTokenizer(args.tok_probe,args.gguf)
    try:
        requests=[encode_generation_input(r,tokenizer) for r in records]
        prefixes=[r.prompt_token_ids for r in requests]; generated=[[] for _ in records]; visible=[[] for _ in records]
        done=[False]*16; stops=tokenizer_stop_ids(tokenizer); raw=[]
        for step in range(32):
            active=[i for i in range(16) if not done[i]]
            if not active: break
            directory=root/f"step-{step:03d}"
            _,logits=extract_prefix_batch(args.prefix_probe,args.gguf,[prefixes[i] for i in active],directory)
            for position,i in enumerate(active):
                token=int(logits[position].argmax()); generated[i].append(token)
                if token in stops: done[i]=True
                else: visible[i].append(token); prefixes[i]=prefixes[i]+(token,)
            raw.append({"step":step,"active_case_ids":[records[i]["id"] for i in active],"input_sha256":sha_file(directory/"input.bin"),
                        "output_sha256":sha_file(directory/"output.bin"),"log_sha256":sha_file(directory/"native.log")})
            print(json.dumps({"step":step+1,"finished":sum(done),"total":16}),flush=True)
        predictions=[]
        for i,record in enumerate(records):
            data=b"".join(tokenizer.decode_pieces(visible[i]))
            try: text=data.decode("utf-8"); valid=True
            except UnicodeDecodeError: text=data.decode("utf-8",errors="replace"); valid=False
            predictions.append({"id":record["id"],"input_sha256":digest(record),"text":text,"token_ids":generated[i],
                                "raw_text_hex":data.hex(),"utf8_complete":valid,"finish_reason":"stop_token" if done[i] else "max_new_tokens",
                                "truncated":not done[i]})
        report={"format":"cg001-fixed-native-baseline-v1","step":0,"feature_package_digest":args.feature_digest,
                "template_version":TEMPLATE_VERSION,"backbone_sha256":BACKBONE_SHA256,"corpus_manifest_sha256":digest(manifest),
                "cached_tokens":0,"reused_tokens":0,"internal_prefill_kv_buffers":True,"oracle_answers_used":False,
                "memory_enabled":False,"predictions":predictions,"raw_steps":raw,"optimizer_steps":0,"memory_accuracy_measured":False,
                "probe_sha256":sha_file(args.prefix_probe),"tokenizer_sha256":sha_file(args.tok_probe),
                "source_sha256":{p:sha_file(Path(__file__).resolve().parents[1]/p) for p in ("python/eval_availability_baseline.py","python/native_prefix_bank.py","python/neural_memory_generation.py","tools/memory_prefix_probe.c")}}
        with (root/"baseline.json").open("x") as f: json.dump(report,f,ensure_ascii=False,indent=2); f.write("\n")
        print(json.dumps({"phase":"done","complete":sum(done),"total":16}),flush=True)
    finally:
        tokenizer._proc.terminate(); tokenizer._proc.wait(); tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ("config","corpus","gguf","prefix-probe","tok-probe","feature-digest","output"): parser.add_argument("--"+n,required=True)
    run(parser.parse_args())


if __name__=="__main__":main()
