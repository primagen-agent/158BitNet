"""Build an identity-bound frozen C training package. No parameter training."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from availability_supervision import encode_reply_targets, teacher_forcing_requests
from c_tokenizer import CTokenizer
from diagnose_native_generation_gradient import exact, native_forward
from ggw import GGUFWeights
from native_memory_encoder import NativeMemoryEncoder, save_bank, sha_file, BACKBONE_SHA256
from native_prefix_bank import extract_prefix_batch
from neural_memory_generation import TEMPLATE_VERSION, encode_generation_input
from prepare_availability_curriculum import materialize
from prepare_neural_memory_protocol import digest
from review_neural_memory import read_rows


def run(args):
    corpus=Path(args.corpus); manifest=materialize(args.config,corpus,verify=True)
    root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
    encoder=NativeMemoryEncoder(args.gguf,args.encoder_probe)
    tokenizer=CTokenizer(args.tok_probe,args.gguf)
    records=[]; prefixes={}; sources={}
    try:
        for split in ("train","dev"):
            labels={r["id"]:r for r in read_rows(corpus/f"{split}.labels.jsonl")}
            meta={r["id"]:r for r in read_rows(corpus/f"{split}.index.jsonl")}
            for runtime in read_rows(corpus/f"{split}.inputs.jsonl"):
                request=encode_generation_input(runtime,tokenizer)
                target=encode_reply_targets(runtime,labels[runtime["id"]],tokenizer)
                requests=list(teacher_forcing_requests(request,target))
                for r in requests: prefixes.setdefault(digest(r.prompt_token_ids),r.prompt_token_ids)
                for source in request.source_texts: sources.setdefault(digest(source),source)
                records.append({"id":runtime["id"],"split":split,"world_id":meta[runtime["id"]]["world_id"],
                                "scenario":meta[runtime["id"]]["scenario"],"language":meta[runtime["id"]]["language"],
                                "input_sha256":digest(runtime),"prompt_ids":request.prompt_token_ids,
                                "source_texts":request.source_texts,"targets":asdict(target)})
        prefix_values=list(prefixes.values()); batches=[]
        (root/"prefixes").mkdir()
        # Before the bulk run, verify model reuse/order/duplicate inputs against
        # fresh-process reference. This loads weights, not chat state, once.
        selected=[prefix_values[i] for i in (0,len(prefix_values)//2,len(prefix_values)-1)]
        control=[selected[2],selected[0],selected[1],selected[0]]
        h,l=extract_prefix_batch(args.prefix_probe,args.gguf,control,root/"batch-parity")
        parity=[]
        for i,ids in enumerate(control):
            ref=native_forward(args.reference_probe,args.gguf,ids,np.zeros(1024,dtype="<f4"),root/"batch-parity",f"reference-{i}",False)
            parity.append(exact(h[i],ref["hidden"]) and exact(l[i],ref["logits"]))
        if not all(parity): raise ValueError("batched prefix extraction changed native bits")
        print(json.dumps({"phase":"parity","passed":len(parity),"unique_prefixes":len(prefix_values),"unique_sources":len(sources)}),flush=True)
        for start in range(0,len(prefix_values),64):
            ids=prefix_values[start:start+64]; directory=root/"prefixes"/f"batch-{start//64:04d}"
            extract_prefix_batch(args.prefix_probe,args.gguf,ids,directory)
            output=directory/"output.bin"
            batches.append({"path":str(output.relative_to(root/"prefixes")),"sha256":sha_file(output),"prefixes":ids})
            print(json.dumps({"phase":"prefixes","completed":min(start+64,len(prefix_values)),"total":len(prefix_values)}),flush=True)
        prefix_manifest={"format":"native-prefix-bank-v1","backbone_sha256":BACKBONE_SHA256,
                         "probe_sha256":sha_file(args.prefix_probe),"encoder_identity":encoder.identity,
                         "template_version":TEMPLATE_VERSION,"training_only":True,"cross_prefix_kv_reuse":False,"batches":batches}
        with (root/"prefixes/manifest.json").open("x") as f: json.dump(prefix_manifest,f,indent=2); f.write("\n")
        texts=list(sources.values()); rows=encoder.encode(texts,root/"source-extraction")
        source_manifest_sha=save_bank(root/"sources",encoder.identity,texts,rows)
        weights=GGUFWeights(args.gguf,args.lib)
        try:
            head=weights.get_f32("token_embd.weight",(73448,1024))
            with (root/"output-head.npy").open("xb") as f: np.save(f,head,allow_pickle=False)
            scale=weights.logit_scale
        finally: weights.close()
        with (root/"records.json").open("x") as f: json.dump(records,f,ensure_ascii=False,indent=2); f.write("\n")
        repo=Path(__file__).resolve().parents[1]
        paths=[p for d in ("src","include") for p in (repo/d).rglob("*") if p.suffix in (".c",".h",".mm")]
        paths += [repo/p for p in ("CMakeLists.txt","tools/memory_prefix_probe.c","tools/memory_feature_probe.c","tools/memory_gradient_probe.c",
                   "python/prepare_availability_features.py","python/native_prefix_bank.py","python/native_memory_encoder.py",
                   "python/availability_supervision.py","python/neural_memory_generation.py","python/c_tokenizer.py","python/ggw.py")]
        package={"format":"availability-training-features-v1","backbone_sha256":sha_file(args.gguf),"encoder_identity":encoder.identity,
                 "template_version":TEMPLATE_VERSION,"corpus_manifest_sha256":digest(manifest),"records_sha256":sha_file(root/"records.json"),
                 "prefix_manifest_sha256":digest(prefix_manifest),"source_manifest_sha256":source_manifest_sha,
                 "output_head_sha256":sha_file(root/"output-head.npy"),"logit_scale":scale,"examples":len(records),
                 "unique_prefixes":len(prefix_values),"unique_sources":len(texts),"reference_parity":parity,"optimizer_steps":0,
                 "source_sha256":{str(p.relative_to(repo)):sha_file(p) for p in paths},
                 "binaries_sha256":{str(p):sha_file(p) for p in (args.prefix_probe,args.reference_probe,args.encoder_probe,args.tok_probe,args.lib)}}
        if package["backbone_sha256"] != BACKBONE_SHA256: raise ValueError("GGUF changed during extraction")
        with (root/"manifest.json").open("x") as f: json.dump(package,f,indent=2); f.write("\n")
        print(json.dumps({"phase":"done","manifest_sha256":digest(package),"examples":len(records)}),flush=True)
    finally:
        tokenizer._proc.terminate(); tokenizer._proc.wait(); tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ("config","corpus","gguf","prefix-probe","reference-probe","encoder-probe","tok-probe","lib","output"):
        parser.add_argument("--"+n,required=True)
    run(parser.parse_args())


if __name__=="__main__":main()
