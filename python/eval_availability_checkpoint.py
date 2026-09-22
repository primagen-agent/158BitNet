"""Fixed CG-001 free-generation panel: trained Python fusion + actual C logits.

No teacher-forced prefixes, labels, source text in prompts, or cached K/V.
This is a supplied-single-episode component test, not C/HTTP deployment.
"""
import argparse
import json
from pathlib import Path

import torch

from c_tokenizer import CTokenizer
from memory_availability import AvailabilityMemoryFusion
from native_continuous_generation import NativeContinuousBackend, tokenizer_stop_ids
from native_memory_encoder import BACKBONE_SHA256, sha_file
from neural_memory_generation import TEMPLATE_VERSION, encode_generation_input, greedy_generate
from prepare_availability_curriculum import materialize
from prepare_neural_memory_protocol import digest
from review_neural_memory import read_rows
from train_availability_memory import source_identity


CHECKPOINT_SHA256 = 'a8976841239f18520842e75551dfbc8681c27f6d25e0784fed03c1efcb5585d9'
BASELINE_SHA256 = 'cb6b2f6292352797b9e33ca4326a6836d9fed8eb8b37ed1423227bf4d36491ef'


def load_checkpoint(path):
    if sha_file(path) != CHECKPOINT_SHA256: raise ValueError('unregistered checkpoint')
    value=torch.load(path,map_location='cpu',weights_only=True)
    if (value['format'],value['step'],value['backbone_sha256'],value['template_version'],value['deployment_approved']) != (
            'availability-memory-pilot-v1',50,BACKBONE_SHA256,TEMPLATE_VERSION,False):
        raise ValueError('checkpoint identity mismatch')
    if value['source_sha256'] != source_identity(): raise ValueError('training implementation changed')
    module=AvailabilityMemoryFusion(1024,24,8)
    module.load_state_dict(value['state_dict'],strict=True); module.eval()
    if any(not torch.isfinite(p).all() for p in module.parameters()): raise ValueError('nonfinite checkpoint')
    return module,value


def fixed_panel(records,index,baseline):
    worlds=sorted({r['world_id'] for r in index})[:2]
    ids={r['id'] for r in index if r['world_id'] in worlds and r['scenario'] in ('original','swapped_values','empty','ordinary_original')}
    selected=[r for r in records if r['id'] in ids]
    before=baseline['predictions']
    if len(selected)!=16 or [r['id'] for r in selected] != [r['id'] for r in before]:
        raise ValueError('panel changed from registered baseline')
    if any(digest(r)!=p['input_sha256'] for r,p in zip(selected,before)): raise ValueError('baseline input changed')
    return selected


def run(args):
    root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    module,checkpoint=load_checkpoint(args.checkpoint)
    if sha_file(args.baseline)!=BASELINE_SHA256: raise ValueError('baseline changed')
    baseline=json.loads(Path(args.baseline).read_text())
    manifest=materialize(args.config,args.corpus,verify=True)
    if baseline['corpus_manifest_sha256']!=digest(manifest) or baseline['feature_package_digest']!=checkpoint['feature_package_digest']:
        raise ValueError('corpus or feature binding changed')
    records=fixed_panel(read_rows(Path(args.corpus)/'dev.inputs.jsonl'),read_rows(Path(args.corpus)/'dev.index.jsonl'),baseline)
    if sha_file(args.tok_probe)!=baseline['tokenizer_sha256']: raise ValueError('tokenizer changed')
    tokenizer=CTokenizer(args.tok_probe,args.gguf); predictions=[]
    try:
        stops=tokenizer_stop_ids(tokenizer)
        for i,record in enumerate(records):
            request=encode_generation_input(record,tokenizer)
            backend=NativeContinuousBackend(gguf=args.gguf,probe=args.probe,reference_probe=args.reference_probe,
                encoder_probe=args.encoder_probe,fusion=module,output=root/f'case-{i:02d}',condition='availability_trained')
            if backend.encoder.identity!=checkpoint['encoder_identity']: raise ValueError('encoder domain changed')
            output=greedy_generate(request,tokenizer,backend,stop_ids=stops,max_new_tokens=32,context_capacity=128)
            trace=backend.finish()
            output['backend_cache_policy_verified']=all(s['base_reference_checked'] for s in trace['steps'])
            prediction={'id':record['id'],'input_sha256':digest(record),**output,
                        'initial_state_probabilities':trace['steps'][0]['availability_probabilities'],
                        'every_base_reference_checked':all(s['base_reference_checked'] for s in trace['steps']),
                        'backend_evidence':f'case-{i:02d}/backend.json'}
            predictions.append(prediction)
            with (root/f'case-{i:02d}'/'generation.json').open('x') as f: json.dump(prediction,f,ensure_ascii=False,indent=2)
            print(json.dumps({'case':i+1,'total':16,'id':record['id'],'text':output['text'],
                              'truncated':output['truncated']},ensure_ascii=False),flush=True)
        report={'format':'cg001-fixed-native-checkpoint-v1','step':50,'checkpoint_sha256':CHECKPOINT_SHA256,
                'baseline_sha256':BASELINE_SHA256,'backbone_sha256':BACKBONE_SHA256,
                'template_version':TEMPLATE_VERSION,'feature_package_digest':checkpoint['feature_package_digest'],
                'corpus_manifest_sha256':digest(manifest),'cached_tokens':0,'reused_tokens':0,
                'internal_prefill_kv_buffers':True,'oracle_answers_used':False,'memory_enabled':True,
                'neural_fusion_implementation':'python','source_episode_selection':'supplied_not_learned',
                'deployment_approved':False,'semantic_review_complete':False,'predictions':predictions,
                'source_sha256':{name:sha_file(Path(__file__).parent/name) for name in
                                ('eval_availability_checkpoint.py','native_continuous_generation.py')},
                'artifact_sha256':{name:sha_file(getattr(args,name)) for name in ('probe','reference_probe','encoder_probe','tok_probe')}}
        with (root/'generation.json').open('x') as f: json.dump(report,f,ensure_ascii=False,indent=2); f.write('\n')
    finally:
        tokenizer._proc.terminate(); tokenizer._proc.wait(); tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','baseline','config','corpus','gguf','probe','reference-probe','encoder-probe','tok-probe','output'):
        p.add_argument('--'+name,required=True)
    run(p.parse_args())


if __name__=='__main__': main()
