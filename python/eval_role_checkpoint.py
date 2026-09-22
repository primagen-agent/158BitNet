"""CG-002 fixed native free-generation panel. No training or oracle routing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import torch

from c_tokenizer import CTokenizer
from eval_availability_checkpoint import BASELINE_SHA256,fixed_panel
from memory_role_separated import FORMAT,RoleSeparatedMemory
from native_continuous_generation import tokenizer_stop_ids
from native_memory_encoder import BACKBONE_SHA256,sha_file
from native_role_generation import NativeRoleBackend
from neural_memory_generation import TEMPLATE_VERSION,encode_generation_input,greedy_generate
from prepare_availability_curriculum import materialize
from prepare_neural_memory_protocol import digest
from review_neural_memory import read_rows
from train_role_separated_memory import FEATURE_DIGEST,source_identity

CHECKPOINT_SHA256='eb6b9a551c616d08a2aa0e8d183a63233c87a3c10e296f290a7b48b99dea716d'


def load_checkpoint(path):
    if sha_file(path)!=CHECKPOINT_SHA256:raise ValueError('unregistered role checkpoint')
    value=torch.load(path,map_location='cpu',weights_only=True)
    if (value['format'],value['step'],value['backbone_sha256'],value['template_version'],value['feature_package_digest'],value['deployment_approved'])!=(FORMAT,50,BACKBONE_SHA256,TEMPLATE_VERSION,FEATURE_DIGEST,False):
        raise ValueError('checkpoint binding mismatch')
    if value['source_sha256']!=source_identity():raise ValueError('training source changed')
    model=RoleSeparatedMemory();model.load_state_dict(value['state_dict'],strict=True);model.eval()
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite weights')
    return model,value


def run(a):
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);model,checkpoint=load_checkpoint(a.checkpoint)
    if sha_file(a.baseline)!=BASELINE_SHA256:raise ValueError('baseline changed')
    baseline=json.loads(Path(a.baseline).read_text());manifest=materialize(a.config,a.corpus,verify=True)
    if digest(manifest)!=baseline['corpus_manifest_sha256'] or baseline['feature_package_digest']!=FEATURE_DIGEST:raise ValueError('data binding mismatch')
    records=fixed_panel(read_rows(Path(a.corpus)/'dev.inputs.jsonl'),read_rows(Path(a.corpus)/'dev.index.jsonl'),baseline)
    if sha_file(a.tok_probe)!=baseline['tokenizer_sha256']:raise ValueError('tokenizer changed')
    def generate(item):
        i,record=item;tokenizer=CTokenizer(a.tok_probe,a.gguf)
        try:
            request=encode_generation_input(record,tokenizer)
            backend=NativeRoleBackend(gguf=a.gguf,probe=a.probe,reference_probe=a.reference_probe,
                encoder_probe=a.encoder_probe,fusion=model,output=root/f'case-{i:02d}')
            if backend.encoder.identity!=checkpoint['encoder_identity']:raise ValueError('encoder identity changed')
            output=greedy_generate(request,tokenizer,backend,stop_ids=tokenizer_stop_ids(tokenizer),max_new_tokens=32,context_capacity=128)
            trace=backend.finish();steps=trace['steps']
            if not steps or sum(s['decision_created'] for s in steps)!=1:raise ValueError('decision lifecycle error')
            output['backend_cache_policy_verified']=all(s['base_reference_checked'] for s in steps)
            prediction={'id':record['id'],'input_sha256':digest(record),**output,
                'initial_state_probabilities':steps[0]['availability_probabilities'],'selected_route':steps[0]['selected_route'],
                'every_base_reference_checked':all(s['base_reference_checked'] for s in steps),'backend_evidence':f'case-{i:02d}/backend.json'}
            with (root/f'case-{i:02d}/generation.json').open('x') as f:json.dump(prediction,f,ensure_ascii=False,indent=2)
            print(json.dumps({'case':i+1,'total':16,'route':prediction['selected_route'],'text':prediction['text'],'truncated':prediction['truncated']},ensure_ascii=False),flush=True)
            return prediction
        finally:
            tokenizer._proc.terminate();tokenizer._proc.wait();tokenizer._proc.stdin.close();tokenizer._proc.stdout.close()
    with ThreadPoolExecutor(max_workers=2) as pool:predictions=list(pool.map(generate,enumerate(records)))
    if sha_file(a.checkpoint)!=CHECKPOINT_SHA256:raise ValueError('checkpoint changed during generation')
    report={'format':'cg002-fixed-native-checkpoint-v1','step':50,'checkpoint_sha256':CHECKPOINT_SHA256,
        'baseline_sha256':BASELINE_SHA256,'backbone_sha256':BACKBONE_SHA256,'template_version':TEMPLATE_VERSION,
        'feature_package_digest':FEATURE_DIGEST,'corpus_manifest_sha256':digest(manifest),'case_workers':2,
        'cached_tokens':0,'reused_tokens':0,'internal_prefill_kv_buffers':True,'oracle_answers_used':False,
        'memory_enabled':True,'neural_fusion_implementation':'python','source_episode_selection':'supplied_not_learned',
        'route_policy':'predicted_initial_prefill_argmax_fixed_for_reply','deployment_approved':False,
        'semantic_review_complete':False,'predictions':predictions,
        'source_sha256':{name:sha_file(Path(__file__).parent/name) for name in ('eval_role_checkpoint.py','native_role_generation.py')},
        'artifact_sha256':{name:sha_file(getattr(a,name)) for name in ('probe','reference_probe','encoder_probe','tok_probe')}}
    with (root/'generation.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','baseline','config','corpus','gguf','probe','reference-probe','encoder-probe','tok-probe','output'):
        p.add_argument('--'+name,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
