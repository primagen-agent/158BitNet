"""CG-003 actual native free replies on the frozen 24-case panel, no labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from c_tokenizer import CTokenizer
from eval_joint_baseline import fixed_panel
from joint_memory_model import create_joint_model
from joint_optimizer import load_checkpoint, FEATURE_DIGEST
from joint_checkpoint_transport import restore_certified_initial
from native_joint_generation import NativeJointBackend
from native_continuous_generation import tokenizer_stop_ids
from native_memory_encoder import sha_file, BACKBONE_SHA256
from native_token_payload import input_binding
from neural_memory_generation import TEMPLATE_VERSION, encode_generation_input, greedy_generate
from prepare_neural_memory_protocol import digest
from review_joint_generation import BASELINE_SHA


def run(a):
    if sha_file(a.status)!=a.status_sha256 or sha_file(a.baseline)!=BASELINE_SHA:
        raise ValueError('status/baseline identity mismatch')
    status=json.loads(Path(a.status).read_text());binding=status['binding']
    if (status['optimizer_steps']!=50 or status['automatic_continuation'] is not False or
            binding['feature_package_digest']!=FEATURE_DIGEST or status['deployment_approved'] is not False):
        raise ValueError('unregistered checkpoint status')
    certs={r['step']:r for r in status['checkpoints']}
    if set(certs)!={0,50}:raise ValueError('initial/final checkpoints required')
    for name,want in binding['source_sha256'].items():
        if sha_file(Path(__file__).parent/name)!=want:raise ValueError('training implementation changed: '+name)
    manifest=json.loads((Path(a.features)/'manifest.json').read_text())
    if digest(manifest)!=FEATURE_DIGEST:raise ValueError('feature binding changed')
    if (sha_file(Path(a.features)/'output-head.npy')!=manifest['output_head_sha256'] or
            sha_file(a.tok_probe)!=manifest['tokenizer_sha256'] or sha_file(a.gguf)!=BACKBONE_SHA256):
        raise ValueError('backbone/head/tokenizer changed')
    identity=input_binding(manifest['encoder_identity'],manifest['tokenizer_sha256'])
    if identity!=binding['encoder_binding']:raise ValueError('encoder binding mismatch')
    torch.set_num_threads(4)
    model=create_joint_model(identity,manifest['blocked_ids'],a.arm).eval()
    restore_certified_initial(Path(a.checkpoint).parent/certs[0]['file'],expected_sha256=certs[0]['sha256'],
        model=model,binding=binding,initial_digest=certs[0]['state_digest'])
    data=load_checkpoint(a.checkpoint,expected_sha256=certs[50]['sha256'],model=model,
                         expected_binding=binding,expected_initial_digest=certs[0]['state_digest'])
    if data['step']!=50:raise ValueError('final checkpoint required')
    head=torch.from_numpy(np.load(Path(a.features)/'output-head.npy',allow_pickle=False))
    baseline=json.loads(Path(a.baseline).read_text());audit=json.loads(Path(a.panel_audit).read_text())
    if sha_file(a.panel_audit)!=baseline['artifact_sha256']['audit']:raise ValueError('panel audit changed')
    records,panel=fixed_panel(a.corpus,audit)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    def generate(item):
        i,r=item;tokenizer=CTokenizer(a.tok_probe,a.gguf)
        try:
            backend=NativeJointBackend(runtime=r,tokenizer=tokenizer,tok_probe=a.tok_probe,gguf=a.gguf,
                probe=a.probe,reference_probe=a.reference_probe,encoder_probe=a.encoder_probe,
                model=model,head=head,scale=manifest['logit_scale'],output=root/f'case-{i:02d}')
            output=greedy_generate(encode_generation_input(r,tokenizer),tokenizer,backend,
                stop_ids=tokenizer_stop_ids(tokenizer),max_new_tokens=64,context_capacity=128)
            trace=backend.finish();steps=trace['steps']
            if not steps:raise ValueError('empty generation trace')
            output['backend_cache_policy_verified']=all(s['base_reference_checked'] for s in steps)
            prediction={'id':r['id'],'input_sha256':digest(r),**output,'selected_route':steps[0]['selected_route'],
                'initial_state_probabilities':steps[0]['state_probabilities'],
                'every_base_reference_checked':all(s['base_reference_checked'] for s in steps),
                'backend_evidence':f'case-{i:02d}/backend.json'}
            with (root/f'case-{i:02d}/generation.json').open('x') as f:json.dump(prediction,f,ensure_ascii=False,indent=2)
            print(json.dumps({'case':i+1,'total':24,'arm':a.arm,'route':prediction['selected_route'],
                              'text':prediction['text'],'truncated':prediction['truncated']},ensure_ascii=False),flush=True)
            return prediction
        finally:
            tokenizer._proc.terminate();tokenizer._proc.wait();tokenizer._proc.stdin.close();tokenizer._proc.stdout.close()
    with ThreadPoolExecutor(max_workers=2) as pool:outputs=list(pool.map(generate,enumerate(records)))
    if sha_file(a.checkpoint)!=certs[50]['sha256']:raise ValueError('checkpoint changed during generation')
    report={'format':'cg003-fixed-native-checkpoint-v1','arm':a.arm,'step':50,'checkpoint_sha256':certs[50]['sha256'],
        'status_sha256':a.status_sha256,'baseline_sha256':BASELINE_SHA,'template_version':TEMPLATE_VERSION,
        'feature_package_digest':FEATURE_DIGEST,'panel':panel,'predictions':outputs,'max_new_tokens':64,'context_capacity':128,
        'cached_tokens':0,'reused_tokens':0,'internal_prefill_kv_buffers':True,'oracle_answers_used':False,
        'memory_enabled':True,'route_policy':'predicted_initial_prefill_argmax_fixed_for_reply',
        'neural_fusion_implementation':'python','source_episode_selection':'supplied_not_learned',
        'deployment_approved':False,'semantic_review_complete':False,
        'source_sha256':sha_file(__file__),
        'artifact_sha256':{n:sha_file(getattr(a,n)) for n in ('gguf','probe','reference_probe','encoder_probe','tok_probe')}}
    with (root/'generation.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('checkpoint','status','status-sha256','baseline','panel-audit','corpus','features','gguf','probe',
              'reference-probe','encoder-probe','tok-probe','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--arm',choices=('joint_aux','joint_product'),required=True)
    run(p.parse_args())
