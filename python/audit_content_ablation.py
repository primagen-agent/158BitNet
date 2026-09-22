"""DG-013 raw C replay: unchanged activation and zero continuous increment."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np
import torch

from ablate_joint_content import ORIGINAL_SHA, selected_indices, read_content_off
from audit_joint_generation import continuous_file, reference_file, close
from c_tokenizer import CTokenizer
from diagnose_native_generation_gradient import exact
from episode_memory_inputs import encoder_texts
from eval_joint_baseline import fixed_panel
from joint_checkpoint_transport import restore_certified_initial
from joint_memory_model import create_joint_model
from joint_optimizer import FEATURE_DIGEST, load_checkpoint, tensor_digest
from native_continuous_generation import tokenizer_stop_ids
from native_memory_encoder import sha_file, read_encoded, BACKBONE_SHA256
from native_token_payload import input_binding, native_token_input
from neural_memory_generation import encode_generation_input
from prepare_neural_memory_protocol import digest
from token_memory_composition import mix_content_copy


def run(a):
    root=Path(a.generation);report=json.loads((root/'generation.json').read_text())
    if sha_file(a.original)!=ORIGINAL_SHA[a.arm] or sha_file(a.status)!=a.status_sha256:raise ValueError('original/status changed')
    original=json.loads(Path(a.original).read_text());selected=selected_indices(original,a.arm)
    if (report['format']!='dg013-content-off-generation-v1' or report['arm']!=a.arm or report['fresh_indices']!=selected or
            report['optimizer_steps']!=0 or report['parameters_unchanged'] is not True or
            report['original_sha256']!=ORIGINAL_SHA[a.arm] or report['cached_tokens']!=0 or report['reused_tokens']!=0):
        raise ValueError('ablation protocol changed')
    status=json.loads(Path(a.status).read_text());binding=status['binding'];certs={c['step']:c for c in status['checkpoints']}
    if report['checkpoint_sha256']!=certs[50]['sha256'] or report['status_sha256']!=a.status_sha256:raise ValueError('checkpoint changed')
    for name,want in binding['source_sha256'].items():
        if sha_file(Path(__file__).parent/name)!=want:raise ValueError('training source changed')
    m=json.loads((Path(a.features)/'manifest.json').read_text())
    if digest(m)!=FEATURE_DIGEST or sha_file(a.gguf)!=BACKBONE_SHA256 or sha_file(a.tok_probe)!=m['tokenizer_sha256']:
        raise ValueError('model/tokenizer identity changed')
    identity=input_binding(m['encoder_identity'],m['tokenizer_sha256'])
    if identity!=binding['encoder_binding']:raise ValueError('encoder changed')
    torch.set_num_threads(4);model=create_joint_model(identity,m['blocked_ids'],a.arm).eval();parent=Path(a.status).parent
    restore_certified_initial(parent/certs[0]['file'],expected_sha256=certs[0]['sha256'],model=model,binding=binding,initial_digest=certs[0]['state_digest'])
    load_checkpoint(parent/certs[50]['file'],expected_sha256=certs[50]['sha256'],model=model,expected_binding=binding,expected_initial_digest=certs[0]['state_digest'])
    before=tensor_digest(model.state_dict());audit=json.loads(Path(a.panel_audit).read_text())
    records,panel=fixed_panel(a.corpus,audit)
    if [r['original_case_index'] for r in report['predictions']]!=selected:raise ValueError('fresh result inventory changed')
    rows=[];tokenizer=CTokenizer(a.tok_probe,a.gguf)
    try:
        stops=tokenizer_stop_ids(tokenizer)
        with torch.no_grad():
            for i,pred in zip(selected,report['predictions']):
                runtime=records[i];old=original['predictions'][i];directory=root/f'case-{i:02d}'
                if json.loads((directory/'generation.json').read_text())!=pred:raise ValueError('case/report mismatch')
                b=json.loads((directory/'backend.json').read_text())
                if (b['format']!='dg013-content-off-backend-v1' or b['intervention']!='supported_content_increment_off' or
                        b['cross_step_kv_reuse'] or b['prefix_bank_used'] or b['gold_answer_prefix_used'] or
                        b['identity']['encoder']!=m['encoder_identity'] or b['route_policy']!=a.arm):raise ValueError('backend changed')
                for n in ('probe','reference_probe'):
                    if b['identity'][n+'_sha256']!=original['artifact_sha256'][n]:raise ValueError('C binary differs from original')
                for name,want in b['raw_sha256'].items():
                    path=directory/name
                    if not path.resolve().is_relative_to(directory.resolve()) or sha_file(path)!=want:raise ValueError('raw file changed')
                    if '-output.' in name or '-qual-' in name:raise ValueError('unexpected residual projection')
                q,sources=encoder_texts(runtime);texts=[q,*sources]
                wire=struct.pack('<I',len(texts))+b''.join(struct.pack('<I',len(t.encode()))+t.encode() for t in texts)
                if (directory/'inputs/batch-0.input').read_bytes()!=wire:raise ValueError('source/query changed')
                encoded=read_encoded(directory/'inputs/batch-0.bin',len(texts))
                if tokenizer.encode(q,True)!=encoded[0].token_ids.tolist():raise ValueError('query tokens changed')
                memory=native_token_input(encoded[0],encoded[1] if sources else None,runtime['episodes'][0] if sources else None,
                    sources[0] if sources else None,tokenizer,identity)
                state=model.prefill(memory);probs=state.core.state_logits.exp().tolist()
                if int(state.core.state_logits.argmax())!=1 or probs!=old['initial_state_probabilities'] or probs!=pred['initial_state_probabilities']:
                    raise ValueError('original neural activation changed')
                prefix=encode_generation_input(runtime,tokenizer).prompt_token_ids;ids=pred['generated_token_ids'];steps=b['steps']
                if (pred['id']!=runtime['id'] or pred['input_sha256']!=digest(runtime) or not 1<=len(ids)<=64 or
                        len(ids)!=len(steps) or any(t in stops for t in ids[:-1])):raise ValueError('reply inventory changed')
                max_error=0.;changes=0
                for j,step in enumerate(steps):
                    if (step['prefix_ids']!=list(prefix) or step['selected_route']!=1 or step['effective_route']!=1 or
                            step['state_probabilities']!=probs or step['continuous_content_enabled'] is not False):
                        raise ValueError('prefix/route/intervention changed')
                    h,base,correction,logits=continuous_file(directory,f'step-{j:03d}-base',prefix,False)
                    rh,rb=reference_file(directory,f'step-{j:03d}-reference',prefix)
                    if not all(exact(x.numpy(),y.numpy()) for x,y in ((h,rh),(base,rb),(base,logits))) or torch.count_nonzero(correction):
                        raise ValueError('nonzero content increment or C reference mismatch')
                    p=model.reader.proposal(h,base,state.core.pointer)
                    actual=mix_content_copy(base,p.position_probabilities,p.copy_mass,memory.token_ids)
                    # None head is intentional: supported intervention cannot use an output projection.
                    error=close(actual,read_content_off(model,h,base,None,m['logit_scale'],state))
                    if int(actual.argmax())!=ids[j] or step['predicted_token_id']!=ids[j] or step['copy_mass']!=float(p.copy_mass[0,0]):
                        raise ValueError('pointer/token replay mismatch')
                    want={'initial_position':0,'final_position':len(prefix),'eval_calls':1,'prefix_tokens':len(prefix)}
                    if step['native_traces']!=[want] or not step['base_reference_checked']:raise ValueError('not a fresh C prefix')
                    changes+=int(actual.argmax())!=int(base.argmax());max_error=max(max_error,error);prefix=prefix+(ids[j],)
                stopped=ids[-1] in stops;visible=ids[:-1] if stopped else ids;raw=b''.join(tokenizer.decode_pieces(visible))
                try:text=raw.decode('utf-8');valid=True
                except UnicodeDecodeError:text=raw.decode('utf-8',errors='replace');valid=False
                if (not stopped and len(ids)!=64 or pred['text']!=text or pred['raw_text_hex']!=raw.hex() or
                        pred['visible_token_ids']!=visible or pred['truncated']!=(not stopped) or pred['utf8_complete']!=valid or
                        pred['finish_reason']!=('stop_token' if stopped else 'max_new_tokens')):raise ValueError('decode/termination mismatch')
                rows.append({'id':pred['id'],'original_case_index':i,'positions':len(ids),'max_error':max_error,
                    'pointer_changes_base_top1':changes,'activation_unchanged':True,'content_increment_zero':True,'passed':True})
    finally:
        tokenizer._proc.terminate();tokenizer._proc.wait();tokenizer._proc.stdin.close();tokenizer._proc.stdout.close()
    if tensor_digest(model.state_dict())!=before:raise ValueError('audit updated weights')
    result={'format':'dg013-content-off-audit-v1','arm':a.arm,'passed':len(rows)==len(selected),'fresh_cases':len(rows),
        'unchanged_cases_not_rerun':24-len(rows),'positions':sum(r['positions'] for r in rows),'cases':rows,
        'generation_sha256':sha_file(root/'generation.json'),'original_sha256':ORIGINAL_SHA[a.arm],
        'semantic_accuracy_measured':False,'optimizer_steps':0,'source_sha256':sha_file(__file__)}
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('arm','passed','fresh_cases','positions')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('original','status','status-sha256','panel-audit','corpus','features','gguf','tok-probe','generation','output'):
        p.add_argument('--'+n,required=True)
    p.add_argument('--arm',choices=('joint_aux','joint_product'),required=True);run(p.parse_args())
