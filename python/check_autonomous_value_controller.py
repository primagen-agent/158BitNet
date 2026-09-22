"""DG-020 label-free first-decision native check; not complete free replies."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from append_value_transport import AppendValueCodec
from audit_joint_generation import continuous_file,reference_file
from autonomous_value_controller import AutonomousValueController,LiveFrame,FrameOrigin,ActionKind,ControllerState
from check_joint_span_interface import source_alignment
from check_value_teacher_trajectories import certificates
from diagnose_continuous_memory import forward
from diagnose_native_generation_gradient import native_forward,exact
from episode_memory_inputs import encoder_texts
from fine_span_reader import FineSpanReader,ByteLayout
from joint_optimizer import FEATURE_DIGEST,tensor_digest
from joint_span_reader import SpanFeatures
from joint_training_data import DATA_DIGEST
from native_memory_encoder import NativeFeatureBank,BACKBONE_SHA256,sha_file,digest,text_key
from native_token_payload import payload_mask
from neural_memory_contract import ModelBinding
from neural_memory_generation import encode_generation_input
from value_transport import PayloadSnapshot,FactPayload,ReplyBinding,TransportLimits

ROOT=Path(__file__).resolve().parents[1];RESEARCH=ROOT/'training/memory/neural-system'


def run(a):
    registration=RESEARCH/'experiments/DG-020.json';reg=json.loads(registration.read_text());before,_=certificates()
    old_path=RESEARCH/'reviews/DG-019/zero-update.json'
    if sha_file(old_path)!='5da3ffa91a74d5d5bfbefb0a7571adb7e4366b46d8e5e24083814466e697b533':raise ValueError('prior evidence changed')
    old=json.loads(old_path.read_text())
    for n,h in old['source_sha256'].items():
        if sha_file(ROOT/'python'/n)!=h:raise ValueError('teacher implementation changed')
    feature_root=ROOT/'build/neural-memory-cg003-full-features';m=json.loads((feature_root/'manifest.json').read_text())
    if digest(m)!=FEATURE_DIGEST:raise ValueError('encoder package changed')
    corpus=RESEARCH/'data/JB-001';manifest=json.loads((corpus/'manifest.json').read_text())
    if digest(manifest)!=DATA_DIGEST:raise ValueError('data identity changed')
    # NO labels file or teacher-trajectory archive is read for runtime decisions.
    data={}
    for name in ('inputs','index'):
        p=corpus/f'train.{name}.jsonl'
        if sha_file(p)!=manifest['file_sha256'][p.name]:raise ValueError('natural input inventory changed')
        data[name]={r['id']:r for r in map(json.loads,p.read_text().splitlines())}
    select=reg['selection'];ids=[]
    for language in select['languages']:
        for scenario in select['scenarios']:
            matches=[r['id'] for r in data['index'].values() if r['world_id']==select['world_id'] and
                r['relation_family']==select['relation_family'] and r['language']==language and r['scenario']==scenario]
            if len(matches)!=1:raise ValueError('nonunique panel')
            ids+=matches
    if len(ids)!=12:raise ValueError('panel size changed')
    root=Path(a.raw);root.mkdir(parents=True,exist_ok=False)
    gguf=ROOT/'models/bitcpm4-0.5b-tq2_0.gguf';codec=AppendValueCodec(ROOT/'build/tok_probe',gguf,m['tokenizer_sha256'])
    tokens=NativeFeatureBank(feature_root/'tokens',expected_encoder_id=m['encoder_identity']['encoder_id'],expected_manifest_sha256=m['token_manifest_digest'])
    binding=ModelBinding(BACKBONE_SHA256,m['tokenizer_sha256'],m['encoder_identity']['encoder_id'],sha_file(registration));key=digest(vars(binding))
    torch.set_num_threads(4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(reg['initialization']['seed']);model=FineSpanReader(2048,1024,key,width=reg['initialization']['width']).eval()
    initial=tensor_digest(model.state_dict())
    if initial!=old['initial_parameter_digest']:raise ValueError('untrained initialization changed')
    rows=[];zero=np.zeros(1024,dtype='<f4');limits=reg['limits']
    try:
        for rid in ids:
            runtime=data['inputs'][rid];query,sources=encoder_texts(runtime);q=tokens.rows[text_key(query)]
            if codec.tokenizer.encode(query,True)!=q.token_ids.tolist():raise ValueError('query token identity changed')
            if sources:
                s=tokens.rows[text_key(sources[0])]
                if codec.tokenizer.encode(sources[0],True)!=s.token_ids.tolist():raise ValueError('source token identity changed')
                pieces=codec.decode_pieces(s.token_ids[1:].tolist());allowed=payload_mask(runtime['episodes'][0],sources[0],pieces)
                ranges=source_alignment(runtime['episodes'][0],sources[0],pieces,allowed);raw=runtime['episodes'][0]['text'].encode()
                source=torch.from_numpy(s.features.copy())
            else:raw=b'';ranges=();allowed=[];source=torch.empty(0,2048)
            x=SpanFeatures(torch.from_numpy(q.features.copy()),source,torch.tensor(allowed,dtype=torch.bool),key,hashlib.sha256(raw).hexdigest(),digest(runtime['context']))
            layout=ByteLayout(x,raw,ranges);snapshot=PayloadSnapshot(binding,initial,'native',rid,0,(FactPayload('source',raw),) if raw else ())
            reply=ReplyBinding('native',rid,x.context_sha256);prefix=encode_generation_input(runtime,codec.tokenizer).prompt_token_ids
            controller=AutonomousValueController(model,x,layout,snapshot,reply,codec,prefix,max_new_tokens=limits['max_new_tokens'],
                max_value_reads=limits['max_value_reads'],limits=TransportLimits(limits['max_value_bytes'],limits['max_value_tokens'],limits['max_total_tokens']))
            directory=root/rid;directory.mkdir()
            forward(str(ROOT/'build/memory_continuous_probe'),str(gguf),prefix,zero,directory,'initial-base',False)
            native_forward(str(ROOT/'build/memory_gradient_reference'),str(gguf),prefix,zero,directory,'initial-reference',False)
            h,b,c,z=continuous_file(directory,'initial-base',prefix,False);rh,rb=reference_file(directory,'initial-reference',prefix)
            if not exact(h.numpy(),rh.numpy()) or not exact(b.numpy(),rb.numpy()) or not exact(b.numpy(),z.numpy()) or torch.count_nonzero(c):
                raise ValueError('native base/reference mismatch')
            frame=LiveFrame(prefix,h[0],b[0],key,FrameOrigin.NATIVE_FRESH);decision=controller.propose(frame)
            if decision.kind is ActionKind.NEEDS_UNCERTAINTY:
                if decision.token is not None or controller.generated_tokens or controller.state is not ControllerState.WAIT_UNCERTAINTY:
                    raise ValueError('uncertainty emitted or fell back')
            else:controller.commit(decision,decision.token)
            row={'id':rid,'input_sha256':digest(runtime),'prefix_ids':prefix,'origin':decision.origin.value,'route':decision.route,
                 'action':decision.kind.value,'selected_token':decision.token,'state':controller.state.value,
                 'committed_token_ids':controller.generated_tokens,'base_top1':int(b.argmax()),'reference_bitwise_equal':True,
                 'fresh_context':True,'passed':True,'raw_sha256':{p.name:sha_file(p) for p in directory.iterdir() if p.is_file()}}
            with (directory/'result.json').open('x') as f:json.dump(row,f,indent=2);f.write('\n')
            rows.append(row);print(json.dumps({'id':rid,'action':decision.kind.value,'emitted':len(controller.generated_tokens)}),flush=True)
        codec.verify_identity()
    finally:codec.close()
    if certificates()[0]!=before or tensor_digest(model.state_dict())!=initial:raise ValueError('model/protected inputs changed')
    report={'format':'dg020-native-first-decision-v1','registration_sha256':sha_file(registration),
            'source_sha256':{n:sha_file(ROOT/'python'/n) for n in ('autonomous_value_controller.py','check_autonomous_value_controller.py')},
            'raw_root':str(root.resolve()),'records':rows,'action_counts':dict(Counter(r['action'] for r in rows)),
            'origin':'native_untrained_first_decision','teacher_labels_used_for_forward':False,'teacher_trajectories_consumed':False,
            'oracle_handles_supplied':False,'optimizer_steps':0,'parameters_unchanged':True,'initial_parameter_digest':initial,
            'C_prefixes':len(rows),'C_forward_calls':2*len(rows),'cross_step_kv_reuse':False,'internal_prefill_kv_buffers':True,
            'full_replies_generated':0,'autonomous_recall_accuracy_measured':False,'uncertainty_generator_implemented':False,
            'passed':len(rows)==12 and all(r['passed'] for r in rows),'V2_complete':False}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True);run(p.parse_args())
