"""DG-019 teacher-only C feature compilation and zero-update mode gradients."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from append_value_transport import AppendValueCodec
from check_fine_span_interface import certificates as old_certificates
from check_joint_span_interface import source_alignment
from diagnose_native_generation_gradient import native_forward,exact
from episode_memory_inputs import encoder_texts
from fine_span_reader import ByteLayout,FineSpanReader
from fine_span_supervision import fine_loss
from joint_optimizer import FEATURE_DIGEST,tensor_digest
from joint_span_reader import SpanFeatures,PrefixFeature
from joint_training_data import DATA_DIGEST
from native_memory_encoder import NativeFeatureBank,BACKBONE_SHA256,digest,sha_file,text_key
from native_prefix_bank import extract_prefix_batch
from native_token_payload import payload_mask
from neural_memory_contract import ModelBinding
from value_teacher_trajectory import compile_teacher

ROOT=Path(__file__).resolve().parents[1];RESEARCH=ROOT/'training/memory/neural-system'
PREFIX_SHA='2735729604b1a52d99d1a60f48234b96ce3683d63aeec6434879bbfe7753e302'


def certificates():
    result,_=old_certificates();p=RESEARCH/'reviews/DG-018/zero-update.json'
    if sha_file(p)!='7c4a8a0efeda31c08fbc315babd8564d7c4179fa294014e106a1aa53d7879f35':raise ValueError('DG-018 report changed')
    report=json.loads(p.read_text())
    for n,h in report['source_sha256'].items():
        if sha_file(ROOT/'python'/n)!=h:raise ValueError('DG-018 implementation changed')
        result['python/'+n]=h
    if sha_file(RESEARCH/'experiments/DG-018.json')!=report['registration_sha256']:raise ValueError('DG-018 registration changed')
    if sha_file(ROOT/'build/memory_prefix_probe')!=PREFIX_SHA:raise ValueError('prefix executable changed')
    result['build/memory_prefix_probe']=PREFIX_SHA
    return result,report


def run(a):
    regpath=RESEARCH/'experiments/DG-019.json';reg=json.loads(regpath.read_text());before_protected,old=certificates()
    root=Path(a.raw);root.mkdir(parents=True,exist_ok=False)
    feature_root=ROOT/'build/neural-memory-cg003-full-features';m=json.loads((feature_root/'manifest.json').read_text())
    if digest(m)!=FEATURE_DIGEST:raise ValueError('source feature package changed')
    corpus=RESEARCH/'data/JB-001';manifest=json.loads((corpus/'manifest.json').read_text())
    if digest(manifest)!=DATA_DIGEST:raise ValueError('data identity changed')
    data={}
    for name in ('inputs','index','labels'):
        path=corpus/f'train.{name}.jsonl'
        if sha_file(path)!=manifest['file_sha256'][path.name]:raise ValueError('training data changed')
        data[name]={r['id']:r for r in map(json.loads,path.read_text().splitlines())}
    selection=reg['selection'];records=[]
    for language in selection['languages']:
        for scenario in selection['scenarios']:
            matches=[r for r in data['index'].values() if r['world_id']==selection['world_id'] and
                r['relation_family']==selection['relation_family'] and r['language']==language and r['scenario']==scenario]
            if len(matches)!=1:raise ValueError('selection is not unique')
            records.append(matches[0]['id'])
    if len(records)!=selection['expected_records']:raise ValueError('fixed inventory changed')
    binding=ModelBinding(BACKBONE_SHA256,m['tokenizer_sha256'],m['encoder_identity']['encoder_id'],sha_file(regpath));key=digest(vars(binding))
    torch.set_num_threads(4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(reg['initialization']['seed']);model=FineSpanReader(2048,1024,key,width=reg['initialization']['width'])
    initial=tensor_digest(model.state_dict())
    if initial!=old['initial_parameter_digest']:raise ValueError('untrained head initialization changed')
    gguf=ROOT/'models/bitcpm4-0.5b-tq2_0.gguf';codec=AppendValueCodec(ROOT/'build/tok_probe',gguf,m['tokenizer_sha256'])
    trajectories=[];rows=[];prefixes={};references=set()
    try:
        for rid in records:
            t=compile_teacher(data['inputs'][rid],data['labels'][rid],data['index'][rid],codec,binding,initial,purpose='training_diagnostic')
            trajectories.append(t)
            for i in range(len(t.completion_ids)):prefixes.setdefault(t.prefix(i,purpose='training_diagnostic'),len(prefixes))
            references.update(t.prefix(i,purpose='training_diagnostic') for i in t.reference_positions())
        state_counts=Counter(t.static_targets.route for t in trajectories)
        if state_counts!=Counter({0:4,1:4,2:4}):raise ValueError('route denominator changed')
        compiled={'format':'dg019-teacher-trajectories-v1','training_only':True,'origin':'teacher_forcing_oracle',
                  'registration_sha256':sha_file(regpath),'records':[asdict(t) for t in trajectories],
                  'unique_prefixes':[list(p) for p in prefixes],'direct_reference_indices':sorted(prefixes[p] for p in references)}
        with (root/'trajectories.json').open('x') as f:json.dump(compiled,f,ensure_ascii=False,indent=2);f.write('\n')
        print(json.dumps({'phase':'compiled','records':len(trajectories),'positions':sum(len(t.phases) for t in trajectories),
                          'unique_prefixes':len(prefixes),'direct_references':len(references)}),flush=True)
        all_ids=list(prefixes);bank={};batches=[]
        for start in range(0,len(all_ids),16):
            group=all_ids[start:start+16];folder=root/'prefixes'/f'batch-{start//16:03d}'
            h,z=extract_prefix_batch(ROOT/'build/memory_prefix_probe',gguf,group,folder)
            for p,hh,zz in zip(group,h,z):bank[p]=(torch.from_numpy(hh.copy()),torch.from_numpy(zz.copy()))
            batches.append({'folder':str(folder.relative_to(root)),'prefixes':group,'output_sha256':sha_file(folder/'output.bin')})
            print(json.dumps({'phase':'C_features','completed':len(bank),'total':len(all_ids)}),flush=True)
        refdir=root/'reference';refdir.mkdir();ref_checks=[]
        for j,p in enumerate(p for p in all_ids if p in references):
            result=native_forward(str(ROOT/'build/memory_gradient_reference'),str(gguf),p,np.zeros(1024,dtype='<f4'),refdir,f'row-{j:03d}',False)
            h,z=bank[p]
            if not exact(h.numpy(),result['hidden']) or not exact(z.numpy(),result['logits']):raise ValueError('C batch/reference mismatch')
            ref_checks.append({'prefix_index':prefixes[p],'prefix_ids':p,'bitwise_equal':True})
            print(json.dumps({'phase':'reference','completed':j+1,'total':len(references)}),flush=True)
        tokens=NativeFeatureBank(feature_root/'tokens',expected_encoder_id=m['encoder_identity']['encoder_id'],expected_manifest_sha256=m['token_manifest_digest'])
        for t in trajectories:
            runtime=data['inputs'][t.record_id];query,sources=encoder_texts(runtime);q=tokens.rows[text_key(query)]
            if codec.tokenizer.encode(query,True)!=q.token_ids.tolist():raise ValueError('query identity changed')
            if sources:
                s=tokens.rows[text_key(sources[0])]
                if codec.tokenizer.encode(sources[0],True)!=s.token_ids.tolist():raise ValueError('source identity changed')
                pieces=codec.decode_pieces(s.token_ids[1:].tolist());allowed=payload_mask(runtime['episodes'][0],sources[0],pieces)
                ranges=source_alignment(runtime['episodes'][0],sources[0],pieces,allowed);raw=runtime['episodes'][0]['text'].encode()
                source=torch.from_numpy(s.features.copy())
            else:source=torch.empty(0,2048);allowed=[];ranges=();raw=b''
            x=SpanFeatures(torch.from_numpy(q.features.copy()),source,torch.tensor(allowed,dtype=torch.bool),key,hashlib.sha256(raw).hexdigest(),digest(runtime['context']))
            layout=ByteLayout(x,raw,ranges);model.zero_grad(set_to_none=True)
            first=t.prefix(0,purpose='training_diagnostic');output=model(x,layout,PrefixFeature(first,bank[first][0],key),first)
            initial_prediction=model.predict(output)
            # Teacher labels only read by loss after the label-free forward.
            static=fine_loss(output,t.static_targets);static['total'].backward()
            idle_count=sum(v is not None for v in t.idle_targets);mode_loss=0.;predictions=[]
            for i,label in enumerate(t.idle_targets):
                if label is None:continue
                p=t.prefix(i,purpose='training_diagnostic');out=model(x,layout,PrefixFeature(p,bank[p][0],key),p)
                predicted=model.predict(out);predictions.append({'position':i,'predicted_route':predicted['route'],
                    'predicted_mode':predicted['mode'],'teacher_phase':t.phases[i]})
                loss=F.cross_entropy(out.mode_logits[None],torch.tensor([label]))/idle_count
                mode_loss+=float(loss.detach());loss.backward()
            grads={n:None if p.grad is None else float(p.grad.abs().sum()) for n,p in model.named_parameters()}
            if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()) or sum(v or 0 for v in grads.values())<=0:
                raise ValueError('invalid controller gradients')
            if t.static_targets.route==1 and not all(grads[n] is not None and grads[n]>0 for n in ('offset.weight','boundaries.0.weight','boundaries.2.weight','fact_value.weight','value.weight','value_query.weight')):
                raise ValueError('positive coarse/fine heads disconnected')
            if grads['mode.0.weight'] is None or grads['mode.0.weight']<=0:raise ValueError('idle mode head disconnected')
            if tensor_digest(model.state_dict())!=initial:raise ValueError('parameters updated')
            row={'id':t.record_id,'route_target':t.static_targets.route,'teacher_positions':len(t.phases),
                'phase_counts':dict(Counter(t.phases)),'idle_positions':idle_count,'initial_random_prediction':initial_prediction,
                'random_idle_predictions':predictions,'static_losses':{k:float(v.detach()) for k,v in static.items()},
                'mean_idle_mode_loss':mode_loss,'gradient_l1':grads,'gradient_gate':True}
            rows.append(row);print(json.dumps({'phase':'zero_update','record':t.record_id,'positions':len(t.phases),'gradient_gate':True}),flush=True)
        codec.verify_identity()
    finally:codec.close()
    if certificates()[0]!=before_protected or tensor_digest(model.state_dict())!=initial:raise ValueError('protected source/weights changed')
    raw_sha={str(p.relative_to(root)):sha_file(p) for p in sorted(root.rglob('*')) if p.is_file()}
    counts=Counter(p for t in trajectories for p in t.phases)
    result={'format':'dg019-teacher-trajectory-check-v1','registration_sha256':sha_file(regpath),
        'source_sha256':{n:sha_file(ROOT/'python'/n) for n in ('value_teacher_trajectory.py','check_value_teacher_trajectories.py')},
        'training_only':True,'origin':'teacher_forcing_oracle','optimizer_steps':0,'parameters_unchanged':True,
        'initial_parameter_digest':initial,'autonomous_recall_accuracy_measured':False,'natural_reply_quality_measured':False,
        'raw_root':str(root.resolve()),'raw_sha256':raw_sha,'trajectories_sha256':sha_file(root/'trajectories.json'),
        'records':rows,'phase_counts':dict(counts),'positions':sum(counts.values()),'unique_C_prefixes':len(prefixes),
        'direct_reference_checks':ref_checks,'C_forward_calls':len(prefixes)+len(ref_checks),'batches':batches,
        'cross_prefix_kv_reuse':False,'internal_prefill_kv_buffers':True,'passed':len(rows)==12 and all(r['gradient_gate'] for r in rows),
        'V2_complete':False,'deployment_approved':False}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True);run(p.parse_args())
