"""DG-017 four fixed C-input zero-update fixtures and actual-prefix lineage."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from audit_joint_generation import continuous_file,reference_file
from c_tokenizer import CTokenizer
from check_append_value_transport import protected
from episode_memory_inputs import encoder_texts
from joint_optimizer import FEATURE_DIGEST,tensor_digest
from joint_span_reader import JointSpanReader,SpanFeatures,PrefixFeature,PayloadAlignment
from joint_span_supervision import SpanTargets,span_loss
from joint_training_data import DATA_DIGEST
from native_memory_encoder import NativeFeatureBank,BACKBONE_SHA256,digest,sha_file,text_key
from native_prefix_bank import PrefixBank
from native_token_payload import payload_mask
from neural_memory_contract import ModelBinding
from neural_memory_generation import encode_generation_input
from prepare_joint_binding_curriculum import sentence
from value_transport import FactPayload,PayloadSnapshot,ReplyBinding

ROOT=Path(__file__).resolve().parents[1];RESEARCH=ROOT/'training/memory/neural-system'


def source_alignment(message,framed,pieces,allowed):
    raw=framed.encode();decoded=b''.join(pieces)
    pad=0 if decoded==raw else 1 if decoded==b' '+raw else None
    literal=json.dumps(message['text'],ensure_ascii=False).encode()
    if pad is None or literal[1:-1]!=message['text'].encode() or not raw.endswith(literal+b'}'):
        raise ValueError('unsupported escaped source alignment')
    start=len(raw)-len(literal)+pad;offset=0;ranges=[]
    for piece,ok in zip(pieces,allowed):
        end=offset+len(piece);ranges.append((offset-start,end-start) if ok else None);offset=end
    return tuple(ranges)


def exact_span(raw,text,ranges):
    needle=text.encode()
    if raw.count(needle)!=1:raise ValueError('ambiguous post-forward annotation')
    start=raw.index(needle);end=start+len(needle)
    starts=[i for i,p in enumerate(ranges) if p is not None and p[0]==start]
    ends=[i+1 for i,p in enumerate(ranges) if p is not None and p[1]==end]
    if len(starts)!=1 or len(ends)!=1:
        raise ValueError(f'unrepresentable exact byte boundary: {text!r}; no trimming or target deletion')
    return starts[0],ends[0]


def run(a):
    regpath=RESEARCH/'experiments/DG-017.json';reg=json.loads(regpath.read_text());before_protected=protected()
    root=ROOT/'build/neural-memory-cg003-full-features';m=json.loads((root/'manifest.json').read_text())
    if digest(m)!=FEATURE_DIGEST:raise ValueError('feature package changed')
    corpus=RESEARCH/'data/JB-001';manifest=json.loads((corpus/'manifest.json').read_text())
    if digest(manifest)!=DATA_DIGEST:raise ValueError('corpus changed')
    for name in ('train.inputs.jsonl','train.index.jsonl'):
        if sha_file(corpus/name)!=manifest['file_sha256'][name]:raise ValueError('data changed')
    inputs={r['id']:r for r in map(json.loads,(corpus/'train.inputs.jsonl').read_text().splitlines())}
    tokens=NativeFeatureBank(root/'tokens',expected_encoder_id=m['encoder_identity']['encoder_id'],expected_manifest_sha256=m['token_manifest_digest'])
    prefixes=PrefixBank(root/'prefixes',expected_manifest_sha256=m['prefix_manifest_digest'])
    binding=ModelBinding(BACKBONE_SHA256,m['tokenizer_sha256'],m['encoder_identity']['encoder_id'],sha_file(regpath));key=digest(vars(binding))
    torch.set_num_threads(4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(reg['initialization']['seed']);model=JointSpanReader(2048,1024,key,width=reg['initialization']['width'])
    initial=tensor_digest(model.state_dict());rows=[]
    if sha_file(ROOT/'build/tok_probe')!=m['tokenizer_sha256'] or sha_file(ROOT/'models/bitcpm4-0.5b-tq2_0.gguf')!=BACKBONE_SHA256:
        raise ValueError('tokenizer/backbone changed')
    tokenizer=CTokenizer(str(ROOT/'build/tok_probe'),str(ROOT/'models/bitcpm4-0.5b-tq2_0.gguf'))
    try:
        for rid in reg['native_records']:
            runtime=inputs[rid];query,sources=encoder_texts(runtime);source=sources[0];message=runtime['episodes'][0]
            q=tokens.rows[text_key(query)];s=tokens.rows[text_key(source)]
            if q.token_ids.tolist()!=tokenizer.encode(query,True) or s.token_ids.tolist()!=tokenizer.encode(source,True):raise ValueError('native token identity mismatch')
            pieces=tokenizer.decode_pieces(s.token_ids[1:].tolist());allowed=payload_mask(message,source,pieces)
            ranges=source_alignment(message,source,pieces,allowed);raw=message['text'].encode()
            x=SpanFeatures(torch.from_numpy(q.features.copy()),torch.from_numpy(s.features.copy()),torch.tensor(allowed),key,
                           hashlib.sha256(raw).hexdigest(),digest(runtime['context']))
            ids=encode_generation_input(runtime,tokenizer).prompt_token_ids
            h,_=prefixes(ids);prefix=PrefixFeature(ids,h,key)
            snapshot=PayloadSnapshot(binding,initial,'fixture',rid,0,(FactPayload('source',raw),));reply=ReplyBinding('fixture',rid,x.context_sha256)
            alignment=PayloadAlignment(x,snapshot,reply,ranges);alignment.validate()
            model.zero_grad(set_to_none=True);output=model(x,prefix,ids);prediction=model.predict(output)
            handle=model.handle(output,alignment)
            # Only now load labels/metadata. They are not fields of forward inputs.
            meta=next(r for r in map(json.loads,(corpus/'train.index.jsonl').read_text().splitlines()) if r['id']==rid)
            row={'id':rid,'scenario':meta['scenario'],'language':meta['language'],'span_candidates':len(output.spans),
                 'random_prediction':prediction,'handle_is_null':handle.span is None,'handle_origin':handle.origin.value,
                 'route_probabilities':output.route_logits.detach().softmax(0).tolist(),'annotation_error':None,
                 'gradient_gate':False,'first_prefix_ids':ids}
            try:
                target=SpanTargets(2,None,None,0)
                if meta['scenario']=='correct':
                    fact=next(f for f in meta['source_facts'] if f['subject']==meta['query_subject'] and f['relation']==meta['query_relation'])
                    fact_range=exact_span(raw,sentence(fact['subject'],fact['relation'],fact['value'],meta['language']),ranges)
                    value_range=exact_span(raw,fact['value'],ranges)
                    target=SpanTargets(1,fact_range,value_range,0)
                losses=span_loss(output,target);losses['total'].backward()
                grads={n:None if p.grad is None else float(p.grad.abs().sum()) for n,p in model.named_parameters()}
                finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                row.update(losses={k:float(v.detach()) for k,v in losses.items()},gradient_l1=grads,
                           gradient_gate=finite and sum(v or 0 for v in grads.values())>0,
                           conditional_span_heads_connected=all(grads[n] is not None for n in ('fact_value.weight','value.weight','value_query.weight')))
                if meta['scenario']=='correct' and not row['conditional_span_heads_connected']:raise ValueError('supported span heads disconnected')
                if not row['gradient_gate']:raise ValueError('invalid gradients')
            except ValueError as e:row['annotation_error']=str(e);row['gradient_gate']=False
            if model.predict(output)!=prediction or tensor_digest(model.state_dict())!=initial:raise ValueError('loss changed prediction or parameters')
            rows.append(row);print(json.dumps({'id':rid,'gradient_gate':row['gradient_gate'],'annotation_error':row['annotation_error']}),flush=True)
        piece_table=tokenizer.decode_pieces(list(range(73448)))
    finally:tokenizer._proc.terminate();tokenizer._proc.wait();tokenizer._proc.stdin.close();tokenizer._proc.stdout.close()
    # Separate DG-016 lineage check, not paired training examples for the four cases.
    file=RESEARCH/'reviews/DG-016/native-fixtures.json';report=json.loads(file.read_text())
    if sha_file(file)!='a5146681a1ac937bb57ca1f0a7bd372895b7f58125d0971e3d6d0cd6eaba2658':raise ValueError('DG-016 changed')
    lineage=[];equal_text_rejections=0
    for case in report['cases']:
        directory=Path(report['raw_root'])/case['id']
        for name,expected in case['raw_sha256'].items():
            if sha_file(directory/name)!=expected:raise ValueError('raw native evidence changed')
        for i,step in enumerate(case['steps']):
            ids=tuple(step['prefix_ids']);h,b,c,z=continuous_file(directory,f'step-{i:03d}-base',ids,False)
            rh,rb=reference_file(directory,f'step-{i:03d}-reference',ids)
            if not torch.equal(h,rh) or not torch.equal(b,rb) or not torch.equal(b,z) or torch.count_nonzero(c):raise ValueError('C mismatch')
            feature=PrefixFeature(ids,h[0],key);feature.validate(ids,key,1024)
            if i==0 and tuple(case['canonical_prefix_ids'])!=ids:
                canonical=tuple(case['canonical_prefix_ids'])
                if b''.join(piece_table[t] for t in ids[1:])!=b''.join(piece_table[t] for t in canonical[1:]):
                    raise ValueError('noncanonical comparison does not have equal text')
                try:feature.validate(canonical,key,1024)
                except ValueError:equal_text_rejections+=1
                else:raise ValueError('equal-text different-prefix feature accepted')
            wrong=ids[:-1]+((ids[-1]+1)%73448,)
            try:feature.validate(wrong,key,1024)
            except ValueError:pass
            else:raise ValueError('mismatched token prefix accepted')
            lineage.append({'case':case['id'],'position':i,'exact_prefix_accepted':True,'different_prefix_rejected':True})
    if len(lineage)!=44 or equal_text_rejections!=1 or tensor_digest(model.state_dict())!=initial or protected()!=before_protected:raise ValueError('inventory or protected identity changed')
    result={'format':'dg017-zero-update-interface-v1','registration_sha256':sha_file(regpath),
            'source_sha256':{n:sha_file(ROOT/'python'/n) for n in ('joint_span_reader.py','joint_span_supervision.py','check_joint_span_interface.py')},
            'optimizer_steps':0,'new_C_generation':False,'parameters_unchanged':True,'initial_parameter_digest':initial,
            'origin':'untrained_neural_head','autonomous_recall_accuracy_measured':False,'rows':rows,'prefix_lineage':lineage,
            'equal_text_different_token_prefix_rejected':equal_text_rejections,
            'full_native_gradient_gate':all(r['gradient_gate'] for r in rows),'V2_complete':False}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    if not result['full_native_gradient_gate']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);run(p.parse_args())
