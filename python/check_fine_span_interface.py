"""DG-018 exact-byte supervision on the unchanged four C-input fixtures."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from c_tokenizer import CTokenizer
from check_append_value_transport import protected
from check_joint_span_interface import source_alignment
from episode_memory_inputs import encoder_texts
from fine_span_reader import ByteLayout,FineSpanReader
from fine_span_supervision import ByteTargets,fine_loss
from joint_optimizer import FEATURE_DIGEST,tensor_digest
from joint_span_reader import SpanFeatures,PrefixFeature
from joint_training_data import DATA_DIGEST
from native_memory_encoder import NativeFeatureBank,BACKBONE_SHA256,digest,sha_file,text_key
from native_prefix_bank import PrefixBank
from native_token_payload import payload_mask
from neural_memory_contract import ModelBinding
from neural_memory_generation import encode_generation_input
from prepare_joint_binding_curriculum import sentence
from value_transport import FactPayload,PayloadSnapshot,ReplyBinding

ROOT=Path(__file__).resolve().parents[1];RESEARCH=ROOT/'training/memory/neural-system'


def certificates():
    result=protected();path=RESEARCH/'reviews/DG-017/zero-update-certified.json'
    expected='8af4a0424a8b055190748a32f166f96cd56a81b1abce64e1b57290120ac5f7d1'
    if sha_file(path)!=expected:raise ValueError('DG-017 evidence changed')
    report=json.loads(path.read_text())
    for n,h in report['source_sha256'].items():
        if sha_file(ROOT/'python'/n)!=h:raise ValueError('DG-017 source changed')
        result['python/'+n]=h
    if sha_file(RESEARCH/'experiments/DG-017.json')!=report['registration_sha256']:raise ValueError('DG-017 registration changed')
    return result,report


def positive_bytes(raw,meta):
    # Training annotation compiler, called strictly AFTER neural forward.
    fact=next(f for f in meta['source_facts'] if f['subject']==meta['query_subject'] and f['relation']==meta['query_relation'])
    clause=sentence(fact['subject'],fact['relation'],fact['value'],meta['language']).encode();value=fact['value'].encode()
    if raw.count(clause)!=1 or clause.count(value)!=1:raise ValueError('ambiguous annotation, preserve as failure')
    fs=raw.index(clause);vs=fs+clause.index(value)
    return ByteTargets(1,(fs,fs+len(clause)),(vs,vs+len(value)),0)


def run(a):
    regpath=RESEARCH/'experiments/DG-018.json';reg=json.loads(regpath.read_text());before_protected,old=certificates()
    if reg['native_records']!=[r['id'] for r in old['rows']]:raise ValueError('original four-case inventory changed')
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
        torch.manual_seed(reg['initialization']['seed']);model=FineSpanReader(2048,1024,key,width=reg['initialization']['width'])
    initial=tensor_digest(model.state_dict());rows=[]
    if sha_file(ROOT/'build/tok_probe')!=m['tokenizer_sha256'] or sha_file(ROOT/'models/bitcpm4-0.5b-tq2_0.gguf')!=BACKBONE_SHA256:
        raise ValueError('tokenizer/backbone changed')
    tokenizer=CTokenizer(str(ROOT/'build/tok_probe'),str(ROOT/'models/bitcpm4-0.5b-tq2_0.gguf'))
    try:
        for rid in reg['native_records']:
            runtime=inputs[rid];query,sources=encoder_texts(runtime);source=sources[0];message=runtime['episodes'][0]
            q=tokens.rows[text_key(query)];s=tokens.rows[text_key(source)]
            if q.token_ids.tolist()!=tokenizer.encode(query,True) or s.token_ids.tolist()!=tokenizer.encode(source,True):raise ValueError('token identity changed')
            pieces=tokenizer.decode_pieces(s.token_ids[1:].tolist());allowed=payload_mask(message,source,pieces)
            ranges=source_alignment(message,source,pieces,allowed);raw=message['text'].encode()
            x=SpanFeatures(torch.from_numpy(q.features.copy()),torch.from_numpy(s.features.copy()),torch.tensor(allowed),key,
                           hashlib.sha256(raw).hexdigest(),digest(runtime['context']))
            layout=ByteLayout(x,raw,ranges);ids=encode_generation_input(runtime,tokenizer).prompt_token_ids
            prior=next(r for r in old['rows'] if r['id']==rid)
            if list(ids)!=prior['first_prefix_ids']:raise ValueError('actual first prefix changed')
            h,_=prefixes(ids);prefix=PrefixFeature(ids,h,key)
            snapshot=PayloadSnapshot(binding,initial,'fixture',rid,0,(FactPayload('source',raw),));reply=ReplyBinding('fixture',rid,x.context_sha256)
            model.zero_grad(set_to_none=True);output=model(x,layout,prefix,ids);prediction=model.predict(output);handle=model.handle(output,snapshot,reply)
            meta=next(r for r in map(json.loads,(corpus/'train.index.jsonl').read_text().splitlines()) if r['id']==rid)
            row={'id':rid,'scenario':meta['scenario'],'language':meta['language'],'span_candidates':len(output.spans),
                 'random_prediction':prediction,'handle_is_null':handle.span is None,'handle_origin':handle.origin.value,
                 'gradient_gate':False,'error':None,'first_prefix_ids':ids}
            try:
                target=positive_bytes(raw,meta) if meta['scenario']=='correct' else ByteTargets(2,None,None,0)
                losses=fine_loss(output,target);losses['total'].backward()
                grads={n:None if p.grad is None else float(p.grad.abs().sum()) for n,p in model.named_parameters()}
                finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                connected=None
                if target.route==1:
                    connected=all(grads[n] is not None and grads[n]>0 for n in ('offset.weight','boundaries.0.weight','boundaries.2.weight','fact_value.weight','value.weight','value_query.weight'))
                    if not connected:raise ValueError('coarse/fine heads disconnected')
                    fs,fe=target.fact_bytes;vs,ve=target.value_bytes;cv=layout.covering_span(target.value_bytes)
                    row.update(target_fact_bytes=[fs,fe],target_value_bytes=[vs,ve],exact_target_value=raw[vs:ve].decode(),
                               coarse_value_tokens=cv,value_start_token_piece_hex=pieces[cv[0]].hex(),
                               value_start_within_token=vs-ranges[cv[0]][0])
                row.update(losses={k:float(v.detach()) for k,v in losses.items()},gradient_l1=grads,
                           fine_heads_connected=connected,gradient_gate=finite and sum(v or 0 for v in grads.values())>0)
                if not row['gradient_gate']:raise ValueError('invalid gradients')
            except ValueError as e:row['error']=str(e);row['gradient_gate']=False
            if model.predict(output)!=prediction or tensor_digest(model.state_dict())!=initial:raise ValueError('labels changed predictions or parameters')
            rows.append(row);print(json.dumps({'id':rid,'gradient_gate':row['gradient_gate'],'error':row['error']}),flush=True)
    finally:tokenizer._proc.terminate();tokenizer._proc.wait();tokenizer._proc.stdin.close();tokenizer._proc.stdout.close()
    if certificates()[0]!=before_protected or tensor_digest(model.state_dict())!=initial:raise ValueError('protected identity changed')
    result={'format':'dg018-fine-boundaries-zero-update-v1','registration_sha256':sha_file(regpath),
            'source_sha256':{n:sha_file(ROOT/'python'/n) for n in ('fine_span_reader.py','fine_span_supervision.py','check_fine_span_interface.py')},
            'optimizer_steps':0,'new_C_generation':False,'parameters_unchanged':True,'initial_parameter_digest':initial,
            'origin':'untrained_neural_head','autonomous_recall_accuracy_measured':False,'rows':rows,
            'native_gradient_gate':len(rows)==4 and all(r['gradient_gate'] for r in rows),'V2_complete':False}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    if not result['native_gradient_gate']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);run(p.parse_args())
