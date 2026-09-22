"""DG-021 native parity/gradient fixtures, not learned refusal or recall."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from append_value_transport import AppendValueCodec
from audit_joint_generation import continuous_file,reference_file,close
from autonomous_value_controller import AutonomousValueController,LiveFrame,FrameOrigin
from check_joint_span_interface import source_alignment
from check_value_teacher_trajectories import certificates
from diagnose_continuous_memory import forward
from diagnose_native_generation_gradient import exact
from episode_memory_inputs import encoder_texts
from fine_span_reader import FineSpanReader,ByteLayout
from joint_optimizer import FEATURE_DIGEST,tensor_digest
from joint_span_reader import SpanFeatures
from joint_training_data import DATA_DIGEST
from native_memory_encoder import NativeFeatureBank,BACKBONE_SHA256,digest,sha_file,text_key
from native_token_payload import payload_mask
from neural_memory_contract import ModelBinding
from neural_memory_generation import encode_generation_input
from query_only_uncertainty import QueryOnlyUncertainty
from uncertainty_reply_controller import ResearchUncertaintyController,UncertaintyDecision
from value_transport import PayloadSnapshot,FactPayload,ReplyBinding

ROOT=Path(__file__).resolve().parents[1];RESEARCH=ROOT/'training/memory/neural-system'


def run(a):
    registration=RESEARCH/'experiments/DG-021.json';reg=json.loads(registration.read_text());before,_=certificates()
    prior_path=RESEARCH/'reviews/DG-020/native-decisions.json'
    if sha_file(prior_path)!='f259c416bc25edaf852b085648c9331d720f0675ed35de3114890805a89a665a':raise ValueError('prior native evidence changed')
    prior=json.loads(prior_path.read_text())
    for name,h in prior['source_sha256'].items():
        if sha_file(ROOT/'python'/name)!=h:raise ValueError('DG-020 implementation changed')
    root=Path(a.raw);root.mkdir(parents=True,exist_ok=False)
    feature_root=ROOT/'build/neural-memory-cg003-full-features';m=json.loads((feature_root/'manifest.json').read_text())
    if digest(m)!=FEATURE_DIGEST or sha_file(feature_root/'output-head.npy')!=m['output_head_sha256']:raise ValueError('frozen feature/head identity changed')
    head=torch.from_numpy(np.load(feature_root/'output-head.npy',allow_pickle=False));scale=m['logit_scale']
    corpus=RESEARCH/'data/JB-001';manifest=json.loads((corpus/'manifest.json').read_text())
    if digest(manifest)!=DATA_DIGEST:raise ValueError('corpus changed')
    p=corpus/'train.inputs.jsonl'
    if sha_file(p)!=manifest['file_sha256'][p.name]:raise ValueError('natural inputs changed')
    inputs={r['id']:r for r in map(json.loads,p.read_text().splitlines())}
    gguf=ROOT/'models/bitcpm4-0.5b-tq2_0.gguf';codec=AppendValueCodec(ROOT/'build/tok_probe',gguf,m['tokenizer_sha256'])
    tokens=NativeFeatureBank(feature_root/'tokens',expected_encoder_id=m['encoder_identity']['encoder_id'],expected_manifest_sha256=m['token_manifest_digest'])
    binding=ModelBinding(BACKBONE_SHA256,m['tokenizer_sha256'],m['encoder_identity']['encoder_id'],sha_file(registration));key=digest(vars(binding))
    torch.set_num_threads(4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(reg['initialization']['reader_seed']);reader=FineSpanReader(2048,1024,key,width=64).eval()
        torch.manual_seed(reg['initialization']['uncertainty_seed']);branch=QueryOnlyUncertainty(1024,key).eval()
    reader_digest=tensor_digest(reader.state_dict());initial=tensor_digest(branch.state_dict())
    if reader_digest!=prior['initial_parameter_digest']:raise ValueError('reader initialization changed')
    # Separate artificial nonzero module; never overwrite the actual zero branch.
    fixture=copy.deepcopy(branch)
    with torch.no_grad():fixture.output.weight.copy_(torch.eye(1024)*.001)
    fixture_digest=tensor_digest(fixture.state_dict());rows=[];nonzero=[];gradient_rows=[];calls=0
    try:
        for index,previous in enumerate(prior['records']):
            rid=previous['id'];runtime=inputs[rid];directory=root/rid;directory.mkdir()
            old_dir=Path(prior['raw_root'])/rid
            for name,h in previous['raw_sha256'].items():
                if sha_file(old_dir/name)!=h:raise ValueError('initial raw C artifact changed')
            prefix=encode_generation_input(runtime,codec.tokenizer).prompt_token_ids
            if list(prefix)!=previous['prefix_ids']:raise ValueError('not original natural prefix')
            h,b,c,z=continuous_file(old_dir,'initial-base',prefix,False);rh,rb=reference_file(old_dir,'initial-reference',prefix)
            if not torch.equal(h,rh) or not torch.equal(b,rb) or torch.count_nonzero(c):raise ValueError('initial reference mismatch')
            query,sources=encoder_texts(runtime);q=tokens.rows[text_key(query)]
            if codec.tokenizer.encode(query,True)!=q.token_ids.tolist():raise ValueError('query tokens changed')
            if sources:
                s=tokens.rows[text_key(sources[0])]
                if codec.tokenizer.encode(sources[0],True)!=s.token_ids.tolist():raise ValueError('source tokens changed')
                pieces=codec.decode_pieces(s.token_ids[1:].tolist());allowed=payload_mask(runtime['episodes'][0],sources[0],pieces)
                ranges=source_alignment(runtime['episodes'][0],sources[0],pieces,allowed);raw=runtime['episodes'][0]['text'].encode();source=torch.from_numpy(s.features.copy())
            else:ranges=();raw=b'';allowed=[];source=torch.empty(0,2048)
            x=SpanFeatures(torch.from_numpy(q.features.copy()),source,torch.tensor(allowed,dtype=torch.bool),key,hashlib.sha256(raw).hexdigest(),digest(runtime['context']))
            layout=ByteLayout(x,raw,ranges);snapshot=PayloadSnapshot(binding,reader_digest,'native',rid,0,(FactPayload('source',raw),) if raw else ())
            reply=ReplyBinding('native',rid,x.context_sha256);core=AutonomousValueController(reader,x,layout,snapshot,reply,codec,prefix)
            controller=ResearchUncertaintyController(core,branch,head,scale,diagnostic_only=True)
            frame=LiveFrame(prefix,h[0],b[0],key,FrameOrigin.NATIVE_FRESH);decision=controller.propose(frame);out=controller.pending_output
            if type(decision) is not UncertaintyDecision or out is None:raise ValueError('actual neural handoff did not occur')
            delta=out.residual.detach().numpy().astype('<f4')
            forward(str(ROOT/'build/memory_continuous_probe'),str(gguf),prefix,delta,directory,'zero',True);calls+=1
            hh,bb,cc,zz=continuous_file(directory,'zero',prefix,True,out.residual[None])
            if not all(exact(u.numpy(),v.numpy()) for u,v in ((h,hh),(b,bb),(b,zz),(out.logits[None],zz))) or torch.count_nonzero(cc) or torch.count_nonzero(out.residual):
                raise ValueError('zero branch is not bitwise identity')
            controller.commit(decision,decision.token)
            # Labels are loaded ONLY AFTER both neural forwards and native parity.
            labels_path=corpus/'train.labels.jsonl'
            if sha_file(labels_path)!=manifest['file_sha256'][labels_path.name]:raise ValueError('training labels changed')
            label=next(r for r in map(json.loads,labels_path.read_text().splitlines()) if r['id']==rid)
            if label['input_sha256']!=digest(runtime):raise ValueError('label belongs to different input')
            if label['state']=='insufficient':
                branch.zero_grad(set_to_none=True);output=branch(frame,prefix,head,scale)
                target=codec.encode_value(label['response'].encode())[0]
                loss=F.cross_entropy(output.logits[None],torch.tensor([target]));loss.backward()
                grads={n:float(p.grad.abs().sum()) for n,p in branch.named_parameters()}
                if not all(bool(torch.isfinite(p.grad).all()) for p in branch.parameters()) or grads['output.weight']<=0 or grads['encoder.weight']!=0 or grads['encoder.bias']!=0:
                    raise ValueError('unexpected zero-initialization gradients')
                gradient_rows.append({'id':rid,'teacher_target_token':target,'loss':float(loss.detach()),'gradient_l1':grads,'passed':True})
            if index in reg['nonzero_fixture']['case_indices']:
                fixture.zero_grad(set_to_none=True);f=fixture(frame,prefix,head,scale);target=int(b.argmax())
                loss=F.cross_entropy(f.logits.double()[None],torch.tensor([target]));g=torch.autograd.grad(loss,f.residual,retain_graph=True)[0]
                loss.backward()
                if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum())>0 for p in fixture.parameters()):raise ValueError('nonzero fixture disconnected')
                fd=reg['nonzero_fixture']['finite_difference'];direction=g/g.norm();epsilon=fd['epsilon']
                values={};errors={}
                for name,residual,enabled in (('nonzero',f.residual,True),('disabled',f.residual,False),
                    ('plus',f.residual.detach()+epsilon*direction,True),('minus',f.residual.detach()-epsilon*direction,True)):
                    d=residual.detach().numpy().astype('<f4')
                    forward(str(ROOT/'build/memory_continuous_probe'),str(gguf),prefix,d,directory,name,enabled);calls+=1
                    # Disabled permits a nonzero input delta, so read this one via the generator's validated return below.
                    if enabled:
                        nh,nb,nc,nz=continuous_file(directory,name,prefix,True,residual[None])
                    else:
                        # The C probe ignores the nonzero delta when disabled; validate output file directly.
                        data=(directory/(name+'.bin')).read_bytes();arr=np.frombuffer(data,dtype='<f4',offset=20).copy()
                        nh=torch.from_numpy(arr[:1024])[None];nb=torch.from_numpy(arr[1024:1024+73448])[None]
                        nc=torch.from_numpy(arr[1024+73448:1024+2*73448])[None];nz=torch.from_numpy(arr[1024+2*73448:])[None]
                    if not exact(nh.numpy(),h.numpy()) or not exact(nb.numpy(),b.numpy()):raise ValueError('residual changed base hidden')
                    if name=='nonzero':errors[name]=close(nz,f.logits[None])
                    if name=='disabled' and (not exact(nz.numpy(),b.numpy()) or torch.count_nonzero(nc)):raise ValueError('disabled path changed base')
                    values[name]=float(F.cross_entropy(nz.double(),torch.tensor([target])))
                derivative=(values['plus']-values['minus'])/(2*epsilon);expected=float(g.norm())
                if abs(derivative-expected)>fd['atol']+fd['rtol']*abs(expected) or derivative<=0:raise ValueError('native finite difference failed')
                nonzero.append({'id':rid,'origin':'artificial_nonzero_fixture','logit_max_error':errors['nonzero'],
                    'derivative_native':derivative,'derivative_autograd':expected,'objective':'CE to base argmax, not answer accuracy','passed':True})
            rows.append({'id':rid,'neural_route':2,'branch':decision.branch,'residual_zero':True,'native_base_bitwise_equal':True,
                         'selected_token':decision.token,'committed_token_ids':controller.generated_tokens,'full_reply':False})
            print(json.dumps({'id':rid,'zero_parity':True,'handoff':True}),flush=True)
        codec.verify_identity()
    finally:codec.close()
    if certificates()[0]!=before or tensor_digest(reader.state_dict())!=reader_digest or tensor_digest(branch.state_dict())!=initial or tensor_digest(fixture.state_dict())!=fixture_digest:
        raise ValueError('parameters or protected artifacts changed')
    if len(rows)!=12 or len(gradient_rows)!=4 or len(nonzero)!=2:raise ValueError('registered inventory incomplete')
    result={'format':'dg021-query-only-uncertainty-check-v1','registration_sha256':sha_file(registration),
        'source_sha256':{n:sha_file(ROOT/'python'/n) for n in ('query_only_uncertainty.py','uncertainty_reply_controller.py','check_query_only_uncertainty.py')},
        'raw_root':str(root.resolve()),'raw_sha256':{str(p.relative_to(root)):sha_file(p) for p in sorted(root.rglob('*')) if p.is_file()},
        'rows':rows,'first_token_gradient_checks':gradient_rows,'nonzero_fixtures':nonzero,'C_forward_calls':calls,
        'optimizer_steps':0,'parameters_unchanged':True,'uncertainty_parameter_digest':initial,'fixture_parameter_digest':fixture_digest,
        'full_replies_generated':0,'autonomous_refusal_accuracy_measured':False,'cross_step_kv_reuse':False,
        'internal_prefill_kv_buffers':True,'passed':True,'V2_complete':False,'deployment_approved':False}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True);run(p.parse_args())
