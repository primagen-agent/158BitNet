"""Independent raw-file replay of CG-003 native generation; no semantic grading."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np
import torch

from c_tokenizer import CTokenizer
from continuous_memory import continuous_logits
from diagnose_native_generation_gradient import exact
from episode_memory_inputs import encoder_texts
from eval_joint_baseline import fixed_panel
from joint_checkpoint_transport import restore_certified_initial
from joint_memory_model import create_joint_model
from joint_optimizer import load_checkpoint, FEATURE_DIGEST, tensor_digest
from native_continuous_generation import tokenizer_stop_ids, check_trace
from native_memory_encoder import sha_file, read_encoded, BACKBONE_SHA256
from native_token_payload import input_binding, native_token_input
from neural_memory_generation import encode_generation_input
from prepare_neural_memory_protocol import digest
from review_joint_generation import BASELINE_SHA
from token_memory_composition import mix_content_copy


def continuous_file(directory, name, prefix, enabled, delta=None):
    n=len(prefix);wire=(directory/(name+'.input')).read_bytes()
    want=b'BNCI0001'+struct.pack('<I',n)+np.asarray(prefix,dtype='<i4').tobytes()+struct.pack('<I',int(enabled))
    if wire[:len(want)]!=want or len(wire)!=len(want)+4096:raise ValueError('continuous input wire changed')
    injected=np.frombuffer(wire,dtype='<f4',offset=len(want)).copy()
    if delta is not None and not exact(injected,delta.detach().cpu().numpy()[0]):raise ValueError('trained residual differs')
    if not enabled and np.any(injected):raise ValueError('nonzero base residual')
    raw=(directory/(name+'.bin')).read_bytes()
    if raw[:8]!=b'BNCO0001' or struct.unpack_from('<III',raw,8)!=(n,1024,73448) or len(raw)!=20+4*(1024+3*73448):
        raise ValueError('continuous output wire changed')
    values=np.frombuffer(raw,dtype='<f4',offset=20)
    if not np.isfinite(values).all():raise ValueError('nonfinite continuous output')
    check_trace(directory/(name+'.log'),n)
    return [torch.from_numpy(v.copy())[None] for v in
        (values[:1024],values[1024:1024+73448],values[1024+73448:1024+2*73448],values[1024+2*73448:])]


def reference_file(directory,name,prefix):
    n=len(prefix);wire=(directory/(name+'.input')).read_bytes()
    want=(b'BNGI0001'+struct.pack('<I',n)+np.asarray(prefix,dtype='<i4').tobytes()+
          struct.pack('<III',23,n-1,0)+np.zeros(1024,dtype='<f4').tobytes())
    if wire!=want:raise ValueError('reference prefix or controls changed')
    raw=(directory/(name+'.bin')).read_bytes()
    if raw[:8]!=b'BNGO0001' or len(raw)!=28+4*(3072+73448) or struct.unpack_from('<IIIII',raw,8)!=(n,1024,73448,23,n-1):
        raise ValueError('reference geometry changed')
    v=np.frombuffer(raw,dtype='<f4',offset=28)
    if not np.isfinite(v).all():raise ValueError('nonfinite reference')
    return torch.from_numpy(v[2048:3072].copy())[None],torch.from_numpy(v[3072:].copy())[None]


def close(a,b):
    if not torch.allclose(a,b,atol=1e-5,rtol=1e-4) or not torch.equal(a.argmax(-1),b.argmax(-1)):
        raise ValueError('native/model numerical mismatch')
    return float((a-b).abs().max())


def audit_case(directory,runtime,prediction,baseline,model,head,scale,manifest,tokenizer):
    directory=Path(directory);backend=json.loads((directory/'backend.json').read_text())
    if (backend['cross_step_kv_reuse'] or backend['prefix_bank_used'] or backend['gold_answer_prefix_used'] or
            backend['enabled'] is not True or backend['route_policy']!=model.route_policy or
            backend['identity']['encoder']!=manifest['encoder_identity'] or
            backend['identity']['backbone_sha256']!=BACKBONE_SHA256):raise ValueError('unqualified backend identity')
    for name,want in backend['raw_sha256'].items():
        path=directory/name
        if not path.resolve().is_relative_to(directory.resolve()) or sha_file(path)!=want:raise ValueError('raw artifact changed')
    query,sources=encoder_texts(runtime);texts=[query,*sources]
    wanted=struct.pack('<I',len(texts))+b''.join(struct.pack('<I',len(t.encode()))+t.encode() for t in texts)
    if (directory/'inputs/batch-0.input').read_bytes()!=wanted:raise ValueError('memory/query source changed')
    encoded=read_encoded(directory/'inputs/batch-0.bin',len(texts))
    if tokenizer.encode(query,True)!=encoded[0].token_ids.tolist():raise ValueError('query encoding mismatch')
    memory=native_token_input(encoded[0],encoded[1] if sources else None,runtime['episodes'][0] if sources else None,
        sources[0] if sources else None,tokenizer,model.reader.binding)
    state=model.prefill(memory);route=int(state.core.state_logits.argmax())
    request=encode_generation_input(runtime,tokenizer);prefix=request.prompt_token_ids
    steps=backend['steps'];ids=prediction['generated_token_ids'];stops=tokenizer_stop_ids(tokenizer)
    if (prediction['id']!=runtime['id'] or prediction['input_sha256']!=digest(runtime) or
            not 1<=len(ids)<=64 or len(ids)!=len(steps) or prediction['selected_route']!=route or
            prediction['initial_state_probabilities']!=state.core.state_logits.exp().tolist() or
            any(t in stops for t in ids[:-1])):raise ValueError('prediction/route inventory changed')
    errors=[];mass_positions=0;copy_effect=[]
    for i,step in enumerate(steps):
        if (step['prefix_ids']!=list(prefix) or step['predicted_token_id']!=ids[i] or
                step['selected_route']!=route or step['effective_route']!=route or
                step['state_probabilities']!=state.core.state_logits.exp().tolist() or not step['base_reference_checked']):
            raise ValueError('not actual prefix and fixed predicted route')
        name=f'step-{i:03d}';h,b,correction,base=continuous_file(directory,name+'-base',prefix,False)
        rh,rb=reference_file(directory,name+'-reference',prefix)
        if not all(exact(x.numpy(),y.numpy()) for x,y in ((h,rh),(b,rb),(base,b))) or torch.count_nonzero(correction):
            raise ValueError('base/reference mismatch')
        def project(suffix,delta):
            hh,bb,cc,out=continuous_file(directory,name+suffix,prefix,True,delta)
            if not exact(hh.numpy(),h.numpy()) or not exact(bb.numpy(),b.numpy()):raise ValueError('projection base changed')
            expected,expected_correction=continuous_logits(b,delta,head,scale)
            close(out,expected)
            if not torch.allclose(cc,expected_correction,atol=1e-5,rtol=1e-4):raise ValueError('correction mismatch')
            return out
        if i==0:
            branches=model.branches(h,b,head,scale,state)
            content,uncertainty=model.roles.branches(h,state.core.prepared)
            cc=project('-qual-content',content);cu=project('-qual-uncertainty',uncertainty)
            close(mix_content_copy(cc,branches.positions,branches.copy_mass,memory.token_ids),branches.supported)
            close(cu,branches.uncertainty)
        elif step['branch_only_qualification'] is not None:raise ValueError('unexpected branch qualification')
        mass=0.
        if route==0:actual=b
        elif route==2:actual=project('-output',model.roles.uncertainty(h))
        else:
            delta=model.roles.content.residual(model.roles.layers-1,h,state.core.prepared)
            content=project('-output',delta);pointer=model.reader.proposal(h,b,state.core.pointer)
            actual=mix_content_copy(content,pointer.position_probabilities,pointer.copy_mass,memory.token_ids)
            mass=float(pointer.copy_mass[0,0])
            token=int(actual.argmax());p=pointer.position_probabilities[0,1:].double()
            w=pointer.copy_mass.double().clamp(torch.finfo(torch.float64).eps,1-torch.finfo(torch.float64).eps)[0,0]
            copy_p=p[memory.token_ids==token].sum()/p.sum().clamp_min(torch.finfo(torch.float64).tiny)
            copy_effect.append({'position':i,'selected_token':token,'base_top1':int(b.argmax()),
                'content_only_top1':int(content.argmax()),'copy_mass':mass,
                'selected_token_content_contribution':float((1-w)*content.double().softmax(-1)[0,token]),
                'selected_token_copy_contribution':float(w*copy_p),
                'scope':'same actual prefix, not an alternative full reply or accuracy score'})
        error=close(actual,model.read(h,b,head,scale,state))
        if int(actual.argmax())!=ids[i] or step['copy_mass']!=mass:raise ValueError('prediction/copy replay mismatch')
        if step['output_equals_base']!=exact(actual.numpy(),base.numpy()):raise ValueError('bypass flag mismatch')
        want_trace={'initial_position':0,'final_position':len(prefix),'eval_calls':1,'prefix_tokens':len(prefix)}
        expected_traces=1+(2 if i==0 else 0)+(route!=0)
        if step['native_traces']!=[want_trace]*expected_traces:raise ValueError('fresh trace count mismatch')
        mass_positions+=mass>0;errors.append(error);prefix=prefix+(ids[i],)
    stopped=ids[-1] in stops;visible=ids[:-1] if stopped else ids;raw=b''.join(tokenizer.decode_pieces(visible))
    try:text=raw.decode('utf-8');valid=True
    except UnicodeDecodeError:text=raw.decode('utf-8',errors='replace');valid=False
    if (not stopped and len(ids)!=64 or prediction['visible_token_ids']!=visible or prediction['raw_text_hex']!=raw.hex() or
            prediction['text']!=text or prediction['truncated']!= (not stopped) or prediction['utf8_complete']!=valid or
            prediction['finish_reason']!=('stop_token' if stopped else 'max_new_tokens')):raise ValueError('reply bytes/termination changed')
    normal_equal=all(prediction[k]==baseline[k] for k in ('generated_token_ids','raw_text_hex'))
    if route==0 and not normal_equal:raise ValueError('normal reply differs from baseline')
    return {'id':runtime['id'],'positions':len(ids),'selected_route':route,'copy_positions':mass_positions,
        'max_error':max(errors),'normal_baseline_equal':normal_equal if route==0 else None,
        'truncated':prediction['truncated'],'passed':True,'copy_effect':copy_effect}


def run(a):
    torch.set_num_threads(4)
    if sha_file(a.status)!=a.status_sha256 or sha_file(a.baseline)!=BASELINE_SHA:raise ValueError('status/baseline changed')
    status=json.loads(Path(a.status).read_text());binding=status['binding'];certs={c['step']:c for c in status['checkpoints']}
    for name,want in binding['source_sha256'].items():
        if sha_file(Path(__file__).parent/name)!=want:raise ValueError('training source changed')
    m=json.loads((Path(a.features)/'manifest.json').read_text())
    if (digest(m)!=FEATURE_DIGEST or sha_file(Path(a.features)/'output-head.npy')!=m['output_head_sha256'] or
            sha_file(a.gguf)!=BACKBONE_SHA256 or sha_file(a.tok_probe)!=m['tokenizer_sha256']):raise ValueError('artifact identity changed')
    identity=input_binding(m['encoder_identity'],m['tokenizer_sha256'])
    if identity!=binding['encoder_binding']:raise ValueError('encoder binding changed')
    model=create_joint_model(identity,m['blocked_ids'],a.arm).eval();parent=Path(a.status).parent
    restore_certified_initial(parent/certs[0]['file'],expected_sha256=certs[0]['sha256'],model=model,binding=binding,initial_digest=certs[0]['state_digest'])
    load_checkpoint(parent/certs[50]['file'],expected_sha256=certs[50]['sha256'],model=model,expected_binding=binding,expected_initial_digest=certs[0]['state_digest'])
    initial=tensor_digest(model.state_dict());head=torch.from_numpy(np.load(Path(a.features)/'output-head.npy',allow_pickle=False))
    baseline=json.loads(Path(a.baseline).read_text());audit=json.loads(Path(a.panel_audit).read_text())
    if sha_file(a.panel_audit)!=baseline['artifact_sha256']['audit']:raise ValueError('panel changed')
    records,panel=fixed_panel(a.corpus,audit);root=Path(a.generation)
    if not a.partial and not (root/'generation.json').exists():raise ValueError('complete report required')
    report=json.loads((root/'generation.json').read_text()) if (root/'generation.json').exists() else None
    if report is not None and (report['checkpoint_sha256']!=certs[50]['sha256'] or report['panel']!=panel or
                              report['status_sha256']!=a.status_sha256 or len(report['predictions'])!=24):raise ValueError('generation report changed')
    rows=[];tokenizer=CTokenizer(a.tok_probe,a.gguf)
    try:
        with torch.no_grad():
            for i,r in enumerate(records):
                directory=root/f'case-{i:02d}';file=directory/'generation.json'
                if not file.exists():
                    if a.partial:continue
                    raise ValueError('missing case')
                prediction=json.loads(file.read_text())
                if report is not None and prediction!=report['predictions'][i]:raise ValueError('per-case/report mismatch')
                row=audit_case(directory,r,prediction,baseline['predictions'][i],model,head,m['logit_scale'],m,tokenizer)
                rows.append(row);print(json.dumps({'verified_cases':len(rows),'positions':row['positions']}),flush=True)
    finally:
        tokenizer._proc.terminate();tokenizer._proc.wait();tokenizer._proc.stdin.close();tokenizer._proc.stdout.close()
    if tensor_digest(model.state_dict())!=initial:raise ValueError('audit changed weights')
    complete=report is not None and len(rows)==24
    result={'format':'cg003-native-generation-audit-v1','arm':a.arm,'expected_cases':24,'complete':complete,
        'passed':complete and all(r['passed'] for r in rows),'cases':rows,'positions':sum(r['positions'] for r in rows),
        'status_sha256':a.status_sha256,'checkpoint_sha256':certs[50]['sha256'],
        'generation_sha256':sha_file(root/'generation.json') if report else None,
        'semantic_accuracy_measured':False,'source_sha256':sha_file(__file__)}
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'complete':complete,'cases':len(rows),'positions':result['positions']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('status','status-sha256','baseline','panel-audit','corpus','features','gguf','tok-probe','generation','output'):
        p.add_argument('--'+n,required=True)
    p.add_argument('--arm',choices=('joint_aux','joint_product'),required=True);p.add_argument('--partial',action='store_true')
    run(p.parse_args())
