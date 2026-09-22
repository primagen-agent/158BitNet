"""DG-007: zero-update controls on frozen C features and recorded prefixes.

Replayed alternative next tokens are off-policy diagnostics, never generated
answers. No optimizer, source selection, reference answer in forward, or model
export. Keep the original CG-001 implementation and checkpoint untouched.
"""
import argparse
import copy
import json
from pathlib import Path
import struct

import numpy as np
import torch

from availability_controls import controlled_read, prefill_decision, initialize_content_output
from availability_supervision import supervised_loss
from continuous_memory import continuous_logits
from eval_availability_checkpoint import BASELINE_SHA256, load_checkpoint
from memory_availability import AvailabilityMemoryFusion
from native_memory_encoder import read_encoded, sha_file
from train_availability_memory import TrainingPackage


def norm(gradients):
    return float(sum((g.detach().square().sum() for g in gradients if g is not None),torch.tensor(0.)).sqrt())


def controller(module):
    return tuple(module.state_encoder.parameters())+tuple(module.state_head.parameters())


def loss_row(module,features,target,head,scale,mode):
    prepared=module.prepare(features.source)
    decision=prefill_decision(module,features.hidden[:1],prepared) if mode=='fixed_prefill' else None
    read=controlled_read(module,23,features.hidden,prepared,decision=decision,isolate=mode=='isolated')
    logits,_=continuous_logits(features.base_logits,read.residual,head,scale)
    losses=supervised_loss(logits,read.state_logits[:1],target,state_coefficient=.2)
    classifier=controller(module)
    branches=(module.content.output.weight,module.uncertainty_output.weight,module.content.layer_gain)
    gradients=torch.autograd.grad(losses['generation'],classifier+branches,allow_unused=True,retain_graph=True)
    state_gradients=torch.autograd.grad(.2*losses['state'],classifier,allow_unused=True)
    if any(g is not None and not torch.isfinite(g).all() for g in (*gradients,*state_gradients)):
        raise ValueError('nonfinite control gradient')
    return {'nll':float(losses['generation'].detach()),'state_ce':float(losses['state'].detach()),
            'generation_classifier_grad_l2':norm(gradients[:-3]),'state_classifier_grad_l2':norm(state_gradients),
            'content_output_grad_l2':norm(gradients[-3:-2]),'uncertainty_output_grad_l2':norm(gradients[-2:-1]),
            'content_gain_grad_l2':norm(gradients[-1:]),
            'zero_residual':not bool(torch.count_nonzero(read.residual)),
            'base_logits_exact':bool(torch.equal(logits,features.base_logits))},logits.detach()


def training_controls(module,package,ids,seed):
    records={r['id']:r for r in package.records}; rows=[]
    torch.manual_seed(seed); initial=AvailabilityMemoryFusion(1024,24,8)
    alternative=copy.deepcopy(initial); initialize_content_output(alternative)
    for cid in ids:
        features,target=package.sample(records[cid],torch.device('cpu'))
        modes={}; reference=None
        for mode in ('unchanged','isolated','fixed_prefill'):
            result,logits=loss_row(module,features,target,package.head,package.scale,mode)
            if mode=='unchanged': reference=logits
            result['same_logits_as_original']=bool(torch.equal(logits,reference))
            modes[mode]=result
        initialization={}
        for name,candidate in (('original',initial),('zero_content_output',alternative)):
            initialization[name],_=loss_row(candidate,features,target,package.head,package.scale,'isolated')
        has_source=bool(len(features.source))
        iso=modes['isolated']; alt=initialization['zero_content_output']
        passed=(iso['same_logits_as_original'] and iso['generation_classifier_grad_l2']==0 and
                iso['state_classifier_grad_l2']>0 and iso['uncertainty_output_grad_l2']>0 and
                (not has_source or iso['content_output_grad_l2']>0) and
                all(v['base_logits_exact'] and v['zero_residual'] for v in initialization.values()) and
                (not has_source or initialization['original']['content_gain_grad_l2']>0) and
                alt['uncertainty_output_grad_l2']>0 and (not has_source or alt['content_output_grad_l2']>0))
        rows.append({'id':cid,'target_state':target.state_index,'has_source':has_source,'modes':modes,
                     'initialization':initialization,'software_gradient_gate_passed':passed})
        print(json.dumps({'phase':'loss_controls','done':len(rows),'total':len(ids),'passed':passed}),flush=True)
    return rows


def recorded_prefix_controls(module,package,root,ids,tolerance):
    root=Path(root); generation=json.loads((root/'generation.json').read_text())
    if [r['id'] for r in generation['predictions']]!=ids: raise ValueError('recorded panel changed')
    rows=[]
    for prediction in generation['predictions']:
        directory=root/Path(prediction['backend_evidence']).parent
        backend=json.loads((directory/'backend.json').read_text())
        if backend['identity']['encoder']!=package.manifest['encoder_identity'] or backend['cross_step_kv_reuse']:
            raise ValueError('source encoder/cache binding mismatch')
        for name,expected in backend['raw_sha256'].items():
            path=directory/name
            if not path.resolve().is_relative_to(directory.resolve()) or sha_file(path)!=expected:
                raise ValueError('raw generation evidence changed')
        source_path=directory/'sources/batch-0.bin'
        steps=[]; decision=None; previous=None
        with torch.no_grad():
            prepared=module.prepare(torch.from_numpy(read_encoded(source_path,1)[0].features)) if source_path.exists() else None
            for i,trace in enumerate(backend['steps']):
                prefix=trace['prefix_ids']; n=len(prefix)
                fresh={'initial_position':0,'final_position':n,'eval_calls':1,'prefix_tokens':n}
                if not trace['base_reference_checked'] or trace['native_traces']!=[fresh,fresh] or (previous is not None and prefix!=previous):
                    raise ValueError('unqualified native generation sequence')
                previous=prefix+[trace['predicted_token_id']]
                wire=(directory/f'step-{i:03d}-output.input').read_bytes()
                raw=(directory/f'step-{i:03d}-output.bin').read_bytes()
                if (wire[:8]!=b'BNCI0001' or struct.unpack_from('<I',wire,8)[0]!=n or
                    struct.unpack_from('<I',wire,12+4*n)[0]!=1 or
                    np.frombuffer(wire,dtype='<i4',count=n,offset=12).tolist()!=prefix or
                    raw[:8]!=b'BNCO0001' or struct.unpack_from('<III',raw,8)!=(n,1024,73448)):
                    raise ValueError('native wire mismatch')
                delta=np.frombuffer(wire,dtype='<f4',offset=16+4*n)
                arrays=np.frombuffer(raw,dtype='<f4',offset=20)
                hidden=torch.from_numpy(arrays[:1024].copy())[None]
                base=torch.from_numpy(arrays[1024:1024+73448].copy())[None]
                actual=arrays[1024+2*73448:]
                read=controlled_read(module,23,hidden,prepared)
                if decision is None: decision=prefill_decision(module,hidden,prepared)
                fixed=controlled_read(module,23,hidden,prepared,decision=decision)
                if not np.array_equal(read.residual[0].numpy(),delta): raise ValueError('checkpoint replay changed')
                residuals=torch.cat((read.residual,read.uncertainty_residual,read.content_residual,fixed.residual))
                logits,correction=continuous_logits(base.expand(4,-1),residuals,package.head,package.scale)
                parity=bool(np.allclose(logits[0].numpy(),actual,**tolerance))
                top=logits.argmax(-1).tolist(); original=int(actual.argmax())
                passed=parity and top[0]==original and original==trace['predicted_token_id']
                steps.append({'position':i,'parity_passed':passed,'max_logit_error':float(np.max(np.abs(logits[0].numpy()-actual))),
                    'next_token_ids':dict(zip(('unchanged','remove_content','remove_uncertainty','fixed_prefill'),top)),
                    'content_logit_l2':float(correction[2].norm()),'uncertainty_logit_l2':float(correction[1].norm()),
                    'current_probabilities':read.state_probabilities[0].tolist(),'fixed_probabilities':decision.probabilities[0].tolist()})
        rows.append({'id':prediction['id'],'positions':steps,'all_native_replay_passed':all(s['parity_passed'] for s in steps),
                     'changed_next_token_counts':{mode:sum(s['next_token_ids'][mode]!=s['next_token_ids']['unchanged'] for s in steps)
                         for mode in ('remove_content','remove_uncertainty','fixed_prefill')}})
        print(json.dumps({'phase':'recorded_prefix_controls','done':len(rows),'total':len(ids)}),flush=True)
    return rows


def run(a):
    root=Path(a.output); root.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(a.experiment).read_text())
    if config['id']!='DG-007' or config['optimizer_steps']!=0: raise ValueError('unregistered diagnostic')
    if sha_file(a.baseline)!=BASELINE_SHA256: raise ValueError('baseline changed')
    repo=Path(__file__).resolve().parents[1]
    if sha_file(Path(a.generation)/'generation.json')!=sha_file(repo/'training/memory/neural-system/reviews/CG-001/step50.json'):
        raise ValueError('generation report changed')
    torch.set_num_threads(4); module,checkpoint=load_checkpoint(a.checkpoint)
    package=TrainingPackage(a.features,checkpoint['feature_package_digest'])
    ids=[r['id'] for r in json.loads(Path(a.baseline).read_text())['predictions']]
    if len(ids)!=16: raise ValueError('fixed denominator changed')
    before={k:p.detach().clone() for k,p in module.named_parameters()}
    training=training_controls(module,package,ids,config['initialization_seed'])
    replay=recorded_prefix_controls(module,package,a.generation,ids,config['numeric_tolerance'])
    unchanged=all(torch.equal(p,before[k]) for k,p in module.named_parameters())
    if not unchanged: raise ValueError('diagnostic modified checkpoint')
    passed=all(r['software_gradient_gate_passed'] for r in training) and all(r['all_native_replay_passed'] for r in replay)
    report={'experiment':config,'training_controls':training,'recorded_prefix_controls':replay,
            'software_gradient_numeric_gate_passed':passed,'original_parameters_unchanged':unchanged,
            'optimizer_steps':0,'new_free_generated_answers':0,'memory_accuracy_measured':False,
            'training_approved':False,'deployment_approved':False,
            'artifact_sha256':{key:sha_file(value) for key,value in {'checkpoint':a.checkpoint,'baseline':a.baseline,'experiment':a.experiment}.items()},
            'feature_package_digest':checkpoint['feature_package_digest'],
            'source_sha256':{name:sha_file(repo/'python'/name) for name in ('availability_controls.py','diagnose_availability_controls.py')}}
    with (root/'report.json').open('x') as f: json.dump(report,f,indent=2); f.write('\n')
    print(json.dumps({'passed':passed,'optimizer_steps':0,'positions':sum(len(r['positions']) for r in replay)}),flush=True)
    if not passed: raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','checkpoint','features','baseline','generation','output'): p.add_argument('--'+name,required=True)
    run(p.parse_args())


if __name__=='__main__': main()
