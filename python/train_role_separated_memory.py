"""CG-002 bounded frozen-C-feature trainer. CPU is preflight-only; no resume."""
import argparse
import json
from pathlib import Path
import random

import torch

from eval_availability_checkpoint import BASELINE_SHA256
from memory_role_separated import FORMAT,RoleSeparatedMemory,forward_roles,role_loss
from native_memory_encoder import BACKBONE_SHA256,sha_file
from neural_memory_generation import TEMPLATE_VERSION
from prepare_neural_memory_protocol import digest
from train_availability_memory import TrainingPackage,ForwardFeatures,SOURCE_FILES as BASE_SOURCES

FEATURE_DIGEST='8f5b2d2ae685a7995d45422e661a8f59811357186d0cff370211eb596d5f36a6'
SOURCES=tuple(sorted(set(BASE_SOURCES)|{'python/memory_role_separated.py','python/availability_controls.py',
             'python/train_role_separated_memory.py','python/qualify_role_separated_native.py'}))


def source_identity():
    root=Path(__file__).resolve().parents[1]
    return {p:sha_file(root/p) for p in SOURCES}


def diagnostic_model(seed):
    torch.manual_seed(seed); model=RoleSeparatedMemory()
    with torch.no_grad():
        model.content.output.weight.normal_(std=.001)
        model.uncertainty_output.weight.normal_(std=.001)
    return model


def qualification_records(package):
    # Cover ordinary chat both without and with supplied memory. Empty content
    # is structurally zero and must not be required to have an output gradient.
    rows=[next(r for r in package.records if r['targets']['state_index']==i) for i in range(3)]
    rows.append(next(r for r in package.records if r['targets']['state_index']==0 and r['source_texts']))
    if len({r['id'] for r in rows})!=4: raise ValueError('qualification fixture inventory changed')
    return rows


def preflight(package,device):
    cpu=diagnostic_model(3180923); candidate=RoleSeparatedMemory().to(device)
    candidate.load_state_dict(cpu.state_dict()); head=package.head.to(device)
    rows=[]
    for record in qualification_records(package):
        state=record['targets']['state_index']
        features,target=package.sample(record,'cpu')
        gpu=ForwardFeatures(*(t.to(device) for t in (features.hidden,features.base_logits,features.source)))
        with torch.no_grad():
            expected=forward_roles(cpu,features,package.head,package.scale)
            actual=forward_roles(candidate,gpu,head,package.scale)
        branch_match=all(torch.allclose(getattr(expected,k),getattr(actual,k).cpu(),atol=1e-5,rtol=1e-4) for k in ('content','uncertainty'))
        state_match=torch.allclose(expected.state_logits.softmax(-1),actual.state_logits.cpu().softmax(-1),atol=1e-5,rtol=1e-4)
        route_match=int(expected.state_logits.argmax())==int(actual.state_logits.argmax())
        candidate.zero_grad(set_to_none=True)
        losses=role_loss(forward_roles(candidate,gpu,head,package.scale),target); losses['weighted_total'].backward()
        grads=[p.grad for p in candidate.parameters() if p.grad is not None]
        finite=bool(grads) and all(torch.isfinite(g).all() for g in grads)
        required=[candidate.state_head.weight]
        required+=(([candidate.content.output.weight] if len(features.source) else [])+[candidate.uncertainty_output.weight] if state==0 else
                   [candidate.content.output.weight] if state==1 else [candidate.uncertainty_output.weight])
        live=all(p.grad is not None and float(p.grad.norm())>0 for p in required)
        rows.append({'id':record['id'],'state':state,'source_present':bool(len(features.source)),'branch_parity':bool(branch_match),'state_parity':bool(state_match),
                     'predicted_route_parity':route_match,'finite_gradients':bool(finite),'required_gradients_live':live,
                     'head_gradient_absent':head.grad is None,'max_logit_error':max(float((getattr(expected,k)-getattr(actual,k).cpu()).abs().max()) for k in ('content','uncertainty'))})
    return {'device':str(device),'torch_version':str(torch.__version__),'cuda_device':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'passed':all(all(r[k] for k in ('branch_parity','state_parity','predicted_route_parity','finite_gradients','required_gradients_live','head_gradient_absent')) for r in rows),
            'cases':rows,'source_sha256':source_identity(),'feature_package_digest':FEATURE_DIGEST,'optimizer_steps':0}


def validate_config(config):
    if config.get('id')!='CG-002' or config.get('backbone_sha256')!=BACKBONE_SHA256 or config.get('first_interval_steps')!=50 or config.get('initial_seed')!=3180923:
        raise ValueError('unregistered experiment/steps/seed/backbone')
    expected={'name':'AdamW','learning_rate':.0001,'weight_decay':.01,'batch_size':4,'gradient_clip_norm':1.,'precision':'float32','amp':False,'tf32':False}
    if config.get('planned_optimizer')!=expected: raise ValueError('unreviewed optimizer change')
    keys={'kv_reuse','lora','svd','nas_ng','backbone_trainable','source_text_in_prompt','locomo_training_or_selection','gold_serving_routing'}
    if config.get('constraints')!={key:False for key in keys}:
        raise ValueError('forbidden training constraint')


def require_ready(config,device_report,baseline,authorization):
    if config.get('training_approved') is not True or config.get('prerequisites_complete') is not True or not config.get('launch_command'):
        raise ValueError('CG-002 training is blocked')
    if not device_report['passed'] or device_report['device']!='cuda': raise ValueError('qualified CUDA required')
    if not baseline or sha_file(baseline)!=BASELINE_SHA256: raise ValueError('fixed native baseline required')
    evidence=json.loads(Path(authorization).read_text()) if authorization else {}
    if (evidence.get('source_sha256')!=source_identity() or evidence.get('feature_package_digest')!=FEATURE_DIGEST or
        evidence.get('baseline_sha256')!=BASELINE_SHA256 or evidence.get('approved_first_stage_steps')!=50 or
        evidence.get('launch_command')!=config['launch_command'] or evidence.get('experiment_sha256')!=digest(config)):
        raise ValueError('launch evidence/source/experiment mismatch')
    cert_path=Path(authorization).parent/evidence['native_certificate']
    if sha_file(cert_path)!=evidence['native_certificate_sha256']: raise ValueError('native certificate changed')
    cert=json.loads(cert_path.read_text())
    if not cert.get('passed') or cert.get('source_sha256')!=source_identity() or cert.get('feature_package_digest')!=FEATURE_DIGEST:
        raise ValueError('native qualification stale/incomplete')


def evaluate(model,package,head,device):
    rows=[]; model.eval()
    with torch.no_grad():
        for record in package.records:
            if record['split']!='dev': continue
            features,target=package.sample(record,device); outputs=forward_roles(model,features,head,package.scale)
            losses=role_loss(outputs,target)
            rows.append({'id':record['id'],'state_target':target.state_index,'state_prediction':int(outputs.state_logits.argmax()),
                         'state_ce':float(losses['state']),'branch_kind':losses['branch_kind'],'branch_objective':float(losses['branch'])})
    return {'measurement':'oracle_branch_teacher_forcing_and_state_not_memory_accuracy','total':len(rows),
            'state_correct':sum(r['state_target']==r['state_prediction'] for r in rows),'rows':rows,
            'branch_means':{kind:sum(r['branch_objective'] for r in rows if r['branch_kind']==kind)/sum(r['branch_kind']==kind for r in rows)
                            for kind in ('base_preservation_kl','content_oracle_ce','uncertainty_oracle_ce')}}


def train(package,config,device,root):
    torch.manual_seed(3180923); rng=random.Random(3180923); model=RoleSeparatedMemory().to(device)
    head=package.head.to(device); version=head._version
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.01)
    records=[r for r in package.records if r['split']=='train']
    def checkpoint(step):
        payload={'format':FORMAT,'step':step,'state_dict':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                 'backbone_sha256':BACKBONE_SHA256,'template_version':TEMPLATE_VERSION,'encoder_identity':package.manifest['encoder_identity'],
                 'feature_package_digest':FEATURE_DIGEST,'source_sha256':source_identity(),'experiment':config,
                 'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state_all(),
                 'python_rng_state':rng.getstate(),'deployment_approved':False}
        with (root/f'step-{step:06d}.pt').open('xb') as f: torch.save(payload,f)
        result=evaluate(model,package,head,device)
        with (root/f'step-{step:06d}-diagnostic.json').open('x') as f: json.dump(result,f,indent=2)
        print(json.dumps({'step':step,'state_correct':result['state_correct'],'total':result['total'],'branch_means':result['branch_means'],'memory_accuracy_measured':False}),flush=True)
    checkpoint(0)
    with (root/'training.jsonl').open('x') as log:
        for step in range(1,51):
            model.train(); optimizer.zero_grad(set_to_none=True); batch=rng.sample(records,4); details=[]
            for record in batch:
                features,target=package.sample(record,device)
                loss=role_loss(forward_roles(model,features,head,package.scale),target)
                objective=loss['weighted_total']/6.
                if not torch.isfinite(objective): raise ValueError('nonfinite objective; update refused')
                objective.backward()
                details.append({'id':record['id'],'objective':float(objective.detach()),'branch_kind':loss['branch_kind'],
                                'branch':float(loss['branch'].detach()),'state':float(loss['state'].detach())})
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True); optimizer.step()
            if head.grad is not None or head._version!=version: raise ValueError('frozen head changed')
            row={'step':step,'gradient_norm_before_clip':float(norm),'examples':details}
            log.write(json.dumps(row)+'\n'); log.flush()
            if step%10==0: print(json.dumps({'step':step,'gradient_norm_before_clip':float(norm)}),flush=True)
    checkpoint(50)
    with (root/'status.json').open('x') as f:
        json.dump({'optimizer_steps':50,'status':'awaiting_fixed_panel_native_generation_review','automatic_continuation':False,
                   'memory_accuracy_measured':False,'deployment_approved':False},f,indent=2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','features','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--baseline');p.add_argument('--authorization')
    a=p.parse_args(); config=json.loads(Path(a.experiment).read_text()); validate_config(config)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    package=TrainingPackage(a.features,FEATURE_DIGEST)
    manifest=Path(__file__).resolve().parents[1]/'training/memory/neural-system'/config['data_manifest']
    if digest(json.loads(manifest.read_text()))!=package.manifest['corpus_manifest_sha256']: raise ValueError('wrong registered corpus')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    result=preflight(package,device)
    with (root/'preflight.json').open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)
    if not result['passed']: raise SystemExit(1)
    if not a.preflight_only:
        require_ready(config,result,a.baseline,a.authorization); train(package,config,device,root)


if __name__=='__main__':main()
