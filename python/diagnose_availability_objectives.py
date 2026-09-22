"""Post-forward loss/branch diagnostics on the unchanged CG-001 fixed panel.

Teacher-forced gradients only. No optimizer, parameter updates or semantic
accuracy claim. References never enter the module's forward signature.
"""
import argparse
import json
from pathlib import Path

import torch

from availability_supervision import supervised_loss
from eval_availability_checkpoint import BASELINE_SHA256, load_checkpoint
from native_memory_encoder import sha_file
from train_availability_memory import TrainingPackage, forward_example


def run(a):
    torch.set_num_threads(4)
    if sha_file(a.baseline)!=BASELINE_SHA256: raise ValueError('fixed baseline required')
    module,checkpoint=load_checkpoint(a.checkpoint)
    package=TrainingPackage(a.features,checkpoint['feature_package_digest'])
    ids=[r['id'] for r in json.loads(Path(a.baseline).read_text())['predictions']]
    records={r['id']:r for r in package.records}; rows=[]
    before={k:p.detach().clone() for k,p in module.named_parameters()}
    gate=(module.state_head.weight,module.state_head.bias)
    for cid in ids:
        features,target=package.sample(records[cid],torch.device('cpu'))
        logits,states=forward_example(module,features,package.head,package.scale)
        losses=supervised_loss(logits,states,target,state_coefficient=.2)
        generation=torch.autograd.grad(losses['generation'],gate,retain_graph=True)
        classification=torch.autograd.grad(.2*losses['state'],gate)
        g=torch.cat([x.flatten() for x in generation]); c=torch.cat([x.flatten() for x in classification])
        denom=g.norm()*c.norm()
        rows.append({'id':cid,'target_state':target.state_index,'teacher_forced_reply_tokens':len(target.completion_token_ids),
                     'generation_nll':float(losses['generation'].detach()),'state_ce':float(losses['state'].detach()),
                     'gate_generation_gradient_l2':float(g.norm()),'gate_weighted_state_gradient_l2':float(c.norm()),
                     'gate_gradient_cosine':float(torch.dot(g,c)/denom) if float(denom)>0 else None,
                     'generation_state_bias_gradient':generation[1].tolist(),
                     'weighted_classification_state_bias_gradient':classification[1].tolist()})
    if any(not torch.equal(p,before[k]) for k,p in module.named_parameters()): raise ValueError('diagnostic changed weights')
    result={'format':'cg001-loss-interaction-diagnostic-v1','checkpoint_sha256':sha_file(a.checkpoint),
            'feature_package_digest':checkpoint['feature_package_digest'],'fixed_panel_total':len(ids),
            'optimizer_steps':0,'parameters_unchanged':True,'teacher_forced':True,'memory_accuracy_measured':False,
            'gradient_scope':'generation CE versus 0.2 times initial state CE, state_head weight and bias only',
            'limitations':'Local gradient alignment is not causal proof of training collapse; no ablation or retraining performed',
            'cases':rows,'source_sha256':sha_file(__file__)}
    with Path(a.output).open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps({'cases':len(rows),'opposing_gate_gradients':sum(r['gate_gradient_cosine'] is not None and r['gate_gradient_cosine']<0 for r in rows)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','features','baseline','output'): p.add_argument('--'+name,required=True)
    run(p.parse_args())


if __name__=='__main__': main()
