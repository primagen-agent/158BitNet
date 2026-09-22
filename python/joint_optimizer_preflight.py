"""Exercise the actual GPU loop; abort before any real-model optimizer step."""
import argparse
import copy
import json
from pathlib import Path
import socket
import tempfile
from unittest.mock import patch

import torch

from joint_cuda_preflight import comparison, move
from joint_memory_model import create_joint_model
from joint_optimizer import (FEATURE_DIGEST, _run_fixed_loop, bind_schedule, tensor_digest,
                             save_checkpoint, load_checkpoint)
from joint_training_data import JointTrainingPackage
from native_memory_encoder import sha_file, BACKBONE_SHA256


class BeforeUpdate(Exception): pass


def dry_batch(model, head, package, records, schedule, device):
    initial = tensor_digest(model.state_dict()); captured = {}
    def sample(pair):
        values=[package.sample(records[pair[k]],device,training=True) for k in ('left_id','right_id')]
        return [v[0] for v in values],[v[1] for v in values]
    def intercept(optimizer, *args, **kwargs):
        captured.update({n:None if p.grad is None else p.grad.detach().cpu().clone()
                         for n,p in model.named_parameters() if p.requires_grad})
        raise BeforeUpdate()
    with patch.object(torch.optim.AdamW,'step',intercept):
        try: _run_fixed_loop(model,head,package.scale,schedule,sample,on_checkpoint=lambda *a:None,on_step=lambda r:None)
        except BeforeUpdate: pass
        else: raise ValueError('dry run failed to intercept optimizer')
    if tensor_digest(model.state_dict())!=initial or any(p.grad is not None for p in model.parameters()) or not captured:
        raise ValueError('dry batch changed parameters or left gradients')
    return captured


def run(a):
    snapshot=json.loads(Path(a.snapshot).read_text()); repo=Path(__file__).resolve().parents[1]
    for name,want in snapshot['source_sha256'].items():
        if sha_file(repo/name)!=want: raise ValueError('source changed: '+name)
    if sha_file(a.gguf)!=BACKBONE_SHA256 or not torch.cuda.is_available():raise ValueError('bound GPU/backbone required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    package=JointTrainingPackage(a.features,FEATURE_DIGEST)
    audit=json.loads(Path(a.audit).read_text());schedule,records=bind_schedule(package,a.corpus,audit)
    head=package.head.to('cuda');rows=[]
    for arm in ('joint_aux','joint_product'):
        cpu=create_joint_model(package.binding,package.manifest['blocked_ids'],arm)
        gpu=copy.deepcopy(cpu).to('cuda')
        c=dry_batch(cpu,package.head,package,records,schedule,'cpu')
        g=dry_batch(gpu,head,package,records,schedule,'cuda')
        checks={n:({'passed':c[n] is None and g[n] is None} if c[n] is None or g[n] is None
                   else comparison(c[n],g[n],1e-4,1e-3)) for n in c}
        rows.append({'arm':arm,'clipped_accumulated_gradients':checks,'passed':all(r['passed'] for r in checks.values())})
        del cpu,gpu,c,g
    # Tiny artificial fixtures exercise actual CUDA AdamW and checkpoint roundtrip.
    # They are not data-bearing 0.5B candidates and cannot enter model selection.
    from availability_supervision import ReplyTargets
    from joint_training_data import JointForwardFeatures
    from memory_token_read import TokenReadInput
    from token_memory_supervision import TokenSupervision
    torch.manual_seed(43)
    q=torch.randn(4,16);h=torch.randn(3,8);b=torch.randn(3,13);small_head=torch.randn(13,8)
    fs=[JointForwardFeatures(TokenReadInput(q,torch.randn(4,16),torch.tensor([4,5,6,7]),
        torch.ones(4,dtype=torch.bool),'fixture'),h,b) for _ in range(2)]
    ts=[TokenSupervision(ReplyTargets((1,2),(3,4,2),i,1.,'fixture'),(True,True),
        (None,(1,),None) if i==1 else (None,)*3) for i in (1,2)]
    tiny_schedule=[{'step':i+1,'pairs':[{'id':f'{i}-{j}','kind':'role_swap','left_id':'a','right_id':'b'} for j in range(4)]} for i in range(50)]
    tiny=[]
    for arm in ('joint_aux','joint_product'):
        models=[create_joint_model('fixture',(0,),arm,hidden=8,vocab=13,layers=2,heads=2) for _ in range(2)]
        models[1].to('cuda');initial=tensor_digest(models[0].state_dict())
        counts=[]
        for m,device in zip(models,('cpu','cuda')):
            completed=[];saved=[]
            with tempfile.TemporaryDirectory() as tmp:
                def checkpoint(step,model,optimizer,init):
                    if step==50:saved.append(save_checkpoint(Path(tmp)/'final.pt',model,optimizer,step,{'fixture':True},init))
                _run_fixed_loop(m,small_head.to(device),4.,tiny_schedule,lambda p:([move(f,device) for f in fs],ts),
                    on_checkpoint=checkpoint,on_step=lambda r:completed.append(r['step']))
                restored=create_joint_model('fixture',(0,),arm,hidden=8,vocab=13,layers=2,heads=2)
                load_checkpoint(Path(tmp)/'final.pt',expected_sha256=saved[0]['sha256'],model=restored,
                                expected_binding={'fixture':True},expected_initial_digest=initial)
                if tensor_digest(restored.state_dict())!=tensor_digest(m.state_dict()):raise ValueError('checkpoint mismatch')
            counts.append(len(completed))
        checks={n:comparison(p,models[1].state_dict()[n],1e-4,1e-3) for n,p in models[0].state_dict().items()}
        tiny.append({'arm':arm,'steps_cpu_cuda':counts,'parameters':checks,'passed':counts==[50,50] and all(c['passed'] for c in checks.values())})
    result={'format':'cg003-optimizer-preflight-v1','host':socket.gethostname(),'torch':str(torch.__version__),
        'gpu':torch.cuda.get_device_name(0),'real_model_optimizer_steps':0,'tiny_fixture_only':True,
        'real_first_batch':rows,'tiny_loops':tiny,'passed':all(r['passed'] for r in rows+tiny),
        'source_snapshot_sha256':sha_file(a.snapshot),'source_sha256':snapshot['source_sha256'],
        'feature_package_digest':FEATURE_DIGEST,'memory_accuracy_measured':False}
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('CG003_OPTIMIZER_PREFLIGHT '+json.dumps({'passed':result['passed'],'sha256':sha_file(a.output)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('snapshot','features','corpus','audit','gguf','output'):p.add_argument('--'+n,required=True)
    run(p.parse_args())
