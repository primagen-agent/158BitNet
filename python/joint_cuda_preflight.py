"""Export fixed C tensors; compare CPU/CUDA without an optimizer or checkpoint."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import platform
import socket
import numpy as np
import torch

from availability_supervision import ReplyTargets
from joint_memory_model import create_joint_model
from joint_training_data import JointTrainingPackage, JointForwardFeatures
from memory_token_read import TokenReadInput
from native_memory_encoder import sha_file, BACKBONE_SHA256
from prepare_neural_memory_protocol import digest
from preflight_joint_full import gradient_requirements
from token_memory_supervision import TokenSupervision
from train_joint_memory import paired_forward, paired_loss

SCENARIOS = ('correct','role_swap','value_swap','empty','ordinary_source','ordinary_empty')


def export(a):
    package = JointTrainingPackage(a.features, a.package_digest)
    if package.manifest['scope'] != 'full': raise ValueError('full features required')
    world = min(r['world_id'] for r in package.records if r['split'] == 'train')
    records = sorted((r for r in package.records if r['split'] == 'train' and r['world_id'] == world and
        r['relation'] == 'home_city' and r['scenario'] in SCENARIOS), key=lambda r: (r['language'], SCENARIOS.index(r['scenario'])))
    if len(records) != 12: raise ValueError('fixture denominator changed')
    root = Path(a.output); root.mkdir(parents=True, exist_ok=False)
    arrays = {}; entries = []
    for i, r in enumerate(records):
        f, target = package.sample(r, training=True)
        for name, value in (('query',f.memory.query),('source',f.memory.source),('ids',f.memory.token_ids),
            ('allowed',f.memory.copy_allowed),('hidden',f.hidden),('base',f.base)):
            arrays[f'{i}_{name}'] = value.numpy()
        entries.append({'id': r['id'], 'language': r['language'], 'scenario': r['scenario'],
                        'reply': asdict(target.reply), 'factors': target.factors, 'positions': target.positions})
    with (root/'data.npz').open('xb') as stream: np.savez_compressed(stream, **arrays)
    with (root/'records.json').open('x') as f: json.dump(entries,f,ensure_ascii=False,indent=2)
    manifest = {'format':'cg003-cuda-fixture-v1','source_package_digest':a.package_digest,'world':world,
        'records':12,'binding':package.binding,'blocked_ids':package.manifest['blocked_ids'],'scale':package.scale,
        'head_sha256':package.manifest['output_head_sha256'],'gguf_sha256':BACKBONE_SHA256,
        'data_sha256':sha_file(root/'data.npz'),'records_sha256':sha_file(root/'records.json'),
        'source_sha256':sha_file(__file__),'optimizer_steps':0,'training_approved':False}
    with (root/'manifest.json').open('x') as f: json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps({'fixture_digest':digest(manifest),'bytes':(root/'data.npz').stat().st_size}),flush=True)


def samples(root, expected):
    root=Path(root);m=json.loads((root/'manifest.json').read_text())
    if digest(m)!=expected or m['format']!='cg003-cuda-fixture-v1':raise ValueError('fixture identity mismatch')
    if sha_file(root/'data.npz')!=m['data_sha256'] or sha_file(root/'records.json')!=m['records_sha256']:raise ValueError('fixture corrupted')
    result={}; entries=json.loads((root/'records.json').read_text())
    with np.load(root/'data.npz',allow_pickle=False) as values:
        for i,r in enumerate(entries):
            get=lambda n:torch.from_numpy(values[f'{i}_{n}'].copy())
            memory=TokenReadInput(get('query'),get('source'),get('ids'),get('allowed'),m['binding'])
            reply=dict(r['reply'])
            for n in ('prompt_token_ids','completion_token_ids'):reply[n]=tuple(reply[n])
            target=TokenSupervision(ReplyTargets(**reply),tuple(r['factors']),tuple(None if p is None else tuple(p) for p in r['positions']))
            result[(r['language'],r['scenario'])]=(JointForwardFeatures(memory,get('hidden'),get('base')),target)
    if set(result)!={(l,s) for l in ('en','zh') for s in SCENARIOS}:raise ValueError('fixture inventory changed')
    return m,result


def move(features, device):
    m=features.memory
    return JointForwardFeatures(TokenReadInput(*(t.to(device) for t in (m.query,m.source,m.token_ids,m.copy_allowed)),m.binding),
                                features.hidden.to(device),features.base.to(device))


def comparison(a,b,atol,rtol):
    a,b=a.detach().cpu(),b.detach().cpu()
    return {'passed':bool(torch.isfinite(a).all() and torch.isfinite(b).all() and torch.allclose(a,b,atol=atol,rtol=rtol)),
            'max_error':float((a-b).abs().max())}


def run(a):
    config=json.loads(Path(a.experiment).read_text())
    if config['id']!='CG-003-platform-preflight' or config['optimizer_steps']!=0:raise ValueError('wrong registration')
    repo=Path(__file__).resolve().parents[1];snapshot=json.loads(Path(a.snapshot).read_text())
    for name,expected in snapshot['source_sha256'].items():
        if sha_file(repo/name)!=expected:raise ValueError('source snapshot mismatch: '+name)
    m,data=samples(a.fixtures,a.fixture_digest)
    if sha_file(a.head)!=m['head_sha256'] or sha_file(a.gguf)!=m['gguf_sha256']:raise ValueError('head/backbone identity mismatch')
    if not torch.cuda.is_available():raise ValueError('CUDA not available')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    head=torch.from_numpy(np.load(a.head,allow_pickle=False));gpu_head=head.to('cuda')
    if head.shape!=(73448,1024) or head.dtype!=torch.float32 or not torch.isfinite(head).all():raise ValueError('invalid head')
    rows=[]
    for arm in ('joint_aux','joint_product'):
        cpu=create_joint_model(m['binding'],m['blocked_ids'],arm).eval();gpu=copy.deepcopy(cpu).to('cuda').eval()
        originals={n:p.detach().clone() for n,p in cpu.named_parameters()}
        for language in ('en','zh'):
            for kind in ('role_swap','value_swap','empty','ordinary_empty'):
                left='ordinary_source' if kind=='ordinary_empty' else 'correct'
                chosen=[data[(language,s)] for s in (left,kind)]
                fs=[s[0] for s in chosen];ts=[s[1] for s in chosen]
                results=[]
                for model,features,output_head in ((cpu,fs,head),(gpu,[move(f,'cuda') for f in fs],gpu_head)):
                    outputs=paired_forward(model,features,output_head,m['scale'])
                    losses=paired_loss(outputs,features,ts,kind)
                    parameters=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
                    grads=torch.autograd.grad(losses['total'],[p for _,p in parameters],allow_unused=True)
                    gradient_requirements([n for n,_ in parameters],grads,ts,kind,arm)
                    results.append((outputs,losses['total'],dict(zip([n for n,_ in parameters],grads))))
                c,g=results;checks=[]
                for co,go in zip(c[0],g[0]):
                    for field in ('supported','uncertainty','positions','copy_mass','factor_logits'):
                        checks.append({'field':field,**comparison(getattr(co,field),getattr(go,field),1e-5,1e-4)})
                    checks.append({'field':'state_probabilities',**comparison(co.state_logits.exp(),go.state_logits.exp(),1e-5,1e-4)})
                    for field in ('supported','uncertainty','state_logits'):
                        checks.append({'field':field+'_top1','passed':bool(torch.equal(getattr(co,field).argmax(-1).cpu(),getattr(go,field).argmax(-1).cpu()))})
                checks.append({'field':'loss',**comparison(c[1],g[1],1e-4,1e-3)})
                gradients={}
                for name,cg in c[2].items():
                    gg=g[2][name]
                    gradients[name]={'passed':cg is None and gg is None,'disconnected':True} if cg is None or gg is None else comparison(cg,gg,1e-4,1e-3)
                row={'arm':arm,'language':language,'kind':kind,'checks':checks,'gradients':gradients,
                     'passed':all(v['passed'] for v in checks) and all(v['passed'] for v in gradients.values())}
                rows.append(row);print(json.dumps({'comparisons':len(rows),'passed':row['passed'],'arm':arm,'kind':kind}),flush=True)
                del results,outputs,losses,grads,c,g,cg,gg
        for model in (cpu,gpu):
            if any(not torch.equal(p.detach().cpu(),originals[n]) or p.grad is not None for n,p in model.named_parameters()):raise ValueError('parameters changed')
        del gpu,cpu;torch.cuda.empty_cache()
    report={'experiment':config,'host':socket.gethostname(),'python':platform.python_version(),'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(0),'tf32':False,'amp':False,'rows':rows,'passed':sum(r['passed'] for r in rows),'total':len(rows),
        'optimizer_steps':0,'parameters_unchanged':True,'memory_accuracy_measured':False,'training_approved':False,
        'fixture_digest':a.fixture_digest,'snapshot_sha256':sha_file(a.snapshot),'head_sha256':sha_file(a.head),'gguf_sha256':sha_file(a.gguf),
        'experiment_sha256':sha_file(a.experiment),'source_sha256':sha_file(__file__)}
    with Path(a.output).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print('CG003_CUDA_DONE '+json.dumps({'passed':report['passed'],'total':report['total'],'sha256':sha_file(a.output)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    e=sub.add_parser('export')
    for n in ('features','package-digest','output'):e.add_argument('--'+n,required=True)
    r=sub.add_parser('run')
    for n in ('experiment','fixtures','fixture-digest','head','gguf','snapshot','output'):r.add_argument('--'+n,required=True)
    a=p.parse_args();export(a) if a.mode=='export' else run(a)
