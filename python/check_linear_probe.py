"""DG-028 linear separability probe on frozen trained representations. Zero model updates."""
import json,sys,itertools
from pathlib import Path
import torch
from torch import nn

sys.path.insert(0,'python')
from append_value_transport import AppendValueCodec
from check_joint_span_interface import source_alignment
from compile_time_scoped_teacher import compile_teacher_time
from episode_memory_inputs import encoder_texts
from fine_span_reader import FineSpanReader, ByteLayout
from joint_optimizer import tensor_digest
from joint_span_reader import SpanFeatures, PrefixFeature
from native_memory_encoder import NativeFeatureBank, BACKBONE_SHA256, digest, sha_file, text_key
from native_token_payload import payload_mask
from neural_memory_contract import ModelBinding

ROOT=Path('.');RESEARCH=ROOT/'training/memory/neural-system'
PKG=ROOT/'build/neural-memory-v3-features';FROZEN=ROOT/'build/neural-memory-cg003-frozen-build'
fm=json.loads((ROOT/'build/neural-memory-cg003-full-features/manifest.json').read_text())
codec=AppendValueCodec(FROZEN/'tok_probe',ROOT/'models/bitcpm4-0.5b-tq2_0.gguf',fm['tokenizer_sha256'])
manifest=json.loads((PKG/'manifest.json').read_text())
hidden=torch.from_numpy(__import__('numpy').load(PKG/'prefix_hidden.npy'))
pindex={tuple(p):i for i,p in enumerate(manifest['prefix_index'])}
bank=NativeFeatureBank(PKG/'tokens',expected_encoder_id=manifest['encoder_id'],expected_manifest_sha256=manifest['bank_manifest_sha256'])
binding=ModelBinding(BACKBONE_SHA256,fm['tokenizer_sha256'],fm['encoder_identity']['encoder_id'],'0'*64)
key=digest(vars(binding))
torch.set_num_threads(4);torch.manual_seed(1018)
model=FineSpanReader(2048,1024,key,width=64).eval()
ck=torch.load('build/neural-memory-v3-002/ckpt-050.pt',weights_only=False)
model.load_state_dict(ck['reader'])
reader_digest=tensor_digest(model.state_dict())

route_z=[];route_y=[];mode_z=[];mode_y=[]
def features(runtime):
    query,sources=encoder_texts(runtime)
    q=bank.rows[text_key(query)]
    if sources:
        s=bank.rows[text_key(sources[0])]
        pieces=codec.decode_pieces(s.token_ids[1:].tolist())
        allowed=payload_mask(runtime['episodes'][0],sources[0],pieces)
        ranges=source_alignment(runtime['episodes'][0],sources[0],pieces,allowed)
        raw=runtime['episodes'][0]['text'].encode();source=torch.from_numpy(s.features.copy())
    else:
        ranges=();raw=b'';allowed=[];source=torch.empty(0,2048)
    x=SpanFeatures(torch.from_numpy(q.features.copy()),source,torch.tensor(allowed,dtype=torch.bool),
                   key,__import__('hashlib').sha256(raw).hexdigest(),digest(runtime['context']))
    return x,ByteLayout(x,raw,ranges)

with torch.no_grad():
    for cid in ('JB-001','JB-002'):
        c=RESEARCH/'data'/cid
        data={n:{r['id']:r for r in map(json.loads,(c/f'train.{n}.jsonl').read_text().splitlines())} for n in ('inputs','labels','index')}
        for meta in data['index'].values():
            rid=meta['id']
            try:
                t=compile_teacher_time(data['inputs'][rid],data['labels'][rid],meta,codec,binding,'0'*64,purpose='training_diagnostic')
            except Exception:continue
            x,layout=features(data['inputs'][rid]);p0=t.prefix(0,purpose='training_diagnostic')
            try:
                out=model(x,layout,PrefixFeature(p0,hidden[pindex[p0]].clone(),key),p0)
            except Exception:continue
            # replicate the route head input exactly: q pooled + evidence
            # (recompute via the model's own submodules to avoid private attr surgery)
            qrows=model.query(x.query).tanh();q=(model.query_pool(qrows).softmax(0)*qrows).sum(0)
            # evidence: reuse the forward output's internals via a second restricted call
            # simplest: monkey-read from out (FineOutput carries spans/route_logits; not raw q/evidence)
            # -> recompute evidence exactly as forward does:
            source=model.source(x.source).tanh();n=len(source)
            from joint_span_reader import candidate_spans
            cands=candidate_spans(x.allowed);starts,ends=layout.endpoints()
            spans=tuple((a,b) for a,b in cands if starts[a] and ends[b-1] and starts[a][0]<ends[b-1][-1])
            if spans:
                s=torch.tensor([a for a,b in spans]);e=torch.tensor([b for a,b in spans])
                sums=torch.cat((source.new_zeros(1,source.shape[1]),source.cumsum(0)))
                h=model.span(torch.cat((source[s],source[e-1],(sums[e]-sums[s])/(e-s)[:,None]),-1)).tanh()
                qq=q.expand_as(h)
                fp=model.fact(torch.cat((qq,h,qq*h,(qq-h).abs()),-1)).flatten().log_softmax(0)
                evidence=fp.exp()@h
            else:
                evidence=torch.zeros_like(q)
            route_z.append(torch.cat((q,evidence)));route_y.append(t.static_targets.route)
            if t.static_targets.route==1:
                for i,lab in enumerate(t.idle_targets):
                    if lab==1:
                        p=t.prefix(i,purpose='training_diagnostic')
                        mode_z.append(torch.cat((hidden[pindex[p]],q,evidence)));mode_y.append(1)
                    elif lab is not None and len(mode_z)<2*sum(1 for l in t.idle_targets if l==1):
                        p=t.prefix(i,purpose='training_diagnostic')
                        mode_z.append(torch.cat((hidden[pindex[p]],q,evidence)));mode_y.append(0)
codec.close()
Z=torch.stack(route_z);Y=torch.tensor(route_y)
M=torch.stack(mode_z);My=torch.tensor(mode_y)

def cv_probe(Z,Y,k=5,seed=1234,epochs=300):
    g=torch.Generator().manual_seed(seed);perm=torch.randperm(len(Y),generator=g)
    accs={c:[0,0] for c in Y.unique().tolist()}
    for f in range(k):
        val=perm[f::k];tr=torch.tensor([i for i in perm.tolist() if i not in val.tolist()])
        probe=nn.Linear(Z.shape[1],int(Y.max())+1)
        opt=torch.optim.LBFGS(probe.parameters(),lr=0.5,max_iter=epochs)
        def closure():
            opt.zero_grad();loss=nn.functional.cross_entropy(probe(Z[tr]),Y[tr]);loss.backward();return loss
        opt.step(closure)
        with torch.no_grad():pred=probe(Z[val]).argmax(1)
        for c in accs:
            m=Y[val]==c;accs[c][0]+=int((pred[m]==c).sum());accs[c][1]+=int(m.sum())
    return {int(c):(v[0],v[1],round(v[0]/max(v[1],1),3)) for c,v in accs.items()}

route_res=cv_probe(Z,Y)
mode_res=cv_probe(M,My) if len(My.unique())>1 else {'note':'single class'}
if tensor_digest(model.state_dict())!=reader_digest:raise ValueError('reader changed')
out={'format':'dg028-linear-probe-v1','registration_sha256':sha_file(RESEARCH/'experiments/DG-028.json'),
 'items':len(Y),'route_probe_cv_recall':route_res,
 'route_actual_recall':{'0':round(96/96,3) if False else None,'note':'actual trained-head recalls from DG-027-style probe: r0=48/96=0.5,r1=268/268=1.0,r2=96/528=0.182'},
 'mode_probe_items':int(len(My)),'mode_probe_cv_recall':mode_res,'mode_actual_note':'start never predicted: 0/268',
 'frozen_reader_digest':reader_digest,'optimizer_steps':0}
p=RESEARCH/'checks/DG-028-probe.json'
with p.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(out,ensure_ascii=False)[:600])
