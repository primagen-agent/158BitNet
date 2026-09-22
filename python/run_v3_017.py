"""V3-010: canonical-segmentation mode augmentation + readout retrain + START trace."""
import json,sys,hashlib
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
sys.path.insert(0,'python')
from append_value_transport import AppendValueCodec
from check_joint_span_interface import source_alignment
from compile_time_scoped_teacher import compile_teacher_time
from diagnose_native_generation_gradient import native_forward
from episode_memory_inputs import encoder_texts
from fine_span_reader import FineSpanReader, ByteLayout
from fine_span_supervision import fine_loss
from joint_optimizer import tensor_digest
from joint_span_reader import SpanFeatures, PrefixFeature
from native_memory_encoder import NativeFeatureBank, BACKBONE_SHA256, digest, sha_file, text_key
from native_token_payload import payload_mask
from neural_memory_contract import ModelBinding
from v3_model_factory import make_struct_route_reader
head=None
ROOT=Path('.');RESEARCH=ROOT/'training/memory/neural-system'
PKG=ROOT/'build/neural-memory-v3-features';FROZEN=ROOT/'build/neural-memory-cg003-frozen-build'
GGUF=ROOT/'models/bitcpm4-0.5b-tq2_0.gguf'
fm=json.loads((ROOT/'build/neural-memory-cg003-full-features/manifest.json').read_text())
codec=AppendValueCodec(FROZEN/'tok_probe',GGUF,fm['tokenizer_sha256'])
man=json.loads((PKG/'manifest.json').read_text())
hidden=torch.from_numpy(np.load(PKG/'prefix_hidden.npy'))
manifest_prefix=man['prefix_index']
pindex={tuple(p):i for i,p in enumerate(manifest_prefix)}
bank=NativeFeatureBank(PKG/'tokens',expected_encoder_id=man['encoder_id'],expected_manifest_sha256=man['bank_manifest_sha256'])
binding=ModelBinding(BACKBONE_SHA256,fm['tokenizer_sha256'],fm['encoder_identity']['encoder_id'],sha_file(RESEARCH/'experiments/V3-017-proposal.json'))
key=digest(vars(binding))
reader=make_struct_route_reader(key,width=64,seed=1018)
ck=torch.load(ROOT/'build/neural-memory-v3-013/ckpt-440.pt',weights_only=False)
reader.load_state_dict(ck['reader'])
torch.manual_seed(1021);branch=QueryOnlyUncertainty(1024,key) if False else None
from query_only_uncertainty import QueryOnlyUncertainty
torch.manual_seed(1021);branch=QueryOnlyUncertainty(1024,key)
ckb=torch.load(ROOT/'build/neural-memory-v3-013/ckpt-440.pt',weights_only=False)
branch.load_state_dict(ckb['branch'])
head=torch.from_numpy(np.load(ROOT/'build/neural-memory-cg003-full-features/output-head.npy'))
scale=fm['logit_scale']
reader_digest=tensor_digest(reader.state_dict())
for n,p in reader.named_parameters():
    if n.startswith('query.') or n.startswith('source.') or n.startswith('query_pool.'): p.requires_grad_(False)
trainable=[p for p in reader.parameters() if p.requires_grad]

def stack_of(runtime):
    query,sources=encoder_texts(runtime)
    q=bank.rows[text_key(query)]
    if sources:
        s=bank.rows[text_key(sources[0])]
        pieces=codec.decode_pieces(s.token_ids[1:].tolist())
        allowed=payload_mask(runtime['episodes'][0],sources[0],pieces)
        ranges=source_alignment(runtime['episodes'][0],sources[0],pieces,allowed)
        raw=runtime['episodes'][0]['text'].encode();source=torch.from_numpy(s.features.copy())
    else:ranges=();raw=b'';allowed=[];source=torch.empty(0,2048)
    x=SpanFeatures(torch.from_numpy(q.features.copy()),source,torch.tensor(allowed,dtype=torch.bool),key,hashlib.sha256(raw).hexdigest(),digest(runtime['context']))
    return x,ByteLayout(x,raw,ranges)

items=[];canon=0;skipped=0
outdir=ROOT/'build/neural-memory-v3-017';outdir.mkdir(parents=True,exist_ok=True)
canon_h={}
for cid in ('JB-001','JB-002'):
    c=RESEARCH/'data'/cid
    data={n:{r['id']:r for r in map(json.loads,(c/f'train.{n}.jsonl').read_text().splitlines())} for n in ('inputs','labels','index')}
    for meta in data['index'].values():
        rid=meta['id']
        try: t=compile_teacher_time(data['inputs'][rid],data['labels'][rid],meta,codec,binding,reader_digest,purpose='training_diagnostic')
        except Exception: continue
        x,layout=stack_of(data['inputs'][rid]);p0=t.prefix(0,purpose='training_diagnostic')
        try:
            with torch.no_grad(): reader(x,layout,PrefixFeature(p0,hidden[pindex[p0]].clone(),key),p0)
        except Exception: continue
        ent={'x':x,'layout':layout,'traj':t,'route':t.static_targets.route,'key':key,'rid':rid}
        if t.static_targets.route==2:
            import numpy as _np
            for pi in range(min(4,len(t.completion_ids))):
                pre_toks=t.prompt_ids+t.completion_ids[:pi]
                text=b''.join(codec.decode_pieces(list(pre_toks))).decode('utf-8',errors='ignore')
                cp=tuple(codec.tokenizer.encode(text,True))
                if len(cp)<=len(t.prompt_ids): continue
                ref=native_forward(str(FROZEN/'memory_gradient_reference'),str(GGUF),cp,_np.zeros(1024,dtype='<f4'),outdir,f'bcan-{len(items)}-{pi}',False)
                items.append({'x':x,'layout':layout,'traj':t,'route':3,'key':key,'rid':rid+'-bcanon','canon2':True,
                              'bp':(cp,torch.from_numpy(ref['hidden'].copy()),torch.from_numpy(ref['logits'].copy()),t.completion_ids[pi])})
        if t.static_targets.route!=2:
            ent['static_prefix']=p0;ent['static_hidden']=hidden[pindex[p0]].clone()
            ent['idle']=[(i,lab,t.prefix(i,purpose='training_diagnostic'),hidden[pindex[t.prefix(i,purpose='training_diagnostic')]].clone()) for i,lab in enumerate(t.idle_targets) if lab is not None]
        else:
            ent['frames_p0']=hidden[pindex[p0]].clone()
        items.append(ent)
        # canonical variant at the START position for supported
        if t.static_targets.route==1:
            si=[i for i,l in enumerate(t.idle_targets) if l==1]
            if si:
                si=si[0]
                pre_toks=t.prompt_ids+t.completion_ids[:si]
                text=b''.join(codec.decode_pieces(list(pre_toks))).decode('utf-8',errors='ignore')
                cp=tuple(codec.tokenizer.encode(text.rstrip(),True))
                if len(cp)>len(t.prompt_ids):
                    ref=native_forward(str(FROZEN/'memory_gradient_reference'),str(GGUF),cp,np.zeros(1024,dtype='<f4'),outdir,f'canon-{canon:04d}',False)
                    canon_h[cp]=torch.from_numpy(ref['hidden'].copy())
                    zero_anchors=[(i,0,t.prefix(i,purpose='training_diagnostic'),hidden[pindex[t.prefix(i,purpose='training_diagnostic')]].clone()) for i,l in enumerate(t.idle_targets) if l==0]
                    ent2={'x':x,'layout':layout,'traj':t,'route':1,'key':key,'rid':rid+'-canon','static_prefix':cp,'static_hidden':canon_h[cp],
                          'idle':[(si,1,cp,canon_h[cp])]+zero_anchors[:4],'canon':True}
                    items.append(ent2);canon+=1
                else: skipped+=1
codec.verify_identity();codec.close()
print(json.dumps({'phase':'items','count':len(items),'canonical':canon,'skipped':skipped}),flush=True)

opt=torch.optim.LBFGS(trainable,lr=0.05,max_iter=20,line_search_fn='strong_wolfe')
def closure():
    opt.zero_grad();tot=0.
    for it in items:
        if it['route']==3:
            from query_only_uncertainty import QueryOnlyUncertainty as _Q
            cp,h,l,tok=it['bp']
            from autonomous_value_controller import LiveFrame as _L, FrameOrigin as _O
            loss=F.cross_entropy(branch(_L(cp,h,l,it['key'],_O.NATIVE_FRESH),cp,head,scale).logits[None],torch.tensor([tok]))*2.0
        elif it['route']==2:
            o=reader(it['x'],it['layout'],PrefixFeature(it['traj'].prefix(0,purpose='training_diagnostic'),it['frames_p0'],it['key']),it['traj'].prefix(0,purpose='training_diagnostic'))
            loss=fine_loss(o,it['traj'].static_targets)['route']
        else:
            o=reader(it['x'],it['layout'],PrefixFeature(it['static_prefix'],it['static_hidden'],it['key']),it['static_prefix'])
            if it.get('canon'):
                fl=fine_loss(o,it['traj'].static_targets)
                loss=fl['route']+2.0*(fl['fact']+fl['value']+fl['boundary'])
            else:
                loss=fine_loss(o,it['traj'].static_targets)['route']
            for i,lab,p,h in it['idle']:
                o2=reader(it['x'],it['layout'],PrefixFeature(p,h,it['key']),p)
                w=3.0 if lab==1 else 0.5
                loss=loss+w*F.cross_entropy(o2.mode_logits[None],torch.tensor([lab]),label_smoothing=0.05)
        (loss/len(items)).backward();tot+=float(loss.detach())
    return tot
for it in range(1,9):
    v=opt.step(closure);print(json.dumps({'iter':it,'loss':round(float(v),3)}),flush=True)
torch.save({'step':480,'reader':reader.state_dict(),'branch':branch.state_dict()},outdir/'ckpt-480.pt')
# decisive trace: canonical-prefix START check on 30 supported items
ok=0;n=0
with torch.no_grad():
    for it in items:
        if '-canon' not in it['rid']: continue
        n+=1
        si,lab,p,h=it['idle'][0]
        o=reader(it['x'],it['layout'],PrefixFeature(p,h,it['key']),p)
        if int(o.mode_logits.argmax())==1: ok+=1
print(json.dumps({'phase':'canonical_start_recall',f'result':f'{ok}/{n}'}),flush=True)
# teacher-position mode recall after retrain
c0=c1=t0=t1=0
with torch.no_grad():
    for it in items:
        if '-canon' in it['rid'] or it['route']!=1: continue
        for i,lab,p,h in it['idle']:
            o=reader(it['x'],it['layout'],PrefixFeature(p,h,it['key']),p)
            pred=int(o.mode_logits.argmax())
            if lab==1: t1+=1; c1+=pred==1
            else: t0+=1; c0+=pred==0
print(json.dumps({'phase':'teacher_mode',f'result':f'start {c1}/{t1} generate {c0}/{t0}'}),flush=True)
