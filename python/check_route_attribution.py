"""DG-027 route-head attribution replay. Zero updates; cached features only."""
import json,sys,hashlib
from pathlib import Path
import torch
from torch.nn import functional as F

sys.path.insert(0,'python')
from append_value_transport import AppendValueCodec
from check_joint_span_interface import source_alignment
from compile_time_scoped_teacher import compile_teacher_time
from episode_memory_inputs import encoder_texts
from fine_span_reader import FineSpanReader, ByteLayout
from fine_span_supervision import fine_loss
from joint_optimizer import tensor_digest
from joint_span_reader import SpanFeatures, PrefixFeature
from native_memory_encoder import NativeFeatureBank, BACKBONE_SHA256, digest, sha_file, text_key
from native_token_payload import payload_mask
from neural_memory_contract import ModelBinding

ROOT=Path('.');RESEARCH=ROOT/'training/memory/neural-system'
PKG=ROOT/'build/neural-memory-v3-features'
FROZEN=ROOT/'build/neural-memory-cg003-frozen-build'
GGUF=ROOT/'models/bitcpm4-0.5b-tq2_0.gguf'
reg=json.loads((RESEARCH/'experiments/DG-027.json').read_text())
fm=json.loads((ROOT/'build/neural-memory-cg003-full-features/manifest.json').read_text())
codec=AppendValueCodec(FROZEN/'tok_probe',GGUF,fm['tokenizer_sha256'])
manifest=json.loads((PKG/'manifest.json').read_text())
hidden=torch.from_numpy(__import__('numpy').load(PKG/'prefix_hidden.npy'))
pindex={tuple(p):i for i,p in enumerate(manifest['prefix_index'])}
bank=NativeFeatureBank(PKG/'tokens',expected_encoder_id=manifest['encoder_id'],
                       expected_manifest_sha256=manifest['bank_manifest_sha256'])
binding=ModelBinding(BACKBONE_SHA256,fm['tokenizer_sha256'],fm['encoder_identity']['encoder_id'],sha_file(RESEARCH/'experiments/DG-027.json'))
key=digest(vars(binding))
torch.set_num_threads(4)

items=[]
for cid in ('JB-001','JB-002'):
    c=RESEARCH/'data'/cid
    data={n:{r['id']:r for r in map(json.loads,(c/f'train.{n}.jsonl').read_text().splitlines())} for n in ('inputs','labels','index')}
    for meta in data['index'].values():
        rid=meta['id']
        try:
            t=compile_teacher_time(data['inputs'][rid],data['labels'][rid],meta,codec,binding,'0'*64,purpose='training_diagnostic')
        except Exception:continue
        runtime=data['inputs'][rid];query,sources=encoder_texts(runtime)
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
                       key,hashlib.sha256(raw).hexdigest(),digest(runtime['context']))
        layout=ByteLayout(x,raw,ranges)
        p0=t.prefix(0,purpose='training_diagnostic')
        probe_model=FineSpanReader(2048,1024,key,width=64).eval()
        try:
            with torch.no_grad():
                probe_model(x,layout,PrefixFeature(p0,hidden[pindex[p0]].clone(),key),p0)
        except Exception:
            continue
        items.append({'rid':rid,'x':x,'layout':layout,'p0':p0,'h0':hidden[pindex[p0]].clone(),
                      'gold':t.static_targets.route,'static':t.static_targets,'meta':meta})
codec.close()
print(json.dumps({'phase':'items','count':len(items)}),flush=True)

ckpts={}
torch.manual_seed(1018);init=FineSpanReader(2048,1024,key,width=64).eval()
ckpts['000']=init
for step in (10,20,30,40,50):
    ck=torch.load(ROOT/f'build/neural-memory-v3-001/ckpt-{step:03d}.pt',weights_only=False)
    torch.manual_seed(1018);m=FineSpanReader(2048,1024,key,width=64).eval()
    m.load_state_dict(ck['reader']);ckpts[f'{step:03d}']=m

conf={};margins={};route_ce={}
with torch.no_grad():
    for tag,model in ckpts.items():
        cm=[[0]*3 for _ in range(3)];mg=[];ces=[]
        for it in items:
            out=model(it['x'],it['layout'],PrefixFeature(it['p0'],it['h0'],key),it['p0'])
            logits=out.route_logits if hasattr(out,'route_logits') else None
            probs=F.softmax(logits, dim=-1) if logits is not None else None
            pred=int(torch.argmax(logits)) if logits is not None else -1
            cm[it['gold']][pred]+=1
            top2=torch.topk(probs,2).values
            mg.append(float(top2[0]-top2[1]))
            ces.append(float(F.cross_entropy(logits[None],torch.tensor([it['gold']]))))
        conf[tag]=cm;margins[tag]={'mean':sum(mg)/len(mg),'p50':sorted(mg)[len(mg)//2]}
        route_ce[tag]=sum(ces)/len(ces)
        print(json.dumps({'phase':'replay','ckpt':tag,'confusion':cm,'margin':margins[tag],'route_ce':route_ce[tag]}),flush=True)

# loss decomposition at step 50 on a fixed sample (first 96 items by deterministic order)
sample=items[:96]
model=ckpts['050']
parts={'route':0.,'total':0.}
with torch.no_grad():
    for it in sample:
        out=model(it['x'],it['layout'],PrefixFeature(it['p0'],it['h0'],key),it['p0'])
        loss=fine_loss(out,it['static'])
        parts['route']+=float(loss.get('route',loss.get('route_ce',0.)))
        parts['total']+=float(loss['total'])
result={'format':'dg027-route-attribution-v1','registration_sha256':sha_file(RESEARCH/'experiments/DG-027.json'),
 'items':len(items),'confusion':conf,'margins':margins,'route_ce':route_ce,
 'loss_sample':{'n':len(sample),'route_mean':parts['route']/len(sample),'total_mean':parts['total']/len(sample),
                'available_keys':list(fine_loss(ckpts['050'](sample[0]['x'],sample[0]['layout'],
                    PrefixFeature(sample[0]['p0'],sample[0]['h0'],key),sample[0]['p0']),sample[0]['static']).keys())},
 'optimizer_steps':0}
out=RESEARCH/'checks/DG-027-route.json'
with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(result['loss_sample'],ensure_ascii=False))
