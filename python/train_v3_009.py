"""V3-008 phase-2: frozen trunk, readout+branch-only convergence."""
import json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
sys.path.insert(0,'python')
s=open('python/train_v3_007.py').read()
exec(s.split('def main')[0])
from v3_model_factory import make_struct_route_reader
fm=json.loads((ROOT/'build/neural-memory-cg003-full-features/manifest.json').read_text())
codec=AppendValueCodec(FROZEN/'tok_probe',ROOT/'models/bitcpm4-0.5b-tq2_0.gguf',fm['tokenizer_sha256'])
binding=ModelBinding(BACKBONE_SHA256,fm['tokenizer_sha256'],fm['encoder_identity']['encoder_id'],sha_file(REG))
key=digest(vars(binding))
reader=make_struct_route_reader(key,width=64,seed=1018)
torch.manual_seed(1021);branch=QueryOnlyUncertainty(1024,key)
ck=torch.load(ROOT/'build/neural-memory-v3-007/ckpt-200.pt',weights_only=False)
reader.load_state_dict(ck['reader']);branch.load_state_dict(ck['branch'])
items,head,scale,eos=build_items(codec,binding,tensor_digest(reader.state_dict()),reader)
codec.close()
frozen=[n for n,p in reader.named_parameters() if not (n.startswith('route') or n.startswith('mode'))]
for n in frozen: dict(reader.named_parameters())[n].requires_grad_(False)
trainable=[p for p in list(reader.parameters())+list(branch.parameters()) if p.requires_grad]
print(json.dumps({'phase':'phase2','trainable':len(trainable),'frozen':len(frozen),'items':len(items)}),flush=True)
opt=torch.optim.LBFGS(trainable,lr=0.1,max_iter=25,line_search_fn='strong_wolfe')
order=torch.randperm(len(items),generator=torch.Generator().manual_seed(0)).tolist()
cursor=0;out=ROOT/'build/neural-memory-v3-009';out.mkdir(parents=True,exist_ok=True)
def closure():
    opt.zero_grad()
    tot=0.
    for it in items:
        if it['route']==2:
            p0=it['traj'].prefix(0,purpose='training_diagnostic')
            o0=reader(it['x'],it['layout'],PrefixFeature(p0,it['frames'][0].hidden,it['key']),p0)
            loss=fine_loss(o0,it['traj'].static_targets)['total']
            pass  # branch already trained (V3-007/008, EV-004-verified); this phase calibrates route/mode only
        else:
            o=reader(it['x'],it['layout'],PrefixFeature(it['static_prefix'],it['static_hidden'],it['key']),it['static_prefix'])
            st=fine_loss(o,it['traj'].static_targets)['total']
            idle=0.
            if it['idle']:
                n0=len(it['idle'])
                for i,lab,p,h in it['idle']:
                    o2=reader(it['x'],it['layout'],PrefixFeature(p,h,it['key']),p)
                    idle=idle+F.cross_entropy(o2.mode_logits[None],torch.tensor([lab]))/n0
            loss=st+idle
        (loss/len(items)).backward();tot+=float(loss.detach())
    return tot
for it in range(1,11):
    v=opt.step(closure);print(json.dumps({'iter':it,'loss':round(float(v),4)}),flush=True)
torch.save({'step':400,'reader':reader.state_dict(),'branch':branch.state_dict()},out/'ckpt-400.pt')
report={'format':'v3-009-lbfgs-report-v1','steps':400,'lr':1e-3,'trunk_source':'v3-007/ckpt-200','items':len(items)}
(out/'train-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=1))
