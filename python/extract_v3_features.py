"""V3-001 feature extraction: all training prefixes + texts for JB-001+JB-002 train.
Frozen probes only; produces build/neural-memory-v3-features with full manifest."""
import json,sys
from pathlib import Path
sys.path.insert(0,'python')
from compile_time_scoped_teacher import compile_teacher_time
from episode_memory_inputs import encoder_texts
from native_memory_encoder import NativeMemoryEncoder,collect_texts,save_bank,sha_file,digest
from native_prefix_bank import extract_prefix_batch
from neural_memory_contract import ModelBinding
from append_value_transport import AppendValueCodec
from native_memory_encoder import BACKBONE_SHA256
import numpy as np, torch

ROOT=Path('.');RESEARCH=ROOT/'training/memory/neural-system'
FROZEN=ROOT/'build/neural-memory-cg003-frozen-build'
fm=json.loads((ROOT/'build/neural-memory-cg003-full-features/manifest.json').read_text())
codec=AppendValueCodec(FROZEN/'tok_probe',ROOT/'models/bitcpm4-0.5b-tq2_0.gguf',fm['tokenizer_sha256'])
binding=ModelBinding(BACKBONE_SHA256,fm['tokenizer_sha256'],fm['encoder_identity']['encoder_id'],'0'*64)
traj={}
for corpus_id in ('JB-001','JB-002'):
    c=RESEARCH/'data'/corpus_id
    data={n:{r['id']:r for r in map(json.loads,(c/f'train.{n}.jsonl').read_text().splitlines())} for n in ('inputs','labels','index')}
    for meta in data['index'].values():
        traj[(corpus_id,meta['id'])]=(data,meta)
    print(corpus_id,'records',len(data['index']),flush=True)
prefixes={}
compile_failures=[]
for (cid,rid),(data,meta) in traj.items():
    try:
        t=compile_teacher_time(data['inputs'][rid],data['labels'][rid],meta,codec,binding,'0'*64,purpose='training_diagnostic')
    except Exception as e:
        compile_failures.append({'id':rid,'corpus':cid,'reason':type(e).__name__});continue
    route=t.static_targets.route
    positions={0}|{i for i,l in enumerate(t.idle_targets) if l is not None}
    if route==2: positions|=set(range(len(t.completion_ids)))
    for i in positions: prefixes.setdefault(t.prefix(i,purpose='training_diagnostic'),len(prefixes))
    if len(prefixes)%500==0: print('prefixes',len(prefixes),flush=True)
ids=list(prefixes)
out=ROOT/'build/neural-memory-v3-features/prefixes';out.mkdir(parents=True)
import numpy as np
all_h=np.zeros((len(ids),1024),dtype='<f4');all_z=np.zeros((len(ids),73448),dtype='<f4')
batches=[]
for start in range(0,len(ids),16):
    group=ids[start:start+16];folder=out/f'batch-{start//16:04d}'
    h,z=extract_prefix_batch(FROZEN/'memory_prefix_probe',ROOT/'models/bitcpm4-0.5b-tq2_0.gguf',group,folder)
    all_h[start:start+len(group)]=h;all_z[start:start+len(group)]=z
    batches.append({'folder':folder.name,'count':len(group),'output_sha256':sha_file(folder/'output.bin')})
    if start%512==0: print('extracted',start+len(group),'/',len(ids),flush=True)
np.save(ROOT/'build/neural-memory-v3-features/prefix_hidden.npy',all_h)
np.save(ROOT/'build/neural-memory-v3-features/prefix_logits.npy',all_z)
texts=collect_texts([d['inputs'][rid] for (cid,rid),(d,m) in traj.items()])
enc=NativeMemoryEncoder('models/bitcpm4-0.5b-tq2_0.gguf',str(FROZEN/'memory_feature_probe'),expected_encoder_id=fm['encoder_identity']['encoder_id'])
encoded=enc.encode(texts,str(ROOT/'build/neural-memory-v3-features/tokens-extraction'),batch_size=32)
bank_digest=save_bank(str(ROOT/'build/neural-memory-v3-features/tokens'),enc.identity,texts,encoded)
codec.verify_identity();codec.close()
manifest={'format':'v3-feature-package-v1','records':len(traj)-len(compile_failures),'compile_failures':compile_failures,'unique_prefixes':len(ids),
 'prefix_index':[list(p) for p in ids],
 'prefix_hidden_sha256':sha_file(ROOT/'build/neural-memory-v3-features/prefix_hidden.npy'),
 'prefix_logits_sha256':sha_file(ROOT/'build/neural-memory-v3-features/prefix_logits.npy'),
 'batches':batches,'texts':len(texts),'bank_manifest_sha256':bank_digest,
 'encoder_id':enc.identity['encoder_id']}
(ROOT/'build/neural-memory-v3-features/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False))
print(json.dumps({'records':len(traj),'prefixes':len(ids),'texts':len(texts)}))
