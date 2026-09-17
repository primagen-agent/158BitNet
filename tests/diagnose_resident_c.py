"""Distinguish native operator errors from the backbone feature domain gap."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from train_resident_identity import IdentityResidentMemorySet,append_lexical_features
from train_resident_memory_set import (CTokenizer,GGUFWeights,TorchBackbone,encode_worlds,
    read_batch,select_sets,evaluate)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('evaluation');p.add_argument('checkpoint');p.add_argument('--probe',default='build/resident_probe')
    a=p.parse_args();root=Path(a.evaluation);out=root/'domain_diagnostic';out.mkdir(exist_ok=False)
    report=json.loads((root/'summary.json').read_text())
    worlds=[json.loads(l) for l in (root/'development.jsonl').read_text().splitlines()]
    gguf=report['artifacts']['gguf']['path'];binary=report['artifacts']['resident']['path']
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert ckpt['backbone_sha256']==report['artifacts']['gguf']['sha256']
    model=IdentityResidentMemorySet(1024).eval().requires_grad_(False);model.load_state_dict(ckpt['state_dict'])
    queries=out/'queries.txt';queries.write_text(''.join(q['text']+'\n' for w in worlds for q in w['queries']))
    feature_file=out/'native_features.bin'
    subprocess.run([a.probe,gguf,str(queries),str(feature_file)],check=True,env={**os.environ,'BITNET_NUM_THREADS':'4'})
    features=[];states=[];hash_bytes=bytes.fromhex(report['artifacts']['resident']['sha256'])
    with feature_file.open('rb') as stream:
        for wi,w in enumerate(worlds):
            row=[]
            for q in w['queries']:
                t=struct.unpack('<I',stream.read(4))[0]
                row.append(torch.from_numpy(np.frombuffer(stream.read(t*2048*4),dtype='<f4').copy().reshape(t,2048)))
            features.append({'queries':row})
            manifest=next((root/'states').glob(f'*resident-dev-{wi}.bnsnapshot'))
            lines=manifest.read_text().splitlines();assert lines[0]=='BNMSNAP2'
            data=Path(str(manifest)[:-len('.bnsnapshot')]+'.'+lines[3]+'.bnresident').read_bytes()
            assert hashlib.sha256(data).hexdigest()==lines[3]
            assert data[:8]==b'BNRSTAT1' and data[8:40]==hash_bytes
            n=struct.unpack_from('<I',data,40)[0];offset=44;slots=[]
            for i in range(n):
                t=struct.unpack_from('<I',data,offset)[0];offset+=4;size=t*1154*4
                slots.append(torch.from_numpy(np.frombuffer(data[offset:offset+size],dtype='<f4').copy().reshape(t,1154)));offset+=size
            assert offset==len(data)
            state=torch.zeros(n,max(len(s) for s in slots),1154)
            for i,s in enumerate(slots):state[i,:len(s)]=s
            states.append(state)
        assert not stream.read()
    differences=[]
    with torch.inference_mode():
        for wi,w in enumerate(worlds):
            indices=[(wi,qi) for qi in range(len(w['queries']))]
            batch=read_batch(features,states,indices,'cpu');scores,counts=model.read(*batch)
            predicted=select_sets(scores,counts,batch[-2])
            for qi in range(len(w['queries'])):
                chosen=predicted[qi].nonzero().flatten().tolist()
                native=report['cases'][wi*12+qi]['selected_event_indices']
                if chosen!=native:differences.append({'world':wi,'query':qi,'native':native,'python_on_native_features':chosen})
    # Fresh local training-formula reference on the identical development corpus.
    weights=GGUFWeights(gguf,'build/libggwshim.so');tok=CTokenizer('build/tok_probe',gguf)
    backbone=TorchBackbone(weights,device='mps',dtype=torch.float32).eval()
    try:
        contextual=encode_worlds(worlds,backbone,tok)
        reference=append_lexical_features(worlds,contextual,weights,tok)
    finally:del backbone;weights.close();tok._proc.terminate();tok._proc.wait()
    model.to('mps');metrics,cases=evaluate(model,worlds,reference,'mps')
    changed=sum(c['selected']!=r['selected_event_indices'] for c,r in zip(cases,report['cases']))
    result={'native_operator_selected_set_mismatches':differences,'questions':len(report['cases']),
        'python_formula_oracle_span_reference':metrics,'c_http_actual_writer_correct':report['correct'],
        'selected_sets_changed_between_backbone_domains':changed,
        'c_native_oracle_span_support_correct':sum(r['selected_event_indices']==q['targets'] for r,q in
            zip(report['cases'],[q for w in worlds for q in w['queries']])),
        'interpretation':'Python reading persisted C addresses and fresh C query features isolates native neural math. The fresh Torch reference includes different backbone features and oracle value spans.',
        'reference_cases':cases}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='reference_cases'}),flush=True)
    assert not differences,'C neural operators disagree with Python on identical native inputs'

if __name__=='__main__':main()
