"""Native operator parity on real locally extracted features, not E2E QA."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from export_resident_identity import export
from train_resident_identity import append_lexical_features, IdentityResidentMemorySet
from train_resident_memory_set import CTokenizer, GGUFWeights, compile_states, read_batch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint');p.add_argument('gguf');p.add_argument('diagnostic_dir')
    p.add_argument('--probe',default='build/tok_probe');p.add_argument('--lib',default='build/libggwshim.so')
    p.add_argument('--native',default='build/test_resident_identity_parity');a=p.parse_args()
    root=Path(a.diagnostic_dir)
    all_worlds=[json.loads(l) for l in (root/'diagnostic_worlds.jsonl').read_text().splitlines()]
    cache=torch.load(root/'diagnostic_features.pt',weights_only=True)
    indices=[0,1,96,97]
    worlds=[all_worlds[i] for i in indices];features=[cache['features'][i] for i in indices]
    item=torch.load(a.checkpoint,weights_only=True,map_location='cpu')
    assert item['backbone_sha256']==cache['backbone_sha256']
    weights=GGUFWeights(a.gguf,a.lib);tok=CTokenizer(a.probe,a.gguf)
    try: features=append_lexical_features(worlds,features,weights,tok)
    finally: weights.close();tok._proc.terminate();tok._proc.wait()
    model=IdentityResidentMemorySet(1024).eval().requires_grad_(False);model.load_state_dict(item['state_dict'])
    with tempfile.TemporaryDirectory(prefix='resident-parity-') as tmp,torch.inference_mode():
        tmp=Path(tmp);binary=tmp/'model.bnresid';export(a.checkpoint,binary)
        sample=tmp/'sample.bin';states=compile_states(model,features,'cpu')
        with sample.open('wb') as f:
            f.write(b'RSPAR001'+struct.pack('<I',len(worlds)))
            for wi,(w,feat,state) in enumerate(zip(worlds,features,states)):
                f.write(struct.pack('<2I',len(w['events']),len(w['queries'])))
                for ei,x in enumerate(feat['events']):
                    f.write(struct.pack('<I',len(x)));f.write(x.float().numpy().astype('<f4').tobytes())
                    f.write(state[ei,:len(x)].numpy().astype('<f4').tobytes())
                for qi,x in enumerate(feat['queries']):
                    data=read_batch(features,states,[(wi,qi)],'cpu');scores,counts=model.read(*data)
                    f.write(struct.pack('<I',len(x)));f.write(x.float().numpy().astype('<f4').tobytes())
                    f.write(scores[0].numpy().astype('<f4').tobytes());f.write(counts[0].numpy().astype('<f4').tobytes())
        subprocess.run([a.native,str(binary),a.gguf,str(sample)],check=True)
        raw=binary.read_bytes()
        for name,data in [('trailing',raw+b'x'),('crc',raw[:-1]+bytes([raw[-1]^1])),('short',raw[:100]),('binding',raw[:24]+bytes([raw[24]^1])+raw[25:])]:
            bad=tmp/(name+'.bnresid');bad.write_bytes(data)
            result=subprocess.run([a.native,str(bad),a.gguf,str(sample)])
            assert result.returncode==3,(name,result.returncode)
        print('Corrupt, truncated, wrong-backbone and trailing model rejection passed')

if __name__=='__main__':main()
