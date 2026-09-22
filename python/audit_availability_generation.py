"""Audit recorded trained-residual C/Torch projection on each initial prefix.

Reads only saved generation tensors and the identity-bound frozen output head;
does not change predictions, train, or score semantics.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np
import torch

from continuous_memory import continuous_logits
from eval_availability_checkpoint import load_checkpoint
from native_memory_encoder import read_encoded, sha_file
from prepare_neural_memory_protocol import digest


def audit(root,features,checkpoint):
    root=Path(root); features=Path(features)
    report=json.loads((root/'generation.json').read_text())
    manifest=json.loads((features/'manifest.json').read_text())
    if digest(manifest)!=report['feature_package_digest'] or sha_file(features/'output-head.npy')!=manifest['output_head_sha256']:
        raise ValueError('frozen projection identity changed')
    head=torch.from_numpy(np.load(features/'output-head.npy',allow_pickle=False))
    torch.set_num_threads(4); module,_=load_checkpoint(checkpoint); cases=[]
    for prediction in report['predictions']:
        path=root/prediction['backend_evidence']; backend=json.loads(path.read_text())
        if backend['cross_step_kv_reuse'] or not backend['internal_prefill_kv_buffers']:
            raise ValueError('unexpected cache policy')
        previous=None
        for step in backend['steps']:
            prefix=step['prefix_ids']; n=len(prefix)
            expected_trace={'initial_position':0,'final_position':n,'eval_calls':1,'prefix_tokens':n}
            if not step['base_reference_checked'] or step['native_traces']!=[expected_trace,expected_trace]:
                raise ValueError('unqualified fresh C/reference trace')
            if previous is not None and prefix!=previous: raise ValueError('not a free-generated prefix')
            previous=prefix+[step['predicted_token_id']]
        for name,expected in backend['raw_sha256'].items():
            if sha_file(path.parent/name)!=expected: raise ValueError('generation evidence changed')
        source=(path.parent/'step-000-output.input').read_bytes()
        n=struct.unpack_from('<I',source,8)[0]
        if source[:8]!=b'BNCI0001' or struct.unpack_from('<I',source,12+4*n)[0]!=1:
            raise ValueError('invalid trained residual input')
        delta=np.frombuffer(source,dtype='<f4',offset=16+4*n).copy()
        raw=(path.parent/'step-000-output.bin').read_bytes()
        if raw[:8]!=b'BNCO0001' or struct.unpack_from('<III',raw,8)!=(n,1024,73448):
            raise ValueError('invalid trained residual output')
        arrays=np.frombuffer(raw,dtype='<f4',offset=20)
        source_file=path.parent/'sources'/'batch-0.bin'
        with torch.no_grad():
            prepared=module.prepare(torch.from_numpy(read_encoded(source_file,1)[0].features)) if source_file.exists() else None
            read=module.inspect(23,torch.from_numpy(arrays[:1024].copy())[None],prepared)
        if not np.array_equal(read.residual[0].numpy(),delta): raise ValueError('checkpoint residual replay changed')
        base=arrays[1024:1024+73448].copy()
        correction=arrays[1024+73448:1024+2*73448]; actual=arrays[1024+2*73448:]
        with torch.no_grad(): expected,change=continuous_logits(torch.from_numpy(base)[None],torch.from_numpy(delta)[None],head,manifest['logit_scale'])
        expected=expected[0].numpy(); change=change[0].numpy()
        checks={name:bool(np.allclose(a,b,atol=1e-5,rtol=1e-4)) for name,a,b in
                (('logits',actual,expected),('correction',correction,change))}
        cases.append({'id':prediction['id'],'parity':checks,'argmax_equal':int(actual.argmax())==int(expected.argmax()),
                      'max_logit_error':float(np.max(np.abs(actual-expected))),
                      'max_correction_error':float(np.max(np.abs(correction-change))),
                      'initial_state_probabilities':prediction['initial_state_probabilities'],
                      'weighted_content_residual_l2':float(read.content_residual.norm()),
                      'weighted_uncertainty_residual_l2':float(read.uncertainty_residual.norm()),
                      'state_changes_during_generation':sum(int(np.argmax(s['availability_probabilities']))!=int(np.argmax(prediction['initial_state_probabilities'])) for s in backend['steps']),
                      'generated_positions':len(backend['steps'])})
    return {'format':'cg001-trained-projection-audit-v1','tolerance':{'atol':1e-5,'rtol':1e-4},
            'positions':'initial query prefill only, all 16 fixed cases','all_raw_hashes_verified':True,
            'all_generation_positions_fresh_and_reference_checked':True,
            'generic_transport_flag_note':'First run retained the generic unqualified transport flag; raw backend traces independently certify each position. Future evaluator reports set the audited flag explicitly.',
            'generation_sha256':sha_file(root/'generation.json'),'cases':cases,
            'learned_content_gain':float(module.content.layer_gain[23].detach().tanh()),
            'passed':all(all(c['parity'].values()) and c['argmax_equal'] for c in cases),
            'semantic_accuracy_measured':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('generation','features','checkpoint','output'): p.add_argument('--'+name,required=True)
    a=p.parse_args(); report=audit(a.generation,a.features,a.checkpoint)
    with Path(a.output).open('x') as f: json.dump(report,f,indent=2); f.write('\n')
    print(json.dumps({'passed':report['passed'],'cases':len(report['cases'])}),flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__': main()
