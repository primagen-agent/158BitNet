"""Verify all CG-002 generated positions without changing or scoring answers."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np
import torch

from continuous_memory import continuous_logits
from diagnose_native_generation_gradient import exact
from eval_availability_checkpoint import BASELINE_SHA256
from eval_role_checkpoint import CHECKPOINT_SHA256,load_checkpoint
from native_memory_encoder import read_encoded,sha_file
from prepare_neural_memory_protocol import digest


def audit(root,features,checkpoint,baseline_path):
    root=Path(root);features=Path(features);torch.set_num_threads(4)
    model,binding=load_checkpoint(checkpoint);report=json.loads((root/'generation.json').read_text())
    if report['checkpoint_sha256']!=CHECKPOINT_SHA256 or sha_file(baseline_path)!=BASELINE_SHA256:raise ValueError('artifact binding mismatch')
    baseline=json.loads(Path(baseline_path).read_text());manifest=json.loads((features/'manifest.json').read_text())
    if digest(manifest)!=binding['feature_package_digest'] or sha_file(features/'output-head.npy')!=manifest['output_head_sha256']:
        raise ValueError('frozen head binding mismatch')
    head=torch.from_numpy(np.load(features/'output-head.npy',allow_pickle=False));rows=[]
    if [r['id'] for r in report['predictions']]!=[r['id'] for r in baseline['predictions']]:raise ValueError('panel changed')
    with torch.no_grad():
        for prediction,before in zip(report['predictions'],baseline['predictions']):
            directory=root/Path(prediction['backend_evidence']).parent
            backend=json.loads((directory/'backend.json').read_text())
            if backend['identity']['encoder']!=binding['encoder_identity'] or backend['cross_step_kv_reuse']:
                raise ValueError('encoder/cache mismatch')
            for name,expected in backend['raw_sha256'].items():
                path=directory/name
                if not path.resolve().is_relative_to(directory.resolve()) or sha_file(path)!=expected:raise ValueError('raw evidence changed')
            source=directory/'sources/batch-0.bin'
            prepared=model.prepare(torch.from_numpy(read_encoded(source,1)[0].features)) if source.exists() else None
            decision=None;previous=None;checks=[]
            for i,step in enumerate(backend['steps']):
                prefix=step['prefix_ids'];n=len(prefix);name=f'step-{i:03d}'
                expected_trace={'initial_position':0,'final_position':n,'eval_calls':1,'prefix_tokens':n}
                if step['native_traces']!=[expected_trace,expected_trace] or not step['base_reference_checked']:
                    raise ValueError('not a fresh ordinary-C-checked prefix')
                if previous is not None and prefix!=previous:raise ValueError('teacher-forced or discontinuous generation')
                previous=prefix+[step['predicted_token_id']]
                wire=(directory/(name+'-output.input')).read_bytes();raw=(directory/(name+'-output.bin')).read_bytes()
                if (wire[:8]!=b'BNCI0001' or struct.unpack_from('<I',wire,8)[0]!=n or
                    np.frombuffer(wire,dtype='<i4',count=n,offset=12).tolist()!=prefix or struct.unpack_from('<I',wire,12+4*n)[0]!=1 or
                    raw[:8]!=b'BNCO0001' or struct.unpack_from('<III',raw,8)!=(n,1024,73448)):
                    raise ValueError('native wire mismatch')
                values=np.frombuffer(raw,dtype='<f4',offset=20)
                hidden=torch.from_numpy(values[:1024].copy())[None];base=torch.from_numpy(values[1024:1024+73448].copy())[None]
                if decision is None:decision=model.decide(hidden,prepared)
                delta=model.selected_residual(hidden,decision)
                route=int(decision.logits.argmax());probabilities=decision.probabilities[0].tolist()
                if step['selected_route']!=route or step['availability_probabilities']!=probabilities or step['decision_created']!=(i==0):
                    raise ValueError('decision not predicted once from initial prefill')
                if not exact(delta[0].numpy(),np.frombuffer(wire,dtype='<f4',offset=16+4*n)):raise ValueError('trained residual replay differs')
                expected,correction=continuous_logits(base,delta,head,manifest['logit_scale'])
                actual=values[1024+2*73448:];adjustment=values[1024+73448:1024+2*73448]
                passed=(np.allclose(actual,expected[0].numpy(),atol=1e-5,rtol=1e-4) and
                        np.allclose(adjustment,correction[0].numpy(),atol=1e-5,rtol=1e-4) and
                        int(actual.argmax())==int(expected[0].argmax())==step['predicted_token_id'])
                if route==0 and not exact(actual,base[0].numpy()):raise ValueError('normal route changed native base')
                checks.append({'position':i,'passed':bool(passed),'max_logit_error':float(np.max(np.abs(actual-expected[0].numpy()))),
                               'memory_changed_next_token':int(actual.argmax())!=int(base[0].argmax())})
            if not checks:raise ValueError('missing generation evidence')
            if prediction['generated_token_ids']!=[s['predicted_token_id'] for s in backend['steps']]:raise ValueError('reported tokens differ')
            normal_same=prediction['generated_token_ids']==before['token_ids'] and prediction['raw_text_hex']==before['raw_text_hex']
            if route==0 and not normal_same:raise ValueError('normal reply differs from registered baseline')
            rows.append({'id':prediction['id'],'selected_route':route,'positions':len(checks),'normal_matches_baseline':normal_same if route==0 else None,
                         'passed':all(c['passed'] for c in checks),'checks':checks})
    return {'format':'cg002-native-generation-audit-v1','passed':all(r['passed'] for r in rows),'cases':rows,
            'positions':sum(r['positions'] for r in rows),'all_raw_hashes_verified':True,'kv_reuse':False,
            'initial_predicted_decision_replayed_and_fixed':True,'normal_route_exact':True,
            'neural_module_implementation':'python','generation_sha256':sha_file(root/'generation.json'),
            'source_sha256':sha_file(__file__),'semantic_accuracy_measured':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('generation','features','checkpoint','baseline','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();report=audit(a.generation,a.features,a.checkpoint,a.baseline)
    with Path(a.output).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'passed':report['passed'],'positions':report['positions']}),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
