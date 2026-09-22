"""Independent DG-019 byte/phase/actual-prefix raw-file audit. No inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct

import numpy as np

from append_value_transport import AppendValueCodec
from audit_joint_generation import reference_file
from check_value_teacher_trajectories import certificates,RESEARCH,ROOT
from diagnose_native_generation_gradient import exact
from native_memory_encoder import sha_file,digest
from native_prefix_bank import read_prefix_batch
from neural_memory_generation import encode_generation_input


def run(a):
    report_path=Path(a.report);report=json.loads(report_path.read_text());root=Path(report['raw_root'])
    registration=RESEARCH/'experiments/DG-019.json'
    if sha_file(registration)!=report['registration_sha256']:raise ValueError('registration changed')
    if not report['passed'] or report['optimizer_steps'] or not report['training_only'] or report['origin']!='teacher_forcing_oracle':
        raise ValueError('unqualified teacher report')
    for name,h in report['source_sha256'].items():
        if sha_file(ROOT/'python'/name)!=h:raise ValueError('source changed')
    certificates()
    for name,h in report['raw_sha256'].items():
        p=root/name
        if not p.resolve().is_relative_to(root.resolve()) or sha_file(p)!=h:raise ValueError('raw artifact changed')
    compiled=json.loads((root/'trajectories.json').read_text());prefixes=[tuple(p) for p in compiled['unique_prefixes']]
    if sha_file(root/'trajectories.json')!=report['trajectories_sha256'] or not compiled['training_only']:
        raise ValueError('compiled archive changed')
    corpus=RESEARCH/'data/JB-001';data={}
    for name in ('inputs','labels','index'):
        data[name]={r['id']:r for r in map(json.loads,(corpus/f'train.{name}.jsonl').read_text().splitlines())}
    selection=json.loads(registration.read_text())['selection'];expected=[]
    for language in selection['languages']:
        for scenario in selection['scenarios']:
            matches=[r['id'] for r in data['index'].values() if r['world_id']==selection['world_id'] and
                r['relation_family']==selection['relation_family'] and r['language']==language and r['scenario']==scenario]
            if len(matches)!=1:raise ValueError('ambiguous panel')
            expected+=matches
    if [r['record_id'] for r in compiled['records']]!=expected or [r['id'] for r in report['records']]!=expected:
        raise ValueError('changed panel')
    m=json.loads((ROOT/'build/neural-memory-cg003-full-features/manifest.json').read_text())
    codec=AppendValueCodec(ROOT/'build/tok_probe',ROOT/'models/bitcpm4-0.5b-tq2_0.gguf',m['tokenizer_sha256'])
    references=set();uses=[];counts=Counter();by_case=[]
    try:
        for t,row in zip(compiled['records'],report['records']):
            runtime=data['inputs'][t['record_id']];label=data['labels'][t['record_id']]
            initial=encode_generation_input(runtime,codec.tokenizer).prompt_token_ids;ids=t['completion_ids'];phases=t['phases'];idle=t['idle_targets']
            if (list(initial)!=t['prompt_ids'] or t['input_sha256']!=digest(runtime) or t['origin']!='teacher_forcing_oracle' or
                    len(ids)!=len(phases) or len(ids)!=len(idle) or ids[-1]!=codec.tokenizer.eos() or
                    b''.join(codec.decode_pieces(ids[:-1]))!=label['response'].encode() or any(codec.forbidden(v) for v in ids[:-1])):
                raise ValueError('teacher input/reply bytes/control changed')
            route={'no_memory_needed':0,'supported':1,'insufficient':2}[label['state']]
            if t['static_targets']['route']!=route or t['static_targets']['idle_mode'] is not None:raise ValueError('static supervision changed')
            for phase,target in zip(phases,idle):
                if target!={'GENERATE':0,'START':1,'CONTINUE':None,'END':0}[phase]:raise ValueError('mode label changed')
            if route==1:
                if phases.count('START')!=1 or phases.count('END')!=1:raise ValueError('missing transitions')
                s,e=phases.index('START'),phases.index('END')
                if not s<e or any(p!='CONTINUE' for p in phases[s+1:e]) or any(p!='GENERATE' for p in phases[:s]+phases[e+1:]):
                    raise ValueError('invalid active path')
                vs,ve=t['static_targets']['value_bytes'];raw=runtime['episodes'][0]['text'].encode()
                if b''.join(codec.decode_pieces(ids[s:e]))!=raw[vs:ve] or raw[vs:ve].decode()!=label['required_claims'][0]['value']:
                    raise ValueError('source value differs from transported value')
            elif any(p!='GENERATE' for p in phases):raise ValueError('negative/normal entered value mode')
            actual=[initial+tuple(ids[:i]) for i in range(len(ids))];uses+=actual;counts.update(phases)
            refs={0,len(ids)//2,len(ids)-1}|{i for i,p in enumerate(phases) if p!='GENERATE'}
            references.update(actual[i] for i in refs)
            if row['phase_counts']!=dict(Counter(phases)) or row['idle_positions']!=sum(i is not None for i in idle):raise ValueError('phase denominator changed')
            if not row['gradient_gate'] or not all(v is None or np.isfinite(v) for v in row['gradient_l1'].values()):raise ValueError('invalid recorded gradients')
            by_case.append({'id':t['record_id'],'positions':len(ids),'full_reply_bytes_exact':True,'phases_valid':True})
    finally:codec.close()
    if list(dict.fromkeys(uses))!=prefixes or sum(counts.values())!=report['positions'] or dict(counts)!=report['phase_counts']:
        raise ValueError('unique prefix inventory changed')
    bank={}
    for batch in report['batches']:
        directory=root/batch['folder'];group=[tuple(p) for p in batch['prefixes']]
        wire=b'BNPI0001'+struct.pack('<I',len(group))+b''.join(struct.pack('<I',len(p))+np.asarray(p,dtype='<i4').tobytes() for p in group)
        if (directory/'input.bin').read_bytes()!=wire or sha_file(directory/'output.bin')!=batch['output_sha256']:raise ValueError('batch wire changed')
        log=(directory/'native.log').read_text()
        if '[bitnet] cpu tier: arm_neon' not in log or f'BNP_TRACE rows={len(group)} fresh_contexts={len(group)}' not in log:raise ValueError('fresh contexts not verified')
        h,z=read_prefix_batch(directory/'output.bin',group)
        for p,hh,zz in zip(group,h,z):
            if p in bank:raise ValueError('duplicate extraction')
            bank[p]=(hh,zz)
    if list(bank)!=prefixes:raise ValueError('extracted prefix inventory changed')
    ordered=[p for p in prefixes if p in references]
    if compiled['direct_reference_indices']!=[prefixes.index(p) for p in ordered] or len(ordered)!=len(report['direct_reference_checks']):
        raise ValueError('direct reference coverage changed')
    for i,(p,row) in enumerate(zip(ordered,report['direct_reference_checks'])):
        if list(p)!=row['prefix_ids'] or row['prefix_index']!=prefixes.index(p):raise ValueError('wrong reference prefix')
        h,z=reference_file(root/'reference',f'row-{i:03d}',p);hh,zz=bank[p]
        if not exact(h[0].numpy(),hh) or not exact(z[0].numpy(),zz):raise ValueError('raw numerical mismatch')
    result={'format':'dg019-independent-raw-audit-v1','report_sha256':sha_file(report_path),'source_sha256':sha_file(__file__),
            'passed':True,'records':by_case,'positions':len(uses),'unique_prefixes':len(prefixes),'direct_references':len(ordered),
            'origin':'teacher_forcing_oracle','autonomous_accuracy_measured':False}
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','source_sha256','report_sha256')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True);p.add_argument('--output',required=True);run(p.parse_args())
