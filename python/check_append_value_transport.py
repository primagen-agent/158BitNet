"""DG-016 oracle suffix transport on fresh actual C prefixes, not recall."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from append_value_transport import AppendValueCodec, compile_append_value
from audit_joint_generation import continuous_file, reference_file
from check_value_transport import frozen_files
from diagnose_continuous_memory import forward
from diagnose_native_generation_gradient import native_forward, exact
from joint_optimizer import FEATURE_DIGEST
from native_memory_encoder import BACKBONE_SHA256, digest, sha_file
from neural_memory_contract import ModelBinding
from value_transport import (ActivatedFact, FactPayload, Origin, PayloadSnapshot, ReplyBinding, ValueSpan,
                             TransportError, TransportLimits, UnsafeToken, TransportBudgetExceeded)

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT/'training/memory/neural-system'
PROBES = {'memory_continuous_probe':'4ce9aa3f21ef95f589fb996f2a43c30a917e038f3fc4e1285bd60769691288ce',
          'memory_gradient_reference':'4c05a9c4faa7b61d03a017a3a80f43c36329c5d8af945ea5c9f1359640de9152'}


def protected():
    result = frozen_files()
    old = json.loads((RESEARCH/'reviews/DG-015/native-fixtures-certified.json').read_text())
    for name, expected in old['source_sha256'].items():
        if sha_file(ROOT/'python'/name) != expected: raise ValueError('DG-015 implementation changed')
        result['python/'+name] = expected
    if sha_file(RESEARCH/'experiments/DG-015.json') != old['registration_sha256']: raise ValueError('old registration changed')
    for name, expected in PROBES.items():
        if sha_file(ROOT/'build'/name) != expected: raise ValueError('native executable changed')
        result['build/'+name] = expected
    return result


def run(a):
    root = Path(a.raw); root.mkdir(parents=True, exist_ok=False)
    registration = RESEARCH/'experiments/DG-016.json'; reg = json.loads(registration.read_text())
    old = json.loads((RESEARCH/'experiments/DG-015.json').read_text())
    fixtures = [dict(f) for f in old['native_fixtures']]
    for f in fixtures:
        if f['id'] in ('prefix-space-merge','prefix-letter-merge'): f['expected']='accept'
    fixtures += reg['extra_fixtures']
    before = protected(); gguf=ROOT/'models/bitcpm4-0.5b-tq2_0.gguf'
    manifest=json.loads((ROOT/'build/neural-memory-cg003-full-features/manifest.json').read_text())
    if digest(manifest) != FEATURE_DIGEST: raise ValueError('feature identity changed')
    codec=AppendValueCodec(ROOT/'build/tok_probe',gguf,manifest['tokenizer_sha256'])
    fixture_sha=hashlib.sha256(b'DG-016 artificial oracle value handles, no learned selector').hexdigest()
    binding=ModelBinding(BACKBONE_SHA256,codec.tokenizer_sha256,fixture_sha,sha_file(registration));rows=[]
    zero=np.zeros(1024,dtype='<f4')
    try:
        for f in fixtures:
            name=f['id']; directory=root/name; directory.mkdir()
            prefix_text=f['prefix'];value=f['value'].encode()
            initial=codec.encode_with_bos(prefix_text)
            if 'prefix_parts' in f:
                parts=f['prefix_parts']
                if ''.join(parts)!=prefix_text: raise ValueError('invalid noncanonical fixture')
                initial=codec.encode_with_bos(parts[0])
                for part in parts[1:]: initial+=codec.encode_value(part.encode())
                if initial==codec.encode_with_bos(prefix_text): raise ValueError('fixture must be noncanonical')
            prefix=initial
            snapshot=PayloadSnapshot(binding,fixture_sha,'fixture-user',name,0,(FactPayload('f',b'value='+value+b'; unrelated'),))
            reply=ReplyBinding('fixture-user',name,hashlib.sha256(prefix_text.encode()).hexdigest())
            handle=ActivatedFact(snapshot.digest,reply,ValueSpan(0,6,6+len(value)),1.,Origin.ORACLE_FIXTURE)
            steps=[];emitted=[];actual='accept';error=None;compiled=None
            try:
                cursor=compile_append_value(handle,snapshot,reply,initial,prefix_text,codec,allow_oracle=True,
                    limits=TransportLimits(max_value_tokens=f.get('max_value_tokens',64)))
                compiled=cursor.value_tokens
                initial_bytes=b''.join(codec.decode_pieces(initial[1:]))
                for i in range(len(compiled)+1):
                    if prefix!=initial+tuple(emitted) or prefix[:len(initial)]!=initial: raise ValueError('prefix rewritten')
                    name_step=f'step-{i:03d}'
                    forward(str(ROOT/'build/memory_continuous_probe'),str(gguf),prefix,zero,directory,name_step+'-base',False)
                    native_forward(str(ROOT/'build/memory_gradient_reference'),str(gguf),prefix,zero,directory,name_step+'-reference',False)
                    # Independently parse the recorded wires, including exact IDs and fresh trace.
                    h,base,correction,out=continuous_file(directory,name_step+'-base',prefix,False)
                    rh,rb=reference_file(directory,name_step+'-reference',prefix)
                    if (not exact(h.numpy(),rh.numpy()) or not exact(base.numpy(),rb.numpy()) or
                            not exact(base.numpy(),out.numpy()) or np.count_nonzero(correction.numpy())):
                        raise ValueError('ordinary/disabled C mismatch')
                    base_token=int(base.argmax());selected=cursor.select(base_token,snapshot,reply,prefix,allow_oracle=True)
                    finished=cursor.done
                    if finished and selected!=base_token: raise ValueError('normal bypass changed')
                    steps.append({'position':i,'prefix_ids':list(prefix),'base_argmax':base_token,'selected_token':selected,
                        'forced_value_token':not finished,'returned_to_base':finished,'reference_bitwise_equal':True,
                        'fresh_context':True,'initial_context_position':0,'prefill_tokens':len(prefix)})
                    if finished: break
                    cursor=cursor.commit(selected,snapshot,reply,prefix,allow_oracle=True)
                    emitted.append(selected);prefix+=(selected,)
                    expected=initial_bytes+b''.join(codec.decode_pieces(emitted))
                    if b''.join(codec.decode_pieces(prefix[1:]))!=expected: raise ValueError('decoded prefix changed')
                if not cursor.done or cursor.committed_text!=f['value'] or b''.join(codec.decode_pieces(emitted))!=value:
                    raise ValueError('incomplete exact value')
            except (UnsafeToken,TransportBudgetExceeded) as e:
                actual='reject_control' if isinstance(e,UnsafeToken) else 'reject_budget'; error=str(e)
                if emitted or steps: raise ValueError('rejected after emission')
            except (TransportError,ValueError) as e:
                actual='unexpected_failure';error=str(e)
            raw={p.name:sha_file(p) for p in sorted(directory.iterdir()) if p.is_file()}
            row={'id':f['id'],'expected':f['expected'],'actual':actual,'passed':actual==f['expected'],
                 'origin':'oracle_fixture','initial_prefix_ids':list(initial),'canonical_prefix_ids':list(codec.encode_with_bos(prefix_text)),
                 'compiled_value_ids':list(compiled) if compiled else None,'emitted_token_ids':emitted,
                 'emitted_piece_hex':[p.hex() for p in codec.decode_pieces(emitted)],'expected_value_hex':value.hex(),
                 'error':error,'steps':steps,'raw_sha256':raw}
            with (directory/'result.json').open('x') as out:json.dump(row,out,indent=2);out.write('\n')
            rows.append(row);print(json.dumps({'fixture':f['id'],'passed':row['passed'],'C_prefixes':len(steps),'actual':actual}),flush=True)
        codec.verify_identity()
    finally:codec.close()
    if protected()!=before:raise ValueError('protected files changed')
    result={'format':'dg016-append-native-fixtures-v1','registration_sha256':sha_file(registration),
        'source_sha256':{n:sha_file(ROOT/'python'/n) for n in ('append_value_transport.py','check_append_value_transport.py')},
        'origin':'oracle_fixture','optimizer_steps':0,'autonomous_recall_measured':False,'natural_reply_quality_measured':False,
        'backbone_sha256':BACKBONE_SHA256,'tokenizer_sha256':manifest['tokenizer_sha256'],
        'cross_step_kv_reuse':False,'internal_prefill_kv_buffers':True,'raw_root':str(root.resolve()),
        'protected_files_unchanged':before,'cases':rows,'C_prefixes':sum(len(r['steps']) for r in rows),
        'C_calls':2*sum(len(r['steps']) for r in rows),'passed':len(rows)==14 and all(r['passed'] for r in rows)}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as out:json.dump(result,out,ensure_ascii=False,indent=2);out.write('\n')
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True);run(p.parse_args())
