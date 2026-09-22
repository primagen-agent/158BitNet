"""DG-015 native tokenizer + oracle software fixtures. No neural recall score."""
import argparse
import hashlib
import json
from pathlib import Path

from joint_optimizer import FEATURE_DIGEST
from native_memory_encoder import BACKBONE_SHA256, digest, sha_file
from native_value_codec import NativeValueCodec
from neural_memory_contract import ModelBinding
from value_transport import (ActivatedFact, FactPayload, Origin, PayloadSnapshot, ReplyBinding, ValueSpan,
    TransportLimits, PrefixRetokenizationRequired, UnsafeToken, TransportBudgetExceeded, compile_value)

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / 'training/memory/neural-system'


def frozen_files():
    result = {}
    expected_status = {'joint_aux':'c08da22f0bb525c7fc794b36eb77da8a1548583df2cb8dd91091ea3221e4672e',
                       'joint_product':'c818f8c99b6eab452a57259db1cc4accbdc9d9c637a025172a8770500aa8a593'}
    for arm in ('joint_aux', 'joint_product'):
        status_path = ROOT / 'build/neural-memory-cg003-training-results/training' / arm / 'status.json'
        if sha_file(status_path) != expected_status[arm]: raise ValueError('training certificate changed')
        status = json.loads(status_path.read_text())
        for name, expected in status['binding']['source_sha256'].items():
            path = ROOT / 'python' / name
            if sha_file(path) != expected: raise ValueError('frozen training source changed')
            result[str(path.relative_to(ROOT))] = expected
        for cert in status['checkpoints']:
            path = status_path.parent / cert['file']
            if sha_file(path) != cert['sha256']: raise ValueError('checkpoint changed')
            result[str(path.relative_to(ROOT))] = cert['sha256']
        result[str(status_path.relative_to(ROOT))] = sha_file(status_path)
    return result


def run(a):
    registration = RESEARCH / 'experiments/DG-015.json'; reg = json.loads(registration.read_text())
    before = frozen_files()
    manifest = json.loads((ROOT / 'build/neural-memory-cg003-full-features/manifest.json').read_text())
    if digest(manifest) != FEATURE_DIGEST: raise ValueError('feature package identity changed')
    codec = NativeValueCodec(ROOT / 'build/tok_probe', ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf', manifest['tokenizer_sha256'])
    fixture_sha = hashlib.sha256(b'DG-015 oracle software fixture, no memory model').hexdigest()
    binding = ModelBinding(BACKBONE_SHA256, codec.tokenizer_sha256, fixture_sha, sha_file(registration))
    rows = []
    try:
        for fixture in reg['native_fixtures']:
            name = fixture['id']; prefix = fixture['prefix']; value = fixture['value'].encode()
            # Artificial bytes, not a trained writer or a reader prediction.
            payload = b'fixture(' + value + b') unrelated payload'
            snapshot = PayloadSnapshot(binding, fixture_sha, 'fixture-user', name, 0, (FactPayload('fixture-fact', payload),))
            reply = ReplyBinding('fixture-user', name, hashlib.sha256(prefix.encode()).hexdigest())
            handle = ActivatedFact(snapshot.digest, reply, ValueSpan(0, 8, 8+len(value)), 1., Origin.ORACLE_FIXTURE)
            prefix_ids = codec.encode_with_bos(prefix); original_prefix = prefix_ids
            combined = codec.encode_with_bos(prefix + value.decode())
            emitted = []; pieces = []; actual = 'accept'; error = None
            try:
                cursor = compile_value(handle, snapshot, reply, prefix_ids, prefix, codec, allow_oracle=True,
                    limits=TransportLimits(max_value_tokens=fixture.get('max_value_tokens', 64)))
                original_ids = cursor.value_tokens
                for _ in range(len(original_ids)):
                    token = cursor.select(codec.bos_id, snapshot, reply, prefix_ids, allow_oracle=True)
                    updated = cursor.commit(token, snapshot, reply, prefix_ids, allow_oracle=True)
                    emitted.append(token); pieces.extend(codec.decode_pieces((token,)))
                    prefix_ids += (token,); cursor = updated
                if not cursor.done or b''.join(pieces) != value: raise AssertionError('transport did not terminate exactly')
                if cursor.committed_text != value.decode(): raise AssertionError('UTF-8 view changed')
                if cursor.select(42, snapshot, reply, prefix_ids, allow_oracle=True) != 42: raise AssertionError('base bypass changed')
            except (PrefixRetokenizationRequired, UnsafeToken, TransportBudgetExceeded) as exc:
                actual = {PrefixRetokenizationRequired:'reject_prefix', UnsafeToken:'reject_control', TransportBudgetExceeded:'reject_budget'}[type(exc)]
                error = str(exc)
                if emitted: raise AssertionError('failed after partial emission')
            rows.append({'id': name, 'expected': fixture['expected'], 'actual': actual, 'passed': actual == fixture['expected'],
                         'original_prefix_ids': original_prefix, 'canonical_combined_ids': combined,
                         'original_prefix_piece_hex': [p.hex() for p in codec.decode_pieces(original_prefix[1:])],
                         'canonical_combined_piece_hex': [p.hex() for p in codec.decode_pieces(combined[1:])],
                         'origin': handle.origin.value, 'emitted_token_ids': emitted, 'emitted_piece_hex': [p.hex() for p in pieces],
                         'expected_value_hex': value.hex(), 'emitted_value_hex': b''.join(pieces).hex(), 'error': error})
        codec.verify_identity()
    finally: codec.close()
    if frozen_files() != before: raise ValueError('protected sources/weights changed during checks')
    report = {'format':'dg015-value-transport-fixtures-v1', 'registration_sha256':sha_file(registration),
              'source_sha256':{n:sha_file(ROOT/'python'/n) for n in ('value_transport.py','native_value_codec.py','check_value_transport.py')},
              'backbone_sha256':BACKBONE_SHA256, 'tokenizer_sha256':manifest['tokenizer_sha256'],
              'origin':'oracle_fixture', 'optimizer_steps':0, 'model_generation_calls':0, 'semantic_accuracy_measured':False,
              'protected_files_unchanged':before, 'cases':rows, 'passed':all(r['passed'] for r in rows)}
    path = Path(a.output); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(report, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({'cases':len(rows), 'passed':sum(r['passed'] for r in rows), 'neural_recall_measured':False}))
    if not report['passed']: raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True); run(p.parse_args())
