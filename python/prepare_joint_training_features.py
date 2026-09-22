"""CG-003 C feature packaging, no training. Explicit smoke/full scope required."""
import argparse
import json
from pathlib import Path
import numpy as np

from c_tokenizer import CTokenizer
from diagnose_native_generation_gradient import native_forward, exact
from ggw import GGUFWeights
from joint_training_data import compile_records
from native_memory_encoder import NativeMemoryEncoder, save_bank, sha_file, BACKBONE_SHA256
from native_prefix_bank import extract_prefix_batch
from neural_memory_generation import TEMPLATE_VERSION
from prepare_neural_memory_protocol import digest


def run(a):
    root = Path(a.output); root.mkdir(parents=True, exist_ok=False)
    encoder = NativeMemoryEncoder(a.gguf, a.encoder_probe)
    tokenizer = CTokenizer(a.tok_probe, a.gguf)
    try:
        compiled = compile_records(a.config, a.corpus, tokenizer, scope=a.scope)
        blocked = [tokenizer.bos(), tokenizer.eos()]
    finally:
        tokenizer._proc.terminate(); tokenizer._proc.wait(); tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()
    binaries = {n: sha_file(getattr(a, n)) for n in ('prefix_probe', 'reference_probe', 'encoder_probe', 'tok_probe', 'lib')}
    with (root / 'records.json').open('x') as f: json.dump(compiled['records'], f, ensure_ascii=False, indent=2)
    with (root / 'inventory.json').open('x') as f: json.dump({k: v for k, v in compiled.items() if k != 'records'}, f, ensure_ascii=False, indent=2)
    prefixes = compiled['prefixes']; texts = compiled['texts']
    print(json.dumps({'phase': 'compiled', 'scope': a.scope, 'records': len(compiled['records']),
                      'prefixes': len(prefixes), 'texts': len(texts), 'plan_only': a.plan_only}), flush=True)
    if a.plan_only: return
    selected = [prefixes[i] for i in (0, len(prefixes)//2, len(prefixes)-1)]
    control = [selected[2], selected[0], selected[1], selected[0]]
    h, z = extract_prefix_batch(a.prefix_probe, a.gguf, control, root / 'parity')
    parity = []
    for i, ids in enumerate(control):
        ref = native_forward(a.reference_probe, a.gguf, ids, np.zeros(1024, dtype='<f4'), root / 'parity', f'reference-{i}', False)
        parity.append(exact(h[i], ref['hidden']) and exact(z[i], ref['logits']))
    if parity != [True]*4: raise ValueError('C prefix extractor changed reference bits')
    print(json.dumps({'phase': 'parity', 'passed': 4}), flush=True)
    (root / 'prefixes').mkdir(); batches = []
    for start in range(0, len(prefixes), 64):
        group = prefixes[start:start+64]; folder = root / 'prefixes' / f'batch-{start//64:04d}'
        extract_prefix_batch(a.prefix_probe, a.gguf, group, folder)
        path = folder / 'output.bin'
        batches.append({'path': str(path.relative_to(root / 'prefixes')), 'sha256': sha_file(path), 'prefixes': group})
        print(json.dumps({'phase': 'prefixes', 'completed': start + len(group), 'total': len(prefixes)}), flush=True)
    pm = {'format': 'native-prefix-bank-v1', 'backbone_sha256': BACKBONE_SHA256, 'probe_sha256': sha_file(a.prefix_probe),
          'encoder_identity': encoder.identity, 'template_version': TEMPLATE_VERSION, 'training_only': True,
          'cross_prefix_kv_reuse': False, 'batches': batches}
    with (root / 'prefixes/manifest.json').open('x') as f: json.dump(pm, f, indent=2)
    rows = encoder.encode(texts, root / 'token-extraction')
    token_digest = save_bank(root / 'tokens', encoder.identity, texts, rows)
    weights = GGUFWeights(a.gguf, a.lib)
    try:
        head = weights.get_f32('token_embd.weight', (73448, 1024))
        with (root / 'output-head.npy').open('xb') as f: np.save(f, head, allow_pickle=False)
        scale = weights.logit_scale
    finally: weights.close()
    if sha_file(a.gguf) != BACKBONE_SHA256 or any(sha_file(getattr(a, n)) != value for n, value in binaries.items()):
        raise ValueError('native artifact changed during extraction')
    m = {'format': 'joint-training-features-v1', 'scope': a.scope, 'splits': compiled['splits'],
        'backbone_sha256': BACKBONE_SHA256, 'corpus_manifest_digest': compiled['corpus_manifest_digest'],
        'template_version': TEMPLATE_VERSION, 'encoder_identity': encoder.identity, 'tokenizer_sha256': binaries['tok_probe'],
        'records_sha256': sha_file(root / 'records.json'), 'output_head_sha256': sha_file(root / 'output-head.npy'),
        'prefix_manifest_digest': digest(pm), 'token_manifest_digest': token_digest, 'logit_scale': scale,
        'blocked_ids': blocked, 'reference_parity': parity, 'records': len(compiled['records']),
        'prefixes': len(prefixes), 'texts': len(texts), 'binaries_sha256': binaries, 'optimizer_steps': 0,
        'source_sha256': {n: sha_file(Path(__file__).parent / n) for n in ('prepare_joint_training_features.py',
            'joint_training_data.py', 'availability_supervision.py', 'native_memory_encoder.py', 'native_prefix_bank.py')}}
    with (root / 'manifest.json').open('x') as f: json.dump(m, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({'phase': 'complete', 'digest': digest(m), 'scope': a.scope}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('config', 'corpus', 'gguf', 'prefix-probe', 'reference-probe', 'encoder-probe', 'tok-probe', 'lib', 'output'):
        p.add_argument('--' + n, required=True)
    p.add_argument('--scope', choices=('smoke', 'full'), required=True)
    p.add_argument('--plan-only', action='store_true')
    run(p.parse_args())
