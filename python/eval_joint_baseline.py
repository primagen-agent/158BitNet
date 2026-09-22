"""CG-003 fixed 24-case disabled C baseline; no labels or optimizer."""
import argparse
import json
from pathlib import Path

from c_tokenizer import CTokenizer
from joint_training_data import DATA_DIGEST
from native_continuous_generation import tokenizer_stop_ids
from native_memory_encoder import BACKBONE_SHA256, sha_file
from native_prefix_bank import extract_prefix_batch
from neural_memory_generation import TEMPLATE_VERSION, encode_generation_input
from prepare_neural_memory_protocol import digest
from review_neural_memory import read_rows


SCENARIOS = ('correct', 'role_swap', 'value_swap', 'empty', 'ordinary_source', 'ordinary_empty')


def fixed_panel(corpus, audit):
    corpus = Path(corpus)
    manifest = json.loads((corpus / 'manifest.json').read_text())
    if digest(manifest) != DATA_DIGEST or audit['dataset_manifest_digest'] != DATA_DIGEST:
        raise ValueError('unregistered dataset')
    for name in ('dev.inputs.jsonl', 'dev.index.jsonl'):
        if sha_file(corpus / name) != manifest['file_sha256'][name]:
            raise ValueError('baseline input changed')
    metadata = read_rows(corpus / 'dev.index.jsonl')
    world = min(r['world_id'] for r in metadata)
    panel = sorted((r for r in metadata if r['world_id'] == world and r['scenario'] in SCENARIOS),
                   key=lambda r: (r['language'], r['relation_family'], SCENARIOS.index(r['scenario'])))
    expected = [{'id': r['id'], 'world': r['world_id'], 'language': r['language'],
                 'relation': r['relation_family'], 'scenario': r['scenario']} for r in panel]
    if len(panel) != 24 or expected != audit['native_panel']:
        raise ValueError('fixed baseline panel changed')
    inputs = {r['id']: r for r in read_rows(corpus / 'dev.inputs.jsonl')}
    return [inputs[r['id']] for r in panel], expected


def advance(prefix, generated, visible, token, stops):
    generated.append(token)
    if token in stops:
        return prefix, True
    visible.append(token)
    return prefix + (token,), False


def run(a):
    audit = json.loads(Path(a.audit).read_text())
    if sha_file(a.audit) != a.audit_sha256:
        raise ValueError('frozen panel audit changed')
    records, panel = fixed_panel(a.corpus, audit)
    if sha_file(a.gguf) != BACKBONE_SHA256 or sha_file(a.tok_probe) != audit['artifact_sha256']['tok_probe']:
        raise ValueError('backbone/tokenizer mismatch')
    identity = {n: sha_file(getattr(a, n)) for n in ('gguf', 'prefix_probe', 'tok_probe', 'audit')}
    source = {n: sha_file(Path(__file__).parent / n) for n in
              ('eval_joint_baseline.py', 'native_prefix_bank.py', 'neural_memory_generation.py', 'episode_memory_inputs.py')}
    root = Path(a.output); root.mkdir(parents=True, exist_ok=False)
    tokenizer = CTokenizer(a.tok_probe, a.gguf)
    try:
        requests = [encode_generation_input(r, tokenizer) for r in records]
        prefixes = [r.prompt_token_ids for r in requests]
        generated = [[] for _ in records]; visible = [[] for _ in records]
        done = [False] * 24; stops = tokenizer_stop_ids(tokenizer); raw = []
        for step in range(64):
            active = [i for i in range(24) if not done[i]]
            if not active: break
            if any(len(prefixes[i]) >= 128 for i in active):
                raise ValueError('registered panel exceeds context capacity')
            directory = root / f'step-{step:03d}'
            _, logits = extract_prefix_batch(a.prefix_probe, a.gguf, [prefixes[i] for i in active], directory)
            for position, i in enumerate(active):
                prefixes[i], done[i] = advance(prefixes[i], generated[i], visible[i],
                                                int(logits[position].argmax()), stops)
            raw.append({'step': step, 'active_case_ids': [records[i]['id'] for i in active],
                        'sha256': {n: sha_file(directory / n) for n in ('input.bin', 'output.bin', 'native.log')}})
            print(json.dumps({'step': step + 1, 'finished': sum(done), 'total': 24}), flush=True)
        predictions = []
        for i, r in enumerate(records):
            data = b''.join(tokenizer.decode_pieces(visible[i]))
            try: text = data.decode('utf-8'); valid = True
            except UnicodeDecodeError: text = data.decode('utf-8', errors='replace'); valid = False
            predictions.append({'id': r['id'], 'input_sha256': digest(r), 'text': text,
                'prompt_token_ids': list(requests[i].prompt_token_ids), 'generated_token_ids': generated[i],
                'visible_token_ids': visible[i], 'raw_text_hex': data.hex(), 'utf8_complete': valid,
                'finish_reason': 'stop_token' if done[i] else 'max_new_tokens', 'truncated': not done[i],
                'forward_calls': len(generated[i]), 'memory_enabled': False, 'backend_cache_policy_verified': True})
        if identity != {n: sha_file(getattr(a, n)) for n in identity}:
            raise ValueError('artifact changed during generation')
        if source != {n: sha_file(Path(__file__).parent / n) for n in source}:
            raise ValueError('source changed during generation')
        report = {'format': 'cg003-fixed-native-baseline-v1', 'panel': panel, 'predictions': predictions,
            'raw_steps': raw, 'artifact_sha256': identity, 'source_sha256': source, 'template_version': TEMPLATE_VERSION,
            'corpus_manifest_digest': DATA_DIGEST, 'max_new_tokens': 64, 'context_capacity': 128,
            'cached_tokens': 0, 'reused_tokens': 0, 'internal_prefill_kv_buffers': True,
            'memory_enabled': False, 'optimizer_steps': 0, 'memory_accuracy_measured': False,
            'semantic_review_complete': False, 'oracle_answers_used': False, 'prefix_bank_used': False}
        with (root / 'baseline.json').open('x') as f:
            json.dump(report, f, ensure_ascii=False, indent=2); f.write('\n')
        print(json.dumps({'finished': sum(done), 'total': 24, 'positions': sum(map(len, generated)),
                          'sha256': sha_file(root / 'baseline.json')}), flush=True)
    finally:
        tokenizer._proc.terminate(); tokenizer._proc.wait()
        tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('corpus', 'audit', 'audit-sha256', 'gguf', 'prefix-probe', 'tok-probe', 'output'):
        p.add_argument('--' + n, required=True)
    run(p.parse_args())
