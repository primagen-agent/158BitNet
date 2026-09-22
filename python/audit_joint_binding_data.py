"""Mechanical train/dev C-token audit; sealed test receives no model/token calls."""
import argparse
from collections import Counter
import json
from pathlib import Path

from availability_supervision import encode_reply_targets
from c_tokenizer import CTokenizer
from diagnose_role_content_path import token_span
from episode_memory_inputs import encoder_texts
from joint_binding_schedule import pair_schedule
from native_memory_encoder import sha_file, BACKBONE_SHA256
from native_token_payload import payload_mask
from prepare_joint_binding_curriculum import materialize
from prepare_neural_memory_protocol import digest
from review_neural_memory import read_rows


def run(a):
    root = Path(a.output); root.mkdir(parents=True, exist_ok=False)
    manifest = materialize(a.config, a.corpus, verify=True)
    if sha_file(a.gguf) != BACKBONE_SHA256: raise ValueError('exact 0.5B GGUF required')
    old = json.loads(Path(a.previous_config).read_text()); new = json.loads(Path(a.config).read_text())
    if set(new['people']) & (set(old['people']) | set(old['excluded_subjects'])) or set(new['cities']) & set(old['cities']):
        raise ValueError('earlier CG-001 identities reused')
    tokenizer = CTokenizer(a.tok_probe, a.gguf); rows = []; metadata = {}
    try:
        for split in ('train', 'dev'):
            inputs = {r['id']: r for r in read_rows(Path(a.corpus) / f'{split}.inputs.jsonl')}
            labels = {r['id']: r for r in read_rows(Path(a.corpus) / f'{split}.labels.jsonl')}
            index = {r['id']: r for r in read_rows(Path(a.corpus) / f'{split}.index.jsonl')}; metadata.update(index)
            for key, runtime in inputs.items():
                reply = encode_reply_targets(runtime, labels[key], tokenizer, context_capacity=128)
                query, sources = encoder_texts(runtime); query_ids = tokenizer.encode(query, True)
                query_bytes = b''.join(tokenizer.decode_pieces(query_ids[1:]))
                if query_bytes not in (query.encode(), b' ' + query.encode()): raise ValueError('query byte mismatch')
                ids, allowed, pieces = [], [], []
                if sources:
                    ids = tokenizer.encode(sources[0], True)[1:]; pieces = tokenizer.decode_pieces(ids)
                    allowed = payload_mask(runtime['episodes'][0], sources[0], pieces)
                    blocked = (tokenizer.bos(), tokenizer.eos())
                    allowed = [ok and t not in blocked for t, ok in zip(ids, allowed)]
                if len(query_ids) >= 512 or len(ids) + 1 >= 512: raise ValueError('native encoder capacity exceeded')
                value_offsets, source_offsets = [], []
                if labels[key]['state'] == 'supported':
                    value = labels[key]['required_claims'][0]['value']
                    value_offsets = token_span(reply.response, tokenizer.decode_pieces(list(reply.completion_token_ids[:-1])), value)
                    source_offsets = token_span(sources[0], pieces, value)
                available = {ids[i] for i in source_offsets if allowed[i]}
                missing = [i for i in value_offsets if reply.completion_token_ids[i] not in available]
                rows.append({'id': key, 'split': split, 'scenario': index[key]['scenario'], 'language': index[key]['language'],
                    'state': labels[key]['state'], 'query_tokens': len(query_ids), 'source_tokens': len(ids),
                    'full_reply_prefix_tokens': len(reply.prompt_token_ids) + len(reply.completion_token_ids),
                    'reply_tokens': len(reply.completion_token_ids), 'value_tokens': len(value_offsets),
                    'uncopyable_value_offsets': missing, 'eos_verified': True, 'all_reply_tokens_retained': True})
        pairs = read_rows(Path(a.corpus) / 'train.pairs.jsonl'); schedule = pair_schedule(pairs)
        selected = [p for step in schedule for p in step['pairs']]
        draws = [key for p in selected for key in (p['left_id'], p['right_id'])]
        row_map = {r['id']: r for r in rows}
        if any(row_map[key]['split'] != 'train' for key in draws): raise ValueError('nontrain sampler access')
        dev_world = min(m['world_id'] for m in metadata.values() if m['split'] == 'dev')
        kinds = ('correct', 'role_swap', 'value_swap', 'empty', 'ordinary_source', 'ordinary_empty')
        panel = sorted((m for m in metadata.values() if m['world_id'] == dev_world and m['scenario'] in kinds),
                       key=lambda m: (m['language'], m['relation_family'], kinds.index(m['scenario'])))
        if len(panel) != 24: raise ValueError('native panel size changed')
        summary = {s: {'records': sum(r['split'] == s for r in rows),
            'reply_tokens': sum(r['reply_tokens'] for r in rows if r['split'] == s),
            'value_tokens': sum(r['value_tokens'] for r in rows if r['split'] == s),
            'uncopyable_value_tokens': sum(len(r['uncopyable_value_offsets']) for r in rows if r['split'] == s),
            'max_full_reply_prefix_tokens': max(r['full_reply_prefix_tokens'] for r in rows if r['split'] == s)} for s in ('train', 'dev')}
        report = {'dataset_manifest_digest': digest(manifest), 'rows': rows, 'summary': summary,
            'all_encoder_decoder_capacity_and_reply_bytes_verified': True, 'test_tokenized_or_evaluated': False,
            'previous_CG001_subjects_and_values_disjoint': True, 'schedule': schedule,
            'schedule_summary': {'steps': len(schedule), 'pairs': len(selected), 'record_uses': len(draws), 'unique_records': len(set(draws)),
                'kind_counts': dict(Counter(p['kind'] for p in selected)), 'state_draws': dict(Counter(row_map[k]['state'] for k in draws)),
                'language_draws': dict(Counter(row_map[k]['language'] for k in draws))},
            'native_panel': [{'id': m['id'], 'world': m['world_id'], 'language': m['language'], 'relation': m['relation_family'],
                              'scenario': m['scenario']} for m in panel],
            'optimizer_steps': 0, 'training_approved': False, 'memory_accuracy_measured': False,
            'artifact_sha256': {'gguf': sha_file(a.gguf), 'tok_probe': sha_file(a.tok_probe)},
            'source_sha256': {n: sha_file(Path(__file__).parent / n) for n in ('audit_joint_binding_data.py', 'joint_binding_schedule.py',
                'prepare_joint_binding_curriculum.py', 'availability_supervision.py', 'native_token_payload.py')}}
        with (root / 'report.json').open('x') as f: json.dump(report, f, ensure_ascii=False, indent=2); f.write('\n')
        print(json.dumps({'summary': summary, 'schedule': report['schedule_summary']}, ensure_ascii=False), flush=True)
    finally:
        tokenizer._proc.terminate(); tokenizer._proc.wait(); tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('config', 'corpus', 'previous-config', 'gguf', 'tok-probe', 'output'): p.add_argument('--' + n, required=True)
    run(p.parse_args())
