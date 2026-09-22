"""Replay the disabled CG-003 baseline's raw C traces; prepare blind packets."""
import argparse
import json
from pathlib import Path
import struct

import numpy as np

from c_tokenizer import CTokenizer
from eval_joint_baseline import fixed_panel
from native_continuous_generation import tokenizer_stop_ids
from native_memory_encoder import sha_file
from native_prefix_bank import read_prefix_batch
from neural_memory_generation import encode_generation_input
from prepare_neural_memory_protocol import digest
from review_availability_generation import predictions
from review_neural_memory import blind_packet


def verify_input_bytes(path, prefixes):
    expected = b'BNPI0001' + struct.pack('<I', len(prefixes))
    for ids in prefixes:
        expected += struct.pack('<I', len(ids)) + np.asarray(ids, dtype='<i4').tobytes()
    if Path(path).read_bytes() != expected:
        raise ValueError('raw C input is not the actual natural/greedy prefix')


def run(a):
    path = Path(a.baseline)
    if sha_file(path) != a.baseline_sha256: raise ValueError('baseline identity mismatch')
    report = json.loads(path.read_text()); root = path.parent
    audit = json.loads(Path(a.panel_audit).read_text())
    if sha_file(a.panel_audit) != report['artifact_sha256']['audit']: raise ValueError('panel audit changed')
    records, panel = fixed_panel(a.corpus, audit)
    if (report['format'] != 'cg003-fixed-native-baseline-v1' or report['panel'] != panel or
            report['max_new_tokens'] != 64 or report['context_capacity'] != 128 or
            report['memory_enabled'] is not False or report['optimizer_steps'] != 0 or
            report['cached_tokens'] != 0 or report['reused_tokens'] != 0 or report['oracle_answers_used'] or report['prefix_bank_used']):
        raise ValueError('unregistered baseline protocol')
    for name in ('gguf', 'tok_probe'):
        if sha_file(getattr(a, name)) != report['artifact_sha256'][name]: raise ValueError('decoder/tokenizer changed')
    for name, expected in report['source_sha256'].items():
        if sha_file(Path(__file__).parent / name) != expected: raise ValueError('baseline source changed')
    rows = report['predictions']
    if [r['id'] for r in rows] != [r['id'] for r in records]: raise ValueError('prediction inventory mismatch')
    tokenizer = CTokenizer(a.tok_probe, a.gguf)
    try:
        stops = tokenizer_stop_ids(tokenizer)
        prefixes = [encode_generation_input(r, tokenizer).prompt_token_ids for r in records]
        for r, row, prefix in zip(records, rows, prefixes):
            if row['input_sha256'] != digest(r) or row['prompt_token_ids'] != list(prefix):
                raise ValueError('natural input/prompt changed')
            ids = row['generated_token_ids']; stopped = bool(ids and ids[-1] in stops)
            if not 1 <= len(ids) <= 64 or any(t in stops for t in ids[:-1]): raise ValueError('invalid termination')
            if not stopped and len(ids) != 64: raise ValueError('unregistered early stop')
            visible = ids[:-1] if stopped else ids
            raw = b''.join(tokenizer.decode_pieces(visible))
            try: text = raw.decode('utf-8'); valid = True
            except UnicodeDecodeError: text = raw.decode('utf-8', errors='replace'); valid = False
            if (row['visible_token_ids'] != visible or row['raw_text_hex'] != raw.hex() or row['text'] != text or
                    row['utf8_complete'] != valid or row['truncated'] != (not stopped) or
                    row['finish_reason'] != ('stop_token' if stopped else 'max_new_tokens') or row['forward_calls'] != len(ids)):
                raise ValueError('response bytes/termination mismatch')
        if len(report['raw_steps']) != max(len(r['generated_token_ids']) for r in rows): raise ValueError('missing/extra raw steps')
        positions = 0
        for step, info in enumerate(report['raw_steps']):
            active = [i for i, row in enumerate(rows) if len(row['generated_token_ids']) > step]
            if info['step'] != step or info['active_case_ids'] != [records[i]['id'] for i in active]:
                raise ValueError('C row order mismatch')
            directory = root / f'step-{step:03d}'
            for name, expected in info['sha256'].items():
                if name not in ('input.bin', 'output.bin', 'native.log') or sha_file(directory / name) != expected:
                    raise ValueError('raw artifact changed')
            if set(info['sha256']) != {'input.bin', 'output.bin', 'native.log'}: raise ValueError('raw inventory changed')
            current = [prefixes[i] + tuple(rows[i]['generated_token_ids'][:step]) for i in active]
            verify_input_bytes(directory / 'input.bin', current)
            _, logits = read_prefix_batch(directory / 'output.bin', current)
            log = (directory / 'native.log').read_text().splitlines()
            if ('[bitnet] cpu tier: arm_neon' not in log or
                    f'BNP_TRACE rows={len(active)} fresh_contexts={len(active)}' not in log):
                raise ValueError('fresh context trace missing')
            for j, i in enumerate(active):
                if int(logits[j].argmax()) != rows[i]['generated_token_ids'][step]: raise ValueError('prediction not C argmax')
            positions += len(active)
        normalized = predictions(report, 'cg003-baseline')
        packets = {p['id']: p for runtime, prediction in zip(records, normalized) for p in [blind_packet(runtime, prediction)]}
        result = {'format': 'cg003-baseline-audit-v1', 'baseline_sha256': a.baseline_sha256,
            'cases': len(rows), 'positions': positions, 'raw_steps': len(report['raw_steps']),
            'stop_token_cases': sum(not r['truncated'] for r in rows),
            'utf8_complete_cases': sum(r['utf8_complete'] for r in rows), 'actual_prefixes_verified': True,
            'raw_artifacts_verified': True, 'cached_tokens': 0, 'reused_tokens': 0,
            'memory_accuracy_measured': False, 'semantic_review_complete': False, 'optimizer_steps': 0,
            'auditor_sha256': sha_file(__file__)}
        out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
        with (out / 'audit.json').open('x') as f: json.dump(result, f, indent=2); f.write('\n')
        with (out / 'blind.json').open('x') as f:
            json.dump({'format': 'neural-memory-blind-review-v1', 'packets': [packets[k] for k in sorted(packets)]},
                      f, ensure_ascii=False, indent=2); f.write('\n')
        print(json.dumps(result), flush=True)
    finally:
        tokenizer._proc.terminate(); tokenizer._proc.wait()
        tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('baseline', 'baseline-sha256', 'panel-audit', 'corpus', 'gguf', 'tok-probe', 'output'):
        p.add_argument('--' + n, required=True)
    run(p.parse_args())
