#!/usr/bin/env python3
"""Independent no-memory/no-KV-reuse tokenizer and full-logit differential gate.

The reference probe is optional and links only to an installed llama.cpp. Run:
  python3 tests/test_llama_reference.py --native build/inference_probe \
    --reference build/llama_reference_probe --model models/bitcpm4-0.5b-tq2_0.gguf \
    --output build/reference-gate
Repeat --model for additional backbones. No output or threshold is selected
based on expected natural-language answers; every forward uses a fresh context.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

TOKENIZER_CASES = [
    "", "hello world", "  a  b\n\tend ", "你好，世界！🙂 café e\u0301",
    "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|><|im_end|>", "hello<|im_end|>world", "<s>test</s><|not_a_token|>\x01",
]
QUERIES = [
    'Reply with exactly this word and nothing else: READY',
    '小林养了一只猫，没有养狗。小林养狗了吗？只回答“是”或“否”。',
    'Mira lives in Oslo. Jonas lives in Lima. Where does Mira live? Answer with only the city.',
]

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def run(args):
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    native, reference = Path(args.native).resolve(), Path(args.reference).resolve()
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('BITNET_', 'LLAMA_', 'GGML_'))}
    environment['BITNET_NUM_THREADS'] = '4'
    report = {'native_sha256': digest(native), 'reference_sha256': digest(reference),
              'max_absolute_logit_error_limit': 1e-4, 'rms_logit_error_limit': 1e-5,
              'fresh_context_per_forward': True, 'models': []}

    def probe(executable, model, source, destination, mode):
        with destination.with_suffix('.log').open('w') as log:
            subprocess.run([str(executable), str(model), str(source), str(destination), mode],
                           check=True, stdout=log, stderr=log, env=environment, timeout=300)
        return [int(x) for x in destination.with_suffix('.ids').read_text().split()]

    for model_arg in args.model:
        model = Path(model_arg).resolve(); root = output / model.stem; root.mkdir()
        row = {'model': str(model), 'sha256': digest(model), 'tokenizer': [], 'forward': []}
        report['models'].append(row)
        print(json.dumps({'model': model.name, 'stage': 'start'}), flush=True)
        try:
            for i, text in enumerate(TOKENIZER_CASES):
                source = root / f'tokenizer-{i}.txt'; source.write_text(text)
                a = probe(native, model, source, root / f'tokenizer-{i}-native', 'tokenize')
                b = probe(reference, model, source, root / f'tokenizer-{i}-reference', 'tokenize')
                row['tokenizer'].append({'id': i, 'text': text, 'ids_equal': a == b, 'native': a, 'reference': b})
            for i, question in enumerate(QUERIES):
                text = '<|im_start|>user\n' + question + '<|im_end|>\n<|im_start|>assistant\n'
                source = root / f'forward-{i}.txt'; source.write_text(text)
                a = probe(native, model, source, root / f'forward-{i}-native', 'text')
                b = probe(reference, model, source, root / f'forward-{i}-reference', 'text')
                compare(root, f'forward-{i}', a, b, row)
            # Long-position numerical stress tests, using actual prose token IDs.
            source = root / 'prose.txt'; source.write_text('A careful test compares identical token sequences and checks every output score. ')
            prose = probe(reference, model, source, root / 'prose-reference', 'tokenize')
            for length in (1, 128, 513):
                ids = [prose[0]] + (prose[1:] * (length + 1))[:length - 1]
                name = f'length-{length}'; source = root / (name + '.tokens')
                source.write_text(' '.join(map(str, ids)))
                a = probe(native, model, source, root / (name + '-native'), 'tokens')
                b = probe(reference, model, source, root / (name + '-reference'), 'tokens')
                compare(root, name, a, b, row)
            row['passed'] = all(x['ids_equal'] for x in row['tokenizer']) and all(x['passed'] for x in row['forward'])
            print(json.dumps({'model': model.name, 'passed': row['passed'], 'tokenizer_cases': len(row['tokenizer']),
                              'forward_cases': len(row['forward'])}), flush=True)
        finally:
            report['passed'] = len(report['models']) == len(args.model) and all(x.get('passed', False) for x in report['models'])
            (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return 0 if report['passed'] else 1

def compare(root, name, ids_a, ids_b, row):
    a = np.fromfile(root / (name + '-native.logits'), dtype='<f4')
    b = np.fromfile(root / (name + '-reference.logits'), dtype='<f4')
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid full-logit output')
    error = a.astype(np.float64) - b.astype(np.float64)
    maximum, rms = float(np.abs(error).max()), float(np.sqrt(np.mean(error**2)))
    result = {'id': name, 'tokens': len(ids_a), 'ids_equal': ids_a == ids_b,
              'vocab': int(a.size), 'max_absolute_error': maximum, 'rms_error': rms,
              'native_argmax': int(a.argmax()), 'reference_argmax': int(b.argmax()),
              'passed': bool(ids_a == ids_b and maximum <= 1e-4 and rms <= 1e-5 and a.argmax() == b.argmax())}
    row['forward'].append(result); print(json.dumps(result), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--model', action='append', required=True)
    parser.add_argument('--output', required=True)
    raise SystemExit(run(parser.parse_args()))
