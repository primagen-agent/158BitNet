"""CG-003 prefix-chain and nonzero route fixtures; NOT semantic acceptance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import numpy as np
import torch

from c_tokenizer import CTokenizer
from joint_memory_model import create_joint_model
from native_joint_generation import NativeJointBackend
from native_continuous_generation import tokenizer_stop_ids
from native_memory_encoder import sha_file
from native_token_payload import input_binding
from neural_memory_generation import encode_generation_input, greedy_generate
from prepare_neural_memory_protocol import digest
from review_neural_memory import read_rows


def fixture_parameters(model, route):
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(4013)
        model.roles.content.output.weight.normal_(std=.001)
        model.roles.uncertainty_output.weight.normal_(std=.001)
        model.reader.state_head.weight.zero_(); model.reader.state_head.bias.fill_(-20.)
        model.reader.state_head.bias[route] = 20.
        for head in model.factor_heads: head.weight.zero_(); head.bias.fill_(20.)


def run(a):
    root = Path(a.output); root.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(a.experiment).read_text())
    if config['id'] != 'CG-003-platform-preflight' or config['seed'] != 1013 or config['optimizer_steps'] != 0:
        raise ValueError('wrong registration')
    m = json.loads((Path(a.features) / 'manifest.json').read_text())
    if digest(m) != a.package_digest or m['scope'] != 'full': raise ValueError('wrong package')
    for name, field in (('output-head.npy', 'output_head_sha256'), ('records.json', 'records_sha256')):
        if sha_file(Path(a.features) / name) != m[field]: raise ValueError('artifact changed')
    head = torch.from_numpy(np.load(Path(a.features) / 'output-head.npy', allow_pickle=False))
    records = json.loads((Path(a.features) / 'records.json').read_text())
    world = min(r['world_id'] for r in records if r['split'] == 'train')
    selected = {r['language']: r for r in records if r['split'] == 'train' and r['world_id'] == world and r['relation'] == 'home_city' and r['scenario'] == 'correct'}
    inputs = {r['id']: r for r in read_rows(Path(a.corpus) / 'train.inputs.jsonl')}
    for r in selected.values():
        if digest(inputs[r['id']]) != r['input_sha256']: raise ValueError('natural input mismatch')
    binding = input_binding(m['encoder_identity'], sha_file(a.tok_probe))
    torch.set_num_threads(4); lock = threading.Lock()
    cases = [(arm, lang, None, True, 4) for arm in ('joint_aux', 'joint_product') for lang in ('en', 'zh')]
    cases += [(arm, 'en', route, enabled, 1) for arm in ('joint_aux', 'joint_product') for route, enabled in ((0, True), (1, True), (2, True), (1, False))]
    def execute(item):
        i, (arm, lang, route, enabled, steps) = item
        tokenizer = CTokenizer(a.tok_probe, a.gguf)
        try:
            with lock:
                model = create_joint_model(binding, m['blocked_ids'], arm).eval()
                if route is not None: fixture_parameters(model, route)
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            runtime = inputs[selected[lang]['id']]
            backend = NativeJointBackend(runtime=runtime, tokenizer=tokenizer, tok_probe=a.tok_probe, gguf=a.gguf,
                probe=a.probe, reference_probe=a.reference_probe, encoder_probe=a.encoder_probe,
                model=model, head=head, scale=m['logit_scale'], output=root/f'case-{i:02d}', enabled=enabled)
            if route is not None and int(backend.state.core.state_logits.argmax()) != route: raise ValueError('fixture route not realized')
            output = greedy_generate(encode_generation_input(runtime, tokenizer), tokenizer, backend,
                stop_ids=tokenizer_stop_ids(tokenizer), max_new_tokens=steps, context_capacity=128)
            audit = backend.finish()
            if any(not torch.equal(p, before[n]) for n, p in model.named_parameters()): raise ValueError('weights changed')
            row = {'arm': arm, 'language': lang, 'numeric_fixture': route is not None, 'fixture_route': route,
                'enabled': enabled, 'generation': output, 'backend': f'case-{i:02d}/backend.json',
                'actual_routes': [s['selected_route'] for s in audit['steps']],
                'copy_positions': sum(s['copy_mass'] > 0 for s in audit['steps']),
                'max_error': max(s['torch_composition_max_error'] for s in audit['steps']),
                'all_equal_base': all(s['output_equals_base'] for s in audit['steps'])}
            if route in (1, 2) and enabled and row['all_equal_base']: raise ValueError('nonzero fixture did not affect output')
            if route == 1 and enabled and row['copy_positions'] != len(audit['steps']): raise ValueError('fixture copy inactive')
            if (not enabled or route == 0) and not row['all_equal_base']: raise ValueError('bypass changed output')
            print(json.dumps({'case': i+1, 'fixture': route is not None, 'arm': arm, 'routes': row['actual_routes']}), flush=True)
            return row
        finally:
            tokenizer._proc.terminate(); tokenizer._proc.wait(); tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()
    with ThreadPoolExecutor(max_workers=2) as pool: rows = list(pool.map(execute, enumerate(cases)))
    report = {'experiment': config, 'package_digest': a.package_digest, 'world': world, 'rows': rows,
        'optimizer_steps': 0, 'candidate_parameters_unchanged': True, 'memory_accuracy_measured': False,
        'training_approved': False, 'prefix_bank_used': False, 'semantic_review_complete': False,
        'source_sha256': {n: sha_file(Path(__file__).parent/n) for n in ('qualify_joint_native.py','native_joint_generation.py','joint_memory_model.py')},
        'artifact_sha256': {n: sha_file(getattr(a,n)) for n in ('experiment','gguf','probe','reference_probe','encoder_probe','tok_probe')}}
    with (root/'report.json').open('x') as f: json.dump(report,f,ensure_ascii=False,indent=2); f.write('\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('experiment','features','package-digest','corpus','gguf','probe','reference-probe','encoder-probe','tok-probe','output'):
        p.add_argument('--'+n,required=True)
    run(p.parse_args())
