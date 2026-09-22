"""DG-012 fixed compound-event counterexamples with actual C free prefixes."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from c_tokenizer import CTokenizer
from eval_role_checkpoint import load_checkpoint
from memory_token_read import TokenReadPrototype
from native_continuous_generation import tokenizer_stop_ids
from native_memory_encoder import sha_file, NativeMemoryEncoder
from native_token_generation import NativeTokenBackend
from native_token_payload import input_binding
from neural_memory_generation import encode_generation_input, greedy_generate
from prepare_neural_memory_protocol import digest, fact, message
from token_memory_composition import ComposedTokenMemory
from train_role_separated_memory import FEATURE_DIGEST


def joint_annotation(facts, subject, relation):
    """Audit labels ONLY; never consumed by NativeTokenBackend or neural read."""
    return {'subject_present': any(f['subject'] == subject for f in facts),
            'relation_present': any(f['relation'] == relation for f in facts),
            'joint_current_actual': any(f['subject'] == subject and f['relation'] == relation and
                                        f['time'] == 'current' and f['status'] == 'actual' for f in facts)}


def panel():
    rows = []
    for language in ('en', 'zh'):
        query = 'Where does Estelle live now?' if language == 'en' else 'Estelle现在住在哪里？'
        for crossed in (False, True):
            home, likes = ('Fabian', 'Estelle') if crossed else ('Estelle', 'Fabian')
            source = (f'{home} currently lives in Graz. {likes} likes the architecture of Graz.' if language == 'en' else
                      f'{home}现在住在Graz。{likes}喜欢Graz的建筑。')
            key = language + ('-crossed' if crossed else '-correct')
            runtime = {'id': key, 'context': [message(query)], 'episodes': [message(source)]}
            annotation = joint_annotation([fact(home, 'home_city', 'Graz'), fact(likes, 'likes_architecture_of', 'Graz')],
                                           'Estelle', 'home_city')
            rows.append({'runtime': runtime, 'annotation': annotation, 'language': language, 'enabled': True})
        a, b = rows[-2:]
        if a['runtime']['context'] != b['runtime']['context'] or Counter(a['runtime']['episodes'][0]['text']) != Counter(b['runtime']['episodes'][0]['text']):
            raise ValueError('counterexample changed query or source character inventory')
    return rows + [{**rows[i], 'enabled': False} for i in (0, 2)]


def run(a):
    config = json.loads(Path(a.experiment).read_text())
    if (config['id'], config['seed'], config['max_new_tokens'], config['optimizer_steps']) != ('DG-012', 1011, 24, 0):
        raise ValueError('unregistered run')
    root = Path(a.output); root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(config['seed'])
    roles, checkpoint = load_checkpoint(a.checkpoint)
    # Deliberately load ONLY the verified head/manifest, never a prefix/answer bank.
    folder = Path(a.features); manifest = json.loads((folder / 'manifest.json').read_text())
    if digest(manifest) != FEATURE_DIGEST or sha_file(folder / 'output-head.npy') != manifest['output_head_sha256']:
        raise ValueError('frozen head identity changed')
    head = torch.from_numpy(np.load(folder / 'output-head.npy', allow_pickle=False))
    encoder = NativeMemoryEncoder(a.gguf, a.encoder_probe, expected_encoder_id=checkpoint['encoder_identity']['encoder_id'])
    if {v for k, v in manifest['binaries_sha256'].items() if Path(k).name == 'tok_probe'} != {sha_file(a.tok_probe)}:
        raise ValueError('tokenizer changed')
    binding = input_binding(encoder.identity, sha_file(a.tok_probe))
    tokenizer = CTokenizer(a.tok_probe, a.gguf)
    try: blocked = (tokenizer.bos(), tokenizer.eos())
    finally:
        tokenizer._proc.terminate(); tokenizer._proc.wait(); tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()
    model = ComposedTokenMemory(TokenReadPrototype(1024, 73448, binding, blocked), roles).eval()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    def generate(item):
        i, case = item; tokenizer = CTokenizer(a.tok_probe, a.gguf)
        try:
            backend = NativeTokenBackend(runtime=case['runtime'], tokenizer=tokenizer, tok_probe=a.tok_probe,
                gguf=a.gguf, probe=a.probe, reference_probe=a.reference_probe, encoder_probe=a.encoder_probe,
                model=model, head=head, scale=manifest['logit_scale'], output=root / f'case-{i:02d}', enabled=case['enabled'])
            request = encode_generation_input(case['runtime'], tokenizer)
            output = greedy_generate(request, tokenizer, backend, stop_ids=tokenizer_stop_ids(tokenizer),
                max_new_tokens=config['max_new_tokens'], context_capacity=config['context_capacity'])
            output['memory_enabled'] = case['enabled']
            audit = backend.finish()
            if not audit['steps'] or len({s['selected_route'] for s in audit['steps']}) != 1:
                raise ValueError('predicted route not fixed')
            output['backend_cache_policy_verified'] = True
            result = {**case, **output, 'backend_evidence': f'case-{i:02d}/backend.json',
                'predicted_route': audit['steps'][0]['selected_route'],
                'state_probabilities': audit['steps'][0]['state_probabilities'],
                'steps_with_copy': sum(s['copy_mass'] > 0 for s in audit['steps']),
                'branch_only_qualifications': sum(s['branch_only_qualification'] is not None for s in audit['steps']),
                'max_torch_composition_error': max(s['torch_composition_max_error'] for s in audit['steps']),
                'all_steps_equal_base': all(s['output_equals_base'] for s in audit['steps']),
                'raw_backend_sha256': sha_file(root / f'case-{i:02d}/backend.json')}
            with (root / f'case-{i:02d}/generation.json').open('x') as f: json.dump(result, f, ensure_ascii=False, indent=2)
            print(json.dumps({'completed_case': i + 1, 'route': result['predicted_route'], 'enabled': case['enabled'],
                'text': result['text'], 'truncated': result['truncated']}, ensure_ascii=False), flush=True)
            return result
        finally:
            tokenizer._proc.terminate(); tokenizer._proc.wait(); tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(generate, enumerate(panel())))
    if any(not torch.equal(p, before[n]) or p.grad is not None for n, p in model.named_parameters()):
        raise ValueError('weights changed')
    report = {'experiment': config, 'results': results, 'optimizer_steps': 0, 'parameters_unchanged': True,
        'prefix_bank_used': False, 'gold_answer_prefix_used': False, 'training_approved': False, 'deployment_approved': False,
        'semantic_review_complete': False, 'memory_accuracy_measured': False,
        'neural_fusion_implementation': 'python', 'cross_step_kv_reuse': False, 'internal_prefill_kv_buffers': True,
        'source_sha256': {n: sha_file(Path(__file__).parent / n) for n in ('diagnose_joint_native_generation.py',
            'native_token_generation.py', 'token_memory_composition.py', 'memory_token_read.py', 'native_token_payload.py')},
        'artifact_sha256': {n: sha_file(getattr(a, n)) for n in ('experiment', 'checkpoint', 'gguf', 'probe', 'reference_probe', 'encoder_probe', 'tok_probe')}}
    with (root / 'report.json').open('x') as f: json.dump(report, f, ensure_ascii=False, indent=2); f.write('\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('experiment', 'features', 'checkpoint', 'gguf', 'probe', 'reference-probe', 'encoder-probe', 'tok-probe', 'output'):
        p.add_argument('--' + n, required=True)
    run(p.parse_args())


if __name__ == '__main__': main()
