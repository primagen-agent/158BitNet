"""DG-010 fresh C-input interface check. No optimizer or generated replies."""
import argparse
import json
from pathlib import Path

import torch

from c_tokenizer import CTokenizer
from episode_memory_inputs import encoder_texts
from memory_token_read import TokenReadPrototype
from native_memory_encoder import NativeMemoryEncoder, HIDDEN, VOCAB, sha_file
from native_token_payload import native_token_input, input_binding
from prepare_availability_curriculum import materialize
from prepare_neural_memory_protocol import digest
from review_neural_memory import read_rows
from train_availability_memory import TrainingPackage
from train_role_separated_memory import FEATURE_DIGEST


def run(a):
    config = json.loads(Path(a.experiment).read_text())
    if (config['id'], config['optimizer_steps'], config['seed'], config['training_approved']) != ('DG-010', 0, 1010, False):
        raise ValueError('unregistered interface run')
    root = Path(a.output); root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(config['seed'])
    package = TrainingPackage(a.features, FEATURE_DIGEST)
    manifest = materialize(a.config, a.corpus, verify=True)
    if digest(manifest) != package.manifest['corpus_manifest_sha256']: raise ValueError('corpus binding changed')
    records = {r['id']: r for r in package.records if r['split'] == 'train'}
    world = min(r['world_id'] for r in records.values())
    panel = sorted((r for r in records.values() if r['world_id'] == world and r['scenario'] in
                    ('original', 'swapped_values', 'wrong_subject')), key=lambda r: (r['language'], r['scenario']))
    if len(panel) != 6: raise ValueError('panel denominator changed')
    inputs = {r['id']: r for r in read_rows(Path(a.corpus) / 'train.inputs.jsonl')}
    texts = []; queries = {}
    for r in panel:
        runtime = inputs[r['id']]
        if r['input_sha256'] != digest(runtime): raise ValueError('natural input changed')
        query, source = encoder_texts(runtime)
        if len(source) != 1 or tuple(r['source_texts']) != source: raise ValueError('supplied source mismatch')
        if query != queries.setdefault(r['language'], query): raise ValueError('source change leaked into query')
        for text in (query, *source):
            if text not in texts: texts.append(text)
    if len(texts) != 8: raise ValueError('unique query/source inventory changed')
    expected_tokenizers = {v for k, v in package.manifest['binaries_sha256'].items() if Path(k).name == 'tok_probe'}
    if expected_tokenizers != {sha_file(a.tok_probe)}: raise ValueError('wrong tokenizer')
    encoder = NativeMemoryEncoder(a.gguf, a.encoder_probe,
        expected_encoder_id=package.manifest['encoder_identity']['encoder_id'])
    encoded = encoder.encode(texts, root / 'fresh-c-inputs')
    by_text = dict(zip(texts, encoded))
    binding = input_binding(encoder.identity, sha_file(a.tok_probe))
    tokenizer = CTokenizer(a.tok_probe, a.gguf)
    try:
        for text, encoded_text in by_text.items():
            if tokenizer.encode(text, True) != encoded_text.token_ids.tolist(): raise ValueError('C token IDs changed')
            raw = b''.join(tokenizer.decode_pieces(encoded_text.token_ids[1:].tolist()))
            if raw not in (text.encode(), b' ' + text.encode()): raise ValueError('C token bytes changed')
        model = TokenReadPrototype(HIDDEN, VOCAB, binding, (tokenizer.bos(), tokenizer.eos())).eval()
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        rows = []; empty_checks = []
        for r in panel:
            runtime = inputs[r['id']]; query, sources = encoder_texts(runtime)
            features = native_token_input(by_text[query], by_text[sources[0]], runtime['episodes'][0], sources[0], tokenizer, binding)
            # Frozen decoder prefix from the independently verified true-C bank;
            # new source/query token encoding above is NOT substituted for it.
            hidden, base = package.prefix(tuple(r['prompt_ids'])); hidden = hidden[None]; base = base[None]
            state = model.prefill(features)
            proposal = model.proposal(hidden, base, state)
            actual = model.read(hidden, base, state)
            off = model.read(hidden, base, state, enabled=False)
            normalized = bool(torch.allclose(proposal.logits.double().exp().sum(-1), torch.ones(1, dtype=torch.float64), atol=1e-6, rtol=1e-6))
            if not normalized or off.logits is not base: raise ValueError('normalization/bypass failed')
            # A numerical probe, not an annotated answer or a training objective.
            loss = -proposal.logits[0, int(base.argmax())] + state.state_logits.square().mean()
            grads = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
            if any(g is None or not torch.isfinite(g).all() for g in grads): raise ValueError('missing/nonfinite interface gradient')
            rows.append({'id': r['id'], 'language': r['language'], 'scenario': r['scenario'],
                'query_tokens': len(features.query), 'source_tokens': len(features.source),
                'copy_eligible_tokens': int(features.copy_allowed.sum()),
                'source_token_ids': features.token_ids.tolist(), 'structural_copy_mask': features.copy_allowed.tolist(),
                'state_probabilities': state.state_logits.detach().softmax(-1).tolist(),
                'predicted_route': actual.route, 'proposal_copy_mass': float(proposal.copy_mass.detach()[0, 0]),
                'actual_copy_mass': float(actual.copy_mass.detach()[0, 0]),
                'normalized_proposal': normalized, 'disabled_exact_identity': True,
                'all_parameter_gradients_finite': True,
                'nonzero_gradient_parameters': sum(float(g.norm()) > 0 for g in grads),
                'parameter_tensors': len(grads), 'semantic_accuracy_measured': False})
            if r['scenario'] == 'original':
                empty = native_token_input(by_text[query], None, None, None, tokenizer, binding)
                empty_state = model.prefill(empty); output = model.read(hidden, base, empty_state)
                if output.logits is not base or output.route == 1 or float(output.copy_mass.sum()) != 0:
                    raise ValueError('empty memory must not copy')
                empty_checks.append({'language': r['language'], 'copy_mass': 0, 'exact_identity': True})
            print(json.dumps({'completed': len(rows), 'total': 6, 'route': actual.route,
                              'eligible': int(features.copy_allowed.sum())}), flush=True)
        if any(not torch.equal(p, before[n]) or p.grad is not None for n, p in model.named_parameters()):
            raise ValueError('prototype weights/gradients changed')
        report = {'experiment': config, 'world': world, 'rows': rows, 'empty_checks': empty_checks,
            'fresh_c_encoded_texts': 8, 'exact_c_token_byte_checks': 8, 'query_texts': 2,
            'query_source_separation_verified': True, 'parameters_unchanged': True, 'optimizer_steps': 0,
            'new_free_generated_answers': 0, 'memory_accuracy_measured': False,
            'training_approved': False, 'deployment_approved': False, 'untrained': True,
            'cross_input_kv_reuse': False, 'internal_prefill_kv_buffers': True,
            'decoder_prefix_source': 'existing verified frozen C prefix bank, not a fresh decoder evaluation',
            'source_episode_selection': 'supplied_not_learned', 'binding': binding,
            'encoder_identity': encoder.identity, 'feature_package_digest': FEATURE_DIGEST,
            'artifact_sha256': {k: sha_file(getattr(a, k)) for k in ('experiment', 'gguf', 'tok_probe', 'encoder_probe')},
            'source_sha256': {p: sha_file(Path(__file__).parent / p) for p in (
                'memory_token_read.py', 'native_token_payload.py', 'diagnose_token_read_interface.py', 'native_memory_encoder.py')},
            'fresh_c_manifest': json.loads((root / 'fresh-c-inputs/manifest.json').read_text())}
        with (root / 'report.json').open('x') as f: json.dump(report, f, ensure_ascii=False, indent=2); f.write('\n')
    finally:
        tokenizer._proc.terminate(); tokenizer._proc.wait(); tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('experiment', 'features', 'config', 'corpus', 'gguf', 'tok-probe', 'encoder-probe', 'output'):
        p.add_argument('--' + key, required=True)
    run(p.parse_args())


if __name__ == '__main__': main()
