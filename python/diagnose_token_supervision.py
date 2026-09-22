"""DG-011 full-training-corpus coverage and fixed six-case gradient audit."""
import argparse
import json
from pathlib import Path
import torch

from c_tokenizer import CTokenizer
from diagnose_role_content_path import token_span
from episode_memory_inputs import encoder_texts
from eval_role_checkpoint import load_checkpoint
from memory_token_read import TokenReadPrototype
from native_memory_encoder import read_encoded, sha_file, text_key
from native_token_payload import payload_mask, native_token_input, input_binding
from prepare_availability_curriculum import materialize
from prepare_neural_memory_protocol import digest
from review_neural_memory import read_rows
from token_memory_composition import ComposedTokenMemory
from token_memory_supervision import TokenSupervision, make_position_targets, supervised_token_loss
from train_availability_memory import TrainingPackage
from train_role_separated_memory import FEATURE_DIGEST


def run(a):
    config = json.loads(Path(a.experiment).read_text())
    if config['id'] != 'DG-011' or config['optimizer_steps'] != 0: raise ValueError('wrong registration')
    root = Path(a.output); root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(1011)
    package = TrainingPackage(a.features, FEATURE_DIGEST)
    manifest = materialize(a.config, a.corpus, verify=True)
    if digest(manifest) != package.manifest['corpus_manifest_sha256']: raise ValueError('wrong corpus')
    roles, checkpoint = load_checkpoint(a.checkpoint)
    if sha_file(a.gguf) != checkpoint['backbone_sha256']: raise ValueError('wrong GGUF')
    if {v for k, v in package.manifest['binaries_sha256'].items() if Path(k).name == 'tok_probe'} != {sha_file(a.tok_probe)}:
        raise ValueError('wrong tokenizer')
    inputs = {r['id']: r for r in read_rows(Path(a.corpus) / 'train.inputs.jsonl')}
    labels = {r['id']: r for r in read_rows(Path(a.corpus) / 'train.labels.jsonl')}
    meta = {r['id']: r for r in read_rows(Path(a.corpus) / 'train.index.jsonl')}
    records = [r for r in package.records if r['split'] == 'train']
    if len(records) != 512: raise ValueError('coverage denominator changed')
    tokenizer = CTokenizer(a.tok_probe, a.gguf)
    try:
        rows = []; targets_by_id = {}
        for r in records:
            runtime, label, info = inputs[r['id']], labels[r['id']], meta[r['id']]
            if r['input_sha256'] != digest(runtime) or label['input_sha256'] != digest(runtime): raise ValueError('input binding mismatch')
            _, sources = encoder_texts(runtime)
            ids, allowed, pieces = [], [], []
            if sources:
                encoded = package.source.rows[text_key(sources[0])]
                if tokenizer.encode(sources[0], True) != encoded.token_ids.tolist(): raise ValueError('source token identity mismatch')
                ids = encoded.token_ids[1:].tolist(); pieces = tokenizer.decode_pieces(ids)
                allowed = payload_mask(runtime['episodes'][0], sources[0], pieces)
                allowed = [ok and i not in (tokenizer.bos(), tokenizer.eos()) for i, ok in zip(ids, allowed)]
            supported = r['targets']['state_index'] == 1
            offsets, source_positions = [], []
            if supported:
                value = label['required_claims'][0]['value']
                reply = r['targets']
                offsets = token_span(reply['response'], tokenizer.decode_pieces(reply['completion_token_ids'][:-1]), value)
                source_positions = token_span(sources[0], pieces, value)
            normal = r['targets']['state_index'] == 0
            factors = (None, None) if normal else (
                any(f['subject'] == info['family_facts'][0]['subject'] for f in info['source_facts']),
                any(f['relation'] == info['family_facts'][0]['relation'] for f in info['source_facts']))
            position_targets = make_position_targets(r['targets']['completion_token_ids'], r['targets']['state_index'],
                offsets, ids, allowed, source_positions)
            targets_by_id[r['id']] = (factors, position_targets)
            rows.append({'id': r['id'], 'scenario': r['scenario'], 'language': r['language'], 'factors': factors,
                'reply_tokens': len(position_targets), 'value_tokens': len(offsets),
                'copyable_value_tokens': sum(position_targets[i] != (0,) for i in offsets),
                'uncopyable_value_offsets': [i for i in offsets if position_targets[i] == (0,)],
                'uncopyable_details': [{'offset': i, 'token_id': r['targets']['completion_token_ids'][i],
                    'target_bytes_hex': tokenizer.decode_pieces([r['targets']['completion_token_ids'][i]])[0].hex(),
                    'source_value_tokens': [{'id': ids[j], 'bytes_hex': pieces[j].hex(), 'eligible': allowed[j]} for j in source_positions],
                    'reason': 'token_id_boundary_mismatch' if r['targets']['completion_token_ids'][i] not in [ids[j] for j in source_positions]
                              else 'structural_payload_exclusion'} for i in offsets if position_targets[i] == (0,)],
                'all_reply_positions_retained': True})
        prior = json.loads((Path(a.native_report) / 'report.json').read_text())
        retained = Path(__file__).resolve().parents[1] / 'training/memory/neural-system/checks/DG-010-result.json'
        if prior != json.loads(retained.read_text()) or prior['encoder_identity'] != package.manifest['encoder_identity']:
            raise ValueError('unverified DG-010 artifact')
        native = Path(a.native_report) / 'fresh-c-inputs'
        for name, want in prior['fresh_c_manifest']['file_sha256'].items():
            if sha_file(native / name) != want: raise ValueError('C evidence corrupted')
        encodings = dict(zip(prior['fresh_c_manifest']['input_sha256'], read_encoded(native / 'batch-0.bin', 8)))
        binding = input_binding(prior['encoder_identity'], sha_file(a.tok_probe))
        model = ComposedTokenMemory(TokenReadPrototype(1024, 73448, binding, (tokenizer.bos(), tokenizer.eos())), roles)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        panel_ids = {r['id'] for r in prior['rows']}; numeric = []
        for r in records:
            if r['id'] not in panel_ids: continue
            runtime = inputs[r['id']]; query, sources = encoder_texts(runtime)
            token_inputs = native_token_input(encodings[text_key(query)], encodings[text_key(sources[0])], runtime['episodes'][0],
                sources[0], tokenizer, binding)
            features, reply = package.sample(r, 'cpu')
            state = model.prefill(token_inputs)
            output = model.branches(features.hidden, features.base_logits, package.head, package.scale, state)
            supervision = TokenSupervision(reply, *targets_by_id[r['id']])
            loss = supervised_token_loss(output, supervision)
            grads = torch.autograd.grad(loss['weighted_total'], tuple(model.parameters()), allow_unused=True)
            if any(g is not None and not torch.isfinite(g).all() for g in grads): raise ValueError('nonfinite gradient')
            off = model.read(features.hidden, features.base_logits, package.head, package.scale, state, enabled=False)
            if off is not features.base_logits: raise ValueError('disabled branch changed base')
            normerr = float((output.supported.detach().double().exp().sum(-1) - 1).abs().max())
            if normerr > 1e-6: raise ValueError('composed distribution not normalized')
            grad_norms = {name: float(g.norm()) if g is not None else None for (name, _), g in zip(model.named_parameters(), grads)}
            numeric.append({'id': r['id'], 'scenario': r['scenario'], 'language': r['language'],
                'predicted_route': int(state.state_logits.argmax()), 'target_state': reply.state_index,
                'losses': {k: float(v.detach()) if isinstance(v, torch.Tensor) else v for k, v in loss.items()},
                'normalization_max_error': normerr, 'disabled_exact_identity': True,
                'finite_gradients': True, 'gradient_norms': grad_norms})
        if len(numeric) != 6 or any(not torch.equal(p, before[n]) or p.grad is not None for n, p in model.named_parameters()):
            raise ValueError('panel/parameters changed')
        report = {'experiment': config, 'coverage': rows, 'numeric_panel': numeric,
            'summary': {'records': len(rows), 'supported_records': sum(r['value_tokens'] > 0 for r in rows),
                'reply_tokens_retained': sum(r['reply_tokens'] for r in rows),
                'value_tokens': sum(r['value_tokens'] for r in rows),
                'copyable_value_tokens': sum(r['copyable_value_tokens'] for r in rows),
                'uncopyable_value_tokens': sum(len(r['uncopyable_value_offsets']) for r in rows)},
            'optimizer_steps': 0, 'parameters_unchanged': True, 'new_free_generated_answers': 0,
            'training_approved': False, 'deployment_approved': False, 'memory_accuracy_measured': False,
            'feature_package_digest': FEATURE_DIGEST, 'native_input_report_sha256': sha_file(Path(a.native_report) / 'report.json'),
            'artifact_sha256': {k: sha_file(getattr(a, k)) for k in ('experiment', 'checkpoint', 'gguf', 'tok_probe')},
            'source_sha256': {n: sha_file(Path(__file__).parent / n) for n in ('token_memory_composition.py', 'token_memory_supervision.py', 'diagnose_token_supervision.py')}}
        with (root / 'report.json').open('x') as f: json.dump(report, f, ensure_ascii=False, indent=2); f.write('\n')
        print(json.dumps(report['summary']), flush=True)
    finally:
        tokenizer._proc.terminate(); tokenizer._proc.wait(); tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('experiment', 'features', 'config', 'corpus', 'checkpoint', 'gguf', 'tok-probe', 'native-report', 'output'):
        p.add_argument('--' + key, required=True)
    run(p.parse_args())


if __name__ == '__main__': main()
