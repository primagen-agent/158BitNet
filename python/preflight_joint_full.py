"""Expanded CPU preflight on real C features. No updates, no accuracy claim."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from continuous_memory import continuous_logits
from joint_binding_schedule import KINDS
from joint_memory_model import create_joint_model
from joint_training_data import JointTrainingPackage
from native_memory_encoder import sha_file
from prepare_neural_memory_protocol import digest
from train_joint_memory import paired_forward, paired_loss


def bit_exact(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def select_pairs(records):
    train = [r for r in records if r['split'] == 'train']
    if not train: raise ValueError('no training records')
    world = min(r['world_id'] for r in train)
    index = {(r['language'], r['relation'], r['scenario']): r for r in train if r['world_id'] == world}
    pairs = []
    for language in ('en', 'zh'):
        for relation in ('home_city', 'work_city'):
            for kind in KINDS:
                left = 'ordinary_source' if kind == 'ordinary_empty' else 'correct'
                try: pair = [index[(language, relation, s)] for s in (left, kind)]
                except KeyError as exc: raise ValueError('incomplete full preflight panel') from exc
                pairs.append((language, relation, kind, pair))
    if len(pairs) != 40: raise ValueError('preflight denominator changed')
    return world, pairs


def gradient_requirements(names, gradients, targets, kind, arm):
    """Disconnected specialists are not missing gradients on the active route."""
    if len(names) != len(gradients) or len(set(names)) != len(names): raise ValueError('gradient inventory mismatch')
    for name, grad in zip(names, gradients):
        if grad is not None and not torch.isfinite(grad).all(): raise ValueError('nonfinite gradient: ' + name)
    required = {'reader.query_encoder.weight', 'reader.source_encoder.weight', 'reader.state_head.weight',
                'reader.position_query.weight', 'reader.copy_gate.weight', 'roles.content.output.weight'}
    if any(t.reply.state_index != 1 for t in targets): required.add('roles.uncertainty_output.weight')
    if arm == 'joint_product' or kind != 'ordinary_empty':
        required.update(('factor_heads.0.weight', 'factor_heads.1.weight'))
    mapping = dict(zip(names, gradients))
    if any(name not in mapping or mapping[name] is None for name in required):
        raise ValueError('active gradient disconnected')
    return {'norms': {n: None if g is None else float(g.norm()) for n, g in mapping.items()},
            'required_present': sorted(required), 'none': [n for n, g in mapping.items() if g is None]}


def audit_package(package, smoke, inventory, data_audit):
    if package.manifest['scope'] != 'full': raise ValueError('full package required')
    if package.manifest['corpus_manifest_digest'] != inventory['dataset_manifest_digest']: raise ValueError('wrong data binding')
    for name, want in inventory['file_sha256'].items():
        if sha_file(package.root / name) != want: raise ValueError('compiled full inventory changed: ' + name)
    expected = {r['id']: r for r in data_audit['rows']}
    if {r['id'] for r in package.records} != set(expected): raise ValueError('missing or extra records')
    positions = 0
    for r in package.records:
        old = expected[r['id']]; ids = r['targets']['completion_token_ids']
        if r['split'] != old['split'] or len(ids) != old['reply_tokens']: raise ValueError('record audit mismatch')
        if [i for i in r['value_offsets'] if r['position_targets'][i] == [0]] != old['uncopyable_value_offsets']:
            raise ValueError('uncopyable targets changed')
        for i in range(len(ids)):
            if digest(tuple(r['prompt_ids'] + ids[:i])) not in package.prefix.rows: raise ValueError('missing teacher prefix')
            positions += 1
        if r['split'] == 'dev':
            try: package.sample(r, training=True)
            except ValueError: pass
            else: raise ValueError('dev accepted for optimizer')
    overlap = 0
    for key, location in smoke.prefix.rows.items():
        if key not in package.prefix.rows: raise ValueError('smoke prefix absent')
        batch, row = location; other_batch, other_row = package.prefix.rows[key]
        if any(not bit_exact(a[row], b[other_row]) for a, b in zip(smoke.prefix.arrays[batch], package.prefix.arrays[other_batch])):
            raise ValueError('full/smoke prefix differs')
        overlap += 1
    token_overlap = 0
    for key, row in smoke.tokens.rows.items():
        other = package.tokens.rows[key]
        if not bit_exact(row.token_ids, other.token_ids) or not bit_exact(row.features, other.features):
            raise ValueError('full/smoke token features differ')
        token_overlap += 1
    return {'records': len(package.records), 'reply_positions': positions, 'unique_prefixes': len(package.prefix.rows),
            'unique_token_texts': len(package.tokens.rows), 'smoke_prefixes_bit_exact': overlap,
            'smoke_token_texts_bit_exact': token_overlap, 'dev_optimizer_rejections': sum(r['split'] == 'dev' for r in package.records),
            'test_present': False}


def run(a):
    config = json.loads(Path(a.experiment).read_text())
    if config['id'] != 'CG-003-full-preflight' or config['optimizer_steps'] != 0 or config['seed'] != 1013:
        raise ValueError('unregistered preflight')
    root = Path(a.output); root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    package = JointTrainingPackage(a.features, a.package_digest)
    smoke = JointTrainingPackage(a.smoke_features, a.smoke_digest)
    checks = Path(__file__).resolve().parents[1] / 'training/memory/neural-system/checks'
    audit = audit_package(package, smoke, json.loads((checks / 'CG-003-feature-inventory.json').read_text()),
                          json.loads((checks / 'CG-003-data.json').read_text()))
    del smoke
    world, pairs = select_pairs(package.records)
    models = [create_joint_model(package.binding, package.manifest['blocked_ids'], arm) for arm in config['arms']]
    if any(not torch.equal(v, models[1].state_dict()[k]) for k, v in models[0].state_dict().items()):
        raise ValueError('initial weights differ')
    rows = []
    for model in models:
        original = {n: p.detach().clone() for n, p in model.named_parameters()}
        for language, relation, kind, records in pairs:
            row = {'arm': model.route_policy, 'language': language, 'relation': relation, 'kind': kind,
                   'record_ids': [r['id'] for r in records], 'passed': False}
            try:
                samples = [package.sample(r, training=True) for r in records]
                features, targets = [s[0] for s in samples], [s[1] for s in samples]
                output = paired_forward(model, features, package.head, package.scale)
                loss = paired_loss(output, features, targets, kind)
                parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
                grads = torch.autograd.grad(loss['total'], [p for _, p in parameters], allow_unused=True)
                gradients = gradient_requirements([n for n, _ in parameters], grads, targets, kind, model.route_policy)
                routes = []; empty = 0; normal = 0; uncertainty = 0
                with torch.no_grad():
                    for f, out in zip(features, output):
                        state = model.prefill(f.memory); route = int(state.core.state_logits.argmax()); routes.append(route)
                        if model.read(f.hidden, f.base, package.head, package.scale, state, enabled=False) is not f.base:
                            raise ValueError('disabled not exact identity')
                        actual = model.read(f.hidden, f.base, package.head, package.scale, state)
                        if route == 0:
                            if actual is not f.base: raise ValueError('normal not exact identity')
                            normal += 1
                        elif route == 2:
                            expected = continuous_logits(f.base, model.roles.uncertainty(f.hidden), package.head, package.scale)[0]
                            if not torch.equal(actual, expected): raise ValueError('uncertainty route changed')
                            uncertainty += 1
                        else:
                            if not torch.equal(actual, out.supported): raise ValueError('supported route changed')
                        if not len(f.memory.source):
                            if route == 1 or torch.count_nonzero(out.copy_mass): raise ValueError('empty memory used copy/support')
                            empty += 1
                        if not torch.isfinite(out.supported).all(): raise ValueError('nonfinite supported logits')
                        # With active copy this is already a log distribution;
                        # applying softmax here would hide a normalization bug.
                        copied_rows = out.copy_mass[:, 0] > 0
                        if copied_rows.any() and not torch.allclose(out.supported[copied_rows].double().exp().sum(-1),
                                torch.ones(int(copied_rows.sum()), dtype=torch.float64), atol=1e-6, rtol=1e-6):
                            raise ValueError('copy mixture not normalized')
                row.update(passed=True, loss=float(loss['total'].detach()), pair_loss=float(loss['pair'].detach()),
                    gradients=gradients, predicted_states=routes, empty_checks=empty, normal_checks=normal, uncertainty_checks=uncertainty)
            except (ValueError, RuntimeError) as exc:
                row['error'] = str(exc)
            rows.append(row)
            with (root / f'pair-{len(rows):03d}.json').open('x') as f: json.dump(row, f, indent=2)
            print(json.dumps({'completed': len(rows), 'total': 80, 'arm': model.route_policy, 'kind': kind, 'passed': row['passed']}), flush=True)
        if any(not torch.equal(p, original[n]) or p.grad is not None for n, p in model.named_parameters()):
            raise ValueError('zero-step preflight modified parameters')
    if len(rows) != 80: raise ValueError('preflight denominator changed')
    report = {'experiment': config, 'audit': audit, 'world': world, 'rows': rows, 'device': 'cpu',
        'pairs_passed': sum(r['passed'] for r in rows), 'pairs_total': len(rows),
        'package_digest': a.package_digest, 'smoke_digest': a.smoke_digest, 'optimizer_steps': 0,
        'parameters_unchanged': True, 'new_free_generated_answers': 0, 'memory_accuracy_measured': False,
        'training_approved': False, 'deployment_approved': False,
        'source_sha256': {n: sha_file(Path(__file__).parent / n) for n in ('preflight_joint_full.py', 'train_joint_memory.py',
            'joint_memory_model.py', 'token_memory_composition.py', 'token_memory_supervision.py')},
        'experiment_sha256': sha_file(a.experiment)}
    with (root / 'report.json').open('x') as f: json.dump(report, f, indent=2); f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('experiment', 'features', 'package-digest', 'smoke-features', 'smoke-digest', 'output'):
        p.add_argument('--' + key, required=True)
    run(p.parse_args())
