"""DG-009: zero-step gradients, not training or a free-recall evaluation.

Labels select post-forward losses and diagnostic reference positions only.
Temporary finite-difference perturbations are always restored and never saved.
"""
import argparse
import json
from pathlib import Path
import statistics

import torch
from torch.nn import functional as F

from c_tokenizer import CTokenizer
from diagnose_role_content_path import first_divergence, token_span, validate_control
from episode_memory_inputs import encoder_texts
from eval_role_checkpoint import CHECKPOINT_SHA256, load_checkpoint
from memory_role_separated import forward_roles, role_loss
from native_memory_encoder import sha_file
from prepare_availability_curriculum import materialize
from prepare_neural_memory_protocol import digest
from review_neural_memory import read_rows
from train_availability_memory import TrainingPackage
from train_role_separated_memory import FEATURE_DIGEST


def pair_metrics(a, b, left_target, right_target, competitors=None):
    """Both correct-side margins are needed; positive contrast alone is weak."""
    if (a.ndim != 1 or b.shape != a.shape or left_target == right_target or
            not 0 <= min(left_target, right_target) <= max(left_target, right_target) < len(a) or
            not torch.isfinite(a).all() or not torch.isfinite(b).all()):
        raise ValueError('finite paired logits and distinct targets required')
    if competitors is None:
        competitors = []
        for logits, target in ((a, left_target), (b, right_target)):
            other = logits.detach().clone(); other[target] = -torch.inf
            competitors.append(int(other.argmax()))
    if (len(competitors) != 2 or any(type(t) is not int or not 0 <= t < len(a) for t in competitors)
            or competitors[0] == left_target or competitors[1] == right_target):
        raise ValueError('invalid fixed competitors')
    a, b = a.double(), b.double()
    ma = a[left_target] - a[right_target]
    mb = b[left_target] - b[right_target]
    return {'difference': ma - mb, 'center': (ma + mb) / 2,
            'left_correct_margin': ma, 'right_correct_margin': -mb,
            'left_full_margin': a[left_target] - a[competitors[0]],
            'right_full_margin': b[right_target] - b[competitors[1]]}, competitors


def objectives_and_metrics(outputs, targets, kind, offset=None, competitors=None):
    """Training-only targets are applied AFTER the label-free neural forward."""
    losses = [role_loss(o, t) for o, t in zip(outputs, targets)]
    old = sum(v['weighted_total'] for v in losses) / sum(t.sample_weight for t in targets)
    balanced = sum(v['state'] + v['branch'] for v in losses) / 2
    if kind == 'value':
        if [t.state_index for t in targets] != [1, 1] or offset is None:
            raise ValueError('supported value pair required')
        ids = [t.completion_token_ids[offset] for t in targets]
        logits = [o.content[offset] for o in outputs]
        focused = sum(v['state'] for v in losses) / 2 + sum(
            F.cross_entropy(z[None], torch.tensor([i], device=z.device)) for z, i in zip(logits, ids)) / 2
        objectives = {'old': old, 'focused': focused}
    elif kind == 'subject':
        if [t.state_index for t in targets] != [1, 2]:
            raise ValueError('matched and insufficient subject pair required')
        logits = [o.state_logits[0] for o in outputs]; ids = [1, 2]
        focused = balanced
        objectives = {'old': old, 'balanced': balanced}
    else:
        raise ValueError('unknown paired diagnostic')
    metrics, competitors = pair_metrics(*logits, *ids, competitors)
    objectives['paired'] = focused + F.softplus(1 - metrics['difference'])
    return objectives, metrics, competitors


def flat_gradient(value, parameters):
    grads = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
    result = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                        for p, g in zip(parameters, grads)])
    if not torch.isfinite(result).all():
        raise ValueError('nonfinite diagnostic gradient')
    return result


def parameter_groups(named, gradient):
    sums = {}; start = 0
    for name, parameter in named:
        group = ('content_output' if name.startswith('content.output.') else
                 'content_gate' if name.startswith(('content.use_gate.', 'content.layer_gain')) else
                 'reader' if name.startswith('content.') else
                 'state' if name.startswith('state_') else 'uncertainty')
        segment = gradient[start:start + parameter.numel()]
        sums[group] = sums.get(group, 0.) + float(segment.double().square().sum())
        start += parameter.numel()
    if start != gradient.numel(): raise ValueError('gradient layout mismatch')
    return {name: value ** .5 for name, value in sums.items()}


def finite_difference(model, direction, evaluate, expected, epsilons):
    """No optimizer. Copy from immutable originals each time; restore on error."""
    parameters = tuple(model.parameters())
    originals = [p.detach().clone() for p in parameters]
    if direction.numel() != sum(p.numel() for p in parameters) or not torch.isfinite(direction).all():
        raise ValueError('invalid diagnostic direction')
    rows = []
    try:
        for epsilon in epsilons:
            if epsilon <= 0: raise ValueError('positive finite-difference radius required')
            sides = []
            for sign in (-1, 1):
                with torch.no_grad():
                    offset = 0
                    for p, original in zip(parameters, originals):
                        step = direction[offset:offset + p.numel()].reshape_as(p)
                        p.copy_(original + sign * epsilon * step); offset += p.numel()
                    # Fresh forward/decision: previous PrefillDecision versions are invalid.
                    sides.append({key: float(v) for key, v in evaluate().items()})
            observations = {}
            for key, analytical in expected.items():
                numerical = (sides[1][key] - sides[0][key]) / (2 * epsilon)
                error = abs(numerical - analytical)
                observations[key] = {'analytical': analytical, 'numerical': numerical,
                    'absolute_error': error, 'passed': error <= .01 + .05 * abs(analytical)}
            rows.append({'epsilon': epsilon, 'metrics': observations})
    finally:
        with torch.no_grad():
            for p, original in zip(parameters, originals): p.copy_(original)
        if any(not torch.equal(p, original) for p, original in zip(parameters, originals)):
            raise ValueError('temporary diagnostic parameters not restored')
    return rows


def analyze_pair(model, samples, head, scale, kind, offset, do_fd):
    parameters = tuple(model.parameters()); named = tuple(model.named_parameters())
    def forward():
        return [forward_roles(model, features, head, scale) for features, _ in samples]
    targets = [target for _, target in samples]
    outputs = forward()
    objectives, metrics, competitors = objectives_and_metrics(outputs, targets, kind, offset)
    gradients = {name: flat_gradient(value, parameters) for name, value in objectives.items()}
    directions = {}
    rows = {}
    for name, g in gradients.items():
        norm = float(g.double().norm())
        if norm <= 0: raise ValueError('zero objective gradient')
        directions[name] = -g / norm
        rows[name] = {'loss': float(objectives[name].detach()), 'gradient_norm': norm,
            'gradient_group_norms': parameter_groups(named, g),
            'gradient_cosine_to_old': float(F.cosine_similarity(g.double()[None], gradients['old'].double()[None])),
            'directional_derivatives': {}}
    for key, value in metrics.items():
        g = flat_gradient(value, parameters)
        for name, direction in directions.items():
            rows[name]['directional_derivatives'][key] = float(torch.dot(g.double(), direction.double()))
    original_metrics = {key: float(value.detach()) for key, value in metrics.items()}
    # Drop the graph before perturbing tensors; autograd cannot reuse old versions.
    del outputs, objectives, metrics, gradients, g
    if do_fd:
        def evaluate():
            return objectives_and_metrics(forward(), targets, kind, offset, competitors)[1]
        for name, direction in directions.items():
            rows[name]['finite_difference'] = finite_difference(model, direction, evaluate,
                rows[name]['directional_derivatives'], (.001, .003))
    return {'kind': kind, 'baseline_metrics': original_metrics, 'fixed_competitors': competitors,
            'objectives': rows, 'finite_difference_selected': do_fd}


def summarize(rows):
    result = {}
    for kind in ('value', 'subject'):
        probes = [r[kind] for r in rows]
        result[kind] = {'pairs': len(probes), 'objectives': {}}
        for name in probes[0]['objectives']:
            obs = [p['objectives'][name] for p in probes]
            ds = [v['directional_derivatives'] for v in obs]
            fd = [m['passed'] for v in obs for e in v.get('finite_difference', []) for m in e['metrics'].values()]
            result[kind]['objectives'][name] = {
                'difference_increases': sum(d['difference'] > 0 for d in ds),
                'both_correct_side_margins_increase': sum(d['left_correct_margin'] > 0 and d['right_correct_margin'] > 0 for d in ds),
                'both_full_vocab_or_state_margins_increase': sum(d['left_full_margin'] > 0 and d['right_full_margin'] > 0 for d in ds),
                'median_directional_derivatives': {k: statistics.median(d[k] for d in ds) for k in ds[0]},
                'median_gradient_norm': statistics.median(v['gradient_norm'] for v in obs),
                'median_gradient_cosine_to_old': statistics.median(v['gradient_cosine_to_old'] for v in obs),
                'finite_difference_passed': sum(fd), 'finite_difference_total': len(fd)}
    return result


def run(a):
    config = json.loads(Path(a.experiment).read_text())
    if (config['id'], config['checkpoint_sha256'], config['feature_package_digest'], config['optimizer_steps']) != (
            'DG-009', CHECKPOINT_SHA256, FEATURE_DIGEST, 0):
        raise ValueError('unregistered diagnostic')
    root = Path(a.output); root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model, binding = load_checkpoint(a.checkpoint)
    package = TrainingPackage(a.features, FEATURE_DIGEST)
    if package.manifest['encoder_identity'] != binding['encoder_identity']: raise ValueError('encoder binding mismatch')
    manifest = materialize(a.config, a.corpus, verify=True)
    if digest(manifest) != package.manifest['corpus_manifest_sha256']: raise ValueError('corpus identity mismatch')
    if sha_file(a.gguf) != binding['backbone_sha256']: raise ValueError('wrong backbone')
    tokenizer_hashes = {v for k, v in package.manifest['binaries_sha256'].items() if Path(k).name == 'tok_probe'}
    if tokenizer_hashes != {sha_file(a.tok_probe)}: raise ValueError('tokenizer identity mismatch')
    records = {r['id']: r for r in package.records if r['split'] == 'train'}
    worlds = sorted({r['world_id'] for r in records.values()})[:4]
    inputs = {r['id']: r for r in read_rows(Path(a.corpus) / 'train.inputs.jsonl')}
    groups = {}
    for row in read_rows(Path(a.corpus) / 'train.index.jsonl'):
        if row['world_id'] in worlds and row['scenario'] in ('original', 'swapped_values', 'wrong_subject'):
            groups.setdefault((row['world_id'], row['language']), {})[row['scenario']] = row
    if len(groups) != 8 or any(len(g) != 3 for g in groups.values()): raise ValueError('pair inventory changed')
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    head_version = package.head._version
    tokenizer = CTokenizer(a.tok_probe, a.gguf)
    rows = []
    try:
        for (world, language), group in sorted(groups.items()):
            runs = {s: records[r['id']] for s, r in group.items()}
            samples = {}
            for scenario, r in runs.items():
                runtime = inputs[r['id']]
                if r['input_sha256'] != digest(runtime) or tuple(r['source_texts']) != encoder_texts(runtime)[1]:
                    raise ValueError('natural runtime input binding changed')
                samples[scenario] = package.sample(r, 'cpu')
            for kind, left, right in (('value', 'original', 'swapped_values'), ('subject', 'swapped_values', 'wrong_subject')):
                validate_control(inputs[runs[left]['id']], inputs[runs[right]['id']],
                    group[left]['source_facts'][0], group[right]['source_facts'][0], kind)
                if runs[left]['prompt_ids'] != runs[right]['prompt_ids']: raise ValueError('paired query changed')
            ta, tb = [runs[s]['targets']['completion_token_ids'] for s in ('original', 'swapped_values')]
            offset = first_divergence(ta, tb)
            for scenario in ('original', 'swapped_values'):
                target = runs[scenario]['targets']; value = group[scenario]['source_facts'][0]['value']
                spans = token_span(target['response'], tokenizer.decode_pieces(target['completion_token_ids'][:-1]), value)
                if offset not in spans: raise ValueError('contrast token is not a value token')
            for field in ('hidden', 'base_logits'):
                av = getattr(samples['original'][0], field)[offset]
                bv = getattr(samples['swapped_values'][0], field)[offset]
                if not torch.equal(av, bv): raise ValueError('first divergent value prefix features changed')
            row = {'world_id': world, 'language': language, 'record_ids': {s: r['id'] for s, r in runs.items()},
                   'value_offset': offset, 'value_target_tokens': [ta[offset], tb[offset]]}
            for kind, left, right in (('value', 'original', 'swapped_values'), ('subject', 'swapped_values', 'wrong_subject')):
                row[kind] = analyze_pair(model, [samples[left], samples[right]], package.head, package.scale,
                    kind, offset if kind == 'value' else None, world == worlds[0])
            rows.append(row)
            with (root / f'pair-{len(rows):02d}.json').open('x') as f: json.dump(row, f, ensure_ascii=False, indent=2)
            print(json.dumps({'completed_pairs': len(rows), 'total': 8, 'world': world, 'language': language}), flush=True)
        if any(not torch.equal(p, before[name]) for name, p in model.named_parameters()): raise ValueError('parameters changed')
        if package.head._version != head_version or package.head.grad is not None: raise ValueError('frozen head changed')
        if any(p.grad is not None for p in model.parameters()): raise ValueError('unexpected accumulated parameter gradients')
        if sha_file(a.checkpoint) != CHECKPOINT_SHA256: raise ValueError('checkpoint file changed')
        report = {'experiment': config, 'rows': rows, 'summary': summarize(rows), 'optimizer_steps': 0,
            'parameters_unchanged': True, 'new_free_generated_answers': 0, 'memory_accuracy_measured': False,
            'training_approved': False, 'deployment_approved': False, 'selected_split': 'train',
            'source_records_verified': 24, 'finite_difference_parameter_restoration': 'exact',
            'direction_normalization': 'negative unit-L2 Euclidean gradient, NOT AdamW or a learning rate',
            'feature_package_digest': FEATURE_DIGEST,
            'artifact_sha256': {key: sha_file(getattr(a, key)) for key in ('experiment', 'checkpoint', 'gguf', 'tok_probe')},
            'source_sha256': {p: sha_file(Path(__file__).parent / p) for p in (
                'diagnose_paired_memory_objectives.py', 'diagnose_role_content_path.py', 'memory_role_separated.py', 'continuous_memory.py')}}
        with (root / 'report.json').open('x') as f: json.dump(report, f, ensure_ascii=False, indent=2); f.write('\n')
    finally:
        tokenizer._proc.terminate(); tokenizer._proc.wait(); tokenizer._proc.stdin.close(); tokenizer._proc.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment', 'checkpoint', 'features', 'config', 'corpus', 'gguf', 'tok-probe', 'output'):
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())


if __name__ == '__main__': main()
