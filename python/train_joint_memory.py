"""CG-003 paired loss/preflight core. Optimizer launch deliberately unavailable."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from diagnose_role_content_path import first_divergence
from joint_memory_model import create_joint_model
from joint_training_data import JointTrainingPackage, JointForwardFeatures
from native_memory_encoder import sha_file
from prepare_neural_memory_protocol import digest
from token_memory_supervision import supervised_token_loss, TokenSupervision


def paired_forward(model, features, head, scale):
    if len(features) != 2 or any(type(f) is not JointForwardFeatures for f in features):
        raise ValueError('two label-free forward examples required')
    if not torch.equal(features[0].memory.query, features[1].memory.query):
        raise ValueError('paired query features changed')
    return [model.branches(f.hidden, f.base, head, scale, model.prefill(f.memory)) for f in features]


def paired_loss(outputs, features, targets, kind):
    if len(outputs) != 2 or len(features) != 2 or len(targets) != 2 or any(type(t) is not TokenSupervision for t in targets):
        raise ValueError('complete post-forward pair supervision required')
    if targets[0].reply.prompt_token_ids != targets[1].reply.prompt_token_ids:
        raise ValueError('paired prompt changed')
    losses = [supervised_token_loss(o, t) for o, t in zip(outputs, targets)]
    if any(t.reply.sample_weight != 1 for t in targets): raise ValueError('unregistered sample weight')
    contrast = outputs[0].base.new_zeros(())
    if kind == 'role_swap':
        if [t.reply.state_index for t in targets] != [1, 2]: raise ValueError('role pair states changed')
        margins = [o.state_logits.double()[1] - o.state_logits.double()[2] for o in outputs]
        contrast = F.softplus(1 - (margins[0] - margins[1]))
    elif kind == 'value_swap':
        if [t.reply.state_index for t in targets] != [1, 1]: raise ValueError('value pair states changed')
        a, b = [t.reply.completion_token_ids for t in targets]; offset = first_divergence(a, b)
        if any(t.positions[offset] is None for t in targets): raise ValueError('first difference is not a supervised value')
        if any(not torch.equal(getattr(features[0], n)[offset], getattr(features[1], n)[offset]) for n in ('hidden', 'base')):
            raise ValueError('value contrast requires identical C causal prefix')
        margins = [o.supported.double()[offset, a[offset]] - o.supported.double()[offset, b[offset]] for o in outputs]
        contrast = F.softplus(1 - (margins[0] - margins[1]))
    elif kind not in ('clause_reorder', 'paraphrase', 'wrong_subject', 'wrong_relation', 'negated', 'historical', 'empty', 'ordinary_empty'):
        raise ValueError('unregistered pair kind')
    return {'total': sum(l['weighted_total'] for l in losses) / 2 + contrast, 'pair': contrast, 'examples': losses}


def require_joint_launch(*args, **kwargs):
    from joint_launch import require_launch
    return require_launch(*args, **kwargs)


def run_preflight(a):
    torch.set_num_threads(4)
    package = JointTrainingPackage(a.features, a.package_digest)
    models = [create_joint_model(package.binding, package.manifest['blocked_ids'], arm) for arm in ('joint_aux', 'joint_product')]
    if any(not torch.equal(v, models[1].state_dict()[k]) for k, v in models[0].state_dict().items()):
        raise ValueError('arms do not share exact initial weights')
    records = package.records
    world = min(r['world_id'] for r in records if r['split'] == 'train')
    panel = {(r['language'], r['scenario']): r for r in records if r['split'] == 'train' and r['world_id'] == world and r['relation'] == 'home_city'}
    rows = []
    for model in models:
        original = {k: p.detach().clone() for k, p in model.named_parameters()}
        for language in ('en', 'zh'):
            samples = [package.sample(panel[(language, s)]) for s in ('correct', 'role_swap')]
            features, targets = [x[0] for x in samples], [x[1] for x in samples]
            output = paired_forward(model, features, package.head, package.scale)
            loss = paired_loss(output, features, targets, 'role_swap')
            parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss['total'], [p for _, p in parameters], allow_unused=True)
            if not grads or any(g is None or not torch.isfinite(g).all() for g in grads):
                raise ValueError('missing or nonfinite trainable gradient')
            actual_states = []
            for f in features:
                state = model.prefill(f.memory)
                actual_states.append(int(state.core.state_logits.argmax()))
                if model.read(f.hidden, f.base, package.head, package.scale, state, enabled=False) is not f.base:
                    raise ValueError('disabled output changed')
            rows.append({'arm': model.route_policy, 'language': language, 'loss': float(loss['total'].detach()),
                'pair_loss': float(loss['pair'].detach()), 'predicted_states': actual_states,
                'gradient_norms': {n: float(g.norm()) for (n, _), g in zip(parameters, grads)},
                'finite_gradients': True, 'disabled_exact_identity': True})
        if any(not torch.equal(p, original[n]) or p.grad is not None for n, p in model.named_parameters()):
            raise ValueError('preflight updated model parameters')
    report = {'format': 'cg003-zero-step-preflight-v1', 'device': 'cpu', 'package_digest': a.package_digest,
        'package_scope': package.manifest['scope'], 'identical_initial_weights': True, 'parameters_unchanged': True,
        'unused_private_classifier_frozen': True, 'optimizer_steps': 0, 'training_approved': False,
        'memory_accuracy_measured': False, 'rows': rows,
        'source_sha256': {n: sha_file(Path(__file__).parent / n) for n in ('train_joint_memory.py', 'joint_training_data.py', 'joint_memory_model.py')}}
    with Path(a.output).open('x') as f: json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps({'preflight_pairs': len(rows), 'optimizer_steps': 0, 'scope': package.manifest['scope']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('preflight', 'train'), required=True)
    for n in ('features', 'package-digest', 'output'): p.add_argument('--' + n, required=True)
    a = p.parse_args()
    if a.mode == 'train': require_joint_launch()
    else: run_preflight(a)
