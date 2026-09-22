"""Bounded CG-003 optimizer implementation. Launch requires bound evidence.

The private engine is tested on tiny artificial tensors only. It is not a
launch authorization, resume path, or a deployable memory artifact exporter.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from joint_binding_schedule import pair_schedule
from joint_memory_model import create_joint_model
from native_memory_encoder import BACKBONE_SHA256, sha_file
from neural_memory_generation import TEMPLATE_VERSION
from preflight_joint_full import gradient_requirements
from prepare_neural_memory_protocol import digest
from review_neural_memory import read_rows
from train_joint_memory import paired_forward, paired_loss, require_joint_launch


FORMAT = 'cg003-joint-research-checkpoint-v1'
OPTIMIZER = {'name': 'AdamW', 'learning_rate': .0001, 'weight_decay': .01,
             'batch_size_pairs': 4, 'gradient_clip_norm': 1., 'precision': 'float32', 'amp': False, 'tf32': False}
FEATURE_DIGEST = 'd299ad9f59c5a8500d751ee71b4f9139232cec9f80a2607e14953442e883b6fb'


def validate_config(config):
    if (config['id'] != 'CG-003' or config['initial_seed'] != 1013 or config['optimizer'] != OPTIMIZER or
            config['backbone_sha256'] != BACKBONE_SHA256 or config['full_feature_package_digest'] != FEATURE_DIGEST or
            [a['id'] for a in config['arms']] != ['joint_aux', 'joint_product'] or
            config['proposed_budget'] != {'candidates': 2, 'optimizer_steps_per_arm': 50, 'total_optimizer_steps': 100,
                                          'additional_seeds': 0, 'automatic_continuation': False}):
        raise ValueError('unregistered optimizer, initialization or budget')
    keys = ('backbone_trainable', 'kv_reuse', 'lora', 'svd', 'nas_ng', 'source_text_in_prompt', 'gold_serving_route', 'locomo')
    if config['constraints'] != {k: False for k in keys}:
        raise ValueError('forbidden training constraint')


def bind_schedule(package, corpus, audit):
    if package.manifest['scope'] != 'full' or digest(package.manifest) != FEATURE_DIGEST:
        raise ValueError('qualified full feature package required')
    corpus = Path(corpus); manifest = json.loads((corpus / 'manifest.json').read_text())
    if digest(manifest) != package.manifest['corpus_manifest_digest']:
        raise ValueError('corpus identity changed')
    if sha_file(corpus / 'train.pairs.jsonl') != manifest['file_sha256']['train.pairs.jsonl']:
        raise ValueError('training pairs changed')
    schedule = pair_schedule(read_rows(corpus / 'train.pairs.jsonl'))
    if schedule != audit['schedule']:
        raise ValueError('predeclared schedule changed')
    records = {r['id']: r for r in package.records}
    for step in schedule:
        for pair in step['pairs']:
            for side, scenario in (('left_id', 'ordinary_source' if pair['kind'] == 'ordinary_empty' else 'correct'),
                                   ('right_id', pair['kind'])):
                r = records[pair[side]]
                if (r['split'], r['world_id'], r['language'], r['relation'], r['scenario']) != (
                        'train', pair['world_id'], pair['language'], pair['relation'], scenario):
                    raise ValueError('pair metadata or training split mismatch')
    return schedule, records


def tensor_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def _finite_tree(value):
    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
        raise ValueError('nonfinite optimizer/checkpoint tensor')
    if isinstance(value, dict):
        for v in value.values(): _finite_tree(v)
    if isinstance(value, (list, tuple)):
        for v in value: _finite_tree(v)


def _atomic_save(payload, path):
    """Publish a complete generated checkpoint without overwriting evidence."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.joint-checkpoint-', delete=False) as stream:
        temp = Path(stream.name)
        try:
            torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
            os.link(temp, path)  # Exclusive publication, also on interruption/retry.
        finally:
            temp.unlink()


def save_checkpoint(path, model, optimizer, step, binding, initial_digest):
    if type(step) is not int or step not in (0, 50):
        raise ValueError('only initial/final checkpoints are registered')
    state = cpu_tree(model.state_dict()); opt = cpu_tree(optimizer.state_dict())
    _finite_tree(state); _finite_tree(opt)
    payload = {'format': FORMAT, 'step': step, 'route_policy': model.route_policy, 'binding': binding,
        'initial_state_digest': initial_digest, 'state_dict': state, 'state_digest': tensor_digest(state),
        'optimizer_state': opt, 'optimizer_parameter_names': [n for n, p in model.named_parameters() if p.requires_grad],
        'backbone_sha256': BACKBONE_SHA256, 'template_version': TEMPLATE_VERSION,
        'torch_version': str(torch.__version__), 'torch_rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
        'resume_allowed': False, 'deployment_approved': False}
    _atomic_save(payload, path)
    return {'file': Path(path).name, 'sha256': sha_file(path), 'step': step, 'state_digest': payload['state_digest']}


def load_checkpoint(path, *, expected_sha256, model, expected_binding, expected_initial_digest):
    if sha_file(path) != expected_sha256: raise ValueError('checkpoint file identity mismatch')
    data = torch.load(path, map_location='cpu', weights_only=True)
    if (data['format'] != FORMAT or type(data['step']) is not int or data['step'] not in (0, 50) or
            data['route_policy'] != model.route_policy or data['binding'] != expected_binding or
            data['initial_state_digest'] != expected_initial_digest or data['resume_allowed'] is not False or
            data['deployment_approved'] is not False or data['backbone_sha256'] != BACKBONE_SHA256 or
            data['template_version'] != TEMPLATE_VERSION or
            data['optimizer_parameter_names'] != [n for n, p in model.named_parameters() if p.requires_grad]):
        raise ValueError('checkpoint model/data/source binding mismatch')
    state = data['state_dict']; current = model.state_dict()
    if tensor_digest(current) != expected_initial_digest:
        raise ValueError('checkpoint loader requires the bound fresh initialization')
    if state.keys() != current.keys() or any(state[n].shape != v.shape or state[n].dtype != v.dtype for n, v in current.items()):
        raise ValueError('checkpoint shape/dtype mismatch')
    _finite_tree(state); _finite_tree(data['optimizer_state'])
    if tensor_digest(state) != data['state_digest']: raise ValueError('checkpoint state digest mismatch')
    if data['step'] == 0 and data['state_digest'] != expected_initial_digest:
        raise ValueError('step-zero checkpoint is not the registered initialization')
    if any(not torch.equal(state[n], p.detach().cpu()) for n, p in model.named_parameters() if not p.requires_grad):
        raise ValueError('checkpoint altered frozen parameters')
    # No model mutation occurs until all validation has succeeded; never resume.
    model.load_state_dict(state, strict=True)
    return data


def _run_fixed_loop(model, head, scale, schedule, sample_pair, *, on_checkpoint, on_step):
    if len(schedule) != 50 or any(s['step'] != i + 1 or len(s['pairs']) != 4 for i, s in enumerate(schedule)):
        raise ValueError('exactly 50 steps/four pairs required; no continuation')
    if head.requires_grad or head.grad is not None or head.dtype != torch.float32:
        raise ValueError('frozen FP32 head required')
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if any(p.dtype != torch.float32 or not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('finite FP32 parameters required')
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    head_version = head._version
    initial = tensor_digest(model.state_dict())
    optimizer = torch.optim.AdamW([p for _, p in named], lr=.0001, weight_decay=.01, foreach=False)
    on_checkpoint(0, model, optimizer, initial)
    try:
        for row in schedule:
            model.train(); optimizer.zero_grad(set_to_none=True); details = []
            for pair in row['pairs']:
                features, targets = sample_pair(pair)
                output = paired_forward(model, features, head, scale)
                loss = paired_loss(output, features, targets, pair['kind'])
                # Sequential pair backprop preserves the graph across BOTH sides,
                # but avoids retaining four large vocabulary graphs on the GPU.
                grads = torch.autograd.grad(loss['total'] / 4, [p for _, p in named], allow_unused=True)
                gradient_requirements([n for n, _ in named], grads, targets, pair['kind'], model.route_policy)
                for (_, p), g in zip(named, grads):
                    if g is not None:
                        if p.grad is None: p.grad = g.detach()
                        else: p.grad.add_(g.detach())
                details.append({'pair_id': pair['id'], 'kind': pair['kind'],
                    'record_ids': [pair['left_id'], pair['right_id']],
                    'pair_total': float(loss['total'].detach()), 'pair_contrast': float(loss['pair'].detach()),
                    'example_totals': [float(v['weighted_total'].detach()) for v in loss['examples']]})
                del output, loss, grads, features, targets
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in named], 1., error_if_nonfinite=True)
            optimizer.step()
            _finite_tree(model.state_dict()); _finite_tree(optimizer.state_dict())
            if head.grad is not None or head._version != head_version or any(
                    not torch.equal(p.detach(), frozen[n]) for n, p in model.named_parameters() if n in frozen):
                raise ValueError('frozen parameter changed')
            on_step({'step': row['step'], 'pairs': details,
                     'batch_mean_loss': sum(d['pair_total'] for d in details) / 4,
                     'gradient_norm_before_clip': float(norm)})
        on_checkpoint(50, model, optimizer, initial)
    finally:
        optimizer.zero_grad(set_to_none=True)
    return {'optimizer_steps': 50, 'record_uses': 400, 'automatic_continuation': False,
            'memory_accuracy_measured': False, 'deployment_approved': False}


def train_candidate(package, config, corpus, audit, arm, root, *, launch_evidence=None):
    qualified = require_joint_launch(config=config, launch_evidence=launch_evidence)
    validate_config(config)
    if arm not in ('joint_aux', 'joint_product') or not torch.cuda.is_available():
        raise ValueError('registered arm on CUDA required')
    schedule, records = bind_schedule(package, corpus, audit)
    # One persistent allocation per approved arm. Failures consume the claim:
    # do not silently restart from a fresh seed or a different output directory.
    budget = Path(qualified['budget_root']); budget.mkdir(exist_ok=True)
    (budget / arm).mkdir(exist_ok=False)
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    model = create_joint_model(package.binding, package.manifest['blocked_ids'], arm).to('cuda')
    head = package.head.to('cuda')
    source = {str(p.relative_to(Path(__file__).parent)): sha_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    binding = {'feature_package_digest': FEATURE_DIGEST, 'encoder_binding': package.binding,
               'schedule_digest': digest(schedule), 'experiment_digest': digest(config),
               'source_sha256': source, 'launch_evidence': launch_evidence}
    checkpoints = []
    def sample(pair):
        values = [package.sample(records[pair[s]], 'cuda', training=True) for s in ('left_id', 'right_id')]
        return [v[0] for v in values], [v[1] for v in values]
    def checkpoint(step, model, optimizer, initial):
        checkpoints.append(save_checkpoint(root / f'step-{step:06d}.pt', model, optimizer, step, binding, initial))
    with (root / 'training.jsonl').open('x') as log:
        def emit(row):
            log.write(json.dumps(row) + '\n'); log.flush(); os.fsync(log.fileno())
            print(json.dumps({'step': row['step'], 'loss': row['batch_mean_loss']}), flush=True)
        result = _run_fixed_loop(model, head, package.scale, schedule, sample, on_checkpoint=checkpoint, on_step=emit)
    if source != {n: sha_file(Path(__file__).parent / n) for n in source}:
        raise ValueError('training source changed')
    from token_memory_supervision import supervised_token_loss
    diagnostic = []
    model.eval()
    with torch.no_grad():
        for record in package.records:
            if record['split'] != 'dev': continue
            f, target = package.sample(record, 'cuda', training=False)
            output = model.branches(f.hidden, f.base, head, package.scale, model.prefill(f.memory))
            loss = supervised_token_loss(output, target)
            diagnostic.append({'id': record['id'], 'target_state': target.reply.state_index,
                'predicted_state': int(output.state_logits.argmax()), 'teacher_forced_loss': float(loss['weighted_total'])})
    if len(diagnostic) != 192: raise ValueError('dev diagnostic denominator changed')
    with (root / 'dev-diagnostic.json').open('x') as f:
        json.dump({'measurement': 'state_and_teacher_forcing_not_memory_accuracy', 'rows': diagnostic}, f, indent=2)
    result.update(checkpoints=checkpoints, binding=binding, status='awaiting_native_generation_and_independent_review',
        dev_state_correct=sum(r['target_state'] == r['predicted_state'] for r in diagnostic), dev_records=192)
    with (root / 'status.json').open('x') as f: json.dump(result, f, indent=2)
    return result
