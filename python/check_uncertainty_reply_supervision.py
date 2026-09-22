"""DG-022 native trajectory supervision check. Zero updates; no serving or free replies."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from append_value_transport import AppendValueCodec
from audit_joint_generation import close, continuous_file
from autonomous_value_controller import AutonomousValueController, LiveFrame, FrameOrigin
from check_joint_span_interface import source_alignment
from diagnose_continuous_memory import forward
from diagnose_native_generation_gradient import exact, native_forward
from episode_memory_inputs import encoder_texts
from fine_span_reader import FineSpanReader, ByteLayout
from joint_optimizer import FEATURE_DIGEST, tensor_digest
from joint_span_reader import PrefixFeature, SpanFeatures
from joint_training_data import DATA_DIGEST
from native_memory_encoder import NativeFeatureBank, BACKBONE_SHA256, digest, sha_file, text_key
from native_prefix_bank import extract_prefix_batch
from native_token_payload import payload_mask
from neural_memory_contract import ModelBinding
from neural_memory_generation import encode_generation_input
from query_only_uncertainty import QueryOnlyUncertainty
from uncertainty_reply_controller import ResearchUncertaintyController, UncertaintyDecision
from uncertainty_reply_supervision import (UncertaintyTrajectory, aggregate_loss, disabled_identity,
                                           gradient_structure, trajectory_losses)
from value_teacher_trajectory import compile_teacher
from value_transport import PayloadSnapshot, FactPayload, ReplyBinding

ROOT = Path(__file__).resolve().parents[1]; RESEARCH = ROOT / 'training/memory/neural-system'
FROZEN = ROOT / 'build/neural-memory-cg003-frozen-build'
FROZEN_SHA = {'tok_probe': 'f3d5fd147895e5875776a579d864f7a36ef16b55a4f59da00838f8b59f416adc',
              'memory_prefix_probe': '2735729604b1a52d99d1a60f48234b96ce3683d63aeec6434879bbfe7753e302',
              'memory_gradient_reference': '4c05a9c4faa7b61d03a017a3a80f43c36329c5d8af945ea5c9f1359640de9152',
              'memory_feature_probe': 'dcee4c7034c2dbc08edda5c05e93318cdc66b4c98eedbb88e3b7822707eff6e0',
              'memory_continuous_probe': '4ce9aa3f21ef95f589fb996f2a43c30a917e038f3fc4e1285bd60769691288ce'}
PRIOR_REPORTS = {RESEARCH / 'reviews/DG-019/zero-update.json': '5da3ffa91a74d5d5bfbefb0a7571adb7e4366b46d8e5e24083814466e697b533',
                 RESEARCH / 'reviews/DG-019/raw-audit.json': '25dd5b98a0a1bc40a2ff4dcf786964b0dd5e4b016935a3677c502799b1e291fa',
                 RESEARCH / 'reviews/DG-020/native-decisions.json': 'f259c416bc25edaf852b085648c9331d720f0675ed35de3114890805a89a665a',
                 RESEARCH / 'reviews/DG-021/zero-update.json': 'ca508eaa66b3eaff19987b9cae07674c61078017cd5782f8d0f4b9f304a1c9a1'}
PANEL = {'uncertainty': [('role_swap', 'en'), ('role_swap', 'zh'), ('empty', 'en'), ('empty', 'zh')],
         'normal': [('ordinary_source', 'en'), ('ordinary_source', 'zh'), ('ordinary_empty', 'en'), ('ordinary_empty', 'zh')],
         'supported': [('correct', 'en'), ('correct', 'zh'), ('value_swap', 'en'), ('value_swap', 'zh')]}
EXPECTED_POSITIONS = {'role_swap/en': 13, 'role_swap/zh': 12, 'empty/en': 13, 'empty/zh': 12,
                      'ordinary_source/en': 8, 'ordinary_source/zh': 5, 'ordinary_empty/en': 8, 'ordinary_empty/zh': 5,
                      'correct/en': 12, 'correct/zh': 10, 'value_swap/en': 11, 'value_swap/zh': 9}
EXPECTED_STATE = {'uncertainty': 'insufficient', 'normal': 'no_memory_needed', 'supported': 'supported'}
FIXTURES = [('role_swap', 'en'), ('role_swap', 'zh')]
CONTINUOUS_CALLS = 0


def select(index, scenario, language):
    matches = [r for r in index.values() if r['world_id'] == 'jb-000' and r['relation_family'] == 'home_city' and
               r['language'] == language and r['scenario'] == scenario]
    if len(matches) != 1: raise ValueError(f'selection {scenario}/{language} is not unique')
    return matches[0]['id']


def runtime_stack(tokens, codec, runtime, binding, reader_digest, record_id, key):
    query, sources = encoder_texts(runtime); q = tokens.rows[text_key(query)]
    if codec.tokenizer.encode(query, True) != q.token_ids.tolist(): raise ValueError('query tokens changed')
    if sources:
        s = tokens.rows[text_key(sources[0])]
        if codec.tokenizer.encode(sources[0], True) != s.token_ids.tolist(): raise ValueError('source tokens changed')
        pieces = codec.decode_pieces(s.token_ids[1:].tolist()); allowed = payload_mask(runtime['episodes'][0], sources[0], pieces)
        ranges = source_alignment(runtime['episodes'][0], sources[0], pieces, allowed); raw = runtime['episodes'][0]['text'].encode()
        source = torch.from_numpy(s.features.copy())
    else:
        ranges = (); raw = b''; allowed = []; source = torch.empty(0, 2048)
    x = SpanFeatures(torch.from_numpy(q.features.copy()), source, torch.tensor(allowed, dtype=torch.bool), key,
                     hashlib.sha256(raw).hexdigest(), digest(runtime['context']))
    layout = ByteLayout(x, raw, ranges)
    snapshot = PayloadSnapshot(binding, reader_digest, 'native', record_id, 0, (FactPayload('source', raw),) if raw else ())
    return x, layout, snapshot, ReplyBinding('native', record_id, x.context_sha256)


def run(a):
    global CONTINUOUS_CALLS
    registration = RESEARCH / 'experiments/DG-022.json'; reg = json.loads(registration.read_text())
    for name, want in FROZEN_SHA.items():
        if sha_file(FROZEN / name) != want: raise ValueError(f'frozen binary identity changed: {name}')
    for path, want in PRIOR_REPORTS.items():
        if sha_file(path) != want: raise ValueError(f'prior report changed: {path.name}')
    dg019 = json.loads((RESEARCH / 'reviews/DG-019/zero-update.json').read_text())
    dg020 = json.loads((RESEARCH / 'reviews/DG-020/native-decisions.json').read_text())
    dg021 = json.loads((RESEARCH / 'reviews/DG-021/zero-update.json').read_text())
    for name, h in {**dg019['source_sha256'], **dg020['source_sha256'], **dg021['source_sha256']}.items():
        if sha_file(ROOT / 'python' / name) != h: raise ValueError('prior implementation changed')
    root = Path(a.raw); root.mkdir(parents=True, exist_ok=False)
    feature_root = ROOT / 'build/neural-memory-cg003-full-features'; m = json.loads((feature_root / 'manifest.json').read_text())
    if digest(m) != FEATURE_DIGEST or sha_file(feature_root / 'output-head.npy') != m['output_head_sha256']: raise ValueError('frozen feature/head identity changed')
    for name, h in m['source_sha256'].items():
        if sha_file(ROOT / 'python' / name) != h: raise ValueError('feature package implementation changed')
    head = torch.from_numpy(np.load(feature_root / 'output-head.npy', allow_pickle=False)); scale = m['logit_scale']
    corpus = RESEARCH / 'data/JB-001'; manifest = json.loads((corpus / 'manifest.json').read_text())
    if digest(manifest) != DATA_DIGEST: raise ValueError('corpus changed')
    data = {}
    for name in ('inputs', 'index', 'labels'):
        path = corpus / f'train.{name}.jsonl'
        if sha_file(path) != manifest['file_sha256'][path.name]: raise ValueError(f'training {name} changed')
        data[name] = {r['id']: r for r in map(json.loads, path.read_text().splitlines())}
    binding = ModelBinding(BACKBONE_SHA256, m['tokenizer_sha256'], m['encoder_identity']['encoder_id'], sha_file(registration))
    key = digest(vars(binding))
    torch.set_num_threads(4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(reg['initialization']['reader_seed']); reader = FineSpanReader(2048, 1024, key, width=reg['initialization']['width']).eval()
        torch.manual_seed(reg['initialization']['uncertainty_seed']); branch = QueryOnlyUncertainty(1024, key).eval()
    reader_digest = tensor_digest(reader.state_dict()); initial = tensor_digest(branch.state_dict())
    if reader_digest != dg020['initial_parameter_digest']: raise ValueError('reader initialization changed')
    if initial != dg021['uncertainty_parameter_digest']: raise ValueError('uncertainty initialization changed')
    fixture = copy.deepcopy(branch)
    with torch.no_grad(): fixture.output.weight.copy_(torch.eye(1024) * .001)
    fixture_digest = tensor_digest(fixture.state_dict())
    gguf = ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf'
    codec = AppendValueCodec(FROZEN / 'tok_probe', gguf, m['tokenizer_sha256'])
    tokens = NativeFeatureBank(feature_root / 'tokens', expected_encoder_id=m['encoder_identity']['encoder_id'],
                               expected_manifest_sha256=m['token_manifest_digest'])
    eos = codec.tokenizer.eos()
    compiled = {}; supervised = []
    try:
        for group, pairs in PANEL.items():
            for scenario, language in pairs:
                rid = select(data['index'], scenario, language)
                if data['labels'][rid]['state'] != EXPECTED_STATE[group]: raise ValueError(f'{scenario}/{language} is not a {group} record')
                t = compile_teacher(data['inputs'][rid], data['labels'][rid], data['index'][rid], codec, binding, reader_digest,
                                    purpose='training_diagnostic')
                if t.prompt_ids != tuple(encode_generation_input(data['inputs'][rid], codec.tokenizer).prompt_token_ids):
                    raise ValueError('teacher prompt is not the natural prompt')
                if len(t.completion_ids) != EXPECTED_POSITIONS[f'{scenario}/{language}']: raise ValueError('pinned trajectory length changed')
                compiled[('uncertainty' if group=='uncertainty' else group, scenario, language)] = (rid, t)
        prefixes = {}
        for (group, scenario, language), (_, t) in compiled.items():
            if group == 'supported': continue
            positions = range(len(t.completion_ids)) if group == 'uncertainty' else sorted(t.reference_positions())
            for i in positions: prefixes.setdefault(t.prefix(i, purpose='training_diagnostic'), len(prefixes))
        bank = {}; batches = []
        ids = list(prefixes)
        for start in range(0, len(ids), 16):
            group = ids[start:start + 16]; folder = root / 'prefixes' / f'batch-{start // 16:03d}'
            h, z = extract_prefix_batch(FROZEN / 'memory_prefix_probe', gguf, group, folder)
            for p, hh, zz in zip(group, h, z): bank[p] = (torch.from_numpy(hh.copy()), torch.from_numpy(zz.copy()))
            batches.append({'folder': str(folder.relative_to(root)), 'prefixes': [list(p) for p in group],
                            'output_sha256': sha_file(folder / 'output.bin')})
            print(json.dumps({'phase': 'C_features', 'completed': len(bank), 'total': len(ids)}), flush=True)
        refdir = root / 'reference'; refdir.mkdir(); reference_rows = []
        checks = [(t, i) for (g, _s, _l), (_, t) in compiled.items() if g == 'uncertainty' for i in t.reference_positions()] + \
                 [(t, 0) for (g, _s, _l), (_, t) in compiled.items() if g == 'normal']
        for j, (t, i) in enumerate(checks):
            p = t.prefix(i, purpose='training_diagnostic'); result = native_forward(str(FROZEN / 'memory_gradient_reference'), str(gguf), p,
                                                                                    np.zeros(1024, dtype='<f4'), refdir, f'row-{j:03d}', False)
            h, z = bank[p]
            if not exact(h.numpy(), result['hidden']) or not exact(z.numpy(), result['logits']): raise ValueError('C batch/reference mismatch')
            reference_rows.append({'record': t.record_id, 'position': i, 'bitwise_equal': True})
            print(json.dumps({'phase': 'reference', 'completed': j + 1, 'total': len(checks)}), flush=True)
        uncertainty_rows = []
        for scenario, language in PANEL['uncertainty']:
            rid, t = compiled[('uncertainty', scenario, language)]
            x, layout, snapshot, reply = runtime_stack(tokens, codec, data['inputs'][rid], binding, reader_digest, rid, key)
            core = AutonomousValueController(reader, x, layout, snapshot, reply, codec, t.prompt_ids)
            controller = ResearchUncertaintyController(core, branch, head, scale, diagnostic_only=True)
            p0 = t.prefix(0, purpose='training_diagnostic')
            decision = controller.propose(LiveFrame(p0, *bank[p0], key, FrameOrigin.NATIVE_FRESH))
            if type(decision) is not UncertaintyDecision or controller.pending_output is None:
                raise ValueError('actual neural handoff did not occur')
            frames = tuple(LiveFrame(p, *bank[p], key, FrameOrigin.NATIVE_FRESH)
                           for p in (t.prefix(i, purpose='training_diagnostic') for i in range(len(t.completion_ids))))
            trajectory = UncertaintyTrajectory(rid, 2, t.prompt_ids, t.completion_ids, eos, 64, frames)
            supervised.append(trajectory)
            labels_path = corpus / 'train.labels.jsonl'
            if sha_file(labels_path) != manifest['file_sha256'][labels_path.name]: raise ValueError('training labels changed')
            if data['labels'][rid]['input_sha256'] != digest(data['inputs'][rid]): raise ValueError('label binding changed')
            with torch.no_grad():
                bitwise = all(torch.equal(branch(f, f.prefix_ids, head, scale).logits.view(torch.int32),
                                          f.logits.view(torch.int32)) for f in frames)
                bypass = disabled_identity(trajectory, branch, head, scale)
                losses = trajectory_losses(branch, trajectory, head, scale)
            uncertainty_rows.append({'id': rid, 'scenario': f'{scenario}/{language}', 'positions': len(t.completion_ids),
                                     'prompt_tokens': len(t.prompt_ids), 'eos_final': t.completion_ids[-1] == eos,
                                     'handoff_anchor': True, 'zero_residual_bitwise_base': bitwise, 'disabled_bypass': bypass,
                                     'per_token_loss': [float(x) for x in losses],
                                     'loss_sum': float(torch.stack(losses).sum()), 'committed_tokens': 0})
            print(json.dumps({'phase': 'uncertainty_trajectory', 'record': rid, 'positions': len(t.completion_ids)}), flush=True)
        branch.zero_grad(set_to_none=True)
        mean, sums = aggregate_loss(supervised, branch, head, scale); mean.backward()
        structure = gradient_structure(branch)
        if not structure['expected_zero_init_chain_rule']: raise ValueError('unexpected zero-initialization trajectory gradients')
        aggregate_row = {'records': len(supervised), 'positions': sum(len(t.target_ids) for t in supervised),
                         'mean_loss': float(mean.detach()), 'per_record_loss': [float(s.detach()) for s in sums],
                         **structure, 'chain_rule_note': 'encoder gradients are exactly zero at Wout=0 by the chain rule, not frozen learning'}
        zero_parity = []
        for scenario, language in PANEL['uncertainty']:
            if (scenario, language) in FIXTURES: continue
            rid, t = compiled[('uncertainty', scenario, language)]; directory = root / rid; directory.mkdir()
            forward(str(FROZEN / 'memory_continuous_probe'), str(gguf), t.prompt_ids, np.zeros(1024, dtype='<f4'), directory, 'zero', True)
            CONTINUOUS_CALLS += 1
            hh, bb, cc, zz = continuous_file(directory, 'zero', t.prompt_ids, True, torch.zeros(1, 1024))
            h0, z0 = bank[t.prompt_ids]
            if not exact(h0.numpy(), hh[0].numpy()) or not exact(z0.numpy(), bb[0].numpy()) or not exact(bb[0].numpy(), zz[0].numpy()) \
                    or torch.count_nonzero(cc): raise ValueError('zero trajectory branch is not bitwise base')
            zero_parity.append({'id': rid, 'position': 0, 'native_zero_bitwise_base': True})
        fixture_rows = []
        for scenario, language in FIXTURES:
            rid, t = compiled[('uncertainty', scenario, language)]; directory = root / rid; directory.mkdir(parents=True, exist_ok=True)
            refs = t.reference_positions()
            frames = [LiveFrame(t.prefix(i, purpose='training_diagnostic'), *bank[t.prefix(i, purpose='training_diagnostic')],
                                key, FrameOrigin.NATIVE_FRESH) for i in refs]
            outputs = [fixture(f, f.prefix_ids, head, scale) for f in frames]
            targets = [t.completion_ids[i] for i in refs]
            partial = torch.stack([F.cross_entropy(o.logits[None], torch.tensor([target]))
                                   for o, target in zip(outputs, targets)]).sum()
            grads = torch.autograd.grad(partial, [o.residual for o in outputs], retain_graph=True)
            norm = float(torch.sqrt(torch.stack([g.norm() ** 2 for g in grads]).sum()))
            fixture.zero_grad(set_to_none=True); partial.backward()
            if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum()) > 0
                       for p in fixture.parameters()): raise ValueError('fixture layers disconnected')
            fd = reg['fixture_finite_difference']; epsilon = fd['epsilon']; values = {}; max_error = 0.
            for name in ('nonzero', 'disabled', 'plus', 'minus'):
                total = 0.
                for j, (o, g, f, target) in enumerate(zip(outputs, grads, frames, targets)):
                    if name == 'plus': residual = o.residual.detach() + epsilon * g / norm
                    elif name == 'minus': residual = o.residual.detach() - epsilon * g / norm
                    else: residual = o.residual.detach()
                    forward(str(FROZEN / 'memory_continuous_probe'), str(gguf), f.prefix_ids, residual.numpy().astype('<f4'),
                            directory, f'{name}-{j}', name != 'disabled')
                    CONTINUOUS_CALLS += 1
                    if name != 'disabled':
                        _nh, nb, _nc, nz = continuous_file(directory, f'{name}-{j}', f.prefix_ids, True, residual[None])
                    else:
                        raw = (directory / (f'{name}-{j}.bin')).read_bytes(); arr = np.frombuffer(raw, dtype='<f4', offset=20).copy()
                        nb = torch.from_numpy(arr[1024:1024 + 73448])[None]; nc = torch.from_numpy(arr[1024 + 73448:1024 + 2 * 73448])[None]
                        nz = torch.from_numpy(arr[1024 + 2 * 73448:])[None]
                    if not exact(nb[0].numpy(), f.logits.numpy()): raise ValueError('fixture path changed base logits')
                    if name == 'disabled' and (not exact(nz[0].numpy(), f.logits.numpy()) or torch.count_nonzero(nc)):
                        raise ValueError('disabled fixture path changed base')
                    if name == 'nonzero': max_error = max(max_error, close(nz, o.logits.detach()[None]))
                    if name in ('plus', 'minus'): total += float(F.cross_entropy(nz.double(), torch.tensor([target])))
                if name in ('plus', 'minus'): values[name] = total
            derivative = (values['plus'] - values['minus']) / (2 * epsilon)
            if derivative <= 0 or abs(derivative - norm) > fd['atol'] + fd['rtol'] * abs(norm): raise ValueError('native trajectory finite difference failed')
            fixture_rows.append({'id': rid, 'origin': 'artificial_nonzero_fixture', 'reference_positions': list(refs),
                                 'objective': 'partial teacher-target CE; calculus fixture on the supervision objective, not answer accuracy',
                                 'partial_loss': float(partial.detach()), 'logit_max_error': max_error,
                                 'derivative_native': derivative, 'derivative_autograd': norm, 'passed': True})
            print(json.dumps({'phase': 'fixture', 'record': rid, 'derivative': derivative, 'autograd': norm}), flush=True)
        normal_rows = []
        for scenario, language in PANEL['normal']:
            rid, t = compiled[('normal', scenario, language)]
            x, layout, _snapshot, _reply = runtime_stack(tokens, codec, data['inputs'][rid], binding, reader_digest, rid, key)
            p0 = t.prefix(0, purpose='training_diagnostic'); h0, _z0 = bank[p0]
            with torch.no_grad():
                prediction = reader.predict(reader(x, layout, PrefixFeature(p0, h0, key), p0))
            bypass = True
            for i in t.reference_positions():
                p = t.prefix(i, purpose='training_diagnostic'); f = LiveFrame(p, *bank[p], key, FrameOrigin.NATIVE_FRESH)
                with torch.no_grad():
                    enabled = branch(f, p, head, scale); disabled = branch(f, p, head, scale, enabled=False)
                bypass = bypass and torch.equal(enabled.logits.view(torch.int32), f.logits.view(torch.int32)) and \
                    torch.equal(disabled.logits.view(torch.int32), f.logits.view(torch.int32))
            normal_rows.append({'id': rid, 'scenario': f'{scenario}/{language}', 'reader_route_prediction': prediction['route'],
                                'kept_wrong_route': prediction['route'] == 2, 'base_bypass_all_reference_frames': bypass,
                                'supervised_by_uncertainty_branch': False})
            print(json.dumps({'phase': 'normal_bypass', 'record': rid, 'route': prediction['route']}), flush=True)
        supported_rows = []
        for scenario, language in PANEL['supported']:
            rid, t = compiled[('supported', scenario, language)]
            if t.static_targets.route != 1 or t.phases.count('START') != 1 or t.phases.count('END') != 1:
                raise ValueError('supported trajectory structure changed')
            supported_rows.append({'id': rid, 'scenario': f'{scenario}/{language}', 'start_end_structure': True,
                                   'supervised_by_uncertainty_branch': False})
        codec.verify_identity()
    finally:
        codec.close()
    for path, want in PRIOR_REPORTS.items():
        if sha_file(path) != want: raise ValueError('prior report changed during run')
    if tensor_digest(reader.state_dict()) != reader_digest or tensor_digest(branch.state_dict()) != initial \
            or tensor_digest(fixture.state_dict()) != fixture_digest: raise ValueError('parameters changed')
    supervised_ids = {t.record_id for t in supervised}
    if len(supervised_ids) != 4 or any(row['id'] in supervised_ids for row in normal_rows + supported_rows):
        raise ValueError('uncertainty supervision boundary violated')
    distinct = sorted({(t.prompt_ids, t.target_ids) for t in supervised})
    members = [[t.record_id for t in supervised if (t.prompt_ids, t.target_ids) == g] for g in distinct]
    result = {'format': 'dg022-uncertainty-reply-supervision-check-v1', 'registration_sha256': sha_file(registration),
              'environment': {'frozen_binary_sha256': FROZEN_SHA, 'live_build_note': reg['environment']['reason']},
              'source_sha256': {n: sha_file(ROOT / 'python' / n) for n in ('uncertainty_reply_supervision.py', 'check_uncertainty_reply_supervision.py')},
              'raw_root': str(root.resolve()), 'raw_sha256': {str(p.relative_to(root)): sha_file(p) for p in sorted(root.rglob('*')) if p.is_file()},
              'uncertainty_trajectories': uncertainty_rows, 'aggregate_gradient': aggregate_row, 'zero_parity': zero_parity,
              'distinct_supervision_trajectories': {'groups': len(distinct), 'group_members': members,
                'note': 'the panel deliberately pairs scenarios over one question text (role_swap/empty and ordinary_source/ordinary_empty differ only in stored episodes); '
                        'the query-only branch never reads the source side, so its supervision signal has fewer distinct trajectories than records'},
              'nonzero_fixtures': fixture_rows, 'normal_bypass': normal_rows, 'copy_exclusion': supported_rows,
              'batches': batches, 'reference_checks': reference_rows,
              'C_forward_calls': {'prefix': len(prefixes), 'reference': len(reference_rows), 'continuous': CONTINUOUS_CALLS,
                                  'total': len(prefixes) + len(reference_rows) + CONTINUOUS_CALLS},
              'optimizer_steps': 0, 'parameters_unchanged': True, 'uncertainty_parameter_digest': initial,
              'fixture_parameter_digest': fixture_digest, 'full_free_replies_generated': 0, 'committed_tokens': 0,
              'semantic_accuracy_measured': False, 'cross_prefix_kv_reuse': False, 'internal_prefill_kv_buffers': True,
              'passed': len(uncertainty_rows) == 4 and len(fixture_rows) == 2 and len(normal_rows) == 4 and len(supported_rows) == 4,
              'V2_complete': False, 'deployment_approved': False}
    path = Path(a.output); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--raw', required=True); p.add_argument('--output', required=True)
    run(p.parse_args())
