"""DG-026 JB-002 zero-update supervision check. No optimizer steps, no serving."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from append_value_transport import AppendValueCodec
from audit_joint_generation import continuous_file
from autonomous_value_controller import LiveFrame, FrameOrigin
from check_joint_span_interface import source_alignment
from compile_time_scoped_teacher import compile_teacher_time
from diagnose_native_generation_gradient import exact, native_forward
from episode_memory_inputs import encoder_texts
from fine_span_reader import FineSpanReader, ByteLayout
from fine_span_supervision import fine_loss
from joint_optimizer import tensor_digest
from joint_span_reader import SpanFeatures, PrefixFeature
from native_memory_encoder import NativeFeatureBank, BACKBONE_SHA256, digest, sha_file, text_key
from native_prefix_bank import extract_prefix_batch
from native_token_payload import payload_mask
from neural_memory_contract import ModelBinding
from neural_memory_generation import encode_generation_input
from query_only_uncertainty import QueryOnlyUncertainty
from uncertainty_reply_supervision import (UncertaintyTrajectory, aggregate_loss, disabled_identity,
                                           gradient_structure, trajectory_losses)
from value_transport import require

ROOT = Path(__file__).resolve().parents[1]; RESEARCH = ROOT / 'training/memory/neural-system'
FROZEN = ROOT / 'build/neural-memory-cg003-frozen-build'
FROZEN_SHA = {'tok_probe': 'f3d5fd147895e5875776a579d864f7a36ef16b55a4f59da00838f8b59f416adc',
              'memory_prefix_probe': '2735729604b1a52d99d1a60f48234b96ce3683d63aeec6434879bbfe7753e302',
              'memory_gradient_reference': '4c05a9c4faa7b61d03a017a3a80f43c36329c5d8af945ea5c9f1359640de9152',
              'memory_feature_probe': 'dcee4c7034c2dbc08edda5c05e93318cdc66b4c98eedbb88e3b7822707eff6e0',
              'memory_continuous_probe': '4ce9aa3f21ef95f589fb996f2a43c30a917e038f3fc4e1285bd60769691288ce'}
PRIOR_REPORTS = {RESEARCH / 'reviews/DG-019/zero-update.json': '5da3ffa91a74d5d5bfbefb0a7571adb7e4366b46d8e5e24083814466e697b533',
                 RESEARCH / 'reviews/DG-020/native-decisions.json': 'f259c416bc25edaf852b085648c9331d720f0675ed35de3114890805a89a665a',
                 RESEARCH / 'reviews/DG-021/zero-update.json': 'ca508eaa66b3eaff19987b9cae07674c61078017cd5782f8d0f4b9f304a1c9a1',
                 RESEARCH / 'reviews/DG-022/zero-update.json': '907d750d0df14ab568e1e9f37a338acd3d0c6995ecf8cad4eca27aa63823c511'}
JB002_MANIFEST = '0e5e1a06e7dff6ecc5e30fa62e15487430e121fe7c81146e3ae37a379f435c22'
JB002_BANK = 'e6bf0382b65bbf42b883232d118f9c9ac04052f555cf02bda7dafacbe9fbf07a'
WORLD = 'jb2-009'
UNCERTAIN = ('wrong_time', 'hypothetical', 'quoted')
SUPPORTED = ('historical_stated', 'multi_fact')
EXPECTED_POSITIONS = {'wrong_time': [16, 17, 16, 17], 'hypothetical': [13, 12, 13, 12],
                      'quoted': [13, 12, 13, 12], 'historical_stated': [17, 15, 17, 15],
                      'multi_fact': [12, 10, 12, 11]}


def select_panel(index):
    panel = {}
    for scenario in UNCERTAIN + SUPPORTED:
        items = [m for m in index.values() if m['world_id'] == WORLD and m['scenario'] == scenario]
        items.sort(key=lambda m: (m['relation_family'], m['language']))
        if len(items) != 4: raise ValueError(f'panel scenario {scenario} is not 4 records')
        panel[scenario] = items
    return panel


def run(a):
    registration = RESEARCH / 'experiments/DG-026.json'; reg = json.loads(registration.read_text())
    for name, want in FROZEN_SHA.items():
        if sha_file(FROZEN / name) != want: raise ValueError(f'frozen binary identity changed: {name}')
    for path, want in PRIOR_REPORTS.items():
        if sha_file(path) != want: raise ValueError(f'prior report changed: {path.name}')
    dg020 = json.loads((RESEARCH / 'reviews/DG-020/native-decisions.json').read_text())
    dg021 = json.loads((RESEARCH / 'reviews/DG-021/zero-update.json').read_text())
    corpus = RESEARCH / 'data/JB-002'
    manifest = json.loads((corpus / 'manifest.json').read_text())
    if sha_file(corpus / 'manifest.json') != JB002_MANIFEST: raise ValueError('JB-002 corpus changed')
    feature_root = ROOT / 'build/neural-memory-cg003-full-features'; m = json.loads((feature_root / 'manifest.json').read_text())
    head = torch.from_numpy(np.load(feature_root / 'output-head.npy', allow_pickle=False)); scale = m['logit_scale']
    data = {}
    for name in ('inputs', 'index', 'labels'):
        path = corpus / f'train.{name}.jsonl'
        if sha_file(path) != manifest['file_sha256'][path.name]: raise ValueError(f'JB-002 {name} changed')
        data[name] = {r['id']: r for r in map(json.loads, path.read_text().splitlines())}
    panel = select_panel(data['index'])
    root = Path(a.raw); root.mkdir(parents=True, exist_ok=False)
    bank_root = ROOT / 'build/neural-memory-jb002-features'
    tokens = NativeFeatureBank(bank_root, expected_encoder_id=m['encoder_identity']['encoder_id'],
                               expected_manifest_sha256=JB002_BANK)
    binding = ModelBinding(BACKBONE_SHA256, m['tokenizer_sha256'], m['encoder_identity']['encoder_id'], sha_file(registration))
    key = digest(vars(binding))
    torch.set_num_threads(4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1018); reader = FineSpanReader(2048, 1024, key, width=64).eval()
        torch.manual_seed(1021); branch = QueryOnlyUncertainty(1024, key).eval()
    reader_digest = tensor_digest(reader.state_dict()); initial = tensor_digest(branch.state_dict())
    if reader_digest != dg020['initial_parameter_digest']: raise ValueError('reader initialization changed')
    if initial != dg021['uncertainty_parameter_digest']: raise ValueError('uncertainty initialization changed')
    gguf = ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf'
    codec = AppendValueCodec(FROZEN / 'tok_probe', gguf, m['tokenizer_sha256'])
    eos = codec.tokenizer.eos()
    trajectories = {}
    for scenario, items in panel.items():
        for i, meta in enumerate(items):
            t = compile_teacher_time(data['inputs'][meta['id']], data['labels'][meta['id']], meta,
                                     codec, binding, reader_digest, purpose='training_diagnostic')
            if len(t.completion_ids) != EXPECTED_POSITIONS[scenario][i]:
                raise ValueError(f'pinned trajectory length changed: {scenario} #{i}')
            trajectories[meta['id']] = t
    prefixes = {}; refs = []
    uncertain_ids = {m['id'] for s in UNCERTAIN for m in panel[s]}
    for rid, t in trajectories.items():
        if rid in uncertain_ids:
            positions = set(range(len(t.completion_ids)))
        else:
            positions = {0} | {i for i, lab in enumerate(t.idle_targets) if lab is not None}
        for i in positions: prefixes.setdefault(t.prefix(i, purpose='training_diagnostic'), len(prefixes))
        refs.extend((rid, i) for i in t.reference_positions() if i in positions)
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
    for j, (rid, i) in enumerate(refs):
        t = trajectories[rid]; p = t.prefix(i, purpose='training_diagnostic')
        result = native_forward(str(FROZEN / 'memory_gradient_reference'), str(gguf), p,
                                np.zeros(1024, dtype='<f4'), refdir, f'row-{j:03d}', False)
        h, z = bank[p]
        if not exact(h.numpy(), result['hidden']) or not exact(z.numpy(), result['logits']):
            raise ValueError('C batch/reference mismatch')
        reference_rows.append({'record': rid, 'position': i, 'bitwise_equal': True})
        print(json.dumps({'phase': 'reference', 'completed': j + 1, 'total': len(refs)}), flush=True)

    def stack_for(meta):
        runtime = data['inputs'][meta['id']]
        query, sources = encoder_texts(runtime); q = tokens.rows[text_key(query)]
        if sources:
            s = tokens.rows[text_key(sources[0])]
            pieces = codec.decode_pieces(s.token_ids[1:].tolist()); allowed = payload_mask(runtime['episodes'][0], sources[0], pieces)
            ranges = source_alignment(runtime['episodes'][0], sources[0], pieces, allowed); raw = runtime['episodes'][0]['text'].encode()
            source = torch.from_numpy(s.features.copy())
        else:
            ranges = (); raw = b''; allowed = []; source = torch.empty(0, 2048)
        x = SpanFeatures(torch.from_numpy(q.features.copy()), source, torch.tensor(allowed, dtype=torch.bool), key,
                         hashlib.sha256(raw).hexdigest(), digest(runtime['context']))
        return x, ByteLayout(x, raw, ranges)

    uncertainty_rows = []; supervised = []
    labels_path = corpus / 'train.labels.jsonl'
    for scenario in UNCERTAIN:
        for meta in panel[scenario]:
            rid = meta['id']; t = trajectories[rid]
            if sha_file(labels_path) != manifest['file_sha256'][labels_path.name]: raise ValueError('labels changed')
            if data['labels'][rid]['input_sha256'] != digest(data['inputs'][rid]): raise ValueError('label binding changed')
            frames = tuple(LiveFrame(p, *bank[p], key, FrameOrigin.NATIVE_FRESH)
                           for p in (t.prefix(i, purpose='training_diagnostic') for i in range(len(t.completion_ids))))
            traj = UncertaintyTrajectory(rid, 2, t.prompt_ids, t.completion_ids, eos, 64, frames)
            supervised.append(traj)
            with torch.no_grad():
                bitwise = all(torch.equal(branch(f, f.prefix_ids, head, scale).logits.view(torch.int32),
                                          f.logits.view(torch.int32)) for f in frames)
                bypass = disabled_identity(traj, branch, head, scale)
                losses = trajectory_losses(branch, traj, head, scale)
            uncertainty_rows.append({'id': rid, 'scenario': scenario, 'positions': len(t.completion_ids),
                                     'eos_final': t.completion_ids[-1] == eos, 'zero_residual_bitwise': bitwise,
                                     'disabled_bypass': bypass, 'per_token_loss': [float(x) for x in losses],
                                     'loss_sum': float(torch.stack(losses).sum())})
            print(json.dumps({'phase': 'uncertainty', 'record': rid, 'positions': len(t.completion_ids)}), flush=True)
    branch.zero_grad(set_to_none=True)
    mean, sums = aggregate_loss(supervised, branch, head, scale); mean.backward()
    structure = gradient_structure(branch)
    if not structure['expected_zero_init_chain_rule']: raise ValueError('zero-init chain rule violated')
    supported_rows = []
    for scenario in SUPPORTED:
        for meta in panel[scenario]:
            rid = meta['id']; t = trajectories[rid]
            x, layout = stack_for(meta)
            reader.zero_grad(set_to_none=True)
            p0 = t.prefix(0, purpose='training_diagnostic')
            output = reader(x, layout, PrefixFeature(p0, bank[p0][0], key), p0)
            static = fine_loss(output, t.static_targets); static['total'].backward()
            idle_count = sum(v is not None for v in t.idle_targets); mode_loss = 0.
            for i, label in enumerate(t.idle_targets):
                if label is None: continue
                p = t.prefix(i, purpose='training_diagnostic')
                out = reader(x, layout, PrefixFeature(p, bank[p][0], key), p)
                loss = torch.nn.functional.cross_entropy(out.mode_logits[None], torch.tensor([label])) / idle_count
                mode_loss += float(loss.detach()); loss.backward()
            grads = {n: None if p.grad is None else float(p.grad.abs().sum()) for n, p in reader.named_parameters()}
            if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in reader.parameters()) \
                    or sum(v or 0 for v in grads.values()) <= 0:
                raise ValueError('invalid reader gradients')
            if not all(grads[n] is not None and grads[n] > 0 for n in
                       ('offset.weight', 'boundaries.0.weight', 'boundaries.2.weight', 'fact_value.weight',
                        'value.weight', 'value_query.weight', 'mode.0.weight')):
                raise ValueError('positive heads disconnected')
            if tensor_digest(reader.state_dict()) != reader_digest: raise ValueError('reader parameters updated')
            supported_rows.append({'id': rid, 'scenario': scenario, 'positions': len(t.completion_ids),
                                   'route_target': t.static_targets.route, 'idle_positions': idle_count,
                                   'static_losses': {k: float(v.detach()) for k, v in static.items()},
                                   'mean_idle_mode_loss': mode_loss, 'gradient_gate': True})
            print(json.dumps({'phase': 'supported', 'record': rid, 'positions': len(t.completion_ids)}), flush=True)
    codec.verify_identity(); codec.close()
    if tensor_digest(reader.state_dict()) != reader_digest or tensor_digest(branch.state_dict()) != initial:
        raise ValueError('parameters changed')
    result = {'format': 'dg026-jb002-supervision-check-v1', 'registration_sha256': sha_file(registration),
              'environment': {'frozen_binary_sha256': FROZEN_SHA, 'bank_manifest_sha256': JB002_BANK},
              'raw_root': str(root.resolve()), 'raw_sha256': {str(p.relative_to(root)): sha_file(p) for p in sorted(root.rglob('*')) if p.is_file()},
              'uncertainty_trajectories': uncertainty_rows,
              'aggregate_gradient': {'records': len(supervised),
                                     'positions': sum(len(t.target_ids) for t in supervised),
                                     'mean_loss': float(mean.detach()),
                                     'per_record_loss': [float(s.detach()) for s in sums], **structure},
              'supported_trajectories': supported_rows, 'batches': batches, 'reference_checks': reference_rows,
              'C_forward_calls': {'prefix': len(prefixes), 'reference': len(reference_rows)},
              'optimizer_steps': 0, 'parameters_unchanged': True, 'semantic_accuracy_measured': False,
              'passed': len(uncertainty_rows) == 12 and len(supported_rows) == 8,
              'V2_complete': False, 'deployment_approved': False}
    path = Path(a.output); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--raw', required=True); p.add_argument('--output', required=True)
    run(p.parse_args())
