"""V3-004 local trainer (supervision-defect fix): supervision-completion arm, 50 steps, cached C features.

Reader (static+idle post-forward losses) and query-only uncertainty branch
(per-token reply CE) on 912 compiled JB-001+JB-002 train trajectories.
Frozen backbone; labels only enter post-forward losses; deterministic."""
import json, sys, time
from pathlib import Path
import torch
from torch.nn import functional as F

sys.path.insert(0, 'python')
from append_value_transport import AppendValueCodec
from check_joint_span_interface import source_alignment
from compile_time_scoped_teacher import compile_teacher_time
from episode_memory_inputs import encoder_texts
from fine_span_reader import FineSpanReader, ByteLayout
from fine_span_supervision import fine_loss
from joint_optimizer import tensor_digest
from joint_span_reader import SpanFeatures, PrefixFeature
from native_memory_encoder import NativeFeatureBank, BACKBONE_SHA256, digest, sha_file, text_key
from native_token_payload import payload_mask
from neural_memory_contract import ModelBinding
from query_only_uncertainty import QueryOnlyUncertainty
from autonomous_value_controller import LiveFrame, FrameOrigin
from uncertainty_reply_supervision import UncertaintyTrajectory
import hashlib

ROOT = Path('.'); RESEARCH = ROOT / 'training/memory/neural-system'
FROZEN = ROOT / 'build/neural-memory-cg003-frozen-build'
PKG = ROOT / 'build/neural-memory-v3-features'
REG = RESEARCH / 'experiments/V3-004-proposal.json'

EXCLUDES=[]
def build_items(codec, binding, reader_digest, reader):
    manifest = json.loads((PKG / 'manifest.json').read_text())
    hidden = torch.from_numpy(__import__('numpy').load(PKG / 'prefix_hidden.npy'))
    logits = torch.from_numpy(__import__('numpy').load(PKG / 'prefix_logits.npy'))
    pindex = {tuple(p): i for i, p in enumerate(manifest['prefix_index'])}
    tokens = NativeFeatureBank(PKG / 'tokens', expected_encoder_id=manifest['encoder_id'],
                               expected_manifest_sha256=manifest['bank_manifest_sha256'])
    fm = json.loads((ROOT / 'build/neural-memory-cg003-full-features/manifest.json').read_text())
    head = torch.from_numpy(__import__('numpy').load(ROOT / 'build/neural-memory-cg003-full-features/output-head.npy'))
    scale = fm['logit_scale']; eos = codec.tokenizer.eos()
    key = digest(vars(binding)); items = []
    for cid in ('JB-001', 'JB-002'):
        c = RESEARCH / 'data' / cid
        cm = json.loads((c / 'manifest.json').read_text())
        for n in ('inputs', 'labels', 'index'):
            if sha_file(c / f'train.{n}.jsonl') != cm['file_sha256'][f'train.{n}.jsonl']:
                raise ValueError(f'{cid} {n} changed')
        ins = {r['id']: r for r in map(json.loads, (c / 'train.inputs.jsonl').read_text().splitlines())}
        lab = {r['id']: r for r in map(json.loads, (c / 'train.labels.jsonl').read_text().splitlines())}
        idx = {r['id']: r for r in map(json.loads, (c / 'train.index.jsonl').read_text().splitlines())}
        for meta in idx.values():
            rid = meta['id']
            try:
                t = compile_teacher_time(ins[rid], lab[rid], meta, codec, binding, reader_digest,
                                         purpose='training_diagnostic')
            except Exception:
                continue
            runtime = ins[rid]; query, sources = encoder_texts(runtime)
            q = tokens.rows[text_key(query)]
            if sources:
                s = tokens.rows[text_key(sources[0])]
                pieces = codec.decode_pieces(s.token_ids[1:].tolist())
                allowed = payload_mask(runtime['episodes'][0], sources[0], pieces)
                ranges = source_alignment(runtime['episodes'][0], sources[0], pieces, allowed)
                raw = runtime['episodes'][0]['text'].encode(); source = torch.from_numpy(s.features.copy())
            else:
                ranges = (); raw = b''; allowed = []; source = torch.empty(0, 2048)
            x = SpanFeatures(torch.from_numpy(q.features.copy()), source, torch.tensor(allowed, dtype=torch.bool),
                             key, hashlib.sha256(raw).hexdigest(), digest(runtime['context']))
            layout = ByteLayout(x, raw, ranges)
            try:
                reader(x, layout, PrefixFeature(t.prefix(0, purpose='training_diagnostic'),
                                                hidden[pindex[t.prefix(0, purpose='training_diagnostic')]].clone(), digest(vars(binding))),
                       t.prefix(0, purpose='training_diagnostic'))
            except Exception as e:
                EXCLUDES.append({'id': rid, 'corpus': cid, 'reason': type(e).__name__}); continue
            entry = {'id': rid, 'corpus': cid, 'traj': t, 'x': x, 'layout': layout,
                     'route': t.static_targets.route, 'key': key}
            if t.static_targets.route == 2:
                frames = tuple(LiveFrame(p, hidden[pindex[p]].clone(), logits[pindex[p]].clone(),
                                         key, FrameOrigin.NATIVE_FRESH)
                               for p in (t.prefix(i, purpose='training_diagnostic')
                                         for i in range(len(t.completion_ids))))
                entry['frames'] = frames
            else:
                entry['static_prefix'] = t.prefix(0, purpose='training_diagnostic')
                entry['static_hidden'] = hidden[pindex[entry['static_prefix']]].clone()
                entry['idle'] = [(i, lab2, t.prefix(i, purpose='training_diagnostic'),
                                  hidden[pindex[t.prefix(i, purpose='training_diagnostic')]].clone())
                                 for i, lab2 in enumerate(t.idle_targets) if lab2 is not None]
            items.append(entry)
    return items, head, scale, eos

def main():
    reg = json.loads(REG.read_text())
    if not reg.get('user_approval', {}).get('approved'): raise ValueError('V3-004 not approved')
    steps = reg['training']['steps_requested']; ckpt_every = 10
    out = ROOT / 'build/neural-memory-v3-004'; out.mkdir(parents=True, exist_ok=True)
    fm = json.loads((ROOT / 'build/neural-memory-cg003-full-features/manifest.json').read_text())
    codec = AppendValueCodec(FROZEN / 'tok_probe', ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf', fm['tokenizer_sha256'])
    binding = ModelBinding(BACKBONE_SHA256, fm['tokenizer_sha256'], fm['encoder_identity']['encoder_id'], sha_file(REG))
    torch.manual_seed(1018); reader = FineSpanReader(2048, 1024, digest(vars(binding)), width=64)
    torch.manual_seed(1021); branch = QueryOnlyUncertainty(1024, digest(vars(binding)))
    reader_digest = tensor_digest(reader.state_dict()); branch_digest = tensor_digest(branch.state_dict())
    items, head, scale, eos = build_items(codec, binding, reader_digest, reader)
    codec.close()
    print(json.dumps({'phase': 'items', 'count': len(items)}), flush=True)
    opt = torch.optim.Adam(list(reader.parameters()) + list(branch.parameters()), lr=1e-4)
    order = torch.randperm(len(items), generator=torch.Generator().manual_seed(0)).tolist()
    log = []
    cursor = 0
    for step in range(1, steps + 1):
        batch = []
        while len(batch) < 16:
            batch.append(items[order[cursor % len(order)]]); cursor += 1
        opt.zero_grad(set_to_none=True)
        losses = []
        for it in batch:
            if it['route'] == 2:
                traj = UncertaintyTrajectory(it['id'], 2, it['traj'].prompt_ids, it['traj'].completion_ids,
                                             eos, 64, it['frames'])
                per = [F.cross_entropy(branch(f, f.prefix_ids, head, scale).logits[None],
                                       torch.tensor([tok])) for f, tok in zip(traj.frames, traj.target_ids)]
                branch_loss = torch.stack(per).sum() / len(per)
                # V3-004 fix: route-2 items ALSO receive the reader static supervision
                # (DG-019/026 protocol) at their prefix-0 frame.
                p0 = it['traj'].prefix(0, purpose='training_diagnostic')
                out0 = reader(it['x'], it['layout'], PrefixFeature(p0, it['frames'][0].hidden, it['key']), p0)
                loss = branch_loss + fine_loss(out0, it['traj'].static_targets)['total']
            else:
                output = reader(it['x'], it['layout'], PrefixFeature(it['static_prefix'], it['static_hidden'], it['key']),
                                it['static_prefix'])
                static = fine_loss(output, it['traj'].static_targets)
                total = static['total']
                if it['idle']:
                    idle_n = len(it['idle'])
                    for i, lab2, p, h in it['idle']:
                        out2 = reader(it['x'], it['layout'], PrefixFeature(p, h, it['key']), p)
                        total = total + F.cross_entropy(out2.mode_logits[None], torch.tensor([lab2])) / idle_n
                loss = total
            (loss / len(batch)).backward(); losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(list(reader.parameters()) + list(branch.parameters()), 1.0)
        opt.step()
        log.append({'step': step, 'mean_loss': sum(losses) / len(losses)})
        print(json.dumps(log[-1]), flush=True)
        if step % ckpt_every == 0 or step == steps:
            torch.save({'step': step, 'reader': reader.state_dict(), 'branch': branch.state_dict(),
                        'reader_digest': reader_digest, 'branch_digest': branch_digest,
                        'binding': vars(binding)}, out / f'ckpt-{step:03d}.pt')
    report = {'format': 'v3-004-train-report-v1', 'registration_sha256': sha_file(REG),
              'items': len(items), 'steps': steps, 'batch': 16, 'lr': 1e-4,
              'initial_reader_digest': reader_digest, 'initial_branch_digest': branch_digest,
              'final_reader_digest': tensor_digest(reader.state_dict()),
              'final_branch_digest': tensor_digest(branch.state_dict()),
              'loss_log': log, 'excluded': EXCLUDES, 'deployed': False}
    with (out / 'train-report.json').open('x') as f: json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
