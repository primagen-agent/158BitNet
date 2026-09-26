"""EC-007 (part 2): retrain the route READOUT (Linear 131->3) on the mixed
corpora (declarative + dialogue + 3B paraphrase), two-stage recipe: trunk
frozen from v3-013, full-batch LBFGS on the readout — the V3-009 method
that originally established routing.

Gate (registered): world-level mixed holdout overall accuracy >= 0.82, and
the dialogue/paraphrase forms must not collapse to a single class.
"""
import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from append_value_transport import AppendValueCodec
from check_joint_span_interface import source_alignment
from fine_span_reader import ByteLayout
from joint_span_reader import candidate_spans
from native_memory_encoder import BACKBONE_SHA256, digest
from native_token_payload import payload_mask
from neural_memory_contract import ModelBinding
from train_episode_relevance import live_extract, split_worlds
from v3_model_factory import make_struct_route_reader, STRUCT_MU, STRUCT_SD

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'build/neural-memory-cg003-frozen-build'
GGUF = ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf'
CKPT = ROOT / 'build/neural-memory-v3-013/ckpt-440.pt'


def load_all_records(codec, binding):
    """All-route records from the five corpora (teacher-routed per corpus)."""
    from compile_time_scoped_teacher import compile_teacher_time
    from compile_dialogue_teacher import compile_teacher_dialogue
    from episode_memory_inputs import encoder_texts
    from train_episode_relevance import load_split
    out = []
    for cid in ('JB-001', 'JB-002', 'JB-003', 'JB-004', 'JB-005'):
        ins, lab, idx = load_split(cid, 'train')
        teacher = compile_teacher_dialogue if cid in ('JB-004', 'JB-005') \
            else compile_teacher_time
        for meta in idx.values():
            rid = meta['id']
            try:
                t = teacher(ins[rid], lab[rid], meta, codec, binding, '0' * 64,
                            purpose='training_diagnostic')
            except Exception:
                continue
            route = t.static_targets.route
            runtime = ins[rid]
            query, sources = encoder_texts(runtime)
            if not sources:
                continue
            out.append({'rid': rid, 'world': meta['world_id'], 'route': route,
                        'query': query, 'source': sources[0],
                        'corpus': cid})
    return out


def route_input(reader, q_feat, s_feat, pieces, episode_msg, source_framed):
    allowed = payload_mask(episode_msg, source_framed, pieces)
    ranges = source_alignment(episode_msg, source_framed, pieces, allowed)
    raw = episode_msg['text'].encode()

    class X:
        pass
    x = X()
    x.allowed = torch.tensor(allowed, dtype=torch.bool)
    x.source = s_feat
    import hashlib as hl
    x.payload_sha256 = hl.sha256(raw).hexdigest()
    layout = ByteLayout(x, raw, tuple(ranges))
    starts, ends = layout.endpoints()
    cands = candidate_spans(x.allowed)
    spans = tuple((s, e) for s, e in cands
                 if starts[s] and ends[e - 1] and starts[s][0] < ends[e - 1][-1])
    with torch.no_grad():
        qrows = reader.query(q_feat).tanh()
        q = (reader.query_pool(qrows).softmax(0) * qrows).sum(0)
        source = reader.source(s_feat).tanh()
        if spans:
            s = torch.tensor([a for a, b in spans])
            e = torch.tensor([b for a, b in spans])
            sums = torch.cat((source.new_zeros(1, 64), source.cumsum(0)))
            h = reader.span(torch.cat((source[s], source[e - 1],
                                       (sums[e] - sums[s]) / (e - s)[:, None]), -1)).tanh()
            qq = q.expand_as(h)
            fp = reader.fact(torch.cat((qq, h, qq * h, (qq - h).abs()), -1)).flatten().log_softmax(0)
            evidence = fp.exp() @ h
        else:
            evidence = torch.zeros_like(q)
        scal = (torch.tensor([float(len(s_feat) > 0), float(len(spans)),
                              float(int(sum(allowed))) / 32.0]) -
                torch.tensor(STRUCT_MU)) / torch.tensor(STRUCT_SD)
        return torch.cat((q, evidence, scal)).detach()


def main():
    import hashlib as hl
    fm = json.loads((ROOT / 'build/neural-memory-cg003-full-features/manifest.json').read_text())
    frozen_codec = AppendValueCodec(FROZEN / 'tok_probe', GGUF, fm['tokenizer_sha256'])
    live_codec = AppendValueCodec(ROOT / 'build/tok_probe', GGUF,
                                  _hl := hl.sha256((ROOT / 'build/tok_probe').read_bytes()).hexdigest())
    binding = ModelBinding(BACKBONE_SHA256, fm['tokenizer_sha256'],
                           fm['encoder_identity']['encoder_id'], '0' * 64)

    records = load_all_records(frozen_codec, binding)
    holdout = split_worlds(records)
    print(f'records: {len(records)} '
          f'(fit={sum(r["split"]=="fit" for r in records)}, '
          f'holdout={sum(r["split"]=="holdout" for r in records)})', flush=True)

    reader = make_struct_route_reader(digest(vars(binding)), width=64, seed=1018)
    ck = torch.load(CKPT, weights_only=False)
    reader.load_state_dict(ck['reader'])
    reader.eval()
    for p in reader.parameters():
        p.requires_grad = False

    items = []
    for r in records:
        q_feat = live_extract(r['query'])
        s_feat = live_extract(r['source'])
        episode_msg = json.loads(r['source'])
        ids = live_codec.tokenizer.encode(r['source'], True)
        pieces = live_codec.decode_pieces(ids[1:])
        try:
            inp = route_input(reader, q_feat, s_feat, pieces, episode_msg, r['source'])
        except Exception:
            continue
        items.append((inp, r['route'], r['split'], r['corpus']))
    print(f'route inputs: {len(items)}', flush=True)

    fit = [(x, y) for x, y, sp, _ in items if sp == 'fit']
    hold = [(x, y, sp, c) for x, y, sp, c in items if sp == 'holdout']

    torch.manual_seed(8192)
    route = nn.Linear(131, 3)
    X = torch.stack([x for x, _ in fit])
    Y = torch.tensor([y for _, y in fit])

    def closure():
        opt.zero_grad()
        logits = route(X)
        loss = nn.functional.cross_entropy(logits, Y)
        loss.backward()
        return loss

    opt = torch.optim.LBFGS(route.parameters(), lr=0.25, max_iter=300,
                            tolerance_grad=1e-7, tolerance_change=1e-9,
                            history_size=50, line_search_fn='strong_wolfe')
    opt.step(closure)

    route.eval()
    holdX = torch.stack([x for x, _, _, _ in hold])
    gold = [y for _, y, _, _ in hold]
    with torch.no_grad():
        pred = route(holdX).argmax(dim=1).tolist()
        # uniform re-score: the ORIGINAL deployed route head on the same holdout
        old_pred = reader.route(holdX).argmax(dim=1).tolist()
    overall = sum(int(p == g) for p, g in zip(pred, gold)) / len(gold)
    from collections import Counter
    per = Counter()
    for p, g in zip(pred, gold):
        per[(g, p)] += 1
    recalls = {g: sum(1 for p, gg in zip(pred, gold) if gg == g and p == g) /
               max(sum(1 for gg in gold if gg == g), 1) for g in (0, 1, 2)}
    print(f'holdout overall={overall:.3f} recalls={recalls}', flush=True)
    print('pred distribution:', Counter(pred), 'gold:', Counter(gold), flush=True)
    old_overall = sum(int(p == g) for p, g in zip(old_pred, gold)) / len(gold)
    print(f'ORIGINAL route head on same holdout: overall={old_overall:.3f} '
          f'pred={dict(Counter(old_pred))}', flush=True)

    gate = overall >= 0.82 and max(Counter(pred).values()) / len(pred) < 0.95
    print('GATE:', 'PASS' if gate else 'FAIL',
          '(>=0.82 overall, no single-class collapse)', flush=True)

    out_dir = ROOT / 'build/neural-memory-ec007-route'
    out_dir.mkdir(exist_ok=True)
    torch.save({'route.weight': route.weight.data, 'route.bias': route.bias.data,
                'holdout_overall': overall, 'recalls': recalls,
                'gate_pass': gate}, out_dir / 'route.pt')
    print(f'saved: {out_dir / "route.pt"}', flush=True)
    frozen_codec.close()
    live_codec.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
