"""EC-005: the learnable EC-002 joint-pair architecture on the expanded
JB-001+002+003 corpus, with the regularization recipe (weight decay, dropout,
100-step holdout early stopping) and same-world negative augmentation.

Registered in training/memory/neural-system/reviews/EC-005/PROCESS.md.
    f_ij = tanh(W1 [q_i; e_j; q_i*e_j; |q_i - e_j|] + b1)   (64-dim)
    agg  = max_ij f_ij (per-dim)
    logit= w tanh(W2 [drop([agg; q_mean; e_mean])] + b2)
"""
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from append_value_transport import AppendValueCodec
from native_memory_encoder import BACKBONE_SHA256, digest
from neural_memory_contract import ModelBinding
from train_episode_relevance import (build_pairs, collect_records, live_extract,
                                     split_worlds)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'build/neural-memory-cg003-frozen-build'
GGUF = ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf'


class JointPairScorer(nn.Module):
    def __init__(self, dim=2048, hidden=64):
        super().__init__()
        self.pair = nn.Linear(4 * dim, hidden)
        self.drop = nn.Dropout(0.1)
        self.mix = nn.Linear(2 * dim + hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, q_rows, e_rows):
        nq, ne = q_rows.shape[0], e_rows.shape[0]
        q = q_rows.unsqueeze(1).expand(nq, ne, q_rows.shape[1])
        e = e_rows.unsqueeze(0).expand(nq, ne, e_rows.shape[1])
        f = torch.tanh(self.pair(torch.cat((q, e, q * e, (q - e).abs()), -1)))
        agg = f.max(dim=0).values.max(dim=0).values
        means = torch.cat((q_rows.mean(0), e_rows.mean(0)))
        feat = self.drop(torch.cat((agg, means)))
        return self.out(torch.tanh(self.mix(feat))).squeeze(-1)


def main():
    fm = json.loads((ROOT / 'build/neural-memory-cg003-full-features/manifest.json').read_text())
    codec = AppendValueCodec(FROZEN / 'tok_probe', GGUF, fm['tokenizer_sha256'])
    binding = ModelBinding(BACKBONE_SHA256, fm['tokenizer_sha256'],
                           fm['encoder_identity']['encoder_id'], '0' * 64)
    records = collect_records(codec, binding,
                              corpora=('JB-001', 'JB-002', 'JB-003'))
    holdout_worlds = split_worlds(records)

    class TextTable:
        def __init__(self):
            self.texts, self.tid = [], {}

        def __call__(self, text):
            if text not in self.tid:
                self.tid[text] = len(self.texts)
                self.texts.append(text)
            return self.tid[text]

    table = TextTable()
    fit_pairs, fit_counts = build_pairs(records, 'fit', table)
    hold_pairs, hold_counts = build_pairs(records, 'holdout', table)
    texts = table.texts
    print(f'records: {len(records)} '
          f'(fit={sum(r["split"]=="fit" for r in records)}, '
          f'holdout={sum(r["split"]=="holdout" for r in records)}, '
          f'worlds holdout={len(holdout_worlds)})', flush=True)
    print(f'pairs fit={fit_counts} holdout={hold_counts}', flush=True)
    print(f'unique texts: {len(texts)}', flush=True)

    feats = {}
    for i, text in enumerate(texts):
        feats[i] = live_extract(text)
        if (i + 1) % 200 == 0:
            print(f'  extracted {i+1}/{len(texts)}', flush=True)

    # same-world negative augmentation (registered)
    rng_aug = random.Random(11)
    fit_records = [r for r in records if r['split'] == 'fit' and r['route'] == 1]
    by_world = {}
    for r in fit_records:
        by_world.setdefault((r['world'], r['lang']), []).append(r)
    extra = []
    for r in fit_records:
        same = [o for o in by_world.get((r['world'], r['lang']), []) if o is not r]
        if len(same) >= 2:
            for pick in rng_aug.sample(same, 2):
                extra.append((table(r['query']), table(pick['source']), 0))
    fit_pos = [p for p in fit_pairs if p[2] == 1]
    fit_neg = [p for p in fit_pairs if p[2] == 0] + extra
    hold_pos = [p for p in hold_pairs if p[2] == 1]
    hold_neg = [p for p in hold_pairs if p[2] == 0]
    print(f'train pos={len(fit_pos)} neg={len(fit_neg)} (extra {len(extra)}) '
          f'holdout pos={len(hold_pos)} neg={len(hold_neg)}', flush=True)

    torch.manual_seed(32768)
    model = JointPairScorer()
    print(f'params: {sum(p.numel() for p in model.parameters())}', flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = random.Random(42)
    best = {'top1': -1.0, 'state': None, 'step': 0, 'pos_rate': 0.0, 'neg_rej': 0.0}

    def evaluate(pairs_pos, pairs_neg):
        model.eval()
        with torch.no_grad():
            correct = 0
            for q, s, _ in pairs_pos:
                cands = [s]
                for _, ns, _ in rng.sample(pairs_neg, min(8, len(pairs_neg))):
                    if ns != s:
                        cands.append(ns)
                scores = [float(model(feats[q], feats[c])) for c in cands]
                correct += int(scores.index(max(scores)) == 0)
            pos_sc = [float(torch.sigmoid(model(feats[q], feats[s])))
                      for q, s, _ in pairs_pos]
            neg_sc = [float(torch.sigmoid(model(feats[q], feats[s])))
                      for q, s, _ in pairs_neg]
        model.train()
        return (correct / max(len(pairs_pos), 1),
                sum(v > 0.5 for v in pos_sc) / max(len(pos_sc), 1),
                sum(v <= 0.5 for v in neg_sc) / max(len(neg_sc), 1))

    print('\ntraining...', flush=True)
    for step in range(1, 2001):
        batch = rng.sample(fit_pos, min(8, len(fit_pos))) + \
                rng.sample(fit_neg, min(8, len(fit_neg)))
        logits = torch.stack([model(feats[q], feats[s]) for q, s, _ in batch])
        labels = torch.tensor([float(l) for _, _, l in batch])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            hold_top1, pos_rate, neg_rej = evaluate(hold_pos, hold_neg)
            if step % 200 == 0:
                fit_top1, _, _ = evaluate(fit_pos[:40], fit_neg)
                print(f'  step {step}: loss={float(loss):.4f} fit={fit_top1:.3f} '
                      f'hold={hold_top1:.3f} pos>tau={pos_rate:.3f} '
                      f'neg_rej={neg_rej:.3f}', flush=True)
            if hold_top1 > best['top1']:
                best = {'top1': hold_top1, 'state': {k: v.clone() for k, v in
                        model.state_dict().items()}, 'step': step,
                        'pos_rate': pos_rate, 'neg_rej': neg_rej}

    model.load_state_dict(best['state'])
    fit_top1, _, _ = evaluate(fit_pos, fit_neg)
    hold_top1, pos_rate, neg_rej = evaluate(hold_pos, hold_neg)
    print(f'\nFINAL best(step {best["step"]}): fit top1={fit_top1:.3f} '
          f'holdout top1={hold_top1:.3f} pos>tau={pos_rate:.3f} '
          f'neg_reject={neg_rej:.3f}', flush=True)
    if fit_top1 < 0.90:
        print('GATE: UNLEARNABLE (fit < 0.90)', flush=True)
    else:
        print('GATE:', 'PASS' if hold_top1 >= 0.80 else 'FAIL', flush=True)

    # threshold operating point from holdout scores (for C-side tau)
    model.eval()
    with torch.no_grad():
        pos_sc = [float(torch.sigmoid(model(feats[q], feats[s]))) for q, s, _ in hold_pos]
        neg_sc = [float(torch.sigmoid(model(feats[q], feats[s]))) for q, s, _ in hold_neg]
    model.train()
    print(f'tau operating point: pos scores mean={sum(pos_sc)/len(pos_sc):.3f} '
          f'min={min(pos_sc):.3f} | neg max={max(neg_sc):.3f} '
          f'mean={sum(neg_sc)/len(neg_sc):.3f}', flush=True)

    out_dir = ROOT / 'build/neural-memory-ec005'
    out_dir.mkdir(exist_ok=True)
    torch.save({'step': best['step'], 'model': best['state'],
                'fit_top1': fit_top1, 'holdout_top1': hold_top1,
                'gate_pass': fit_top1 >= 0.90 and hold_top1 >= 0.80},
               out_dir / 'ckpt-best.pt')
    print(f'saved: {out_dir / "ckpt-best.pt"}', flush=True)
    codec.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
