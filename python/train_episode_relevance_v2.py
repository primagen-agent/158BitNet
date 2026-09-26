"""EC-002: token-level max-interaction episode relevance head.

Registered in training/memory/neural-system/reviews/EC-002/PROCESS.md.
Motivated by EC-001's structural failure: mean pooling dilutes the subject
signal (fit top-1 0.371). Per-pair interaction with max aggregation keeps
"an identical name token exists somewhere in both texts" — position-free and
dilution-free.

    f_ij = tanh(W1 [q_i; e_j; q_i*e_j; |q_i - e_j|] + b1)   (64-dim)
    agg  = max_ij f_ij                                       (per-dim max)
    logit= w tanh(W2 [agg; q_mean; e_mean] + b2)

Pairs, split, gate and features are identical to EC-001 (same registry).
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
from train_episode_relevance import (build_pairs, collect_records,
                                     live_extract, split_worlds)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'build/neural-memory-cg003-frozen-build'
GGUF = ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf'


class TokenInteractionScorer(nn.Module):
    def __init__(self, dim=2048, hidden=64):
        super().__init__()
        self.pair = nn.Linear(4 * dim, hidden)
        self.mix = nn.Linear(2 * dim + hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, q_rows, e_rows):
        """q_rows [nq, 2048], e_rows [ne, 2048] -> logit scalar."""
        nq, ne = q_rows.shape[0], e_rows.shape[0]
        q = q_rows.unsqueeze(1).expand(nq, ne, q_rows.shape[1])
        e = e_rows.unsqueeze(0).expand(nq, ne, e_rows.shape[1])
        f = torch.tanh(self.pair(torch.cat((q, e, q * e, (q - e).abs()), -1)))
        agg = f.max(dim=0).values.max(dim=0).values      # [hidden]
        means = torch.cat((q_rows.mean(0), e_rows.mean(0)))
        h = torch.tanh(self.mix(torch.cat((agg, means))))
        return self.out(h).squeeze(-1)


def main():
    fm = json.loads((ROOT / 'build/neural-memory-cg003-full-features/manifest.json').read_text())
    codec = AppendValueCodec(FROZEN / 'tok_probe', GGUF, fm['tokenizer_sha256'])
    binding = ModelBinding(BACKBONE_SHA256, fm['tokenizer_sha256'],
                           fm['encoder_identity']['encoder_id'], '0' * 64)

    records = collect_records(codec, binding)
    holdout = split_worlds(records)

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
    print(f'records fit={sum(r["split"]=="fit" for r in records)} '
          f'holdout={sum(r["split"]=="holdout" for r in records)}', flush=True)
    print(f'pairs fit={fit_counts} holdout={hold_counts}', flush=True)
    print(f'unique texts: {len(texts)}', flush=True)

    feats = {}
    for i, text in enumerate(texts):
        feats[i] = live_extract(text)
        if (i + 1) % 100 == 0:
            print(f'  extracted {i+1}/{len(texts)}', flush=True)

    torch.manual_seed(4096)
    model = TokenInteractionScorer()
    print(f'params: {sum(p.numel() for p in model.parameters())}', flush=True)

    fit_pos = [p for p in fit_pairs if p[2] == 1]
    fit_neg = [p for p in fit_pairs if p[2] == 0]
    hold_pos = [p for p in hold_pairs if p[2] == 1]
    hold_neg = [p for p in hold_pairs if p[2] == 0]

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = random.Random(42)

    def pair_logit(pair):
        q, s, _ = pair
        return model(feats[q], feats[s])

    def evaluate(pairs_pos, pairs_neg, label):
        model.eval()
        with torch.no_grad():
            correct = 0
            for pr in pairs_pos:
                q, s, _ = pr
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
        top1 = correct / max(len(pairs_pos), 1)
        pos_rate = sum(v > 0.5 for v in pos_sc) / max(len(pos_sc), 1)
        neg_rej = sum(v <= 0.5 for v in neg_sc) / max(len(neg_sc), 1)
        return top1, pos_rate, neg_rej

    print('\ntraining...', flush=True)
    for step in range(1, 2001):
        batch = rng.sample(fit_pos, min(8, len(fit_pos))) + \
                rng.sample(fit_neg, min(8, len(fit_neg)))
        logits = torch.stack([pair_logit(p) for p in batch])
        labels = torch.tensor([float(p[2]) for p in batch])
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            fit_top1, _, _ = evaluate(fit_pos[:60], fit_neg, 'fit')
            hold_top1, pos_rate, neg_rej = evaluate(hold_pos, hold_neg, 'holdout')
            print(f'  step {step}: loss={float(loss):.4f} fit_top1={fit_top1:.3f} '
                  f'holdout_top1={hold_top1:.3f} pos>tau={pos_rate:.3f} '
                  f'neg_reject={neg_rej:.3f}', flush=True)

    fit_top1, _, _ = evaluate(fit_pos, fit_neg, 'fit')
    hold_top1, pos_rate, neg_rej = evaluate(hold_pos, hold_neg, 'holdout')
    print(f'\nFINAL fit top1={fit_top1:.3f}  holdout top1={hold_top1:.3f} '
          f'pos>tau={pos_rate:.3f} neg_reject={neg_rej:.3f}', flush=True)
    if fit_top1 < 0.90:
        print('GATE: UNLEARNABLE (fit < 0.90) — stop condition', flush=True)
    else:
        print('GATE:', 'PASS' if hold_top1 >= 0.80 else 'FAIL',
              '(deploy threshold 0.80)', flush=True)

    out_dir = ROOT / 'build/neural-memory-ec002'
    out_dir.mkdir(exist_ok=True)
    torch.save({'step': 2000, 'model': model.state_dict(),
                'fit_top1': fit_top1, 'holdout_top1': hold_top1,
                'gate_pass': fit_top1 >= 0.90 and hold_top1 >= 0.80},
               out_dir / 'ckpt-2000.pt')
    print(f'saved: {out_dir / "ckpt-2000.pt"}', flush=True)
    codec.close()


if __name__ == '__main__':
    main()
