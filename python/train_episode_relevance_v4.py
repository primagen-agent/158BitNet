"""EC-004: multi-channel bilinear max-interaction episode relevance.

Registered in training/memory/neural-system/reviews/EC-004/PROCESS.md.
Combines EC-002's structural aggregation (per-channel max, not a scalar)
with EC-003's deployability (episode-side projection precomputable at write
time). Regularized + early-stopped on the world-level holdout.

    a_i = tanh(Wa q_i)  [K*h];  b_j = tanh(Wb e_j)  [K*h]
    s_k(i,j) = <a_k(i), b_k(j)>/sqrt(h);  agg_k = max_ij s_k
    logit = w tanh(W2 [agg; q_mean; e_mean] + b2)
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
from train_episode_relevance import build_pairs, collect_records, live_extract, split_worlds

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'build/neural-memory-cg003-frozen-build'
GGUF = ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf'
K, H = 8, 32


class MultiChannelBilinear(nn.Module):
    def __init__(self, dim=2048, k=K, h=H):
        super().__init__()
        self.k, self.h = k, h
        self.proj_q = nn.Linear(dim, k * h)
        self.proj_e = nn.Linear(dim, k * h)
        self.drop = nn.Dropout(0.1)
        self.mix = nn.Linear(2 * dim + k, 64)
        self.out = nn.Linear(64, 1)

    def episode_side(self, e_rows):
        return torch.tanh(self.proj_e(e_rows))

    def forward(self, q_rows, e_rows, e_side=None):
        a = torch.tanh(self.proj_q(q_rows))                       # [nq, K*h]
        b = self.episode_side(e_rows) if e_side is None else e_side
        nq, ne = a.shape[0], b.shape[0]
        av = a.view(nq, self.k, self.h)
        bv = b.view(ne, self.k, self.h)
        # s_k(i,j) = <a_k(i), b_k(j)>/sqrt(h)  -> [nq, ne, K]
        scores = torch.einsum('ikh,jkh->ijk', av, bv) / (self.h ** 0.5)
        agg = scores.amax(dim=(0, 1))                             # [K]
        means = torch.cat((q_rows.mean(0), e_rows.mean(0)))
        feat = self.drop(torch.cat((agg, means)))
        return self.out(torch.tanh(self.mix(feat))).squeeze(-1)


def main():
    fm = json.loads((ROOT / 'build/neural-memory-cg003-full-features/manifest.json').read_text())
    codec = AppendValueCodec(FROZEN / 'tok_probe', GGUF, fm['tokenizer_sha256'])
    binding = ModelBinding(BACKBONE_SHA256, fm['tokenizer_sha256'],
                           fm['encoder_identity']['encoder_id'], '0' * 64)
    records = collect_records(codec, binding)
    split_worlds(records)

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
    print(f'pairs fit={fit_counts} holdout={hold_counts}', flush=True)
    print(f'unique texts: {len(texts)}', flush=True)

    feats = {}
    for i, text in enumerate(texts):
        feats[i] = live_extract(text)
        if (i + 1) % 100 == 0:
            print(f'  extracted {i+1}/{len(texts)}', flush=True)

    torch.manual_seed(16384)
    model = MultiChannelBilinear()
    print(f'params: {sum(p.numel() for p in model.parameters())}', flush=True)

    # extra same-world negatives (registered augmentation)
    rng_aug = random.Random(11)
    fit_records = [r for r in records if r['split'] == 'fit' and r['route'] == 1]
    by_world = {}
    for r in fit_records:
        by_world.setdefault((r['world'], r['lang']), []).append(r)
    extra_negB = []
    qid_of = {r['query']: table(r['query']) for r in records}
    for r in fit_records:
        same = [o for o in by_world.get((r['world'], r['lang']), []) if o is not r]
        if len(same) >= 2:
            for pick in rng_aug.sample(same, 2):
                extra_negB.append((qid_of[r['query']], table(pick['source']), 0))
    fit_neg_all = [p for p in fit_pairs if p[2] == 0] + extra_negB
    fit_pos = [p for p in fit_pairs if p[2] == 1]
    hold_pos = [p for p in hold_pairs if p[2] == 1]
    hold_neg = [p for p in hold_pairs if p[2] == 0]
    print(f'train pos={len(fit_pos)} neg={len(fit_neg_all)} '
          f'(extra negB: {len(extra_negB)})', flush=True)

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
                rng.sample(fit_neg_all, min(8, len(fit_neg_all)))
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
                fit_top1, _, _ = evaluate(fit_pos[:40], fit_neg_all)
                print(f'  step {step}: loss={float(loss):.4f} fit={fit_top1:.3f} '
                      f'hold={hold_top1:.3f} pos>tau={pos_rate:.3f} '
                      f'neg_rej={neg_rej:.3f}', flush=True)
            if hold_top1 > best['top1']:
                best = {'top1': hold_top1, 'state': {k: v.clone() for k, v in
                        model.state_dict().items()}, 'step': step,
                        'pos_rate': pos_rate, 'neg_rej': neg_rej}

    model.load_state_dict(best['state'])
    fit_top1, _, _ = evaluate(fit_pos, fit_neg_all)
    hold_top1, pos_rate, neg_rej = evaluate(hold_pos, hold_neg)
    print(f'\nFINAL best(step {best["step"]}): fit top1={fit_top1:.3f} '
          f'holdout top1={hold_top1:.3f} pos>tau={pos_rate:.3f} '
          f'neg_reject={neg_rej:.3f}', flush=True)
    if fit_top1 < 0.90:
        print('GATE: UNLEARNABLE (fit < 0.90) — bilinear family closed', flush=True)
    else:
        print('GATE:', 'PASS' if hold_top1 >= 0.80 else 'FAIL', flush=True)

    out_dir = ROOT / 'build/neural-memory-ec004'
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
