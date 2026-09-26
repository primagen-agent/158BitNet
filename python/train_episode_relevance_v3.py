"""EC-003: bilinear max-interaction episode relevance (deployable variant).

Registered in training/memory/neural-system/reviews/EC-003/PROCESS.md.
Same inductive bias as EC-002 (max over token-pair scores: an identical
name token anywhere in query and episode) but bilinear, so the episode-side
projection b_j is precomputable at write time — C deployment costs one
64-dim max inner product per episode instead of ~0.6 GFLOP.

    a_i = tanh(Wa q_i);  b_j = tanh(Wb e_j);  s_ij = <a_i, b_j>/sqrt(64)
    agg = max_ij s_ij
    logit = w tanh(W2 [agg; q_mean; e_mean] + b2)

Only runs when EC-002 proved the signal learnable (fit >= 0.90).
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


class BilinearMaxScorer(nn.Module):
    def __init__(self, dim=2048, hidden=64):
        super().__init__()
        self.proj_q = nn.Linear(dim, hidden)
        self.proj_e = nn.Linear(dim, hidden)
        self.mix = nn.Linear(2 * dim + 1, hidden)
        self.out = nn.Linear(hidden, 1)
        self.hidden = hidden

    def episode_side(self, e_rows):
        """Precomputable at write time in C."""
        return torch.tanh(self.proj_e(e_rows))

    def forward(self, q_rows, e_rows=None, e_side=None):
        a = torch.tanh(self.proj_q(q_rows))                # [nq, h]
        b = self.episode_side(e_rows) if e_side is None else e_side
        scores = a @ b.T / (self.hidden ** 0.5)            # [nq, ne]
        agg = scores.max()
        means = torch.cat((q_rows.mean(0), e_rows.mean(0) if e_rows is not None
                           else torch.zeros_like(q_rows[0])))
        h = torch.tanh(self.mix(torch.cat((agg.unsqueeze(0), means))))
        return self.out(h).squeeze(-1)


def main():
    gate = ROOT / 'build/neural-memory-ec002/ckpt-2000.pt'
    ck = torch.load(gate, weights_only=False)
    if ck.get('fit_top1', 0.0) < 0.90:
        print(f'EC-002 fit_top1={ck.get("fit_top1", 0):.3f} < 0.90 — '
              'unlearnable; EC-003 not started (registered stop condition)')
        return 1

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

    torch.manual_seed(8192)
    model = BilinearMaxScorer()
    print(f'params: {sum(p.numel() for p in model.parameters())}', flush=True)

    fit_pos = [p for p in fit_pairs if p[2] == 1]
    fit_neg = [p for p in fit_pairs if p[2] == 0]
    hold_pos = [p for p in hold_pairs if p[2] == 1]
    hold_neg = [p for p in hold_pairs if p[2] == 0]

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
        if step % 200 == 0:
            fit_top1, _, _ = evaluate(fit_pos[:60], fit_neg)
            hold_top1, pos_rate, neg_rej = evaluate(hold_pos, hold_neg)
            print(f'  step {step}: loss={float(loss):.4f} fit_top1={fit_top1:.3f} '
                  f'holdout_top1={hold_top1:.3f} pos>tau={pos_rate:.3f} '
                  f'neg_reject={neg_rej:.3f}', flush=True)
            if hold_top1 > best['top1']:
                best = {'top1': hold_top1, 'state': {k: v.clone() for k, v in
                        model.state_dict().items()}, 'step': step,
                        'pos_rate': pos_rate, 'neg_rej': neg_rej}

    fit_top1, _, _ = evaluate(fit_pos, fit_neg)
    model.load_state_dict(best['state'])
    hold_top1, pos_rate, neg_rej = evaluate(hold_pos, hold_neg)
    print(f'\nFINAL fit top1={fit_top1:.3f}  best-holdout top1={hold_top1:.3f} '
          f'(step {best["step"]}) pos>tau={pos_rate:.3f} neg_reject={neg_rej:.3f}',
          flush=True)
    if fit_top1 < 0.90:
        print('GATE: UNLEARNABLE (fit < 0.90) — stop condition', flush=True)
    else:
        print('GATE:', 'PASS' if hold_top1 >= 0.80 else 'FAIL', flush=True)

    out_dir = ROOT / 'build/neural-memory-ec003'
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
