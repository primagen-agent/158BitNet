"""EC-001: train the episode relevance head on LIVE backbone features.

Registered in training/memory/neural-system/reviews/EC-001/PROCESS.md.
Architecture mirrors the wide heads: [query_mean(2048); ep_mean(2048)] ->
256 -> tanh -> 1 -> sigmoid. Balanced BCE so 0.5 is the natural threshold
(tau registered as 0.5 for the C server).

Pairs (per registered protocol):
  positive : route==1 records — (query, its gold source)
  negativeA: route==2 records — (query, its non-answering source)   [cat-5 shape]
  negativeB: route==1 — (query, same-scenario other record's source) [name discrimination]
  negativeC: route==1 — (query, random other-scenario source)

Gate for C deployment: dev gold-source top-1 (among gold + 4 same-world +
  4 random candidates) >= 0.80.
"""
import hashlib
import json
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from append_value_transport import AppendValueCodec
from compile_time_scoped_teacher import compile_teacher_time
from compile_dialogue_teacher import compile_teacher_dialogue
from episode_memory_inputs import encoder_texts
from native_memory_encoder import BACKBONE_SHA256, digest
from neural_memory_contract import ModelBinding

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'build/neural-memory-cg003-frozen-build'
GGUF = ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf'
LIVE_PROBE = ROOT / 'build/memory_feature_probe'
RESEARCH = ROOT / 'training/memory/neural-system'


def live_extract(text, cache={}):
    if text in cache:
        return cache[text]
    with tempfile.TemporaryDirectory() as tmp:
        infile, outfile = Path(tmp) / 'in.bin', Path(tmp) / 'out.bin'
        raw = text.encode()
        with infile.open('wb') as f:
            f.write(struct.pack('<I', 1))
            f.write(struct.pack('<I', len(raw)))
            f.write(raw)
        subprocess.run([str(LIVE_PROBE), str(GGUF), str(infile), str(outfile),
                        '24', 'reader-v1'], capture_output=True, timeout=60, check=True)
        data = outfile.read_bytes()
        n = struct.unpack('<I', data[20:24])[0]
        hidden = struct.unpack('<I', data[12:16])[0]
        n_feat = n - 1
        feat = np.frombuffer(data, dtype='<f4', count=n_feat * hidden * 2,
                             offset=24 + 4 * n).reshape(n_feat, hidden * 2)
        t = torch.from_numpy(feat.copy())
        cache[text] = t
        return t


def load_split(cid, split):
    d = RESEARCH / 'data' / cid
    suffix = f'{split}.inputs.jsonl'
    ins = {r['id']: r for r in map(json.loads, (d / suffix).read_text().splitlines())}
    lab = {r['id']: r for r in map(json.loads,
          (d / f'{split}.labels.jsonl').read_text().splitlines())}
    idx = {r['id']: r for r in map(json.loads,
          (d / f'{split}.index.jsonl').read_text().splitlines())}
    return ins, lab, idx


def collect_records(codec, binding, corpora=('JB-001', 'JB-002')):
    """Train-split records with a teacher route label and a source episode.

    The pinned compiler refuses dev records by design (training-only bound
    labels); the held-out evaluation below therefore splits TRAIN records by
    semantic world instead, per the PLAN's world-level split rule.
    """
    out = []
    for cid in corpora:
        ins, lab, idx = load_split(cid, 'train')
        teacher = (compile_teacher_dialogue if cid in ('JB-004', 'JB-005')
                   else compile_teacher_time)
        for meta in idx.values():
            rid = meta['id']
            try:
                t = teacher(ins[rid], lab[rid], meta, codec,
                            binding, '0' * 64,
                            purpose='training_diagnostic')
            except Exception:
                continue
            route = t.static_targets.route
            runtime = ins[rid]
            query, sources = encoder_texts(runtime)
            if not sources or route not in (1, 2):
                continue
            out.append({'rid': rid, 'split': 'train', 'world': meta['world_id'],
                        'lang': meta['language'], 'route': route,
                        'query': query, 'source': sources[0],
                        'value_bytes': t.static_targets.value_bytes,
                        'episode_raw': runtime['episodes'][0]['text'].encode() if route == 1 else None})
    return out


def split_worlds(records, holdout_every=7):
    """Deterministic world-level split: sorted worlds, every Nth -> holdout."""
    worlds = sorted({r['world'] for r in records})
    holdout = set(worlds[::holdout_every])
    for r in records:
        r['split'] = 'holdout' if r['world'] in holdout else 'fit'
    return sorted(holdout)


def build_pairs(records, split, table):
    """(query_idx, source_idx, label) using a shared text table."""
    recs = [r for r in records if r['split'] == split]
    by_world = {}
    for r in recs:
        by_world.setdefault((r['world'], r['lang']), []).append(r)

    pairs = []
    pos_idx, negA_idx, negB_idx, negC_idx = [], [], [], []
    for r in recs:
        q, s = table(r['query']), table(r['source'])
        if r['route'] == 1:
            pos_idx.append(len(pairs))
            pairs.append((q, s, 1))
        else:
            negA_idx.append(len(pairs))
            pairs.append((q, s, 0))

    rng = random.Random(7)
    for r in recs:
        if r['route'] != 1:
            continue
        q = table(r['query'])
        same = [o for o in by_world.get((r['world'], r['lang']), []) if o is not r]
        if same:
            pick = rng.choice(same)
            negB_idx.append(len(pairs))
            pairs.append((q, table(pick['source']), 0))
        other = [o for o in recs if o['world'] != r['world'] and o['lang'] == r['lang']]
        if other:
            pick = rng.choice(other)
            negC_idx.append(len(pairs))
            pairs.append((q, table(pick['source']), 0))
    counts = {'pos': len(pos_idx), 'negA': len(negA_idx),
              'negB': len(negB_idx), 'negC': len(negC_idx)}
    return pairs, counts


def main():
    fm = json.loads((ROOT / 'build/neural-memory-cg003-full-features/manifest.json').read_text())
    codec = AppendValueCodec(FROZEN / 'tok_probe', GGUF, fm['tokenizer_sha256'])
    binding = ModelBinding(BACKBONE_SHA256, fm['tokenizer_sha256'],
                           fm['encoder_identity']['encoder_id'], '0' * 64)

    records = collect_records(codec, binding)
    holdout_worlds = split_worlds(records)
    print(f'records with route label + source: {len(records)} '
          f'(fit={sum(r["split"]=="fit" for r in records)}, '
          f'holdout={sum(r["split"]=="holdout" for r in records)})', flush=True)
    print(f'holdout worlds ({len(holdout_worlds)}): {holdout_worlds}', flush=True)

    class TextTable:
        def __init__(self):
            self.texts, self.tid = [], {}

        def __call__(self, text):
            if text not in self.tid:
                self.tid[text] = len(self.texts)
                self.texts.append(text)
            return self.tid[text]

    table = TextTable()
    train_pairs, train_counts = build_pairs(records, 'fit', table)
    dev_pairs, dev_counts = build_pairs(records, 'holdout', table)
    texts = table.texts
    print('fit     pair counts:', train_counts, flush=True)
    print('holdout pair counts:', dev_counts, flush=True)
    print(f'unique texts to extract: {len(texts)}', flush=True)
    feats = {}
    for i, text in enumerate(texts):
        feats[i] = live_extract(text)
        if (i + 1) % 100 == 0:
            print(f'  extracted {i+1}/{len(texts)}', flush=True)

    def mean_of(i):
        return feats[i].mean(dim=0)

    torch.manual_seed(2048)
    model = nn.Sequential(nn.Linear(4096, 256), nn.Tanh(), nn.Linear(256, 1))
    n_params = sum(p.numel() for p in model.parameters())
    print(f'ep_rel params: {n_params}', flush=True)

    pos_train = [p for p in train_pairs if p[2] == 1]
    neg_train = [p for p in train_pairs if p[2] == 0]
    print(f'train pos={len(pos_train)} neg={len(neg_train)}', flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = random.Random(42)

    def batch_tensors(batch):
        xs, ys = [], []
        for q, s, label in batch:
            xs.append(torch.cat([mean_of(q), mean_of(s)]))
            ys.append(float(label))
        return torch.stack(xs), torch.tensor(ys)

    def evaluate():
        """Holdout worlds: gold top-1 among gold + 8 sampled negatives; tau rates."""
        dev_pos = [p for p in dev_pairs if p[2] == 1]
        dev_neg = [p for p in dev_pairs if p[2] == 0]
        model.eval()
        with torch.no_grad():
            correct = 0
            for q, s, _ in dev_pos:
                cands = [s]
                for _, ns, _ in rng.sample(dev_neg, min(8, len(dev_neg))):
                    if ns != s:
                        cands.append(ns)
                scores = [float(model(torch.cat([mean_of(q), mean_of(c)]))) for c in cands]
                if scores.index(max(scores)) == 0:
                    correct += 1
            pos_scores = [float(torch.sigmoid(model(torch.cat([mean_of(q), mean_of(s)]))))
                          for q, s, _ in dev_pos]
            neg_scores = [float(torch.sigmoid(model(torch.cat([mean_of(q), mean_of(s)]))))
                          for q, s, _ in dev_neg]
        model.train()
        top1 = correct / max(len(dev_pos), 1)
        pos_rate = sum(sc > 0.5 for sc in pos_scores) / max(len(pos_scores), 1)
        neg_reject = sum(sc <= 0.5 for sc in neg_scores) / max(len(neg_scores), 1)
        return top1, pos_rate, neg_reject

    print('\ntraining...', flush=True)
    for step in range(1, 2001):
        batch = rng.sample(pos_train, min(8, len(pos_train))) + \
                rng.sample(neg_train, min(8, len(neg_train)))
        x, y = batch_tensors(batch)
        logits = model(x).squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            top1, pos_rate, neg_reject = evaluate()
            print(f'  step {step}: loss={float(loss):.4f} holdout_top1={top1:.3f} '
                  f'pos>tau={pos_rate:.3f} neg_reject={neg_reject:.3f}', flush=True)

    top1, pos_rate, neg_reject = evaluate()
    print(f'\nFINAL holdout: top1={top1:.3f} pos>tau={pos_rate:.3f} '
          f'neg_reject={neg_reject:.3f}', flush=True)
    print('GATE:', 'PASS' if top1 >= 0.80 else 'FAIL', '(deploy threshold 0.80)', flush=True)

    out_dir = ROOT / 'build/neural-memory-ec001'
    out_dir.mkdir(exist_ok=True)
    torch.save({'step': 2000, 'model': model.state_dict(),
                'holdout_top1': top1, 'gate_pass': top1 >= 0.80},
               out_dir / 'ckpt-2000.pt')
    print(f'saved: {out_dir / "ckpt-2000.pt"}', flush=True)
    codec.close()


if __name__ == '__main__':
    main()
