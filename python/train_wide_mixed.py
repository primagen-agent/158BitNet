"""EC-006 (part 2): retrain the wide value-span heads on the MIXED corpus
(declarative JB-001..003 + dialogue JB-004), live features + live tokenizer,
softmax CE over span candidates — the train_wide_live_ce recipe with mixed
forms and a world-level holdout split.

Gates (registered): dialogue-form holdout exact-span >= 0.70 AND declarative
holdout exact-span >= 0.90 (the old JB-only head reached 1.00 on its form).
"""
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from append_value_transport import AppendValueCodec
from native_memory_encoder import BACKBONE_SHA256, digest
from neural_memory_contract import ModelBinding
from train_episode_relevance import collect_records, live_extract, split_worlds
from v3_model_factory import make_struct_route_reader
from wide_reader import WideFactValueReader

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'build/neural-memory-cg003-frozen-build'
GGUF = ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf'


def main():
    import hashlib as _hl
    fm = json.loads((ROOT / 'build/neural-memory-cg003-full-features/manifest.json').read_text())
    frozen_codec = AppendValueCodec(FROZEN / 'tok_probe', GGUF, fm['tokenizer_sha256'])
    live_codec = AppendValueCodec(ROOT / 'build/tok_probe', GGUF,
                                  _hl.sha256((ROOT / 'build/tok_probe').read_bytes()).hexdigest())
    binding = ModelBinding(BACKBONE_SHA256, fm['tokenizer_sha256'],
                           fm['encoder_identity']['encoder_id'], '0' * 64)

    records = collect_records(frozen_codec, binding,
                              corpora=('JB-001', 'JB-002', 'JB-003', 'JB-004'))
    holdout_worlds = split_worlds(records)
    items = [r for r in records if r['route'] == 1 and r['value_bytes'] is not None]
    print(f'route==1 items with spans: {len(items)} '
          f'(fit={sum(r["split"]=="fit" for r in items)}, '
          f'holdout={sum(r["split"]=="holdout" for r in items)}, '
          f'dialogue={sum(r["rid"].startswith("jb4-") for r in items)})', flush=True)

    # Map gold value bytes to live-tokenizer span indices (reconstruct search)
    mapped = []
    skipped = 0
    for r in items:
        s_text = r['source']
        raw = r['episode_raw']
        vs, ve = r['value_bytes']
        val_bytes = raw[vs:ve]
        ids = live_codec.tokenizer.encode(s_text, True)
        pieces = live_codec.decode_pieces(ids[1:])
        content = pieces
        recon = b''.join(content)
        pos = recon.find(val_bytes)
        if pos < 0:
            skipped += 1
            continue
        pos_end = pos + len(val_bytes)
        offsets = [0]
        for piece in content:
            offsets.append(offsets[-1] + len(piece))
        ts = next((ti for ti in range(len(content)) if offsets[ti] <= pos < offsets[ti + 1]), -1)
        if ts < 0:
            skipped += 1
            continue
        te = next((ti for ti in range(ts, len(content)) if offsets[ti] <= pos_end - 1 < offsets[ti + 1]), -1) + 1
        if te <= ts:
            skipped += 1
            continue
        while te < len(content) and val_bytes not in b''.join(content[ts:te]):
            te += 1
        if val_bytes not in b''.join(content[ts:te]):
            skipped += 1
            continue
        while te - 1 > ts and pos_end <= offsets[te - 1]:
            te -= 1
        mapped.append({'rid': r['rid'], 'query': r['query'], 'source': s_text,
                       'f_start': ts, 'f_end': te, 'split': r['split'],
                       'dialogue': r['rid'].startswith('jb4-')})
    print(f'mapped {len(mapped)} items, skipped {skipped}', flush=True)

    texts = []
    tid = {}
    def T(t):
        if t not in tid:
            tid[t] = len(texts)
            texts.append(t)
        return tid[t]
    for m in mapped:
        m['q_idx'] = T(m['query'])
        m['s_idx'] = T(m['source'])
    print(f'unique texts: {len(texts)}', flush=True)
    feats = {}
    for i, t in enumerate(texts):
        feats[i] = live_extract(t)
        if (i + 1) % 300 == 0:
            print(f'  extracted {i+1}/{len(texts)}', flush=True)

    torch.manual_seed(2048)
    base_reader = make_struct_route_reader(digest(vars(binding)), width=64, seed=1018)
    ck = torch.load(ROOT / 'build/neural-memory-v3-013/ckpt-440.pt', weights_only=False)
    base_reader.load_state_dict(ck['reader'])
    base_reader.eval()
    for p in base_reader.parameters():
        p.requires_grad = False
    torch.manual_seed(4096)
    wide = WideFactValueReader(base_reader, hidden_dim=2048)
    trainable = list(wide.wide_fact.parameters()) + list(wide.wide_value.parameters())
    opt = torch.optim.Adam(trainable, lr=1e-3, weight_decay=1e-4)
    rng = random.Random(42)
    fit = [m for m in mapped if m['split'] == 'fit']

    def score_all_spans(q_feat, s_feat, max_len=6):
        n_s = s_feat.shape[0]
        spans, scores = [], []
        q_mean = q_feat.mean(dim=0)
        for start in range(n_s):
            for length in range(1, max_len + 1):
                end = start + length
                if end > n_s:
                    break
                span_src = s_feat[start:end].mean(dim=0)
                wide_in = torch.cat([q_mean, span_src])
                h1 = wide.wide_fact[0](wide_in)
                h1 = torch.tanh(h1)
                sc = wide.wide_fact[2](h1).squeeze()
                spans.append((start, end))
                scores.append(sc)
        return torch.stack(scores), spans

    def evaluate(rows):
        correct = 0
        by_form = {'dialogue': [0, 0], 'declarative': [0, 0]}
        for m in rows:
            with torch.no_grad():
                scores, spans = score_all_spans(feats[m['q_idx']], feats[m['s_idx']])
                best = int(scores.argmax())
                a, b = spans[best]
                ok = a == m['f_start'] and b == m['f_end']
                correct += ok
                form = 'dialogue' if m['dialogue'] else 'declarative'
                by_form[form][0] += ok
                by_form[form][1] += 1
        return correct / max(len(rows), 1), by_form

    print('\ntraining (softmax CE over spans, mixed forms)...', flush=True)
    for step in range(1, 2001):
        opt.zero_grad()
        batch = rng.sample(fit, min(16, len(fit)))
        total, n = 0., 0
        for m in batch:
            q_feat, s_feat = feats[m['q_idx']], feats[m['s_idx']]
            scores, spans = score_all_spans(q_feat, s_feat)
            gold = next(i for i, (a, b) in enumerate(spans)
                        if a == m['f_start'] and b == m['f_end'])
            loss = -scores.log_softmax(0)[gold]
            total = total + loss
            n += 1
        if n:
            (total / n).backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
        if step % 400 == 0:
            sample = rng.sample(mapped, min(80, len(mapped)))
            acc, by_form = evaluate(sample)
            print(f'  step {step}: loss={float(total)/max(n,1):.4f} sample_exact={acc:.3f} '
                  f'dialogue={by_form["dialogue"][0]}/{by_form["dialogue"][1]} '
                  f'declarative={by_form["declarative"][0]}/{by_form["declarative"][1]}', flush=True)

    hold = [m for m in mapped if m['split'] == 'holdout']
    acc, by_form = evaluate(hold)
    dial_ok = by_form['dialogue'][0] / max(by_form['dialogue'][1], 1)
    decl_ok = by_form['declarative'][0] / max(by_form['declarative'][1], 1)
    print(f'\nFINAL holdout exact-span: overall={acc:.3f} '
          f'dialogue={dial_ok:.3f} ({by_form["dialogue"][1]} items) '
          f'declarative={decl_ok:.3f} ({by_form["declarative"][1]} items)', flush=True)
    gate = dial_ok >= 0.70 and decl_ok >= 0.90
    print('GATE:', 'PASS' if gate else 'FAIL',
          '(dialogue>=0.70 AND declarative>=0.90)', flush=True)

    out_dir = ROOT / 'build/neural-memory-ec006-wide'
    out_dir.mkdir(exist_ok=True)
    torch.save({'step': 2000, 'wide_fact': wide.wide_fact.state_dict(),
                'wide_value': wide.wide_value.state_dict(),
                'dialogue_exact': dial_ok, 'declarative_exact': decl_ok,
                'gate_pass': gate}, out_dir / 'ckpt-best.pt')
    print(f'saved: {out_dir / "ckpt-best.pt"}', flush=True)
    frozen_codec.close()
    live_codec.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
