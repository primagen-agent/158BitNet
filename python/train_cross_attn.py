"""Train cross-attention selector for value extraction on LIVE features.

The cross-attention model scores each source token's relevance to the query.
The value is the contiguous high-scoring region. Unlike mean-pooled spans,
attention can align "Brenna" in the query with "Brenna" in the source,
encoding subject-ownership.
"""
import json, sys, struct, subprocess, tempfile
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import random

from append_value_transport import AppendValueCodec
from compile_time_scoped_teacher import compile_teacher_time
from episode_memory_inputs import encoder_texts
from joint_optimizer import tensor_digest
from native_memory_encoder import BACKBONE_SHA256, digest
from neural_memory_contract import ModelBinding
from v3_model_factory import make_struct_route_reader
from cross_attn_selector import CrossAttentionSelector

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'build/neural-memory-cg003-frozen-build'
GGUF = ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf'
LIVE_PROBE = ROOT / 'build/memory_feature_probe'


def live_extract(text, cache={}):
    if text in cache:
        return cache[text]
    with tempfile.TemporaryDirectory() as tmp:
        infile, outfile = Path(tmp) / 'in.bin', Path(tmp) / 'out.bin'
        raw = text.encode()
        with infile.open('wb') as f:
            f.write(struct.pack('<I', 1)); f.write(struct.pack('<I', len(raw))); f.write(raw)
        subprocess.run([str(LIVE_PROBE), str(GGUF), str(infile), str(outfile), '24', 'reader-v1'],
                       capture_output=True, timeout=60, check=True)
        data = outfile.read_bytes()
        n = struct.unpack('<I', data[20:24])[0]
        hidden = struct.unpack('<I', data[12:16])[0]
        feat = np.frombuffer(data, dtype='<f4', count=n_feat * hidden * 2,
                             offset=24 + 4 * n).reshape(n_feat, hidden * 2) if (n_feat := n - 1) else None
        t = torch.from_numpy(feat.copy())
        cache[text] = t
        return t


def main():
    fm = json.loads((ROOT / 'build/neural-memory-cg003-full-features/manifest.json').read_text())
    codec = AppendValueCodec(FROZEN / 'tok_probe', GGUF, fm['tokenizer_sha256'])
    binding = ModelBinding(BACKBONE_SHA256, fm['tokenizer_sha256'],
                           fm['encoder_identity']['encoder_id'], '0' * 64)

    rd_init_key = digest(vars(binding))

    # Cross-attention model (fresh, no base reader needed for scoring)
    torch.manual_seed(1024)
    selector = CrossAttentionSelector(hidden_dim=2048, attn_dim=256)
    trainable = list(selector.parameters())
    print(f'Cross-attention: {sum(p.numel() for p in trainable)} params')

    # LIVE codec for tokenization (matches C server)
    import hashlib as _hl
    live_tok_sha = _hl.sha256((ROOT / 'build/tok_probe').read_bytes()).hexdigest()
    live_codec = AppendValueCodec(ROOT / 'build/tok_probe', GGUF, live_tok_sha)

    RESEARCH = ROOT / 'training/memory/neural-system'
    items = []
    for cid in ('JB-001', 'JB-002'):
        c = RESEARCH / 'data' / cid
        ins = {r['id']: r for r in map(json.loads, (c / 'train.inputs.jsonl').read_text().splitlines())}
        lab = {r['id']: r for r in map(json.loads, (c / 'train.labels.jsonl').read_text().splitlines())}
        idx = {r['id']: r for r in map(json.loads, (c / 'train.index.jsonl').read_text().splitlines())}
        for meta in idx.values():
            rid = meta['id']
            try:
                t = compile_teacher_time(ins[rid], lab[rid], meta, codec, binding, '0'*64,
                                         purpose='training_diagnostic')
            except Exception:
                continue
            if t.static_targets.route != 1 or t.static_targets.value_bytes is None:
                continue
            runtime = ins[rid]
            query, sources = encoder_texts(runtime)
            if not sources:
                continue
            s_text = sources[0]
            raw = runtime['episodes'][0]['text'].encode()
            vs, ve = t.static_targets.value_bytes
            val_bytes = raw[vs:ve]

            q_feat = live_extract(query)
            s_feat = live_extract(s_text)
            n_s = s_feat.shape[0]
            if n_s < 2:
                continue

            # Map gold value to source token positions (content piece space)
            ids = live_codec.tokenizer.encode(s_text, True)
            pieces = live_codec.decode_pieces(ids)
            content_pieces = pieces[1:]
            recon = b''.join(content_pieces)
            abs_pos = recon.find(val_bytes)
            if abs_pos < 0:
                continue
            abs_ve = abs_pos + len(val_bytes)
            offsets = [0]
            for piece in content_pieces:
                offsets.append(offsets[-1] + len(piece))
            ts = next((ti for ti in range(len(content_pieces))
                       if offsets[ti] <= abs_pos < offsets[ti+1]), -1)
            if ts < 0:
                continue
            te = next((ti for ti in range(ts, len(content_pieces))
                       if offsets[ti] <= abs_ve - 1 < offsets[ti+1]), -1) + 1
            if te <= ts:
                continue
            while te < len(content_pieces) and val_bytes not in b''.join(content_pieces[ts:te]):
                te += 1
            if val_bytes not in b''.join(content_pieces[ts:te]):
                continue
            while te - 1 > ts and abs_ve <= offsets[te - 1]:
                te -= 1
            if ts >= n_s or te > n_s:
                continue

            # Content mask: True for content tokens (skip JSON wrapper)
            # The JSON wrapper is roughly the first 11-12 content pieces
            # Find where the episode text starts in the content pieces
            ep_start_byte = recon.find(raw[:20])  # first 20 bytes of episode text
            if ep_start_byte < 0:
                continue
            content_mask = torch.zeros(n_s, dtype=torch.bool)
            for ti in range(n_s):
                content_mask[ti] = offsets[ti] >= ep_start_byte

            items.append({'rid': rid, 'q_feat': q_feat, 's_feat': s_feat,
                          'gold_start': ts, 'gold_end': te,
                          'content_mask': content_mask})

    print(f'Loaded {len(items)} items')

    opt = torch.optim.Adam(trainable, lr=1e-3)
    random.seed(42)

    def train_step(batch):
        opt.zero_grad()
        total, n = 0., 0
        for it in batch:
            scores = selector(it['q_feat'], it['s_feat'], it['content_mask'])
            # Per-token BCE: gold tokens should score high, others low
            target = torch.zeros_like(scores)
            target[it['gold_start']:it['gold_end']] = 1.0
            # Weight positives higher (they're rare)
            pos_weight = torch.where(target > 0, torch.tensor(10.0), torch.tensor(1.0))
            loss = F.binary_cross_entropy_with_logits(scores, target, weight=pos_weight)
            total = total + loss
            n += 1
        if n > 0:
            (total / n).backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
        return float(total.detach()) / max(n, 1)

    def evaluate(count=50):
        """Check if the highest-scoring contiguous region matches the gold span."""
        correct = 0
        for it in random.sample(items, min(count, len(items))):
            with torch.no_grad():
                scores = selector(it['q_feat'], it['s_feat'], it['content_mask'])
            # Find contiguous region with highest mean score (same as C decode)
            best_start, best_len, best_mean = -1, 0, -1e30
            n_s = scores.shape[0]
            for start in range(n_s):
                if not it['content_mask'][start]:
                    continue
                for length in range(1, 7):
                    end = start + length
                    if end > n_s or not it['content_mask'][end-1]:
                        break
                    mean = float(scores[start:end].mean())
                    if mean > best_mean:
                        best_mean, best_start, best_len = mean, start, length
            # Correct if the best region is within/overlapping the gold span
            if best_start >= 0:
                gold_s, gold_e = it['gold_start'], it['gold_end']
                if best_start >= gold_s and best_start + best_len <= gold_e:
                    correct += 1
                elif best_start < gold_e and best_start + best_len > gold_s:
                    # Overlapping — count as half-correct (value likely extractable)
                    correct += 0.5
        return correct

    print('\nTraining cross-attention selector...')
    for step in range(1, 2001):
        batch = random.sample(items, min(16, len(items)))
        loss = train_step(batch)
        if step % 200 == 0:
            acc = evaluate()
            print(f'  Step {step}: loss={loss:.4f}, overlap={acc}/50', flush=True)

    acc = evaluate(len(items))
    print(f'\nFinal: {acc}/{len(items)} ({100*acc/len(items):.0f}%)')

    out_dir = ROOT / 'build/neural-memory-xattn'
    out_dir.mkdir(exist_ok=True)
    torch.save({'step': 2000, 'selector': selector.state_dict()}, out_dir / 'ckpt-2000.pt')
    print(f'Saved: {out_dir / "ckpt-2000.pt"}')
    codec.close()


if __name__ == '__main__':
    main()
