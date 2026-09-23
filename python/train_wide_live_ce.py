"""V3-023: softmax CE loss (V3-021 style) on LIVE backbone features.

Combines what worked:
- V3-021's loss: softmax CE over span candidates (gold span should have
  the highest probability) — proven to reach 95% on frozen features.
- V3-022's features: LIVE backbone via build/memory_feature_probe (matches
  the C server's runtime feature distribution exactly).
"""
import json, sys, struct, subprocess, tempfile
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from pathlib import Path
import numpy as np
import torch
import random

from append_value_transport import AppendValueCodec
from compile_time_scoped_teacher import compile_teacher_time
from episode_memory_inputs import encoder_texts
from joint_optimizer import tensor_digest
from native_memory_encoder import BACKBONE_SHA256, digest
from neural_memory_contract import ModelBinding
from v3_model_factory import make_struct_route_reader
from wide_reader import WideFactValueReader

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
        n_feat = n - 1
        feat = np.frombuffer(data, dtype='<f4', count=n_feat * hidden * 2,
                             offset=24 + 4 * n).reshape(n_feat, hidden * 2)
        t = torch.from_numpy(feat.copy())
        cache[text] = t
        return t


def main():
    fm = json.loads((ROOT / 'build/neural-memory-cg003-full-features/manifest.json').read_text())
    codec = AppendValueCodec(FROZEN / 'tok_probe', GGUF, fm['tokenizer_sha256'])
    # LIVE codec: matches the C server's runtime tokenizer (the frozen probe's
    # tokenizer differs due to the O(1) BPE merge optimization in the live build).
    # Use it for gold-span token mapping so training == deployment.
    import hashlib as _hl
    live_tok_sha = _hl.sha256((ROOT / 'build/tok_probe').read_bytes()).hexdigest()
    live_codec = AppendValueCodec(ROOT / 'build/tok_probe', GGUF, live_tok_sha)
    binding = ModelBinding(BACKBONE_SHA256, fm['tokenizer_sha256'],
                           fm['encoder_identity']['encoder_id'], '0' * 64)
    key = digest(vars(binding))

    torch.manual_seed(1018)
    base_reader = make_struct_route_reader(key, width=64, seed=1018)
    ck = torch.load(ROOT / 'build/neural-memory-v3-013/ckpt-440.pt', weights_only=False)
    base_reader.load_state_dict(ck['reader'])
    base_reader.eval()
    rd_init = tensor_digest(base_reader.state_dict())
    for p in base_reader.parameters():
        p.requires_grad = False

    torch.manual_seed(1022)
    wide = WideFactValueReader(base_reader, hidden_dim=2048)
    trainable = list(wide.wide_fact.parameters()) + list(wide.wide_value.parameters())
    print(f'Wide heads: {sum(p.numel() for p in trainable)} params')

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
                t = compile_teacher_time(ins[rid], lab[rid], meta, codec, binding, rd_init,
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

            q_feat = live_extract(query)
            s_feat = live_extract(s_text)
            n_s = s_feat.shape[0]
            if n_s < 2:
                continue

            # Map gold value bytes → feature row indices.
            # Find the token range whose decoded text CONTAINS the gold value exactly.
            # (Piece-level byte overlap can include a preceding piece like " in".)
            ids = live_codec.tokenizer.encode(s_text, True)
            pieces = live_codec.decode_pieces(ids)
            # BOS decodes with a leading artifact; content pieces are pieces[1:].
            # Reconstructed text may differ from original by whitespace —
            # so we search for the value in the RECONSTRUCTED text, not the original.
            content_pieces = pieces[1:]
            n_content = len(content_pieces)
            recon = b''.join(content_pieces)
            val_bytes = raw[vs:ve]
            # Find the value in the reconstructed text
            abs_pos = recon.find(val_bytes)
            if abs_pos < 0:
                continue
            abs_ve = abs_pos + len(val_bytes)
            # Byte offsets of each content piece within the reconstructed text
            offsets = [0]
            for piece in content_pieces:
                offsets.append(offsets[-1] + len(piece))
            # Find the token range [ts, te) covering [abs_pos, abs_ve)
            # with ts aligned to the first piece that does not start before abs_pos
            ts = -1
            for ti in range(n_content):
                if offsets[ti] <= abs_pos < offsets[ti + 1]:
                    # This piece contains the value start.
                    # If the piece starts exactly at abs_pos, it's aligned.
                    # If it starts before (like " in Rig"), start from THIS piece
                    # only if the value is a suffix; otherwise start from next.
                    ts = ti
                    break
            if ts < 0:
                continue
            # END: the last piece that CONTAINS any byte of the value.
            # The value's last byte is at abs_ve-1; find the piece covering it
            # and extend te past it (value may end mid-piece, e.g. "ache"+"n").
            te = -1
            for ti in range(ts, n_content):
                if offsets[ti] <= abs_ve - 1 < offsets[ti + 1]:
                    te = ti + 1
                    break
            if te < 0:
                continue
            # Extend te while the value continues into subsequent pieces:
            # the decoded range must fully contain the value bytes
            while te < n_content and b''.join(content_pieces[ts:te]).find(val_bytes) < 0:
                te += 1
            decoded = b''.join(content_pieces[ts:te])
            if val_bytes not in decoded:
                continue
            # Shrink te: drop trailing pieces that start at/after the value end
            while te - 1 > ts and abs_ve <= offsets[te - 1]:
                te -= 1
            # tok_start/tok_end are CONTENT piece indices; feature rows are
            # content_piece_index (since feature row r = content piece r)
            tok_start, tok_end = ts, te
            # Feature row r corresponds to content piece r (0-based)
            f_start = tok_start
            f_end = tok_end
            if f_start >= n_s or f_end > n_s:
                continue

            items.append({'rid': rid, 'q_feat': q_feat, 's_feat': s_feat,
                          'f_start': f_start, 'f_end': f_end,
                          'gold_value': raw[vs:ve].decode()})

    print(f'Loaded {len(items)} items (LIVE features, gold spans mapped)')
    random.seed(42)

    opt = torch.optim.Adam(trainable, lr=1e-3)

    def score_all_spans(q_mean, s_feat, max_len=6):
        """Score all candidate spans (start, len) with the wide head.
        Returns: scores tensor [n_spans], list of (start, end) spans."""
        n_s = s_feat.shape[0]
        spans, scores = [], []
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

    def train_step(batch):
        opt.zero_grad()
        total, n = 0., 0
        for it in batch:
            q_mean = it['q_feat'].mean(dim=0)
            scores, spans = score_all_spans(q_mean, it['s_feat'])
            # Find gold span index in the candidate list
            gold_idx = -1
            for si, (a, b) in enumerate(spans):
                if a == it['f_start'] and b == it['f_end']:
                    gold_idx = si
                    break
            if gold_idx < 0:
                continue
            # Softmax CE: gold span should have the highest probability
            logp = scores.log_softmax(0)
            loss = -logp[gold_idx]
            total = total + loss
            n += 1
        if n > 0:
            (total / n).backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
        return float(total.detach()) / max(n, 1)

    def evaluate(count=50):
        """Check if argmax span contains the gold value rows."""
        correct = 0
        sample = random.sample(items, min(count, len(items)))
        for it in sample:
            with torch.no_grad():
                q_mean = it['q_feat'].mean(dim=0)
                scores, spans = score_all_spans(q_mean, it['s_feat'])
                best = int(scores.argmax())
                a, b = spans[best]
                # Correct if the selected span overlaps the gold span exactly
                if a == it['f_start'] and b == it['f_end']:
                    correct += 1
        return correct

    print('\nV3-023: softmax CE on LIVE features...')
    for step in range(1, 2001):
        batch = random.sample(items, min(16, len(items)))
        loss = train_step(batch)
        if step % 200 == 0:
            acc = evaluate()
            print(f'  Step {step}: loss={loss:.4f}, exact_span={acc}/50', flush=True)

    acc = evaluate(len(items))
    print(f'\nFinal: {acc}/{len(items)} ({100*acc/max(len(items),1):.0f}%) exact span match')

    out_dir = ROOT / 'build/neural-memory-v3-023'
    out_dir.mkdir(exist_ok=True)
    torch.save({'step': 2000, 'wide_fact': wide.wide_fact.state_dict(),
                'wide_value': wide.wide_value.state_dict(),
                'base_reader': base_reader.state_dict()}, out_dir / 'ckpt-600.pt')
    print(f'Saved: {out_dir / "ckpt-600.pt"}')
    assert tensor_digest(base_reader.state_dict()) == rd_init
    print('Base reader unchanged')
    codec.close()


if __name__ == '__main__':
    main()
