"""Export route-parity samples: features + expected route logits from the
trained reader (build/neural-memory-v3-013/ckpt-440.pt), computed with the
REAL span-based evidence (candidate_spans + ByteLayout endpoints + fact
probability-weighted evidence). tests/test_route_parity.c must reproduce
these logits from the raw texts alone.

Sample file layout (little-endian):
  [4B magic "NMRP"] [4B version 1] [4B n_records]
  per record:
    [4B n_query] [n_query*2048 f32 query features]
    [4B n_source] [n_source*2048 f32 source features]
    [4B query_text_len] [bytes]      (framed query JSON)
    [4B episode_text_len] [bytes]    (raw episode text, not framed)
    [3 f32 expected route logits]
    [4B expected route decision]
"""
import json
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from append_value_transport import AppendValueCodec
from joint_span_reader import candidate_spans
from native_memory_encoder import BACKBONE_SHA256, digest
from neural_memory_contract import ModelBinding
from fine_span_reader import ByteLayout
from train_episode_relevance import collect_records, live_extract
from v3_model_factory import STRUCT_MU, STRUCT_SD, make_struct_route_reader

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'build/neural-memory-cg003-frozen-build'
GGUF = ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf'
CKPT = ROOT / 'build/neural-memory-v3-013/ckpt-440.pt'


def frame(message):
    return json.dumps(message, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'))


def main():
    import hashlib as hl
    fm = json.loads((ROOT / 'build/neural-memory-cg003-full-features/manifest.json').read_text())
    frozen_codec = AppendValueCodec(FROZEN / 'tok_probe', GGUF, fm['tokenizer_sha256'])
    live_codec = AppendValueCodec(ROOT / 'build/tok_probe', GGUF,
                                  hl.sha256((ROOT / 'build/tok_probe').read_bytes()).hexdigest())
    binding = ModelBinding(BACKBONE_SHA256, fm['tokenizer_sha256'],
                           fm['encoder_identity']['encoder_id'], '0' * 64)
    key = digest(vars(binding))

    reader = make_struct_route_reader(key, width=64, seed=1018)
    ck = torch.load(CKPT, weights_only=False)
    reader.load_state_dict(ck['reader'])
    reader.eval()

    records = collect_records(frozen_codec, binding)
    picked, seen_routes = [], {0: 0, 1: 0, 2: 0}
    for r in records:
        if seen_routes[r['route']] < 5:
            picked.append(r)
            seen_routes[r['route']] += 1
        if all(v >= 5 for v in seen_routes.values()):
            break
    print(f'picked {len(picked)} records by route {seen_routes}', flush=True)

    out_dir = ROOT / 'build/neural-memory-route-parity'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'samples.bin'
    meta = []
    with out_path.open('wb') as f:
        f.write(b'NMRP')
        f.write(struct.pack('<II', 1, len(picked)))
        for k, r in enumerate(picked):
            query_framed = r['query']
            source_framed = r['source']
            episode_msg = json.loads(source_framed)

            q_feat = live_extract(query_framed)
            s_feat = live_extract(source_framed)

            ids = live_codec.tokenizer.encode(source_framed, True)
            pieces = live_codec.decode_pieces(ids[1:])
            from native_token_payload import payload_mask
            from check_joint_span_interface import source_alignment
            allowed = payload_mask(episode_msg, source_framed, pieces)
            ranges = source_alignment(episode_msg, source_framed, pieces, allowed)
            raw = episode_msg['text'].encode()

            class SpanFeaturesStub:  # minimal carrier for ByteLayout
                pass

            x = SpanFeaturesStub()
            x.source = s_feat
            x.allowed = torch.tensor(allowed, dtype=torch.bool)
            import hashlib as _hl
            x.payload_sha256 = _hl.sha256(raw).hexdigest()
            x.ranges = None  # not used by endpoints (layout carries ranges)
            layout = ByteLayout(x, raw, tuple(ranges))
            starts, ends = layout.endpoints()
            cands = candidate_spans(x.allowed)
            spans = tuple((s, e) for s, e in cands
                          if starts[s] and ends[e - 1] and starts[s][0] < ends[e - 1][-1])

            # ---- exact reader math (v3_model_factory forward, route part) ----
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
                raw_scal = torch.tensor([float(len(s_feat) > 0), float(len(spans)),
                                         float(x.allowed.sum().item()) / 32.0])
                scal = (raw_scal - torch.tensor(STRUCT_MU)) / torch.tensor(STRUCT_SD)
                logits = reader.route(torch.cat((q, evidence, scal)))
                if not spans:
                    logits = logits.masked_fill(torch.tensor([False, True, False]), float('-inf'))

            route = int(logits.argmax())
            f.write(struct.pack('<i', q_feat.shape[0]))
            f.write(q_feat.numpy().astype('<f4').tobytes())
            f.write(struct.pack('<i', s_feat.shape[0]))
            f.write(s_feat.numpy().astype('<f4').tobytes())
            question = json.loads(query_framed)[0]['text']
            qraw = question.encode()
            f.write(struct.pack('<i', len(qraw)))
            f.write(qraw)
            qb = query_framed.encode()
            f.write(struct.pack('<i', len(qb))); f.write(qb)
            eb = episode_msg['text'].encode()
            f.write(struct.pack('<i', len(eb))); f.write(eb)
            f.write(struct.pack('<3f', *logits.tolist()))
            f.write(struct.pack('<i', route))
            # pieces + framed source for tokenizer-free span reconstruction in C
            f.write(struct.pack('<i', len(pieces)))
            for piece in pieces:
                f.write(struct.pack('<i', len(piece)))
                f.write(piece)
            sb = source_framed.encode()
            f.write(struct.pack('<i', len(sb)))
            f.write(sb)
            meta.append({'rid': r['rid'], 'route': r['route'], 'n_spans': len(spans),
                         'logits': logits.tolist(), 'computed_route': route,
                         'n_query': q_feat.shape[0], 'n_source': s_feat.shape[0]})
            print(f'  [{k}] rid={r["rid"]} spans={len(spans)} '
                  f'logits={["%.3f" % v for v in logits.tolist()]} route={route}', flush=True)

    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(f'wrote {out_path} ({out_path.stat().st_size} bytes)', flush=True)
    frozen_codec.close()
    live_codec.close()


if __name__ == '__main__':
    main()
