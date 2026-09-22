"""EV-001: V3-001 step-50 candidate free-reply evaluation (DG-023 protocol)."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from append_value_transport import AppendValueCodec
from audit_joint_generation import close as close_tol, continuous_file
from autonomous_value_controller import (AutonomousValueController, LiveFrame, FrameOrigin,
                                         ActionKind, Decision)
from check_fine_span_interface import positive_bytes
from check_joint_span_interface import source_alignment
from diagnose_continuous_memory import forward
from diagnose_native_generation_gradient import exact, native_forward
from episode_memory_inputs import encoder_texts
from fine_span_reader import FineSpanReader, ByteLayout
from free_reply_protocol import FreeReply, packet_sha256
from joint_optimizer import tensor_digest
from joint_span_reader import SpanFeatures
from native_memory_encoder import NativeFeatureBank, BACKBONE_SHA256, digest, sha_file, text_key
from native_token_payload import payload_mask
from neural_memory_contract import ModelBinding
from neural_memory_generation import encode_generation_input
from query_only_uncertainty import QueryOnlyUncertainty
from uncertainty_reply_controller import ResearchUncertaintyController, UncertaintyDecision

ROOT = Path(__file__).resolve().parents[1]; RESEARCH = ROOT / 'training/memory/neural-system'
FROZEN = ROOT / 'build/neural-memory-cg003-frozen-build'
GGUF = ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf'
FROZEN_SHA = {'tok_probe': 'f3d5fd147895e5875776a579d864f7a36ef16b55a4f59da00838f8b59f416adc',
              'memory_prefix_probe': '2735729604b1a52d99d1a60f48234b96ce3683d63aeec6434879bbfe7753e302',
              'memory_gradient_reference': '4c05a9c4faa7b61d03a017a3a80f43c36329c5d8af945ea5c9f1359640de9152',
              'memory_feature_probe': 'dcee4c7034c2dbc08edda5c05e93318cdc66b4c98eedbb88e3b7822707eff6e0',
              'memory_continuous_probe': '4ce9aa3f21ef95f589fb996f2a43c30a917e038f3fc4e1285bd60769691288ce'}
PANEL = [('JB-001', 'jb-000', ['role_swap', 'empty', 'ordinary_source', 'ordinary_empty', 'correct', 'value_swap']),
         ('JB-002', 'jb2-009', ['wrong_time', 'hypothetical', 'quoted', 'historical_stated', 'multi_fact', 'correct'])]
ROUTE = {'insufficient': 2, 'no_memory_needed': 0, 'supported': 1}
BUDGET = 24


def utf8_prefix(data):
    for cut in range(0, min(4, len(data)) + 1):
        try:
            return data[:len(data) - cut].decode('utf-8', errors='strict'), cut
        except UnicodeDecodeError:
            continue
    return '', len(data)


def run(a):
    reg = json.loads((RESEARCH / 'experiments/EV-001.json').read_text())
    ck = torch.load(ROOT / 'build/neural-memory-v3-010/ckpt-410.pt', weights_only=False)
    tr = json.loads((ROOT / 'build/neural-memory-v3-010/train-report.json').read_text())
    for name, want in FROZEN_SHA.items():
        if sha_file(FROZEN / name) != want: raise ValueError(f'frozen binary changed: {name}')
    fm = json.loads((ROOT / 'build/neural-memory-cg003-full-features/manifest.json').read_text())
    head = torch.from_numpy(np.load(ROOT / 'build/neural-memory-cg003-full-features/output-head.npy', allow_pickle=False))
    scale = fm['logit_scale']
    banks = {'JB-001': NativeFeatureBank(ROOT / 'build/neural-memory-cg003-full-features/tokens',
                                         expected_encoder_id=fm['encoder_identity']['encoder_id'],
                                         expected_manifest_sha256=fm['token_manifest_digest']),
             'JB-002': NativeFeatureBank(ROOT / 'build/neural-memory-jb002-features',
                                         expected_encoder_id=fm['encoder_identity']['encoder_id'],
                                         expected_manifest_sha256='e6bf0382b65bbf42b883232d118f9c9ac04052f555cf02bda7dafacbe9fbf07a')}
    binding = ModelBinding(BACKBONE_SHA256, fm['tokenizer_sha256'], fm['encoder_identity']['encoder_id'],
                           sha_file(RESEARCH / 'experiments/EV-006.json'))
    key = digest(vars(binding))
    torch.set_num_threads(4)
    from v3_model_factory import make_struct_route_reader
    reader = make_struct_route_reader(key, width=64, seed=1018).eval()
    torch.manual_seed(1021); branch = QueryOnlyUncertainty(1024, key).eval()
    reader.load_state_dict(ck['reader']); branch.load_state_dict(ck['branch'])
    if tensor_digest(reader.state_dict()) != tr['final_reader_digest'] or \
            tensor_digest(branch.state_dict()) != tr['final_branch_digest']:
        raise ValueError('checkpoint digests do not match the train report')
    codec = AppendValueCodec(FROZEN / 'tok_probe', GGUF, fm['tokenizer_sha256'])
    eos = codec.tokenizer.eos()
    root = Path(a.raw); root.mkdir(parents=True, exist_ok=False)
    replies = []; rows = []; parity_fail = 0
    try:
        for corpus, world, scenarios in PANEL:
            c = RESEARCH / 'data' / corpus
            cm = json.loads((c / 'manifest.json').read_text())
            data = {}
            for n in ('inputs', 'labels', 'index'):
                p = c / f'train.{n}.jsonl'
                if sha_file(p) != cm['file_sha256'][p.name]: raise ValueError(f'{corpus} {n} changed')
                data[n] = {r['id']: r for r in map(json.loads, p.read_text().splitlines())}
            for scenario in scenarios:
                for language in ('en', 'zh'):
                    meta = next(m for m in data['index'].values()
                                if m['world_id'] == world and m['relation_family'] == 'home_city'
                                and m['scenario'] == scenario and m['language'] == language)
                    rid = meta['id']; runtime = data['inputs'][rid]
                    prompt = tuple(encode_generation_input(runtime, codec.tokenizer).prompt_token_ids)
                    question = runtime['context']['messages'][-1]['text'] if isinstance(runtime['context'], dict) \
                        and runtime['context'].get('messages') else runtime['context'][0]['text']
                    gold_value = ''
                    if meta['scenario'] == 'supported':
                        raw_b = runtime['episodes'][0]['text'].encode()
                        vs, ve = positive_bytes(raw_b, meta).value_bytes if corpus == 'JB-001' else \
                            (None, None)
                        if corpus == 'JB-001':
                            gold_value = raw_b[vs:ve].decode()
                        else:
                            from compile_time_scoped_teacher import positive_bytes_time
                            gold_value = raw_b[positive_bytes_time(raw_b, meta).value_bytes[0]:
                                               positive_bytes_time(raw_b, meta).value_bytes[1]].decode()
                    query, sources = encoder_texts(runtime)
                    q = banks[corpus].rows[text_key(query)]
                    if sources:
                        s = banks[corpus].rows[text_key(sources[0])]
                        pieces = codec.decode_pieces(s.token_ids[1:].tolist())
                        allowed = payload_mask(runtime['episodes'][0], sources[0], pieces)
                        ranges = source_alignment(runtime['episodes'][0], sources[0], pieces, allowed)
                        raw_b = runtime['episodes'][0]['text'].encode(); source = torch.from_numpy(s.features.copy())
                    else:
                        ranges = (); raw_b = b''; allowed = []; source = torch.empty(0, 2048)
                    x = SpanFeatures(torch.from_numpy(q.features.copy()), source, torch.tensor(allowed, dtype=torch.bool),
                                     key, hashlib.sha256(raw_b).hexdigest(), digest(runtime['context']))
                    layout = ByteLayout(x, raw_b, ranges)
                    from value_transport import PayloadSnapshot, FactPayload, ReplyBinding
                    snapshot = PayloadSnapshot(binding, tensor_digest(reader.state_dict()), 'native', rid, 0,
                                               (FactPayload('source', raw_b),) if raw_b else ())
                    core = AutonomousValueController(reader, x, layout, snapshot,
                                                     ReplyBinding('native', rid, x.context_sha256),
                                                     codec, prompt, max_new_tokens=BUDGET, max_value_reads=4)
                    wrapper = ResearchUncertaintyController(core, branch, head, scale, diagnostic_only=True)
                    directory = root / rid; directory.mkdir()
                    prefix = list(prompt); emitted = []; pieces_out = []; stop = 'budget'
                    parity_ok = True; route_kinds = []; overtrigger = False
                    for pos in range(BUDGET):
                        ref = native_forward(str(FROZEN / 'memory_gradient_reference'), str(GGUF), tuple(prefix),
                                             np.zeros(1024, dtype='<f4'), directory, f'base-{pos:02d}', False)
                        h = torch.from_numpy(ref['hidden'].copy()); b = torch.from_numpy(ref['logits'].copy())
                        frame = LiveFrame(tuple(prefix), h, b, key, FrameOrigin.NATIVE_FRESH)
                        try:
                            decision = wrapper.propose(frame)
                        except Exception:
                            stop = 'budget'; overtrigger = True; break
                        overtrigger = False
                        route_kinds.append(type(decision).__name__ + ':' +
                                            (decision.kind.value if isinstance(decision, Decision) else 'unc'))
                        if isinstance(decision, UncertaintyDecision):
                            out = wrapper.pending_output
                            delta = out.residual.detach().numpy().astype('<f4')
                            forward(str(FROZEN / 'memory_continuous_probe'), str(GGUF), tuple(prefix), delta,
                                    directory, f'unc-{pos:02d}', True)
                            _nh, nb, _nc, nz = continuous_file(directory, f'unc-{pos:02d}', tuple(prefix), True,
                                                               out.residual.detach()[None])
                            if not exact(nb[0].numpy(), b.numpy()):
                                parity_ok = False; parity_fail += 1
                            try:
                                close_tol(nz, out.logits.detach()[None])
                            except ValueError:
                                parity_ok = False; parity_fail += 1
                        wrapper.commit(decision, decision.token)
                        emitted.append(decision.token); prefix.append(decision.token)
                        pieces_out.append(bytes(codec.decode_pieces([decision.token])[0]))
                        if decision.token == eos: stop = 'eos'; break
                    blob = b''.join(pieces_out); text, tail = utf8_prefix(blob)
                    reply = FreeReply(rid, ROUTE[data['labels'][rid]['state']], language, question,
                                      meta.get('query_subject') or '', gold_value, prompt, tuple(emitted),
                                      text, stop, len(emitted), parity_ok, True)
                    replies.append(reply)
                    rows.append({'id': rid, 'corpus': corpus, 'scenario': f'{scenario}/{language}',
                                 'route_target': reply.route_target, 'stop_reason': stop, 'tokens': len(emitted),
                                 'route_kinds': route_kinds[:3] + (['...'] if len(route_kinds) > 3 else []),
                                 'token_ids': list(emitted), 'prompt_ids': list(prompt),
                                 'parity_ok': parity_ok, 'start_overtrigger': overtrigger, 'utf8_tail_dropped': tail, 'reply': text})
                    print(json.dumps({'phase': 'reply', 'record': rid, 'stop': stop, 'tokens': len(emitted),
                                      'parity': parity_ok}), flush=True)
        codec.verify_identity()
    finally:
        codec.close()
    packets = [r.packet() for r in replies]
    with (root / 'blind-packets.jsonl').open('x') as f:
        for p in sorted(packets, key=lambda z: z['id']):
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    result = {'format': 'ev006-candidate-eval-v1', 'registration_sha256': sha_file(RESEARCH / 'experiments/EV-001.json'),
              'raw_root': str(root.resolve()), 'raw_sha256': {str(p.relative_to(root)): sha_file(p) for p in sorted(root.rglob('*')) if p.is_file()},
              'rows': rows, 'packet_sha256': {p['id']: packet_sha256(p) for p in packets},
              'parity_failures': parity_fail, 'truncated': sum(1 for r in replies if r.stop_reason == 'budget'),
              'route_distribution': {str(t): sum(1 for r in replies if r.route_target == t) for t in (0, 1, 2)},
              'protocol_passed': parity_fail == 0 and len(replies) == 24,
              'semantic_review_pending': True, 'deployed': False}
    path = Path(a.output); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', required=True); p.add_argument('--output', required=True)
    run(p.parse_args())
