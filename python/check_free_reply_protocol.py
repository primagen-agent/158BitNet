"""DG-023 native free-reply protocol. Zero updates; no serving, no training."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from append_value_transport import AppendValueCodec
from audit_joint_generation import continuous_file
from autonomous_value_controller import AutonomousValueController, LiveFrame, FrameOrigin
from check_fine_span_interface import positive_bytes
from check_joint_span_interface import source_alignment
from diagnose_continuous_memory import forward
from episode_memory_inputs import encoder_texts
from fine_span_reader import FineSpanReader, ByteLayout
from free_reply_protocol import FreeReply, packet_sha256
from joint_optimizer import FEATURE_DIGEST, tensor_digest
from joint_span_reader import SpanFeatures
from joint_training_data import DATA_DIGEST
from native_memory_encoder import NativeFeatureBank, BACKBONE_SHA256, digest, sha_file, text_key
from native_token_payload import payload_mask
from neural_memory_contract import ModelBinding
from neural_memory_generation import encode_generation_input
from query_only_uncertainty import QueryOnlyUncertainty
from uncertainty_reply_controller import ResearchUncertaintyController, UncertaintyDecision
from value_transport import PayloadSnapshot, FactPayload, ReplyBinding

ROOT = Path(__file__).resolve().parents[1]; RESEARCH = ROOT / 'training/memory/neural-system'
FROZEN = ROOT / 'build/neural-memory-cg003-frozen-build'
FROZEN_SHA = {'tok_probe': 'f3d5fd147895e5875776a579d864f7a36ef16b55a4f59da00838f8b59f416adc',
              'memory_prefix_probe': '2735729604b1a52d99d1a60f48234b96ce3683d63aeec6434879bbfe7753e302',
              'memory_gradient_reference': '4c05a9c4faa7b61d03a017a3a80f43c36329c5d8af945ea5c9f1359640de9152',
              'memory_feature_probe': 'dcee4c7034c2dbc08edda5c05e93318cdc66b4c98eedbb88e3b7822707eff6e0',
              'memory_continuous_probe': '4ce9aa3f21ef95f589fb996f2a43c30a917e038f3fc4e1285bd60769691288ce'}
PRIOR_REPORTS = {RESEARCH / 'reviews/DG-019/zero-update.json': '5da3ffa91a74d5d5bfbefb0a7571adb7e4366b46d8e5e24083814466e697b533',
                 RESEARCH / 'reviews/DG-020/native-decisions.json': 'f259c416bc25edaf852b085648c9331d720f0675ed35de3114890805a89a665a',
                 RESEARCH / 'reviews/DG-021/zero-update.json': 'ca508eaa66b3eaff19987b9cae07674c61078017cd5782f8d0f4b9f304a1c9a1',
                 RESEARCH / 'reviews/DG-022/zero-update.json': '907d750d0df14ab568e1e9f37a338acd3d0c6995ecf8cad4eca27aa63823c511'}
PANEL = [('uncertainty', 'role_swap', 'en'), ('uncertainty', 'role_swap', 'zh'),
         ('uncertainty', 'empty', 'en'), ('uncertainty', 'empty', 'zh'),
         ('normal', 'ordinary_source', 'en'), ('normal', 'ordinary_source', 'zh'),
         ('normal', 'ordinary_empty', 'en'), ('normal', 'ordinary_empty', 'zh'),
         ('supported', 'correct', 'en'), ('supported', 'correct', 'zh'),
         ('supported', 'value_swap', 'en'), ('supported', 'value_swap', 'zh')]
EXPECTED_STATE = {'uncertainty': 'insufficient', 'normal': 'no_memory_needed', 'supported': 'supported'}
ROUTE = {'uncertainty': 2, 'normal': 0, 'supported': 1}
REPLY_BUDGET = 24
CONTINUOUS_CALLS = 0


def select(index, scenario, language):
    matches = [r for r in index.values() if r['world_id'] == 'jb-000' and r['relation_family'] == 'home_city' and
               r['language'] == language and r['scenario'] == scenario]
    if len(matches) != 1: raise ValueError(f'selection {scenario}/{language} is not unique')
    return matches[0]['id']


def runtime_stack(tokens, codec, runtime, binding, reader_digest, record_id, key):
    query, sources = encoder_texts(runtime); q = tokens.rows[text_key(query)]
    if codec.tokenizer.encode(query, True) != q.token_ids.tolist(): raise ValueError('query tokens changed')
    if sources:
        s = tokens.rows[text_key(sources[0])]
        pieces = codec.decode_pieces(s.token_ids[1:].tolist()); allowed = payload_mask(runtime['episodes'][0], sources[0], pieces)
        ranges = source_alignment(runtime['episodes'][0], sources[0], pieces, allowed); raw = runtime['episodes'][0]['text'].encode()
        source = torch.from_numpy(s.features.copy())
    else:
        ranges = (); raw = b''; allowed = []; source = torch.empty(0, 2048)
    x = SpanFeatures(torch.from_numpy(q.features.copy()), source, torch.tensor(allowed, dtype=torch.bool), key,
                     hashlib.sha256(raw).hexdigest(), digest(runtime['context']))
    layout = ByteLayout(x, raw, ranges)
    snapshot = PayloadSnapshot(binding, reader_digest, 'native', record_id, 0, (FactPayload('source', raw),) if raw else ())
    return x, layout, snapshot, ReplyBinding('native', record_id, x.context_sha256)


def frame_at(directory, name, prefix):
    """Fresh full-prefix C forward with zero residual; hard-gates zero correction."""
    global CONTINUOUS_CALLS
    forward(str(FROZEN / 'memory_continuous_probe'), str(ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf'),
            tuple(prefix), np.zeros(1024, dtype='<f4'), directory, name, True)
    CONTINUOUS_CALLS += 1
    h, b, c, z = continuous_file(directory, name, tuple(prefix), True, torch.zeros(1, 1024))
    if torch.count_nonzero(c) or not torch.equal(b[0], z[0]): raise ValueError('zero-residual C frame is not base')
    return h[0], b[0], z[0]


def utf8_prefix(data):
    """Longest strict-UTF-8 prefix; returns text and dropped tail length."""
    for cut in range(0, min(4, len(data)) + 1):
        try:
            return data[:len(data) - cut].decode('utf-8', errors='strict'), cut
        except UnicodeDecodeError:
            continue
    return '', len(data)


def run(a):
    global CONTINUOUS_CALLS
    registration = RESEARCH / 'experiments/DG-023.json'; reg = json.loads(registration.read_text())
    for name, want in FROZEN_SHA.items():
        if sha_file(FROZEN / name) != want: raise ValueError(f'frozen binary identity changed: {name}')
    for path, want in PRIOR_REPORTS.items():
        if sha_file(path) != want: raise ValueError(f'prior report changed: {path.name}')
    dg020 = json.loads((RESEARCH / 'reviews/DG-020/native-decisions.json').read_text())
    dg021 = json.loads((RESEARCH / 'reviews/DG-021/zero-update.json').read_text())
    dg022 = json.loads((RESEARCH / 'reviews/DG-022/zero-update.json').read_text())
    for name, h in {**dg020['source_sha256'], **dg021['source_sha256'], **dg022['source_sha256']}.items():
        if sha_file(ROOT / 'python' / name) != h and name != 'check_uncertainty_reply_supervision.py':
            if name in ('free_reply_protocol.py', 'check_free_reply_protocol.py'): continue
            raise ValueError(f'prior implementation changed: {name}')
    root = Path(a.raw); root.mkdir(parents=True, exist_ok=False)
    feature_root = ROOT / 'build/neural-memory-cg003-full-features'; m = json.loads((feature_root / 'manifest.json').read_text())
    if digest(m) != FEATURE_DIGEST or sha_file(feature_root / 'output-head.npy') != m['output_head_sha256']:
        raise ValueError('frozen feature/head identity changed')
    head = torch.from_numpy(np.load(feature_root / 'output-head.npy', allow_pickle=False)); scale = m['logit_scale']
    corpus = RESEARCH / 'data/JB-001'; manifest = json.loads((corpus / 'manifest.json').read_text())
    if digest(manifest) != DATA_DIGEST: raise ValueError('corpus changed')
    data = {}
    for name in ('inputs', 'index', 'labels'):
        path = corpus / f'train.{name}.jsonl'
        if sha_file(path) != manifest['file_sha256'][path.name]: raise ValueError(f'training {name} changed')
        data[name] = {r['id']: r for r in map(json.loads, path.read_text().splitlines())}
    binding = ModelBinding(BACKBONE_SHA256, m['tokenizer_sha256'], m['encoder_identity']['encoder_id'], sha_file(registration))
    key = digest(vars(binding))
    torch.set_num_threads(4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(reg['initialization']['reader_seed']); reader = FineSpanReader(2048, 1024, key, width=reg['initialization']['width']).eval()
        torch.manual_seed(reg['initialization']['uncertainty_seed']); branch = QueryOnlyUncertainty(1024, key).eval()
    reader_digest = tensor_digest(reader.state_dict()); initial = tensor_digest(branch.state_dict())
    if reader_digest != dg020['initial_parameter_digest']: raise ValueError('reader initialization changed')
    if initial != dg021['uncertainty_parameter_digest']: raise ValueError('uncertainty initialization changed')
    codec = AppendValueCodec(FROZEN / 'tok_probe', ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf', m['tokenizer_sha256'])
    tokens = NativeFeatureBank(feature_root / 'tokens', expected_encoder_id=m['encoder_identity']['encoder_id'],
                               expected_manifest_sha256=m['token_manifest_digest'])
    eos = codec.tokenizer.eos()
    replies = []; rows = []
    try:
        for group, scenario, language in PANEL:
            rid = select(data['index'], scenario, language)
            if data['labels'][rid]['state'] != EXPECTED_STATE[group]: raise ValueError(f'{scenario}/{language} is not {group}')
            runtime = data['inputs'][rid]; meta = data['index'][rid]
            prompt = tuple(encode_generation_input(runtime, codec.tokenizer).prompt_token_ids)
            question = runtime['context']['messages'][-1]['text'] if isinstance(runtime['context'], dict) \
                and runtime['context'].get('messages') else runtime['context'][0]['text']
            gold_value = ''
            if group == 'supported':
                raw = runtime['episodes'][0]['text'].encode()
                vs, ve = positive_bytes(raw, meta).value_bytes
                gold_value = raw[vs:ve].decode()
            elif group == 'normal':
                import re
                text = data['labels'][rid]['response']
                mm = re.search(r'\d+', text)
                gold_value = mm.group() if mm else ''
            x, layout, snapshot, reply_binding = runtime_stack(tokens, codec, runtime, binding, reader_digest, rid, key)
            directory = root / rid; directory.mkdir()
            core = AutonomousValueController(reader, x, layout, snapshot, reply_binding, codec, prompt, max_new_tokens=REPLY_BUDGET)
            wrapper = ResearchUncertaintyController(core, branch, head, scale, diagnostic_only=True)
            prefix = list(prompt); emitted = []; pieces = []; stop = 'budget'
            zero_bits = True; argmax_ok = True
            for pos in range(REPLY_BUDGET):
                h, b, z = frame_at(directory, f'pos-{pos:02d}', prefix)
                decision = wrapper.propose(LiveFrame(tuple(prefix), h, b, key, FrameOrigin.NATIVE_FRESH))
                if type(decision) is not UncertaintyDecision:
                    raise ValueError(f'{rid}: untrained model did not route to uncertainty at position {pos}')
                argmax_ok = argmax_ok and decision.token == int(z.argmax())
                out = wrapper.pending_output; out.validate(branch)
                zero_bits = zero_bits and torch.equal(out.logits.view(torch.int32), b.view(torch.int32))
                wrapper.commit(decision, decision.token)
                emitted.append(decision.token); prefix.append(decision.token)
                pieces.append(bytes(codec.decode_pieces([decision.token])[0]))
                if decision.token == eos: stop = 'eos'; break
            blob = b''.join(pieces); text, tail = utf8_prefix(blob)
            reply = FreeReply(rid, ROUTE[group], language, question, meta['query_subject'], gold_value,
                              prompt, tuple(emitted), text, stop, len(emitted), zero_bits, argmax_ok)
            replies.append(reply)
            base_tokens = []
            if group in ('normal', 'supported'):
                bprefix = list(prompt); bstop = 'budget'
                for pos in range(REPLY_BUDGET):
                    _h, _b, bz = frame_at(directory, f'base-{pos:02d}', bprefix)
                    tok = int(bz.argmax()); base_tokens.append(tok); bprefix.append(tok)
                    if tok == eos: bstop = 'eos'; break
                if tuple(base_tokens) != tuple(emitted):
                    raise ValueError(f'{rid}: untrained branch reply diverged from base reply')
            rows.append({'id': rid, 'group': group, 'scenario': f'{scenario}/{language}',
                         'route_target': ROUTE[group], 'stop_reason': stop, 'tokens': len(emitted),
                         'token_ids': list(emitted), 'prompt_ids': list(prompt),
                         'utf8_tail_dropped': tail, 'reply': text,
                         'branch_equals_base': tuple(base_tokens) == tuple(emitted) if base_tokens else None})
            print(json.dumps({'phase': 'free_reply', 'record': rid, 'stop': stop, 'tokens': len(emitted)}), flush=True)
        codec.verify_identity()
    finally:
        codec.close()
    if tensor_digest(reader.state_dict()) != reader_digest or tensor_digest(branch.state_dict()) != initial:
        raise ValueError('parameters changed during generation')
    packets = [r.packet() for r in replies]
    with (root / 'blind-packets.jsonl').open('x') as f:
        for p in sorted(packets, key=lambda x: x['id']):
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    result = {'format': 'dg023-free-reply-protocol-v1', 'registration_sha256': sha_file(registration),
              'environment': {'frozen_binary_sha256': FROZEN_SHA},
              'raw_root': str(root.resolve()), 'raw_sha256': {str(p.relative_to(root)): sha_file(p) for p in sorted(root.rglob('*')) if p.is_file()},
              'rows': rows, 'C_forward_calls': CONTINUOUS_CALLS,
              'packet_sha256': {p['id']: packet_sha256(p) for p in packets},
              'optimizer_steps': 0, 'parameters_unchanged': True,
              'replies_generated': len(replies), 'truncated': sum(1 for r in replies if r.stop_reason == 'budget'),
              'eos_stopped': sum(1 for r in replies if r.stop_reason == 'eos'),
              'zero_residual_bitwise_all': all(r.zero_residual_bitwise for r in replies),
              'decision_matches_argmax_all': all(r.decision_matches_argmax for r in replies),
              'branch_equals_base_all': all(row['branch_equals_base'] is not False for row in rows),
              'semantic_review_pending': True, 'V2_complete': False, 'deployment_approved': False,
              'passed_protocol': len(replies) == 12}
    path = Path(a.output); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--raw', required=True); p.add_argument('--output', required=True)
    run(p.parse_args())
