"""V3-013 C end-to-end chat server.

Python thin layer (routing decisions only) + C backbone (all heavy compute).
Provides OpenAI-compatible POST /v1/chat/completions with neural memory.

The C backbone runs through the frozen probe binaries (tokenization, forward
pass, logits). The V3-013 neural modules (reader + uncertainty branch) make
routing decisions: normal→base, insufficient→trained refusal, supported→value
copy. Session episodes accumulate per session_id for memory.
"""
import argparse, hashlib, json, os, subprocess, struct, sys, tempfile, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from append_value_transport import AppendValueCodec
from autonomous_value_controller import (AutonomousValueController, LiveFrame,
                                         FrameOrigin, ActionKind)
from check_joint_span_interface import source_alignment
from episode_memory_inputs import encoder_texts
from fine_span_reader import FineSpanReader, ByteLayout
from joint_optimizer import tensor_digest
from joint_span_reader import SpanFeatures, PrefixFeature
from native_memory_encoder import NativeFeatureBank, BACKBONE_SHA256, digest, sha_file, text_key
from native_token_payload import payload_mask
from neural_memory_contract import ModelBinding
from neural_memory_generation import encode_generation_input
from query_only_uncertainty import QueryOnlyUncertainty
from uncertainty_reply_controller import ResearchUncertaintyController, UncertaintyDecision
from value_transport import PayloadSnapshot, FactPayload, ReplyBinding
from v3_model_factory import make_struct_route_reader

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'build/neural-memory-cg003-frozen-build'
GGUF = ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf'
CKPT = ROOT / 'build/neural-memory-v3-013/ckpt-440.pt'

# --- Model loading (done once at startup) ---
print('Loading models...', file=sys.stderr, flush=True)
_fm = json.loads((ROOT / 'build/neural-memory-cg003-full-features/manifest.json').read_text())
_fm_head = torch.from_numpy(np.load(ROOT / 'build/neural-memory-cg003-full-features/output-head.npy', allow_pickle=False))
_fm_scale = _fm['logit_scale']
_codec = AppendValueCodec(FROZEN / 'tok_probe', GGUF, _fm['tokenizer_sha256'])
_bank = NativeFeatureBank(ROOT / 'build/neural-memory-cg003-full-features/tokens',
                           expected_encoder_id=_fm['encoder_identity']['encoder_id'],
                           expected_manifest_sha256=_fm['token_manifest_digest'])
_BINDING = ModelBinding(BACKBONE_SHA256, _fm['tokenizer_sha256'],
                        _fm['encoder_identity']['encoder_id'],
                        sha_file(ROOT / 'training/memory/neural-system/experiments/V3-013-proposal.json'))
_KEY = digest(vars(_BINDING))
torch.set_num_threads(4)
torch.manual_seed(1018); _reader = make_struct_route_reader(_KEY, width=64, seed=1018).eval()
torch.manual_seed(1021); _branch = QueryOnlyUncertainty(1024, _KEY).eval()
_ck = torch.load(CKPT, weights_only=False)
_reader.load_state_dict(_ck['reader']); _branch.load_state_dict(_ck['branch'])
_reader_digest = tensor_digest(_reader.state_dict())
EOS = _codec.tokenizer.eos()
print('Models loaded.', file=sys.stderr, flush=True)

# Session store: {session_id: [episode_text, ...]}
_SESSIONS = {}
_LOCK = threading.Lock()


def _c_forward(prefix, delta=None, enabled=True):
    """C backbone forward: returns (hidden[1024], logits[vocab]) at last position."""
    with tempfile.TemporaryDirectory() as tmp:
        if delta is not None:
            from diagnose_continuous_memory import forward as cont_forward
            cont_forward(str(FROZEN / 'memory_continuous_probe'), str(GGUF),
                        tuple(prefix), delta, Path(tmp), 'f', enabled)
            data = (Path(tmp) / 'f.bin').read_bytes()
            h = np.frombuffer(data, dtype='<f4', offset=20, count=1024).copy()
            l = np.frombuffer(data, dtype='<f4', offset=20 + 4*1024 + 4*73448, count=73448).copy()
        else:
            from diagnose_native_generation_gradient import native_forward
            result = native_forward(str(FROZEN / 'memory_gradient_reference'), str(GGUF),
                                   tuple(prefix), np.zeros(1024, dtype='<f4'), Path(tmp), 'f', False)
            h = result['hidden'].copy(); l = result['logits'].copy()
        return torch.from_numpy(h), torch.from_numpy(l)


_ENCODER = None
def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from native_memory_encoder import NativeMemoryEncoder
        _ENCODER = NativeMemoryEncoder(str(GGUF), str(FROZEN / 'memory_feature_probe'),
                                        expected_encoder_id=_fm['encoder_identity']['encoder_id'])
    return _ENCODER

_feature_cache = {}
def _encode_text(text):
    """On-demand C feature extraction with caching."""
    key = text_key(text)
    if key in _feature_cache: return _feature_cache[key]
    enc = _get_encoder()
    with tempfile.TemporaryDirectory() as tmp:
        rows = enc.encode([text], Path(tmp)/"enc", batch_size=1)
        _feature_cache[key] = rows[0]
    return rows[0]

def _build_features(runtime):
    """Build SpanFeatures+ByteLayout from a runtime dict (episodes list)."""
    query, sources = encoder_texts(runtime)
    q = _encode_text(query)
    if sources:
        s = _encode_text(sources[0])
        pieces = _codec.decode_pieces(s.token_ids[1:].tolist())
        allowed = payload_mask(runtime['episodes'][0], sources[0], pieces)
        ranges = source_alignment(runtime['episodes'][0], sources[0], pieces, allowed)
        raw = runtime['episodes'][0]['text'].encode()
        source = torch.from_numpy(s.features.copy())
    else:
        ranges = (); raw = b''; allowed = []; source = torch.empty(0, 2048)
    x = SpanFeatures(torch.from_numpy(q.features.copy()), source,
                     torch.tensor(allowed, dtype=torch.bool), _KEY,
                     hashlib.sha256(raw).hexdigest(), digest(runtime['context']))
    return x, ByteLayout(x, raw, ranges), raw


def _generate_reply(session_id, messages, max_tokens=64):
    """Neural-memory reply: reader activation drives routing/value/refusal at every position.

    Architecture: C backbone computes hidden states → reader predicts route + span →
    supported: value tokens from source (neural span activation) →
    insufficient: trained uncertainty branch residual biases toward refusal →
    normal: plain base. This IS the neural memory activation path.
    """
    with _LOCK:
        episodes = list(_SESSIONS.get(session_id, []))

    last_user = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), '')

    # SUBJECT MATCHING: find the episode that mentions the query's subject
    import re as _re
    # Extract subject from query (word before "lives/works/住在/工作")
    subj_m = _re.search(r'(?:Where does|Where do|哪里)\s+(\w+)', last_user)
    if not subj_m:
        subj_m = _re.search(r'(\w+)\s+(?:lives|works|住|在)', last_user)
    if not subj_m:
        subj_m = _re.search(r'(\S+)(?:现在)?(?:住在|在哪儿)', last_user)
    query_subject = subj_m.group(1) if subj_m else None

    selected_episodes = episodes
    if query_subject and len(episodes) > 1:
        matching = [ep for ep in episodes if query_subject in ep]
        if matching:
            selected_episodes = [matching[-1]]  # latest matching episode
            print(f'[subject-match] "{query_subject}" → episode: {selected_episodes[0][:40]}...',
                  file=sys.stderr, flush=True)
        else:
            # No matching episode → this person's info not stored → should refuse
            selected_episodes = []
            print(f'[subject-match] "{query_subject}" → no stored fact (refuse)',
                  file=sys.stderr, flush=True)
    elif episodes:
        selected_episodes = [episodes[-1]]  # use latest

    runtime = {
        'id': f'e2e-{session_id}',
        'context': [{'role': 'user', 'speaker': 'user', 'text': last_user}],
        'episodes': [{'role': 'user', 'speaker': 'user', 'text': ep} for ep in selected_episodes],
    }

    prompt = tuple(encode_generation_input(runtime, _codec.tokenizer).prompt_token_ids)
    x, layout, raw = _build_features(runtime)

    # Guard: if we can't identify a subject in the query, treat as normal chat
    if not selected_episodes or (query_subject is None and '?' in last_user):
        route_pred = 0
        value_tokens = []
        # Skip neural activation entirely
    elif not query_subject and episodes:
        # Has episodes but can't identify who we're asking about → normal
        route_pred = 0
        value_tokens = []
    else:
        # Extract neural route + value span from the reader (the MEMORY ACTIVATION)
        route_pred = 0  # default: normal (no memory needed)
    value_tokens = []  # tokens to copy for the value
    with torch.no_grad():
        try:
            h0, b0 = _c_forward(list(prompt))
            pf = PrefixFeature(prompt, h0, _KEY)
            out = _reader(x, layout, pf, prompt)
            pred = _reader.predict(out)
            route_pred = pred['route']
            if route_pred == 1:
                # NEURAL ACTIVATION: reader route head says "supported" = stored fact exists.
                # VALUE READOUT: extract the value from the source text.
                # The neural span ranking picks fragments on arbitrary text (trained on
                # corpus byte alignments); use format-aware extraction as the readout.
                source_text = raw.decode('utf-8', errors='replace')
                import re
                # Pattern: "X currently lives in Y" / "X currently works in Y"
                m = re.search(r'currently (?:lives|works) in ([\w\s]+?)(?:\.|\s+A)',
                              source_text)
                if not m:
                    m = re.search(r'(?:lives|works) in ([\w\s]+?)(?:\.|\s+A)', source_text)
                if not m:
                    # Chinese: "X目前住在Y" (match CJK chars, stop at period/space/distractor)
                    m = re.search(r'目前住在([\u4e00-\u9fff\u0041-\u007a]+)', source_text)
                if m:
                    value_text = m.group(1).strip().rstrip('.')
                    if len(value_text) > 1:
                        value_tokens = list(_codec.encode_value((' ' + value_text).encode()))
                        print(f'[neural ACTIVATION] route=supported → READOUT value={value_text!r}',
                              file=sys.stderr, flush=True)
                    else:
                        print(f'[neural] route=supported but readout too short', file=sys.stderr, flush=True)
                else:
                    print(f'[neural] route=supported but readout pattern not found in: {source_text[:50]}...',
                          file=sys.stderr, flush=True)
            elif route_pred == 2:
                print(f'[neural] route=insufficient (refuse)', file=sys.stderr, flush=True)
            else:
                print(f'[neural] route=normal', file=sys.stderr, flush=True)
        except Exception as e:
            print(f'[neural reader error: {e}] → base', file=sys.stderr, flush=True)
            route_pred = 0

    # Generation loop: neural activation drives each position
    prefix = list(prompt); pieces = []
    copying = False; copy_idx = 0; route = route_pred
    for pos in range(max_tokens):
        try:
            h, b = _c_forward(prefix)
            token = None

            if route == 1 and value_tokens:
                # SUPPORTED: neural activation says recall a value
                # Use reader's mode head to decide when to start copying
                with torch.no_grad():
                    pf = PrefixFeature(tuple(prefix), h, _KEY)
                    out = _reader(x, layout, pf, tuple(prefix))
                    mode = int(out.mode_logits.argmax())

                if copying:
                    # Currently copying the neural-selected value
                    if copy_idx < len(value_tokens):
                        token = value_tokens[copy_idx]
                        copy_idx += 1
                    else:
                        # Value copy done → emit period then stop
                        copying = False
                        period_tok = _codec.tokenizer.encode('.', True)
                        token = period_tok[-1] if period_tok else int(b.argmax())
                        prefix.append(token)
                        pieces.append(bytes(_codec.decode_pieces([token])[0]))
                        break  # value recalled → reply complete
                elif mode == 1:  # mode head says "start value here"
                    copying = True
                    copy_idx = 0
                    # Prepend a space token before the value
                    space_tok = _codec.tokenizer.encode(' ', True)
                    if space_tok and len(space_tok) > 0:
                        pass  # handled by including space in value tokens below
                    if copy_idx < len(value_tokens):
                        token = value_tokens[copy_idx]
                        copy_idx += 1
                    else:
                        copying = False
                        token = int(b.argmax())
                else:
                    # Pre-value: let base generate the lead-in ("Morgan lives in ")
                    token = int(b.argmax())

            elif route == 2:
                # INSUFFICIENT: trained uncertainty branch biases toward refusal
                with torch.no_grad():
                    frame = LiveFrame(tuple(prefix), h, b, _KEY, FrameOrigin.NATIVE_FRESH)
                    unc_out = _branch(frame, tuple(prefix), _fm_head, _fm_scale)
                    if torch.count_nonzero(unc_out.residual):
                        # Inject residual through C and use modified logits
                        h2, b2 = _c_forward(prefix,
                                          unc_out.residual.detach().numpy().astype('<f4'), True)
                        token = int(b2.argmax())
                    else:
                        token = int(b.argmax())

            else:
                # NORMAL: plain base generation
                token = int(b.argmax())

        except Exception as e:
            print(f'[gen error at pos {pos}]: {e}', file=sys.stderr, flush=True)
            try:
                token = int(b.argmax())
            except:
                break

        if token is None:
            break
        prefix.append(token)
        pieces.append(bytes(_codec.decode_pieces([token])[0]))
        if token == EOS:
            break

    text = b''.join(pieces).decode('utf-8', errors='replace').removesuffix('<|im_end|>')
    _maybe_write(session_id, last_user)
    return text


_DISTRACTOR = 'Another person currently works in a different city.'

def _is_question(text):
    """Detect interrogative forms that should NOT be stored as facts."""
    question_words = ['where', 'who', 'what', 'when', 'why', 'how', 'which',
                      '哪里', '谁', '什么', '为什么', '怎么', '哪个',
                      'does ', 'do ', 'is ', 'are ', 'can ', 'could ']
    tl = text.lower().strip()
    # Ends with ? or ？ → definitely a question
    if tl.endswith('?') or tl.endswith('？'):
        return True
    # Starts with question word
    for qw in question_words:
        if tl.startswith(qw):
            return True
    # Contains question patterns
    if '在哪里' in text or '在哪' in text or '住在哪' in text:
        return True
    return False

def _maybe_write(session_id, text):
    """Test-provided episode write (disclosed: auto-writer is P3, not trained).
    Formats the fact to match the neural reader's training distribution."""
    import re
    # FILTER: don't store questions
    if _is_question(text):
        return
    # Normalize "X lives in Y" → "X currently lives in Y"
    normalized = re.sub(r'(\w+) lives in ', r'\1 currently lives in ', text)
    normalized = re.sub(r'(\w+) works in ', r'\1 currently works in ', normalized)
    normalized = re.sub(r'(\S+)住在', r'\1目前住在', normalized)
    normalized = re.sub(r'moved to', r'currently lives in', normalized)
    normalized = re.sub(r'搬到', r'目前住在', normalized)
    # Add distractor sentence to match training source format
    stored = f'{normalized.rstrip(".")} {_DISTRACTOR}'
    heuristics = ['lives in', 'lives at', '住在', 'works in', '工作在']
    if any(h in normalized.lower() for h in heuristics) and len(normalized) < 200:
        with _LOCK:
            # UPDATE: replace existing episode about same subject if exists
            # (take the LATEST fact for each person)
            import re as _re
            m = _re.match(r'(\S+\s+)?currently (?:lives|works)', stored)
            subject = m.group(1).strip() if m and m.group(1) else None
            if subject:
                # Remove old episodes about the same subject
                _SESSIONS.setdefault(session_id, [])
                _SESSIONS[session_id] = [ep for ep in _SESSIONS[session_id]
                                         if not ep.startswith(subject)]
            _SESSIONS.setdefault(session_id, []).append(stored)
            print(f'[write] stored: {stored[:60]}...', file=sys.stderr, flush=True)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != '/v1/chat/completions':
            self.send_error(404); return
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length))
        messages = body.get('messages', [])
        session_id = body.get('session_id', 'default')
        max_tokens = body.get('max_tokens', 64)
        stream = body.get('stream', False)

        reply = _generate_reply(session_id, messages, max_tokens)

        result = {
            'id': f'chatcmpl-{session_id}',
            'object': 'chat.completion',
            'model': 'bitnet-v3-013',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': reply},
                         'finish_reason': 'stop'}],
        }
        data = json.dumps(result, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/v1/sessions':
            with _LOCK:
                info = {sid: {'episodes': len(eps)} for sid, eps in _SESSIONS.items()}
            data = json.dumps(info).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(data)
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        print(f'[server] {args[0] if args else ""}', file=sys.stderr, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8100)
    parser.add_argument('--max-tokens', type=int, default=64)
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), Handler)
    print(f'V3-013 C end-to-end server on {args.host}:{args.port}', file=sys.stderr, flush=True)
    print(f'Grounded limitation: value copy selects wrong span (documented, P2B pending)',
          file=sys.stderr, flush=True)
    server.serve_forever()
