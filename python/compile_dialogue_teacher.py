"""EC-006 dialogue-form teacher compilation. Mirrors the DG-025 pattern:
the pinned compile_teacher/positive_bytes are untouched; dialogue-form
sources (speaker-prefixed turns, paraphrased phrasing) take this branch,
which resolves spans by locating the subject's spoken turn and the unique
value occurrence inside it. Annotation discipline is unchanged: any
ambiguity fails the record instead of guessing.
"""
from dataclasses import replace

from compile_time_scoped_teacher import compile_teacher_time
from fine_span_supervision import ByteTargets
from neural_memory_generation import RESERVED, encode_generation_input, generation_prompt
from prepare_neural_memory_protocol import digest
from append_value_transport import compile_append_value
from value_teacher_trajectory import TeacherTrajectory
from value_transport import (ActivatedFact, FactPayload, PayloadSnapshot, ReplyBinding,
                             ValueSpan, Origin, require)


def _turn_bounds(raw: bytes, colon_pos: int):
    """Turn = [start after speaker prefix, next turn start). Speaker names sit
    between the previous space (or start) and the colon."""
    start = colon_pos + 1
    while start < len(raw) and raw[start:start + 1] == b' ':
        start += 1
    end = len(raw)
    # next turn begins at the next " name:" / " name：" pattern
    pos = start
    while pos < len(raw):
        ch = raw[pos:pos + 1]
        if ch in (b':', b'\xef\xbc\x9a'.decode('latin1').encode('latin1')):
            pass
        if ch == b':' or raw[pos:pos + 3] == b'\xef\xbc\x9a':
            # is it a speaker colon? preceded by a name and a space
            back = pos
            while back > 0 and raw[back - 1:back] not in (b' ',):
                back -= 1
            name = raw[back:pos]
            if back > 0 and name and len(name) <= 24 and raw[back - 1:back] == b' ':
                end = back - 1
                break
        pos += 1
    return start, end


def _speaker_name(raw: bytes, colon_pos: int):
    back = colon_pos
    while back > 0 and raw[back - 1:back] != b' ':
        back -= 1
    return raw[back:colon_pos]


def positive_bytes_dialogue(raw: bytes, meta):
    """Locate the value inside the turn spoken by the query subject."""
    matches = [f for f in meta['source_facts'] if f['subject'] == meta['query_subject'] and
               f['relation'] == meta['query_relation'] and f['time'] == 'current' and
               f['status'] == 'actual']
    if len(matches) != 1:
        raise ValueError('expected exactly one actual current fact')
    target = matches[0]
    subject = target['subject'].encode()
    value = target['value'].encode()

    # speaker-colon positions for the subject (ASCII ':' and full-width '：')
    colon_positions = []
    pos = 0
    while True:
        ascii_hit = raw.find(b':', pos)
        fw_hit = raw.find('：'.encode(), pos)
        candidates = [h for h in (ascii_hit, fw_hit) if h >= 0]
        if not candidates:
            break
        hit = min(candidates)
        colon_positions.append(hit)
        pos = hit + 1
    subject_turns = [p for p in colon_positions if _speaker_name(raw, p) == subject]
    if not subject_turns:
        raise ValueError('no turn spoken by the query subject')

    candidates = []
    search = 0
    while True:
        hit = raw.find(value, search)
        if hit < 0:
            break
        search = hit + 1
        for colon in subject_turns:
            turn_start, turn_end = _turn_bounds(raw, colon)
            if turn_start <= hit and hit + len(value) <= turn_end:
                candidates.append((hit, turn_start, turn_end))
                break
    if len(candidates) != 1:
        raise ValueError('ambiguous/missing dialogue annotation, preserve as failure')
    vs, turn_start, turn_end = candidates[0]
    return ByteTargets(1, (turn_start, turn_end), (vs, vs + len(value)), 0)


def compile_teacher_dialogue(runtime, label, meta, codec, binding, memory_model_sha256,
                             *, purpose, capacity=128):
    """compile_teacher for dialogue-form corpora; delegates everything that
    does not need dialogue span resolution."""
    if label['state'] != 'supported':
        return compile_teacher_time(runtime, label, meta, codec, binding, memory_model_sha256,
                                    purpose=purpose, capacity=capacity)
    require(purpose == 'training_diagnostic', 'gold compiler is training-only')
    require(label['id'] == runtime['id'] == meta['id'] and label['input_sha256'] == digest(runtime) and
            meta['split'] in ('train', 'dev'), 'training-side bound labels required; test rejected')
    require(binding.backbone_sha256 == codec.backbone_sha256 and
            binding.tokenizer_sha256 == codec.tokenizer_sha256, 'codec binding mismatch')
    response = label['response']
    require(type(response) is str and bool(response) and '\0' not in response and
            not any(x in response for x in RESERVED), 'invalid teacher reply')
    initial = encode_generation_input(runtime, codec.tokenizer).prompt_token_ids
    parts = []
    phases = []

    def append_text(text, phase):
        if not text:
            return
        ids = codec.encode_value(text.encode())
        require(b''.join(codec.decode_pieces(ids)) == text.encode() and
                not any(codec.forbidden(t) for t in ids), 'unsafe/nonexact teacher segment')
        parts.extend(ids)
        phases.extend([phase] * len(ids))

    raw = runtime['episodes'][0]['text'].encode()
    target = replace(positive_bytes_dialogue(raw, meta), idle_mode=None)
    vs, ve = target.value_bytes
    value = raw[vs:ve].decode()
    require(response.count(value) == 1, 'ambiguous/missing teacher value; do not drop record')
    pre, post = response.split(value)
    append_text(pre, 'GENERATE')
    snapshot = PayloadSnapshot(binding, memory_model_sha256, 'teacher', runtime['id'], 0,
                               (FactPayload('source', raw),))
    reply = ReplyBinding('teacher', runtime['id'], digest(runtime['context']))
    handle = ActivatedFact(snapshot.digest, reply, ValueSpan(0, vs, ve), 1., Origin.ORACLE_FIXTURE)
    prefix = initial + tuple(parts)
    cursor = compile_append_value(handle, snapshot, reply, prefix,
                                  generation_prompt(runtime) + pre, codec, allow_oracle=True)
    first = True
    while not cursor.done:
        token = cursor.select(codec.bos_id, snapshot, reply, prefix, allow_oracle=True)
        cursor = cursor.commit(token, snapshot, reply, prefix, allow_oracle=True)
        parts.append(token)
        phases.append('START' if first else 'CONTINUE')
        first = False
        prefix += (token,)
    require(cursor.committed_text == value, 'teacher cursor changed value')
    end = len(parts)
    append_text(post, 'GENERATE')
    stop = codec.tokenizer.eos()
    require(codec.decode_pieces((stop,)) == (b'<|im_end|>',), 'unqualified EOS')
    parts.append(stop)
    phases.append('GENERATE')
    phases[end] = 'END'
    require(b''.join(codec.decode_pieces(parts[:-1])) == response.encode(),
            'full teacher reply bytes changed')
    require(len(initial) + len(parts) <= capacity, 'teacher trajectory exceeds capacity')
    idle = tuple(None if p == 'CONTINUE' else int(p == 'START') for p in phases)
    return TeacherTrajectory(runtime['id'], digest(runtime), initial, tuple(parts), tuple(phases),
                             idle, response, target)
