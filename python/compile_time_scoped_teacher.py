"""DG-025 time-scoped teacher compilation. The pinned current-time compiler is
delegated to byte-identically; only explicit past-time targets take the new
branch, reusing the same cursor machinery and trajectory format."""
from dataclasses import replace

from append_value_transport import compile_append_value
from fine_span_supervision import ByteTargets
from prepare_expanded_memory_worlds import time_sentence
from prepare_neural_memory_protocol import digest
from neural_memory_generation import RESERVED, encode_generation_input, generation_prompt
from value_teacher_trajectory import TeacherTrajectory, compile_teacher
from value_transport import (ActivatedFact, FactPayload, PayloadSnapshot, ReplyBinding,
                             ValueSpan, Origin, require)


def positive_bytes_time(raw, meta):
    """Time-scoped variant of the pinned span resolver: selects the actual fact
    matching meta['query_time'] and locates its past-style clause and value."""
    time = meta.get('query_time', 'current')
    matches = [f for f in meta['source_facts'] if f['subject'] == meta['query_subject'] and
               f['relation'] == meta['query_relation'] and f['time'] == time and f['status'] == 'actual']
    if len(matches) != 1:
        raise ValueError(f'expected exactly one actual fact at time {time!r}')
    target = matches[0]
    clause = time_sentence(target['subject'], target['relation'], target['value'],
                           meta['language'], 'past').encode()
    value = target['value'].encode()
    if raw.count(clause) != 1 or clause.count(value) != 1:
        raise ValueError('ambiguous time-scoped annotation, preserve as failure')
    fact_start = raw.index(clause)
    value_start = fact_start + clause.index(value)
    return ByteTargets(1, (fact_start, fact_start + len(clause)), (value_start, value_start + len(value)), 0)


def compile_teacher_time(runtime, label, meta, codec, binding, memory_model_sha256,
                         *, purpose, capacity=128):
    """compile_teacher for current-time targets; past-time branch for query_time='2020'."""
    # The original compiler never resolves spans unless the target is supported,
    # so time-qualified refusals (e.g. wrong_time) delegate unchanged too.
    if meta.get('query_time', 'current') == 'current' or label['state'] != 'supported':
        return compile_teacher(runtime, label, meta, codec, binding, memory_model_sha256,
                               purpose=purpose, capacity=capacity)
    require(purpose == 'training_diagnostic', 'gold compiler is training-only')
    require(label['id'] == runtime['id'] == meta['id'] and label['input_sha256'] == digest(runtime) and
            meta['split'] in ('train', 'dev'), 'training-side bound labels required; test rejected')
    require(binding.backbone_sha256 == codec.backbone_sha256 and
            binding.tokenizer_sha256 == codec.tokenizer_sha256, 'codec binding mismatch')
    response = label['response']
    require(type(response) is str and bool(response) and '\0' not in response and
            not any(x in response for x in RESERVED), 'invalid teacher reply')
    state = {'no_memory_needed': 0, 'supported': 1, 'insufficient': 2}[label['state']]
    require(state == 1, 'time-scoped compilation is only defined for supported targets')
    require(meta['query_time'] != 'current', 'current-time targets must delegate')
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
    target = replace(positive_bytes_time(raw, meta), idle_mode=None)
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
    require(len(initial) + len(parts) <= capacity, 'teacher trajectory exceeds capacity; no truncation')
    idle = tuple(None if p == 'CONTINUE' else int(p == 'START') for p in phases)
    return TeacherTrajectory(runtime['id'], digest(runtime), initial, tuple(parts), tuple(phases),
                             idle, response, target)
