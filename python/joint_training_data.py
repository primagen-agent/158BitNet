"""Version-bound CG-003 compiler/loader. Test records never become features."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch

from availability_supervision import ReplyTargets, encode_reply_targets, teacher_forcing_requests
from diagnose_role_content_path import token_span
from episode_memory_inputs import encoder_texts
from memory_token_read import TokenReadInput
from native_memory_encoder import NativeFeatureBank, BACKBONE_SHA256, sha_file, text_key
from native_prefix_bank import PrefixBank
from native_token_payload import payload_mask, input_binding
from neural_memory_generation import TEMPLATE_VERSION, encode_generation_input
from prepare_joint_binding_curriculum import materialize
from prepare_neural_memory_protocol import digest
from review_neural_memory import read_rows
from token_memory_supervision import TokenSupervision, make_position_targets

DATA_DIGEST = 'fcff455f690e360298701b123102e3a90527d83bf12ef390ca59d035fbd2bf2c'


def allowed_splits(splits):
    if type(splits) is not tuple or not splits or len(set(splits)) != len(splits) or any(s not in ('train', 'dev') for s in splits):
        raise ValueError('only train/dev feature compilation allowed; test is sealed')


def compile_records(config, corpus, tokenizer, *, scope='full', splits=('train', 'dev')):
    allowed_splits(splits)
    if scope not in ('full', 'smoke'): raise ValueError('unknown package scope')
    manifest = materialize(config, corpus, verify=True)
    if digest(manifest) != DATA_DIGEST: raise ValueError('unregistered corpus identity')
    records = []; prefixes = {}; texts = {}; inputs = {}
    indexes = {s: read_rows(Path(corpus) / f'{s}.index.jsonl') for s in splits}
    first_train = min(r['world_id'] for r in indexes.get('train', [])) if scope == 'smoke' else None
    for split in splits:
        labels = {r['id']: r for r in read_rows(Path(corpus) / f'{split}.labels.jsonl')}
        index = {r['id']: r for r in indexes[split]}
        for runtime in read_rows(Path(corpus) / f'{split}.inputs.jsonl'):
            meta = index[runtime['id']]
            if scope == 'smoke' and not (split == 'train' and meta['world_id'] == first_train and
                meta['relation_family'] == 'home_city' and meta['scenario'] in ('correct', 'role_swap')): continue
            label = labels[runtime['id']]
            request = encode_generation_input(runtime, tokenizer)
            target = encode_reply_targets(runtime, label, tokenizer)
            query, sources = encoder_texts(runtime)
            qids = tokenizer.encode(query, True)
            qbytes = b''.join(tokenizer.decode_pieces(qids[1:]))
            if qbytes not in (query.encode(), b' ' + query.encode()): raise ValueError('query bytes mismatch')
            if len(qids) >= 512: raise ValueError('query encoder capacity exceeded')
            ids, allowed, source_span, offsets = [], [], [], []
            if sources:
                ids = tokenizer.encode(sources[0], True)[1:]
                if len(ids) + 1 >= 512: raise ValueError('source encoder capacity exceeded')
                pieces = tokenizer.decode_pieces(ids)
                allowed = payload_mask(runtime['episodes'][0], sources[0], pieces)
                blocked = (tokenizer.bos(), tokenizer.eos())
                allowed = [ok and token not in blocked for token, ok in zip(ids, allowed)]
                if target.state_index == 1:
                    value = label['required_claims'][0]['value']
                    source_span = token_span(sources[0], pieces, value)
                    offsets = token_span(target.response, tokenizer.decode_pieces(list(target.completion_token_ids[:-1])), value)
            positions = make_position_targets(target.completion_token_ids, target.state_index, offsets, ids, allowed, source_span)
            for r in teacher_forcing_requests(request, target): prefixes.setdefault(digest(r.prompt_token_ids), r.prompt_token_ids)
            for text in (query, *sources): texts.setdefault(text_key(text), text)
            records.append({'id': runtime['id'], 'split': split, 'world_id': meta['world_id'], 'language': meta['language'],
                'scenario': meta['scenario'], 'relation': meta['relation_family'], 'input_sha256': digest(runtime),
                'query_text': query, 'query_ids': qids, 'source_texts': sources, 'source_token_ids': ids, 'copy_allowed': allowed,
                'prompt_ids': request.prompt_token_ids, 'targets': asdict(target), 'factor_targets': label['factor_targets'],
                'position_targets': positions, 'value_offsets': offsets})
            inputs[runtime['id']] = runtime
    expected = 4 if scope == 'smoke' else sum(576 if s == 'train' else 192 for s in splits)
    if len(records) != expected: raise ValueError('compiled record inventory changed')
    return {'scope': scope, 'splits': splits, 'corpus_manifest_digest': DATA_DIGEST, 'records': records,
            'prefixes': list(prefixes.values()), 'texts': list(texts.values())}


@dataclass(frozen=True)
class JointForwardFeatures:
    memory: TokenReadInput
    hidden: torch.Tensor
    base: torch.Tensor


class JointTrainingPackage:
    def __init__(self, root, expected_digest):
        self.root = Path(root)
        m = json.loads((self.root / 'manifest.json').read_text()); self.manifest = m
        if (digest(m) != expected_digest or m['format'] != 'joint-training-features-v1' or
            m['backbone_sha256'] != BACKBONE_SHA256 or m['corpus_manifest_digest'] != DATA_DIGEST or
            m['template_version'] != TEMPLATE_VERSION or m['reference_parity'] != [True] * 4):
            raise ValueError('unqualified feature package')
        allowed_splits(tuple(m['splits']))
        if m['scope'] not in ('smoke', 'full'): raise ValueError('unknown scope')
        for name, field in (('records.json', 'records_sha256'), ('output-head.npy', 'output_head_sha256')):
            if sha_file(self.root / name) != m[field]: raise ValueError('feature artifact corrupted')
        self.records = json.loads((self.root / 'records.json').read_text())
        expected = 4 if m['scope'] == 'smoke' else sum(576 if s == 'train' else 192 for s in m['splits'])
        if len(self.records) != expected or len({r['id'] for r in self.records}) != expected or any(r['split'] not in m['splits'] for r in self.records):
            raise ValueError('unregistered record inventory')
        self.prefix = PrefixBank(self.root / 'prefixes', expected_manifest_sha256=m['prefix_manifest_digest'])
        self.tokens = NativeFeatureBank(self.root / 'tokens', expected_encoder_id=m['encoder_identity']['encoder_id'],
                                       expected_manifest_sha256=m['token_manifest_digest'])
        self.head = torch.from_numpy(np.load(self.root / 'output-head.npy', allow_pickle=False))
        if self.head.shape != (73448, 1024) or self.head.dtype != torch.float32 or not torch.isfinite(self.head).all():
            raise ValueError('invalid frozen head')
        self.scale = m['logit_scale']; self.binding = input_binding(m['encoder_identity'], m['tokenizer_sha256'])
        for r in self.records:
            if self.tokens.rows[text_key(r['query_text'])].token_ids.tolist() != r['query_ids']: raise ValueError('query tokens changed')
            if r['source_texts'] and self.tokens.rows[text_key(r['source_texts'][0])].token_ids[1:].tolist() != r['source_token_ids']:
                raise ValueError('source tokens changed')

    def sample(self, record, device='cpu', *, training=False):
        if record not in self.records: raise ValueError('record not in bound package')
        if training and (self.manifest['scope'] != 'full' or record['split'] != 'train'):
            raise ValueError('optimizer input requires full package and train split')
        values = dict(record['targets'])
        for name in ('prompt_token_ids', 'completion_token_ids'): values[name] = tuple(values[name])
        reply = ReplyTargets(**values)
        if tuple(record['prompt_ids']) != reply.prompt_token_ids: raise ValueError('target/prompt binding changed')
        rows = [self.prefix(reply.prompt_token_ids + reply.completion_token_ids[:i]) for i in range(len(reply.completion_token_ids))]
        source = self.tokens(record['source_texts'][0]) if record['source_texts'] else torch.empty(0, 2048)
        memory = TokenReadInput(self.tokens(record['query_text']).to(device), source.to(device),
            torch.tensor(record['source_token_ids'], dtype=torch.long, device=device),
            torch.tensor(record['copy_allowed'], dtype=torch.bool, device=device), self.binding)
        features = JointForwardFeatures(memory, torch.stack([r[0] for r in rows]).to(device), torch.stack([r[1] for r in rows]).to(device))
        targets = TokenSupervision(reply, tuple(record['factor_targets']), tuple(None if p is None else tuple(p) for p in record['position_targets']))
        return features, targets
