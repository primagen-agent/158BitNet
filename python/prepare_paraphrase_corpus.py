"""EC-007 Phase B: assemble the JB-005 paraphrase corpus from the 3B cache.

Records keep the JB schema (inputs/labels/index/pairs). Episodes render as
"{person}: {paraphrase}" turns (variant chosen by record hash for
determinism); queries come from the paraphrased question cache. Every
supported record is teacher-verified via the dialogue branch (value unique
in the subject's turn); failures are dropped and counted, never repaired.

Run AFTER generate_paraphrase_corpus.py completes.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from episode_memory_inputs import validate_input
from prepare_dialogue_memory_worlds import worlds as dialogue_worlds
from prepare_neural_memory_protocol import canonical, digest, fact, message

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / 'training/memory/neural-system'
SCENARIOS = ('correct', 'role_swap', 'empty', 'multi_fact')
SUPPORTED = ('correct', 'multi_fact')


def make_records(config, cache):
    records = []
    for world, split in dialogue_worlds(config):
        for relation_index, relation in enumerate(('home_city', 'work_city')):
            target_index = world['number'] % 2
            person, other = world['people'][target_index], world['people'][1 - target_index]
            city, alternate = world['cities'][relation_index], world['cities'][1 - relation_index]
            other_relation = ('work_city', 'home_city')[relation_index]
            for language in ('en', 'zh'):
                qkey = f'q:{person}:{relation}:{language}:'
                qvariants = [cache[qkey + str(v)] for v in range(3)
                             if qkey + str(v) in cache]
                if not qvariants:
                    continue
                for scenario in SCENARIOS:
                    target = fact(person, relation, city)
                    distractor = fact(other, other_relation, alternate)
                    facts = [target, distractor]
                    if scenario == 'role_swap':
                        facts = [fact(other, relation, city),
                                 fact(person, other_relation, alternate)]
                    elif scenario == 'empty':
                        facts = []
                    elif scenario == 'multi_fact':
                        facts = [target, distractor,
                                 fact(person, other_relation, alternate)]

                    def turn(f):
                        keys = [f'ep:{f["subject"]}:{f["relation"]}:{language}:{v}'
                                for v in range(3)]
                        vlist = [cache[k] for k in keys if k in cache]
                        if not vlist:
                            return None
                        pick = vlist[int(hashlib.sha256(
                            (world['id'] + f['subject'] + f['relation'] + scenario +
                             language).encode()).hexdigest(), 16) % len(vlist)]
                        return f'{f["subject"]}: {pick}' if language == 'en' \
                            else f'{f["subject"]}：{pick}'

                    turns = [turn(f) for f in facts]
                    if any(t is None for f, t in zip(facts, turns) if f) or \
                       (facts and any(t is None for t in turns)):
                        continue
                    source = ' '.join(turns)
                    query = qvariants[int(hashlib.sha256(
                        (world['id'] + scenario + language).encode()).hexdigest(), 16)
                        % len(qvariants)]
                    state = 'supported' if scenario in SUPPORTED else 'insufficient'
                    joint = [f for f in facts if f['subject'] == person
                             and f['relation'] == relation
                             and f['time'] == 'current' and f['status'] == 'actual']
                    if bool(joint) != (state == 'supported'):
                        raise ValueError('truth label inconsistent')
                    key = 'jb5-' + digest([world['id'], relation, language, scenario])[:20]
                    zh = language == 'zh'
                    home = relation == 'home_city'
                    if state == 'supported':
                        response = (f'{joint[0]["subject"]}现在' +
                                    ('住在' if home else '在') + joint[0]['value'] +
                                    ('。' if home else '工作。')) if zh else (
                            f'{joint[0]["subject"]} currently ' +
                            ('lives in ' if home else 'works in ') +
                            f'{joint[0]["value"]}.')
                        required = joint
                    else:
                        response = (f'我还不知道{person}目前' +
                                    ('住在哪座城市。' if home else '在哪座城市工作。')) if zh else (
                            f'I do not know which city {person} currently ' +
                            ('lives in.' if home else 'works in.'))
                        required = []
                    runtime = {'id': key, 'context': [message(query)],
                               'episodes': [message(source)] if facts else []}
                    label = {'id': key, 'input_sha256': digest(runtime),
                             'state': state, 'response': response,
                             'sample_weight': 1., 'required_claims': required,
                             'allowed_claims': required if not facts else facts,
                             'memory_needed': True,
                             'evidence_missing': state == 'insufficient',
                             'relevant_episode_indices': [0] if state == 'supported' else [],
                             'factor_targets': [any(f['subject'] == person for f in facts),
                                                any(f['relation'] == relation for f in facts)],
                             'joint_current_actual': bool(joint)}
                    meta = {'id': key, 'world_id': world['id'], 'split': split,
                            'language': language, 'scenario': scenario,
                            'query_subject': person, 'query_relation': relation,
                            'query_time': 'current', 'relation_family': relation,
                            'world_people': world['people'],
                            'world_cities': world['cities'], 'source_facts': facts}
                    records.append((runtime, label, meta))
    return records


def main():
    config = json.loads((RESEARCH / 'data/JB-005.json').read_text())
    cache = json.loads((ROOT / 'build/neural-memory-ec007/paraphrases.json').read_text())
    records = make_records(config, cache)
    print(f'assembled {len(records)} records', flush=True)

    # teacher-verify every supported record (dialogue branch); drop failures
    from append_value_transport import AppendValueCodec
    from compile_dialogue_teacher import compile_teacher_dialogue
    from native_memory_encoder import BACKBONE_SHA256
    from neural_memory_contract import ModelBinding
    fm = json.loads((ROOT / 'build/neural-memory-cg003-full-features/manifest.json').read_text())
    codec = AppendValueCodec(ROOT / 'build/neural-memory-cg003-frozen-build/tok_probe',
                             ROOT / 'models/bitcpm4-0.5b-tq2_0.gguf',
                             fm['tokenizer_sha256'])
    binding = ModelBinding(BACKBONE_SHA256, fm['tokenizer_sha256'],
                           fm['encoder_identity']['encoder_id'], '0' * 64)
    kept, dropped = [], Counter()
    for runtime, label, meta in records:
        validate_input(runtime)
        if label['state'] == 'supported':
            try:
                compile_teacher_dialogue(runtime, label, meta, codec, binding,
                                         '0' * 64, purpose='training_diagnostic')
            except Exception as e:
                dropped[type(e).__name__] += 1
                continue
        kept.append((runtime, label, meta))
    print(f'teacher verification: kept {len(kept)}, dropped {sum(dropped.values())} '
          f'({dict(dropped)})', flush=True)
    codec.close()

    # world-level inventory isolation check + paired groups
    splits = {}
    for _, _, meta in kept:
        for value in meta['world_people'] + meta['world_cities']:
            splits.setdefault(value, meta['split'])
            if splits[value] != meta['split']:
                raise ValueError('cross-split leakage')

    files = {}
    for split in ('train', 'dev', 'test'):
        selected = [r for r in kept if r[2]['split'] == split]
        for i, name in enumerate(('inputs', 'labels', 'index')):
            files[f'{split}.{name}.jsonl'] = ''.join(
                canonical(r[i]) + '\n' for r in selected).encode()
    counts = Counter(r[2]['split'] + '/' + r[1]['state'] for r in kept)
    manifest = {'format': 'paraphrase-dialogue-worlds-v1',
                'source_corpus_config_sha256': digest(config),
                'file_sha256': {n: hashlib.sha256(d).hexdigest()
                                for n, d in files.items()},
                'source_sha256': {
                    'prepare_paraphrase_corpus.py':
                        hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'generate_paraphrase_corpus.py':
                        hashlib.sha256((ROOT / 'python/generate_paraphrase_corpus.py')
                                       .read_bytes()).hexdigest()},
                'purpose': 'paraphrase_self_distillation_dialogue_supervision',
                'test_policy': 'sealed_no_training_selection_or_model_forward',
                'locomo_used': False, 'backbone': 'bitcpm4-3b-tq2_0 (paraphrase only)',
                'records': len(kept), 'worlds': len({r[2]['world_id'] for r in kept}),
                'counts': dict(sorted(counts.items())),
                'dropped_by_teacher': dict(dropped)}
    files['manifest.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    out = RESEARCH / 'data/JB-005'
    if out.exists():
        raise SystemExit('JB-005 already exists')
    out.mkdir(parents=True)
    for name, data in files.items():
        (out / name).write_bytes(data)
    print(f'JB-005 written: {len(kept)} records, {manifest["worlds"]} worlds', flush=True)


if __name__ == '__main__':
    main()
