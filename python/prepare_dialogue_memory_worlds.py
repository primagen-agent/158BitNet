"""EC-006 dialogue-form memory worlds (JB-004). Labels never enter runtime.

Same record/label/index skeleton as prepare_expanded_memory_worlds (so the
pinned teacher compiler loads it unchanged), but episodes render as spoken
turns with speaker prefixes and colloquial templates, and queries use
varied natural question forms. Targets the LC-003 finding: every learned
head is a specialist of the two-fact declarative form; natural dialogue is
out-of-distribution for all of them.

Scenario subset: correct / role_swap / empty / multi_fact (form diversity
only — time-qualified and hypothetical/quoted machinery is untouched).
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
from collections import Counter

from episode_memory_inputs import validate_input
from prepare_joint_binding_curriculum import RELATIONS, sentence
from prepare_neural_memory_protocol import canonical, digest, fact, message

SCENARIOS = ('correct', 'role_swap', 'empty', 'multi_fact')
SUPPORTED = ('correct', 'multi_fact')

EN_HOME = ("{p}: I've been living in {c} for a few months now.",
           "{p}: guess what, I finally moved to {c}!",
           "{p}: I'm based in {c} these days, really enjoying it.")
EN_WORK = ("{p}: work update — I'm working in {c} now.",
           "{p}: the new job is in {c}, so that's where I am.",
           "{p}: I've been doing my thing in {c} lately.")
ZH_HOME = ("{p}：我最近搬到{c}了。",
           "{p}：我现在住在{c}，挺喜欢的。",
           "{p}：对了，我搬家了，现在在{c}。")
ZH_WORK = ("{p}：我在{c}工作。",
           "{p}：新工作在{c}，还不错。",
           "{p}：最近都在{c}忙项目。")

EN_Q_HOME = ("Where is {p} living these days?",
             "What city is {p} in right now?",
             "Where does {p} call home?")
EN_Q_WORK = ("Where is {p} working these days?",
             "What city does {p} work in?",
             "{p} is based where for work?")
ZH_Q_HOME = ("{p}现在住在哪？", "{p}最近在哪个城市住？", "{p}住在哪儿来着？")
ZH_Q_WORK = ("{p}现在在哪里工作？", "{p}在哪个城市上班？", "{p}工作地点在哪？")


def stable_pick(rng, seq, key):
    return seq[hash(key) % len(seq)] if not rng else rng.choice(seq)


def dialogue_turn(f, language, rng_seed_source):
    home = f['relation'] == 'home_city'
    if language == 'zh':
        templates = ZH_HOME if home else ZH_WORK
    else:
        templates = EN_HOME if home else EN_WORK
    return stable_pick(None, templates, rng_seed_source).format(
        p=f['subject'], c=f['value'])


def dialogue_query(person, relation, language):
    home = relation == 'home_city'
    if language == 'zh':
        templates = ZH_Q_HOME if home else ZH_Q_WORK
    else:
        templates = EN_Q_HOME if home else EN_Q_WORK
    return stable_pick(None, templates, person + relation + language).format(p=person)


def make_cases(world, split):
    for relation_index, relation in enumerate(RELATIONS):
        target_index = world['number'] % 2
        person, other = world['people'][target_index], world['people'][1 - target_index]
        city, alternate = world['cities'][relation_index], world['cities'][1 - relation_index]
        other_relation = RELATIONS[1 - relation_index]
        for language in ('en', 'zh'):
            query = dialogue_query(person, relation, language)
            for scenario in SCENARIOS:
                target = fact(person, relation, city)
                distractor = fact(other, other_relation, alternate)
                facts = [target, distractor]
                if scenario == 'role_swap':
                    facts = [fact(other, relation, city), fact(person, other_relation, alternate)]
                elif scenario == 'empty':
                    facts = []
                elif scenario == 'multi_fact':
                    facts = [target, distractor, fact(person, other_relation, alternate)]
                source = ' '.join(dialogue_turn(f, language, f['subject'] + f['value'])
                                  for f in facts)
                state = 'supported' if scenario in SUPPORTED else 'insufficient'
                joint = [f for f in facts if f['subject'] == person and f['relation'] == relation
                         and f['time'] == 'current' and f['status'] == 'actual']
                if bool(joint) != (state == 'supported'):
                    raise ValueError('truth label inconsistent')
                if state == 'supported':
                    response = sentence(person, relation, joint[0]['value'], language)
                    required = joint
                else:
                    zh = language == 'zh'; home = relation == 'home_city'
                    response = (f'我还不知道{person}目前' + ('住在哪座城市。' if home else '在哪座城市工作。')) if zh else (
                        f'I do not know which city {person} currently ' + ('lives in.' if home else 'works in.'))
                    required = []
                key = 'jb4-' + digest([world['id'], relation, language, scenario])[:20]
                runtime = {'id': key, 'context': [message(query)], 'episodes': [message(source)] if facts else []}
                label = {'id': key, 'input_sha256': digest(runtime), 'state': state, 'response': response,
                         'sample_weight': 1., 'required_claims': required,
                         'allowed_claims': required if not facts else facts,
                         'memory_needed': True, 'evidence_missing': state == 'insufficient',
                         'relevant_episode_indices': [0] if state == 'supported' else [],
                         'factor_targets': [any(f['subject'] == person for f in facts),
                                            any(f['relation'] == relation for f in facts)],
                         'joint_current_actual': bool(joint)}
                meta = {'id': key, 'world_id': world['id'], 'split': split, 'language': language,
                        'scenario': scenario, 'query_subject': person, 'query_relation': relation,
                        'query_time': 'current', 'relation_family': relation,
                        'world_people': world['people'], 'world_cities': world['cities'], 'source_facts': facts}
                yield runtime, label, meta


def worlds(config):
    want = {'format', 'seed', 'train_worlds', 'dev_worlds', 'test_worlds', 'people', 'cities'}
    if set(config) != want or config['format'] != 'dialogue-memory-worlds-v1':
        raise ValueError('invalid dialogue worlds config')
    sizes = [config[s + '_worlds'] for s in ('train', 'dev', 'test')]
    if any(type(n) is not int or n < 1 for n in sizes) or type(config['seed']) is not int:
        raise ValueError('positive explicit split sizes required')
    for key in ('people', 'cities'):
        values = config[key]
        if len(values) != 2 * sum(sizes) or len(set(values)) != len(values) or \
                any(type(v) is not str or not v.strip() for v in values):
            raise ValueError('unique disjoint world inventories required')
    if set(config['people']) & set(config['cities']):
        raise ValueError('subject/value inventory overlap')
    inventory = {'people': list(config['people']), 'cities': list(config['cities'])}
    number = 0
    splits = ['train'] * sizes[0] + ['dev'] * sizes[1] + ['test'] * sizes[2]
    random.Random(config['seed']).shuffle(splits)
    for split in splits:
        people = [inventory['people'].pop() for _ in range(2)]
        cities = [inventory['cities'].pop() for _ in range(2)]
        yield {'id': 'jb4-world-' + digest([config['seed'], number])[:16],
               'number': number, 'split': split, 'people': people, 'cities': cities}, split
        number += 1
    if inventory['people'] or inventory['cities']:
        raise ValueError('unconsumed inventory')


def validate(rows):
    ids = set(); inventories = {'people': {}, 'cities': {}, 'world': {}}; groups = {}; counts = Counter()
    for runtime, label, meta in rows:
        validate_input(runtime)
        if runtime['id'] in ids or runtime['id'] != label['id'] or runtime['id'] != meta['id'] \
                or digest(runtime) != label['input_sha256']:
            raise ValueError('duplicate or mismatched example')
        ids.add(runtime['id']); split = meta['split']
        if split not in ('train', 'dev', 'test'):
            raise ValueError('invalid split')
        for kind, values in (('people', meta['world_people']), ('cities', meta['world_cities']), ('world', [meta['world_id']])):
            for value in values:
                if inventories[kind].setdefault(value, split) != split:
                    raise ValueError('cross-split identity/value leakage')
        if meta['scenario'] not in SCENARIOS:
            raise ValueError('unknown scenario')
        expected = 'supported' if meta['scenario'] in SUPPORTED else 'insufficient'
        if label['state'] != expected or label['evidence_missing'] != (expected == 'insufficient'):
            raise ValueError('state supervision mismatch')
        facts = meta['source_facts']
        joint = [f for f in facts if f['subject'] == meta['query_subject'] and f['relation'] == meta['query_relation']
                 and f['time'] == 'current' and f['status'] == 'actual']
        if bool(joint) != (expected == 'supported'):
            raise ValueError('joint binding label mismatch')
        groups.setdefault((meta['world_id'], meta['language'], meta['relation_family']), {})[meta['scenario']] = (runtime, label, meta)
        counts[split + '/' + label['state']] += 1
    pairs = []
    for group in groups.values():
        if set(group) != set(SCENARIOS):
            raise ValueError('missing paired scenario')
        for kind in ('role_swap', 'empty'):
            left, right = group['correct'], group[kind]
            if left[0]['context'] != right[0]['context']:
                raise ValueError(f'paired query changed: {kind}')
            pairs.append({'id': 'pair-' + digest([left[0]['id'], right[0]['id']])[:20],
                          'split': left[2]['split'], 'world_id': left[2]['world_id'],
                          'language': left[2]['language'], 'relation': left[2]['relation_family'],
                          'kind': kind, 'left_id': left[0]['id'], 'right_id': right[0]['id']})
    return pairs, {'records': len(rows), 'worlds': len(inventories['world']), 'pairs': len(pairs),
                   'counts': dict(sorted(counts.items())), 'split_subjects_and_values_disjoint': True,
                   'form': 'dialogue'}


def build(config):
    rows = [row for world, split in worlds(config) for row in make_cases(world, split)]
    pairs, report = validate(rows)
    files = {}
    for split in ('train', 'dev', 'test'):
        selected = [r for r in rows if r[2]['split'] == split]
        for i, name in enumerate(('inputs', 'labels', 'index')):
            files[f'{split}.{name}.jsonl'] = ''.join(canonical(r[i]) + '\n' for r in selected).encode()
        files[f'{split}.pairs.jsonl'] = ''.join(canonical(p) + '\n' for p in pairs if p['split'] == split).encode()
    return files, report


def materialize(config_path, output, verify=False):
    path = Path(config_path); root = Path(output)
    files, report = build(json.loads(path.read_text()))
    manifest = {'format': 'dialogue-memory-worlds-v1',
                'config_sha256': digest(json.loads(path.read_text())),
                'file_sha256': {n: hashlib.sha256(data).hexdigest() for n, data in files.items()},
                'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ('prepare_dialogue_memory_worlds.py',
                                               'episode_memory_inputs.py',
                                               'prepare_joint_binding_curriculum.py',
                                               'prepare_neural_memory_protocol.py')},
                'purpose': 'dialogue_form_diversity_supervision_not_general_memory',
                'test_policy': 'sealed_no_training_selection_or_model_forward',
                'locomo_used': False, 'automatic_writer': False, **report}
    files['manifest.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    if verify:
        if {p.name for p in root.iterdir()} != set(files):
            raise ValueError('unexpected corpus files')
        if any((root / name).read_bytes() != data for name, data in files.items()):
            raise ValueError('frozen corpus changed')
    else:
        root.mkdir(parents=True, exist_ok=False)
        for name, data in files.items():
            with (root / name).open('xb') as f:
                f.write(data)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    print(json.dumps(materialize(a.config, a.output, a.verify), ensure_ascii=False, indent=2))
