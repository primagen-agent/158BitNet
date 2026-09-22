"""Versioned synthetic joint-binding pilot. Labels never enter runtime inputs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from episode_memory_inputs import validate_input
from prepare_neural_memory_protocol import canonical, digest, fact, message


SCENARIOS = ('correct', 'role_swap', 'value_swap', 'clause_reorder', 'paraphrase', 'wrong_subject',
             'wrong_relation', 'negated', 'historical', 'empty', 'ordinary_source', 'ordinary_empty')
SUPPORTED = ('correct', 'value_swap', 'clause_reorder', 'paraphrase')
RELATIONS = ('home_city', 'work_city')


def worlds(config):
    if set(config) != {'format', 'seed', 'train_worlds', 'dev_worlds', 'test_worlds', 'people', 'cities'} or config['format'] != 'joint-binding-curriculum-v1':
        raise ValueError('invalid joint curriculum config')
    sizes = [config[s + '_worlds'] for s in ('train', 'dev', 'test')]
    if any(type(n) is not int or n < 1 for n in sizes) or type(config['seed']) is not int:
        raise ValueError('positive explicit split sizes required')
    for key in ('people', 'cities'):
        values = config[key]
        if len(values) != 2 * sum(sizes) or len(set(values)) != len(values) or any(type(v) is not str or not v.strip() for v in values):
            raise ValueError('unique disjoint world inventories required')
    if set(config['people']) & set(config['cities']): raise ValueError('subject/value inventory overlap')
    order = list(range(sum(sizes))); random.Random(config['seed']).shuffle(order)
    for position, i in enumerate(order):
        split = 'train' if position < sizes[0] else 'dev' if position < sum(sizes[:2]) else 'test'
        yield {'id': f'jb-{i:03d}', 'people': config['people'][2*i:2*i+2],
               'cities': config['cities'][2*i:2*i+2], 'number': i + 3}, split


def sentence(person, relation, city, language, style='current'):
    home = relation == 'home_city'
    if language == 'zh':
        verb = '住在' if home else '在'; tail = '' if home else '工作'
        if style == 'paraphrase': return f'{city}是{person}目前' + ('居住' if home else '工作') + '的城市。'
        when = '在2020年' if style == 'historical' else '现在'
        negative = '不' if style == 'negated' else ''
        return f'{person}{when}{negative}{verb}{city}{tail}。'
    verb = 'lives' if home else 'works'
    if style == 'paraphrase': return f'{city} is where {person} currently ' + ('resides.' if home else 'works.')
    if style == 'historical': return f'{person} ' + ('lived' if home else 'worked') + f' in {city} in 2020.'
    if style == 'negated': return f'{person} does not currently ' + ('live' if home else 'work') + f' in {city}.'
    return f'{person} currently {verb} in {city}.'


def make_cases(world, split):
    for relation_index, relation in enumerate(RELATIONS):
        # Ask BOTH relations of the same person. Alternate which person is the
        # target across worlds, rather than encoding relation in their identity.
        target_index = world['number'] % 2
        person, other = world['people'][target_index], world['people'][1-target_index]
        city, alternate = world['cities'][relation_index], world['cities'][1-relation_index]
        other_relation = RELATIONS[1-relation_index]
        for language in ('en', 'zh'):
            home = relation == 'home_city'; zh = language == 'zh'
            if zh:
                query = (f'{person}现在' + ('住在哪里？' if home else '在哪里工作？')) if split == 'train' else (
                    f'能告诉我{person}目前' + ('居住' if home else '工作') + '的城市吗？')
            else:
                query = (f'Where does {person} ' + ('live' if home else 'work') + ' now?') if split == 'train' else (
                    f'In which city is {person} currently ' + ('living?' if home else 'working?'))
            for scenario in SCENARIOS:
                facts = [fact(person, relation, city), fact(other, other_relation, alternate)]
                style = 'current'
                if scenario == 'role_swap': facts = [fact(other, relation, city), fact(person, other_relation, alternate)]
                elif scenario == 'value_swap': facts = [fact(person, relation, alternate), fact(other, other_relation, city)]
                elif scenario == 'clause_reorder': facts.reverse()
                elif scenario == 'paraphrase': style = 'paraphrase'
                elif scenario == 'wrong_subject': facts = [fact(other, relation, city)]
                elif scenario == 'wrong_relation': facts = [fact(person, other_relation, city)]
                elif scenario == 'negated': facts = [fact(person, relation, city, status='negated')]; style = 'negated'
                elif scenario == 'historical': facts = [fact(person, relation, city, time='2020')]; style = 'historical'
                elif scenario in ('empty', 'ordinary_empty'): facts = []
                source = (' ' if not zh else '').join(sentence(f['subject'], f['relation'], f['value'], language, style) for f in facts)
                normal = scenario.startswith('ordinary_')
                state = 'no_memory_needed' if normal else 'supported' if scenario in SUPPORTED else 'insufficient'
                joint = [f for f in facts if f['subject'] == person and f['relation'] == relation and f['time'] == 'current' and f['status'] == 'actual']
                if not normal and bool(joint) != (state == 'supported'): raise ValueError('truth label inconsistent')
                question = (f'{world["number"]}加7等于多少？' if zh else f'What is {world["number"]} plus 7?') if normal else query
                if state == 'supported':
                    response = sentence(person, relation, joint[0]['value'], language)
                    required = joint
                elif normal:
                    total = str(world['number'] + 7)
                    response = f'结果是{total}。' if zh else f'The sum is {total}.'
                    required = [fact(f'{world["number"]}+7', 'equals', total, 'timeless')]
                else:
                    response = (f'我还不知道{person}目前' + ('住在哪座城市。' if home else '在哪座城市工作。')) if zh else (
                        f'I do not know which city {person} currently ' + ('lives in.' if home else 'works in.'))
                    required = []
                key = 'joint-' + digest([world['id'], relation, language, scenario])[:20]
                runtime = {'id': key, 'context': [message(question)], 'episodes': [message(source)] if facts else []}
                label = {'id': key, 'input_sha256': digest(runtime), 'state': state, 'response': response,
                    'sample_weight': 1., 'required_claims': required, 'allowed_claims': required if normal else facts,
                    'memory_needed': not normal, 'evidence_missing': state == 'insufficient',
                    'relevant_episode_indices': [0] if state == 'supported' else [],
                    'factor_targets': [None, None] if normal else [any(f['subject'] == person for f in facts), any(f['relation'] == relation for f in facts)],
                    'joint_current_actual': None if normal else bool(joint)}
                meta = {'id': key, 'world_id': world['id'], 'split': split, 'language': language,
                    'scenario': scenario, 'query_subject': None if normal else person, 'query_relation': None if normal else relation,
                    'relation_family': relation, 'world_people': world['people'], 'world_cities': world['cities'], 'source_facts': facts}
                yield runtime, label, meta


def validate(rows):
    ids = set(); inventories = {'people': {}, 'cities': {}, 'world': {}}; groups = {}; counts = Counter()
    for runtime, label, meta in rows:
        validate_input(runtime)
        if runtime['id'] in ids or runtime['id'] != label['id'] or runtime['id'] != meta['id'] or digest(runtime) != label['input_sha256']:
            raise ValueError('duplicate or mismatched example')
        ids.add(runtime['id']); split = meta['split']
        if split not in ('train', 'dev', 'test'): raise ValueError('invalid split')
        for kind, values in (('people', meta['world_people']), ('cities', meta['world_cities']), ('world', [meta['world_id']])):
            for value in values:
                if inventories[kind].setdefault(value, split) != split: raise ValueError('cross-split identity/value leakage')
        if meta['scenario'] not in SCENARIOS: raise ValueError('unknown scenario')
        normal = meta['scenario'].startswith('ordinary_')
        expected = 'no_memory_needed' if normal else 'supported' if meta['scenario'] in SUPPORTED else 'insufficient'
        if label['state'] != expected or label['memory_needed'] != (not normal) or label['evidence_missing'] != (expected == 'insufficient'):
            raise ValueError('state supervision mismatch')
        if label['relevant_episode_indices'] != ([0] if expected == 'supported' else []):
            raise ValueError('source supervision mismatch')
        facts = meta['source_facts']
        joint = [f for f in facts if f['subject'] == meta['query_subject'] and f['relation'] == meta['query_relation']
                 and f['time'] == 'current' and f['status'] == 'actual']
        if not normal and (label['joint_current_actual'] != bool(joint) or bool(joint) != (expected == 'supported')):
            raise ValueError('joint binding label mismatch')
        factors = [None, None] if normal else [any(f['subject'] == meta['query_subject'] for f in facts), any(f['relation'] == meta['query_relation'] for f in facts)]
        if label['factor_targets'] != factors: raise ValueError('presence label mismatch')
        groups.setdefault((meta['world_id'], meta['language'], meta['relation_family']), {})[meta['scenario']] = (runtime, label, meta)
        counts[split + '/' + label['state']] += 1
    pairs = []
    for group in groups.values():
        if set(group) != set(SCENARIOS): raise ValueError('missing paired scenario')
        base = group['correct']; swapped = group['role_swap']
        if (base[0]['context'] != swapped[0]['context'] or Counter(base[0]['episodes'][0]['text']) != Counter(swapped[0]['episodes'][0]['text'])
                or base[1]['factor_targets'] != [True, True] or swapped[1]['factor_targets'] != [True, True]
                or base[1]['joint_current_actual'] is not True or swapped[1]['joint_current_actual'] is not False):
            raise ValueError('joint binding control corrupted')
        for scenario in SCENARIOS[1:]:
            if scenario == 'ordinary_source': continue
            left, right = (group['ordinary_source'], group['ordinary_empty']) if scenario == 'ordinary_empty' else (base, group[scenario])
            if left[0]['context'] != right[0]['context']: raise ValueError('paired query changed')
            pairs.append({'id': 'pair-' + digest([left[0]['id'], right[0]['id']])[:20], 'split': left[2]['split'],
                'world_id': left[2]['world_id'], 'language': left[2]['language'], 'relation': left[2]['relation_family'],
                'kind': scenario, 'left_id': left[0]['id'], 'right_id': right[0]['id']})
    return pairs, {'records': len(rows), 'worlds': len(inventories['world']), 'pairs': len(pairs),
                   'counts': dict(sorted(counts.items())), 'split_subjects_and_values_disjoint': True,
                   'role_swap_groups_verified': len(groups)}


def build(config):
    rows = [row for world, split in worlds(config) for row in make_cases(world, split)]
    pairs, report = validate(rows); files = {}
    for split in ('train', 'dev', 'test'):
        selected = [r for r in rows if r[2]['split'] == split]
        for i, name in enumerate(('inputs', 'labels', 'index')):
            files[f'{split}.{name}.jsonl'] = ''.join(canonical(r[i]) + '\n' for r in selected).encode()
        files[f'{split}.pairs.jsonl'] = ''.join(canonical(p) + '\n' for p in pairs if p['split'] == split).encode()
    return files, report


def materialize(config_path, output, verify=False):
    path, root = Path(config_path), Path(output)
    files, report = build(json.loads(path.read_text()))
    manifest = {'format': 'joint-binding-curriculum-v1', 'config_sha256': digest(json.loads(path.read_text())),
        'file_sha256': {n: hashlib.sha256(data).hexdigest() for n, data in files.items()},
        'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
                          ('prepare_joint_binding_curriculum.py', 'episode_memory_inputs.py', 'prepare_neural_memory_protocol.py')},
        'purpose': 'bounded_joint_binding_pilot_not_general_memory', 'test_policy': 'sealed_no_training_selection_or_model_forward',
        'locomo_used': False, 'automatic_writer': False, **report}
    files['manifest.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    if verify:
        if {p.name for p in root.iterdir()} != set(files): raise ValueError('unexpected corpus files')
        if any((root / name).read_bytes() != data for name, data in files.items()): raise ValueError('frozen corpus changed')
    else:
        root.mkdir(parents=True, exist_ok=False)
        for name, data in files.items():
            with (root / name).open('xb') as f: f.write(data)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--output', required=True); p.add_argument('--verify', action='store_true')
    a = p.parse_args(); print(json.dumps(materialize(a.config, a.output, a.verify), ensure_ascii=False, indent=2))
