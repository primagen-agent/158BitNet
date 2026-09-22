"""DG-024 expanded semantic worlds (JB-002). Labels never enter runtime inputs.

Counter-example curriculum for the neural memory line: time-qualified
questions, explicit-historical sources, multi-fact distractors, hypotheses
and unconfirmed relays — with paired controls that change one clause at a
time. Reuses the JB-001 record/label/index formats so the existing pipeline
loads it unchanged.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from episode_memory_inputs import validate_input
from prepare_joint_binding_curriculum import RELATIONS, sentence
from prepare_neural_memory_protocol import canonical, digest, fact, message

SCENARIOS = ('correct', 'role_swap', 'empty', 'wrong_time', 'historical_stated',
             'multi_fact', 'hypothetical', 'quoted')
SUPPORTED = ('correct', 'historical_stated', 'multi_fact')
TIME_SCOPED = ('wrong_time', 'historical_stated')
JB1_CONFIG = Path(__file__).resolve().parents[1] / 'training/memory/neural-system/data/JB-001.json'


def worlds(config):
    want = {'format', 'seed', 'train_worlds', 'dev_worlds', 'test_worlds', 'people', 'cities'}
    if set(config) != want or config['format'] != 'expanded-memory-worlds-v1':
        raise ValueError('invalid expanded worlds config')
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
    jb1 = json.loads(JB1_CONFIG.read_text())
    if set(config['people']) & set(jb1['people']) or set(config['cities']) & set(jb1['cities']):
        raise ValueError('inventory overlaps JB-001')
    order = list(range(sum(sizes))); random.Random(config['seed']).shuffle(order)
    for position, i in enumerate(order):
        split = 'train' if position < sizes[0] else 'dev' if position < sum(sizes[:2]) else 'test'
        yield {'id': f'jb2-{i:03d}', 'people': config['people'][2 * i:2 * i + 2],
               'cities': config['cities'][2 * i:2 * i + 2], 'number': i + 3}, split


def time_sentence(person, relation, city, language, style='past'):
    home = relation == 'home_city'
    if language == 'zh':
        verb = '住在' if home else '在'; tail = '' if home else '工作'
        if style == 'hypothetical': return f'如果{person}获奖，{person}将搬到{city}。'
        if style == 'quoted': return f'{person}告诉我{person}现在{verb}{city}{tail}。'
        return f'{person}在2020年{verb}{city}{tail}。'
    verb = 'lives' if home else 'works'
    if style == 'hypothetical': return f'If {person} wins the award, {person} will move to {city}.'
    if style == 'quoted': return f'{person} told me that {person} currently {verb} in {city}.'
    return f'{person} ' + ('lived' if home else 'worked') + f' in {city} in 2020.'


def styled(f, language, relayer):
    if f['status'] == 'hypothetical':
        return time_sentence(f['subject'], f['relation'], f['value'], language, 'hypothetical')
    if f['status'] == 'quoted':
        home = f['relation'] == 'home_city'
        if language == 'zh':
            verb = '住在' if home else '在'; tail = '' if home else '工作'
            return f'{relayer}告诉我{f["subject"]}现在{verb}{f["value"]}{tail}。'
        verb = 'lives' if home else 'works'
        return f'{relayer} told me that {f["subject"]} currently {verb} in {f["value"]}.'
    if f['time'] == '2020':
        return time_sentence(f['subject'], f['relation'], f['value'], language, 'past')
    return sentence(f['subject'], f['relation'], f['value'], language)


def make_cases(world, split):
    for relation_index, relation in enumerate(RELATIONS):
        target_index = world['number'] % 2
        person, other = world['people'][target_index], world['people'][1 - target_index]
        city, alternate = world['cities'][relation_index], world['cities'][1 - relation_index]
        other_relation = RELATIONS[1 - relation_index]
        for language in ('en', 'zh'):
            home = relation == 'home_city'; zh = language == 'zh'
            if zh:
                current_q = (f'{person}现在' + ('住在哪里？' if home else '在哪里工作？')) if split == 'train' else (
                    f'能告诉我{person}目前' + ('居住' if home else '工作') + '的城市吗？')
                time_q = (f'{person}在2020年' + ('住在哪里？' if home else '在哪里工作？')) if split == 'train' else (
                    f'能告诉我{person}2020年' + ('居住' if home else '工作') + '的城市吗？')
            else:
                current_q = (f'Where does {person} ' + ('live' if home else 'work') + ' now?') if split == 'train' else (
                    f'In which city is {person} currently ' + ('living?' if home else 'working?'))
                time_q = (f'Where did {person} ' + ('live' if home else 'work') + ' in 2020?') if split == 'train' else (
                    f'In which city did {person} ' + ('live' if home else 'work') + ' in 2020?')
            for scenario in SCENARIOS:
                target = fact(person, relation, city)
                distractor = fact(other, other_relation, alternate)
                facts = [target, distractor]
                query, query_time, status = current_q, 'current', 'actual'
                if scenario == 'role_swap':
                    facts = [fact(other, relation, city), fact(person, other_relation, alternate)]
                elif scenario == 'empty':
                    facts = []
                elif scenario == 'wrong_time':
                    query, query_time = time_q, '2020'
                elif scenario == 'historical_stated':
                    facts = [target, fact(person, relation, alternate, time='2020'), distractor]
                    query, query_time = time_q, '2020'
                elif scenario == 'multi_fact':
                    facts = [target, distractor, fact(person, other_relation, alternate)]
                elif scenario == 'hypothetical':
                    facts = [fact(person, relation, city, status='hypothetical'), distractor]
                elif scenario == 'quoted':
                    facts = [fact(person, relation, city, status='quoted'), distractor]
                source = (' ' if not zh else '').join(styled(f, language, other) for f in facts)
                state = 'supported' if scenario in SUPPORTED else 'insufficient'
                joint = [f for f in facts if f['subject'] == person and f['relation'] == relation
                         and f['time'] == query_time and f['status'] == 'actual']
                if bool(joint) != (state == 'supported'): raise ValueError('truth label inconsistent')
                if state == 'supported' and scenario == 'historical_stated':
                    response = time_sentence(person, relation, joint[0]['value'], language)
                    required = joint
                elif state == 'supported':
                    response = sentence(person, relation, joint[0]['value'], language)
                    required = joint
                elif query_time == '2020':
                    response = (f'我还不知道{person}在2020年' + ('住在哪座城市。' if home else '在哪座城市工作。')) if zh else (
                        f'I do not know where {person} ' + ('lived' if home else 'worked') + ' in 2020.')
                    required = []
                else:
                    response = (f'我还不知道{person}目前' + ('住在哪座城市。' if home else '在哪座城市工作。')) if zh else (
                        f'I do not know which city {person} currently ' + ('lives in.' if home else 'works in.'))
                    required = []
                key = 'jb2-' + digest([world['id'], relation, language, scenario])[:20]
                runtime = {'id': key, 'context': [message(query)], 'episodes': [message(source)] if facts else []}
                label = {'id': key, 'input_sha256': digest(runtime), 'state': state, 'response': response,
                         'sample_weight': 1., 'required_claims': required, 'allowed_claims': required if not facts else facts,
                         'memory_needed': True, 'evidence_missing': state == 'insufficient',
                         'relevant_episode_indices': [0] if state == 'supported' else [],
                         'factor_targets': [any(f['subject'] == person for f in facts),
                                            any(f['relation'] == relation for f in facts)],
                         'joint_current_actual': None if query_time != 'current' else bool(joint)}
                meta = {'id': key, 'world_id': world['id'], 'split': split, 'language': language,
                        'scenario': scenario, 'query_subject': person, 'query_relation': relation,
                        'query_time': query_time, 'relation_family': relation,
                        'world_people': world['people'], 'world_cities': world['cities'], 'source_facts': facts}
                yield runtime, label, meta


def validate(rows):
    ids = set(); inventories = {'people': {}, 'cities': {}, 'world': {}}; groups = {}; counts = Counter()
    for runtime, label, meta in rows:
        validate_input(runtime)
        if runtime['id'] in ids or runtime['id'] != label['id'] or runtime['id'] != meta['id'] \
                or digest(runtime) != label['input_sha256']:
            raise ValueError('duplicate or mismatched example')
        ids.add(runtime['id']); split = meta['split']
        if split not in ('train', 'dev', 'test'): raise ValueError('invalid split')
        for kind, values in (('people', meta['world_people']), ('cities', meta['world_cities']), ('world', [meta['world_id']])):
            for value in values:
                if inventories[kind].setdefault(value, split) != split:
                    raise ValueError('cross-split identity/value leakage')
        if meta['scenario'] not in SCENARIOS: raise ValueError('unknown scenario')
        expected = 'supported' if meta['scenario'] in SUPPORTED else 'insufficient'
        if label['state'] != expected or label['evidence_missing'] != (expected == 'insufficient'):
            raise ValueError('state supervision mismatch')
        facts = meta['source_facts']
        joint = [f for f in facts if f['subject'] == meta['query_subject'] and f['relation'] == meta['query_relation']
                 and f['time'] == meta['query_time'] and f['status'] == 'actual']
        if bool(joint) != (expected == 'supported'): raise ValueError('joint binding label mismatch')
        factors = [any(f['subject'] == meta['query_subject'] for f in facts),
                   any(f['relation'] == meta['query_relation'] for f in facts)]
        if label['factor_targets'] != factors: raise ValueError('presence label mismatch')
        groups.setdefault((meta['world_id'], meta['language'], meta['relation_family']), {})[meta['scenario']] = (runtime, label, meta)
        counts[split + '/' + label['state']] += 1
    pairs = []
    for group in groups.values():
        if set(group) != set(SCENARIOS): raise ValueError('missing paired scenario')
        for left_name, right_name, kind in (('correct', 'role_swap', 'role_swap'),
                                            ('correct', 'empty', 'empty'),
                                            ('wrong_time', 'historical_stated', 'time_pair'),
                                            ('correct', 'hypothetical', 'hypothetical'),
                                            ('correct', 'quoted', 'quoted')):
            left, right = group[left_name], group[right_name]
            if left[0]['context'] != right[0]['context']: raise ValueError(f'paired query changed: {kind}')
        wt_facts = group['wrong_time'][2]['source_facts']; hs_facts = group['historical_stated'][2]['source_facts']
        extra = [f for f in hs_facts if f not in wt_facts]
        if len(extra) != 1 or extra[0]['time'] != '2020' or len(hs_facts) != len(wt_facts) + 1:
            raise ValueError('time pair must differ by exactly the 2020 clause')
        for kind, right_name in (('role_swap', 'role_swap'), ('empty', 'empty'), ('time_pair', 'historical_stated'),
                                 ('hypothetical', 'hypothetical'), ('quoted', 'quoted')):
            left = group['correct'] if kind != 'time_pair' else group['wrong_time']
            right = group[right_name]
            pairs.append({'id': 'pair-' + digest([left[0]['id'], right[0]['id']])[:20], 'split': left[2]['split'],
                          'world_id': left[2]['world_id'], 'language': left[2]['language'],
                          'relation': left[2]['relation_family'], 'kind': kind,
                          'left_id': left[0]['id'], 'right_id': right[0]['id']})
    return pairs, {'records': len(rows), 'worlds': len(inventories['world']), 'pairs': len(pairs),
                   'counts': dict(sorted(counts.items())), 'split_subjects_and_values_disjoint': True,
                   'jb001_disjoint': True, 'paired_groups_verified': len(groups)}


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
    manifest = {'format': 'expanded-memory-worlds-v1', 'config_sha256': digest(json.loads(path.read_text())),
                'file_sha256': {n: hashlib.sha256(data).hexdigest() for n, data in files.items()},
                'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in
                                  ('prepare_expanded_memory_worlds.py', 'episode_memory_inputs.py',
                                   'prepare_joint_binding_curriculum.py', 'prepare_neural_memory_protocol.py')},
                'purpose': 'expanded_counter_example_supervision_not_general_memory',
                'test_policy': 'sealed_no_training_selection_or_model_forward',
                'locomo_used': False, 'automatic_writer': False, **report}
    files['manifest.json'] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode()
    if verify:
        if {p.name for p in root.iterdir()} != set(files): raise ValueError('unexpected corpus files')
        if any((root / name).read_bytes() != data for name, data in files.items()):
            raise ValueError('frozen corpus changed')
    else:
        root.mkdir(parents=True, exist_ok=False)
        for name, data in files.items():
            with (root / name).open('xb') as f: f.write(data)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--output', required=True); p.add_argument('--verify', action='store_true')
    a = p.parse_args(); print(json.dumps(materialize(a.config, a.output, a.verify), ensure_ascii=False, indent=2))
