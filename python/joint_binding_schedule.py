"""Fixed proposed 50-step pair schedule. No optimizer or launch authorization."""
import random

KINDS = ('role_swap', 'value_swap', 'clause_reorder', 'paraphrase', 'wrong_subject',
         'wrong_relation', 'negated', 'historical', 'empty', 'ordinary_empty')


def pair_schedule(pairs, seed=1013, steps=50):
    if seed != 1013 or steps != 50: raise ValueError('unregistered schedule variation')
    if not pairs or any(p['split'] != 'train' for p in pairs): raise ValueError('train pairs only')
    if len({p['id'] for p in pairs}) != len(pairs) or {p['kind'] for p in pairs} != set(KINDS):
        raise ValueError('duplicate/incomplete pair inventory')
    rng = random.Random(seed); groups = {}
    for kind in KINDS:
        group = sorted((p for p in pairs if p['kind'] == kind), key=lambda p: p['id'])
        if len(group) != 48: raise ValueError('expected all 12 training worlds in each kind')
        rng.shuffle(group); groups[kind] = group
    return [{'step': step + 1, 'pairs': [groups[KINDS[i % 10]][i // 10] for i in range(4 * step, 4 * step + 4)]}
            for step in range(steps)]
