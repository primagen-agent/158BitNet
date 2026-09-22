"""CG-003 fixed-panel semantic aggregation. No self-grading or answer matching."""
import argparse
from collections import Counter
import json
from pathlib import Path

from eval_joint_baseline import fixed_panel
from native_memory_encoder import sha_file
from prepare_neural_memory_protocol import digest
from review_availability_generation import predictions
from review_neural_memory import blind_packet, read_rows, score_bundle

BASELINE_SHA = '4389fc1c29a28f81353dac9ecc52bec8bb024fd54c658ceecc8485db5874433d'


def repeated_block(ids):
    return any(ids[i:i+n] == ids[i+n:i+2*n] == ids[i+2*n:i+3*n]
               for n in range(4, len(ids)//3+1) for i in range(len(ids)-3*n+1))


def summarize(rows):
    return {'total': len(rows), **{s: sum(r['status'] == s for r in rows) for s in ('pass','fail','needs_review')}}


def apply_gates(score, report, baseline, panel):
    output = {r['id']: r for r in report['predictions']}
    base = {r['id']: r for r in baseline['predictions']}
    meta = {r['id']: r for r in panel}
    invalid = []
    for case in score['cases']:
        row = output[case['id']]; scenario = meta[case['id']]['scenario']
        failures = []
        if row['truncated'] or not row['utf8_complete']: failures.append('incomplete_generation')
        if repeated_block(row['generated_token_ids']): failures.append('repeated_token_block')
        if report['memory_enabled'] and scenario.startswith('ordinary_'):
            reference = base[case['id']]
            if row['selected_route'] != 0 or any(row[k] != reference[k] for k in ('generated_token_ids','raw_text_hex')):
                failures.append('normal_route_or_baseline_changed')
        if failures:
            invalid.append(case['id']); case['reasons'].extend(failures)
            # Retain unresolved semantic review even if a mechanical failure exists.
            if case['status'] == 'pass': case['status'] = 'fail'
    score.update(summarize(score['cases']))
    score['confirmed_success_fraction'] = score['pass'] / score['total']
    score['adjudication_complete'] = score['needs_review'] == 0
    for key in score['groups']:
        score['groups'][key] = summarize([c for c in score['cases'] if c['scenario']+'/'+c['language'] == key])
    for key in score['worlds']:
        score['worlds'][key] = summarize([c for c in score['cases'] if c['world_id'] == key])
    categories = {'grounded': ('correct','value_swap'), 'role_swap_negative': ('role_swap',),
                  'empty': ('empty',), 'normal': ('ordinary_source','ordinary_empty')}
    score['denominators'] = {k: summarize([c for c in score['cases'] if c['scenario'] in kinds]) for k,kinds in categories.items()}
    pairs = []
    for language in ('en','zh'):
        for relation in ('home_city','work_city'):
            sides = [c for c in score['cases'] if meta[c['id']]['language'] == language and
                     meta[c['id']]['relation'] == relation and c['scenario'] in ('correct','value_swap')]
            if len(sides) != 2: raise ValueError('value pair denominator changed')
            status = ('needs_review' if any(c['status']=='needs_review' for c in sides) else
                      'pass' if all(c['status']=='pass' for c in sides) else 'fail')
            pairs.append({'language':language,'relation':relation,'status':status})
    score['value_pairs'] = {**summarize(pairs), 'pairs':pairs}
    d = score['denominators']
    if {k:v['total'] for k,v in d.items()} != {'grounded':8,'role_swap_negative':4,'empty':4,'normal':8}:
        raise ValueError('panel denominator changed')
    score['mechanically_invalid_ids'] = invalid
    score['pilot_gates_passed'] = bool(report['memory_enabled'] and not invalid and not score['needs_review'] and
        d['grounded']['pass']>=6 and score['value_pairs']['pass']>=3 and d['role_swap_negative']['pass']==4 and
        d['empty']['pass']==4 and d['normal']['pass']==8)
    score['promotion_eligible'] = False  # A pilot pass never authorizes deployment.
    return score


def run(a):
    if sha_file(a.baseline) != BASELINE_SHA: raise ValueError('fixed baseline changed')
    base = json.loads(Path(a.baseline).read_text())
    audit = json.loads(Path(a.panel_audit).read_text())
    if sha_file(a.panel_audit) != base['artifact_sha256']['audit']: raise ValueError('panel audit changed')
    inputs, panel = fixed_panel(a.corpus,audit); ids = {r['id'] for r in inputs}
    report = json.loads(Path(a.candidate).read_text()) if a.candidate else base
    if [r['id'] for r in report['predictions']] != [r['id'] for r in inputs]: raise ValueError('prediction inventory changed')
    if a.candidate:
        if (report['format']!='cg003-fixed-native-checkpoint-v1' or report['baseline_sha256']!=BASELINE_SHA or
                report['memory_enabled'] is not True or report['max_new_tokens']!=64 or report['context_capacity']!=128 or
                report['route_policy']!='predicted_initial_prefill_argmax_fixed_for_reply'):
            raise ValueError('unqualified candidate protocol')
        if not all(r['backend_cache_policy_verified'] and r['every_base_reference_checked'] and
                   type(r['selected_route']) is int and r['selected_route'] in (0,1,2) for r in report['predictions']):
            raise ValueError('unqualified candidate traces')
    normalized = predictions(report,'cg003-candidate' if a.candidate else 'cg003-baseline')
    packets = {p['id']:p for r,prediction in zip(inputs,normalized) for p in [blind_packet(r,prediction)]}
    if a.mode=='pack': result={'format':'neural-memory-blind-review-v1','packets':[packets[k] for k in sorted(packets)]}
    else:
        corpus=Path(a.corpus); manifest=json.loads((corpus/'manifest.json').read_text())
        if sha_file(corpus/'dev.labels.jsonl')!=manifest['file_sha256']['dev.labels.jsonl']: raise ValueError('labels changed')
        labels=[r for r in read_rows(corpus/'dev.labels.jsonl') if r['id'] in ids]
        index=[r for r in read_rows(corpus/'dev.index.jsonl') if r['id'] in ids]
        reviews=[r for path in a.reviews for r in read_rows(path)]
        result=apply_gates(score_bundle(inputs,labels,index,normalized,reviews,a.reviewer),report,base,panel)
        result.update(baseline_sha256=BASELINE_SHA, report_sha256=sha_file(a.candidate or a.baseline),
            review_sha256={str(p):sha_file(p) for p in a.reviews}, source_sha256=sha_file(__file__),
            independence_limit='Isolated reviewers of the same model family, not human or cross-family validation',
            additional_training_approved=False)
    with Path(a.output).open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps({k:result[k] for k in ('total','pass','fail','needs_review','denominators','value_pairs','pilot_gates_passed') if k in result}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('pack','score'))
    for n in ('baseline','panel-audit','corpus','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--candidate');p.add_argument('--reviews',action='append',default=[]);p.add_argument('--reviewer',action='append',default=[])
    run(p.parse_args())
