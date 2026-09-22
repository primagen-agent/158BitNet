"""Blind, audited comparison of the 16 fresh DG-013 interventions only."""
import argparse
import json
from pathlib import Path

from ablate_joint_content import ORIGINAL_SHA, selected_indices
from eval_joint_baseline import fixed_panel
from native_memory_encoder import sha_file
from review_availability_generation import predictions
from review_joint_generation import BASELINE_SHA, repeated_block
from review_neural_memory import blind_packet, read_rows, score_bundle


def qualify_fresh(report,audit,original,arm):
    indices=selected_indices(original,arm)
    if (report['format']!='dg013-content-off-generation-v1' or report['arm']!=arm or report['fresh_indices']!=indices or
            report['optimizer_steps']!=0 or report['parameters_unchanged'] is not True or
            report['checkpoint_sha256']!=original['checkpoint_sha256'] or report['original_sha256']!=ORIGINAL_SHA[arm] or
            report['all_predicted_routes_unchanged'] is not True or report['max_new_tokens']!=64 or report['context_capacity']!=128):
        raise ValueError('unregistered intervention')
    ids=[original['predictions'][i]['id'] for i in indices]
    if ([r['id'] for r in report['predictions']]!=ids or audit['format']!='dg013-content-off-audit-v1' or
            audit['arm']!=arm or audit['passed'] is not True or audit['fresh_cases']!=len(ids) or
            audit['unchanged_cases_not_rerun']!=24-len(ids) or audit['original_sha256']!=ORIGINAL_SHA[arm] or
            [r['id'] for r in audit['cases']]!=ids or not all(r['passed'] for r in audit['cases'])):
        raise ValueError('complete fresh-subset audit required')
    if audit['positions']!=sum(len(r['generated_token_ids']) for r in report['predictions']):raise ValueError('position count changed')
    return indices


def finish_score(score,report):
    outputs={r['id']:r for r in report['predictions']}
    for c in score['cases']:
        r=outputs[c['id']];bad=r['truncated'] or not r['utf8_complete'] or repeated_block(r['generated_token_ids'])
        if bad:
            c['reasons'].append('incomplete_or_repetitive_generation')
            if c['status']=='pass':c['status']='fail'
    def summary(rows):return {'total':len(rows),**{s:sum(c['status']==s for c in rows) for s in ('pass','fail','needs_review')}}
    score.update(summary(score['cases']));score['confirmed_success_fraction']=score['pass']/score['total']
    score['adjudication_complete']=score['needs_review']==0
    score['grounded']=summary([c for c in score['cases'] if c['scenario'] in ('correct','value_swap')])
    score['role_swap_negative']=summary([c for c in score['cases'] if c['scenario']=='role_swap'])
    score['truncated']=sum(r['truncated'] for r in outputs.values())
    for key in score['groups']:score['groups'][key]=summary([c for c in score['cases'] if c['scenario']+'/'+c['language']==key])
    for key in score['worlds']:score['worlds'][key]=summary([c for c in score['cases'] if c['world_id']==key])
    score['promotion_eligible']=False
    return score


def run(a):
    repo=Path(__file__).resolve().parents[1];research=repo/'training/memory/neural-system'
    baseline=research/'reviews/CG-003/baseline.json'
    if sha_file(baseline)!=BASELINE_SHA:raise ValueError('baseline changed')
    base=json.loads(baseline.read_text());panel_audit=research/'checks/CG-003-data.json'
    if sha_file(panel_audit)!=base['artifact_sha256']['audit']:raise ValueError('panel changed')
    corpus=research/'data/JB-001';inputs,panel=fixed_panel(corpus,json.loads(panel_audit.read_text()));runtime={r['id']:r for r in inputs}
    reports={};sets={};provenance={}
    for arm,short,directory,audit_file in (('joint_aux','aux',a.aux,a.aux_audit),('joint_product','product',a.product,a.product_audit)):
        oldpath=research/f'reviews/CG-003/{short}-generation.json'
        if sha_file(oldpath)!=ORIGINAL_SHA[arm]:raise ValueError('original report changed')
        old=json.loads(oldpath.read_text());path=Path(directory)/'generation.json';new=json.loads(path.read_text())
        audit=json.loads(Path(audit_file).read_text())
        if audit['generation_sha256']!=sha_file(path):raise ValueError('audit/report identity mismatch')
        indices=qualify_fresh(new,audit,old,arm)
        reports[arm+'_original_selected']={**old,'predictions':[old['predictions'][i] for i in indices]}
        reports[arm+'_content_off']=new
        provenance[arm]={'original_sha256':sha_file(oldpath),'fresh_report_sha256':sha_file(path),'audit_sha256':sha_file(audit_file),
                         'fresh_indices':indices,'historical_unchanged_controls_not_rerun':24-len(indices)}
    normalized={name:predictions(report,'dg013-'+name) for name,report in reports.items()};packets={}
    for name,rows in normalized.items():
        sets[name]=set()
        for pred in rows:
            packet=blind_packet(runtime[pred['id']],pred);packets[packet['id']]=packet;sets[name].add(packet['id'])
    if a.mode=='pack':result={'format':'neural-memory-blind-review-v1','packets':[packets[k] for k in sorted(packets)]}
    else:
        manifest=json.loads((corpus/'manifest.json').read_text())
        if sha_file(corpus/'dev.labels.jsonl')!=manifest['file_sha256']['dev.labels.jsonl']:raise ValueError('labels changed')
        labels=read_rows(corpus/'dev.labels.jsonl');index=read_rows(corpus/'dev.index.jsonl')
        reviews=[r for path in a.reviews for r in read_rows(path)]
        if any(r['id'] not in packets for r in reviews):raise ValueError('unknown review packet')
        scores={}
        for name,rows in normalized.items():
            ids={r['id'] for r in rows}
            score=score_bundle([r for r in inputs if r['id'] in ids],[r for r in labels if r['id'] in ids],
                [r for r in index if r['id'] in ids],rows,[r for r in reviews if r['id'] in sets[name]],a.reviewer)
            scores[name]=finish_score(score,reports[name])
        result={'format':'dg013-reviewed-content-ablation-v1','runs':scores,'provenance':provenance,'fresh_case_uses':16,
            'old_case_uses':16,'unchanged_controls_not_rerun':32,'unique_blind_packets':len(packets),
            'scope':'Predicted-supported subset of one development world, not a full-panel rerun or general accuracy estimate',
            'reviewers':a.reviewer,'review_sha256':{str(p):sha_file(p) for p in a.reviews},
            'independence_limit':'Two isolated contexts of the same model family; not human or cross-family validation',
            'optimizer_steps':0,'deployment_approved':False,'additional_training_approved':False,'source_sha256':sha_file(__file__)}
    with Path(a.output).open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'mode':a.mode,'unique_packets':len(packets),'runs':list(result.get('runs',{}))}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('pack','score'))
    for n in ('aux','aux-audit','product','product-audit','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--reviews',action='append',default=[]);p.add_argument('--reviewer',action='append',default=[])
    run(p.parse_args())
