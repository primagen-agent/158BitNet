"""Bind complete C audits, blind both arms together, then apply fixed gates."""
import argparse
import json
from pathlib import Path

from eval_joint_baseline import fixed_panel
from native_memory_encoder import sha_file
from review_availability_generation import predictions
from review_joint_generation import BASELINE_SHA, apply_gates
from review_neural_memory import blind_packet, read_rows, score_bundle


def qualify(report,audit,arm,ids):
    if (report['format']!='cg003-fixed-native-checkpoint-v1' or report['arm']!=arm or report['step']!=50 or
            report['baseline_sha256']!=BASELINE_SHA or report['memory_enabled'] is not True or
            report['route_policy']!='predicted_initial_prefill_argmax_fixed_for_reply' or
            report['max_new_tokens']!=64 or report['context_capacity']!=128 or
            [r['id'] for r in report['predictions']]!=ids):raise ValueError('candidate protocol changed')
    if (audit['format']!='cg003-native-generation-audit-v1' or audit['arm']!=arm or
            audit['complete'] is not True or audit['passed'] is not True or audit['expected_cases']!=24 or
            [r['id'] for r in audit['cases']]!=ids or not all(r['passed'] for r in audit['cases']) or
            audit['checkpoint_sha256']!=report['checkpoint_sha256'] or audit['status_sha256']!=report['status_sha256']):
        raise ValueError('complete bound native audit required; partial audits cannot qualify')
    if sum(len(r['generated_token_ids']) for r in report['predictions'])!=audit['positions']:
        raise ValueError('native position denominator mismatch')
    if not all(r['backend_cache_policy_verified'] and r['every_base_reference_checked'] for r in report['predictions']):
        raise ValueError('missing native checks')


def select(scores):
    eligible=[arm for arm in ('joint_aux','joint_product') if scores[arm]['pilot_gates_passed']]
    if not eligible:return None
    return max(eligible,key=lambda arm:(scores[arm]['value_pairs']['pass'],scores[arm]['denominators']['grounded']['pass'],arm=='joint_aux'))


def run(a):
    if sha_file(a.baseline)!=BASELINE_SHA:raise ValueError('baseline changed')
    base=json.loads(Path(a.baseline).read_text());panel_audit=json.loads(Path(a.panel_audit).read_text())
    if sha_file(a.panel_audit)!=base['artifact_sha256']['audit']:raise ValueError('panel audit changed')
    inputs,panel=fixed_panel(a.corpus,panel_audit);ids=[r['id'] for r in inputs]
    reports={'baseline':base};report_hashes={'baseline':BASELINE_SHA};audit_hashes={}
    for arm,directory,audit_file in (('joint_aux',a.aux,a.aux_audit),('joint_product',a.product,a.product_audit)):
        path=Path(directory)/'generation.json';report=json.loads(path.read_text());audit=json.loads(Path(audit_file).read_text())
        if audit['generation_sha256']!=sha_file(path):raise ValueError('generation/audit identity mismatch')
        qualify(report,audit,arm,ids)
        reports[arm]=report;report_hashes[arm]=sha_file(path);audit_hashes[arm]=sha_file(audit_file)
    normalized={name:predictions(r,'cg003-'+name) for name,r in reports.items()}
    packets={};run_packet_ids={}
    for name,rows in normalized.items():
        run_packet_ids[name]=set()
        for runtime,pred in zip(inputs,rows):
            p=blind_packet(runtime,pred);packets[p['id']]=p;run_packet_ids[name].add(p['id'])
    if a.mode=='pack':
        result={'format':'neural-memory-blind-review-v1','packets':[packets[k] for k in sorted(packets)]}
    else:
        corpus=Path(a.corpus);manifest=json.loads((corpus/'manifest.json').read_text())
        if sha_file(corpus/'dev.labels.jsonl')!=manifest['file_sha256']['dev.labels.jsonl']:raise ValueError('labels changed')
        labels=[r for r in read_rows(corpus/'dev.labels.jsonl') if r['id'] in ids]
        index=[r for r in read_rows(corpus/'dev.index.jsonl') if r['id'] in ids]
        reviews=[r for path in a.reviews for r in read_rows(path)]
        if any(r['id'] not in packets for r in reviews):raise ValueError('unknown review packet')
        scores={}
        for name,rows in normalized.items():
            votes=[r for r in reviews if r['id'] in run_packet_ids[name]]
            scores[name]=apply_gates(score_bundle(inputs,labels,index,rows,votes,a.reviewer),reports[name],base,panel)
        result={'format':'cg003-fixed-native-final-review-v1','runs':scores,'selected_arm':select(scores),
            'unique_blind_packets':len(packets),'case_uses':72,'report_sha256':report_hashes,'native_audit_sha256':audit_hashes,
            'review_sha256':{str(p):sha_file(p) for p in a.reviews},'reviewers':a.reviewer,
            'independence_limit':'Two isolated contexts of the same model family, not human or cross-family validation',
            'deployment_approved':False,'additional_training_approved':False,'remaining_optimizer_steps':0,
            'source_sha256':sha_file(__file__)}
    with Path(a.output).open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'mode':a.mode,'unique_packets':len(packets),'selected_arm':result.get('selected_arm')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('pack','score'))
    for n in ('baseline','panel-audit','corpus','aux','aux-audit','product','product-audit','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--reviews',action='append',default=[]);p.add_argument('--reviewer',action='append',default=[])
    run(p.parse_args())
