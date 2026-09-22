"""Blind annotation and conservative paired scoring for the fixed CG-001 panel."""
import argparse
import json
from pathlib import Path

from eval_availability_checkpoint import fixed_panel, BASELINE_SHA256, CHECKPOINT_SHA256
from native_memory_encoder import sha_file
from prepare_neural_memory_protocol import digest
from review_neural_memory import blind_packet, read_rows, score_bundle


def validate_candidate(candidate,kind):
    expected=CHECKPOINT_SHA256
    if kind=='cg002':
        expected='eb6b9a551c616d08a2aa0e8d183a63233c87a3c10e296f290a7b48b99dea716d'
        if candidate.get('format')!='cg002-fixed-native-checkpoint-v1' or candidate.get('route_policy')!='predicted_initial_prefill_argmax_fixed_for_reply':
            raise ValueError('wrong role generation protocol')
        if not all(type(r.get('selected_route')) is int and r['selected_route'] in (0,1,2) and
                   r.get('backend_cache_policy_verified') is True and r.get('every_base_reference_checked') is True
                   for r in candidate.get('predictions',[])):
            raise ValueError('unqualified role generation trace')
    elif kind!='cg001':raise ValueError('unknown candidate registration')
    if candidate.get('checkpoint_sha256')!=expected or candidate.get('baseline_sha256')!=BASELINE_SHA256:
        raise ValueError('candidate binding mismatch')


def predictions(report,producer):
    if report['cached_tokens']!=0 or report['reused_tokens']!=0 or report['oracle_answers_used']:
        raise ValueError('unqualified generation report')
    trace={'level':'predicted_component','backbone_kv_cache_enabled':False,'cached_tokens':0,'reused_tokens':0,
           'label_fields_in_forward':[],'future_messages_in_forward':[],'oracle_activation':False,
           'supplied_write_boundaries':True,'source_text_in_prompt':False,'whole_answer_bypass':False,
           'generated_by_decoder':True}
    return [{'id':r['id'],'text':r['text'],'input_sha256':r['input_sha256'],'producer_id':producer,'trace':trace}
            for r in report['predictions']]


def paired_summary(cases):
    groups={}
    for c in cases: groups.setdefault((c['world_id'],c['language']),{})[c['scenario']]=c
    result={}
    for name,left,right in (('value_swap','original','swapped_values'),('remove_memory','original','empty')):
        rows=[]
        for (world,language),group in sorted(groups.items()):
            sides=[group.get(left),group.get(right)]
            status=('needs_review' if any(s is None or s['status']=='needs_review' for s in sides)
                    else 'pass' if all(s['status']=='pass' for s in sides) else 'fail')
            rows.append({'world_id':world,'language':language,'status':status})
        result[name]={'total':len(rows),**{s:sum(r['status']==s for r in rows) for s in ('pass','fail','needs_review')},'pairs':rows}
    return result


def qualify_complete_generation(score,report):
    clipped={r['id'] for r in report['predictions'] if r['truncated'] or not r['utf8_complete']}
    score['incomplete_generation_ids']=sorted(clipped)
    for case in score['cases']:
        if case['id'] in clipped:
            case['reasons'].append('incomplete_generation')
            if case['status']=='pass': case['status']='fail'
    summarize=lambda rows: {'total':len(rows),**{s:sum(c['status']==s for c in rows) for s in ('pass','fail','needs_review')}}
    score.update(summarize(score['cases']))
    score['confirmed_success_fraction']=score['pass']/score['total']
    score['adjudication_complete']=not score['needs_review']
    for key in score['groups']:
        score['groups'][key]=summarize([c for c in score['cases'] if c['scenario']+'/'+c['language']==key])
    for key in score['worlds']:
        score['worlds'][key]=summarize([c for c in score['cases'] if c['world_id']==key])
    score['paired_controls']=paired_summary(score['cases'])
    return score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('pack','score'))
    for name in ('corpus','baseline','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--candidate'); p.add_argument('--reviews',action='append',default=[])
    p.add_argument('--candidate-kind',choices=('cg001','cg002'),default='cg001')
    p.add_argument('--reviewer',action='append',default=[])
    a=p.parse_args()
    if sha_file(a.baseline)!=BASELINE_SHA256: raise ValueError('baseline changed')
    baseline=json.loads(Path(a.baseline).read_text())
    corpus=Path(a.corpus); index=read_rows(corpus/'dev.index.jsonl')
    inputs=fixed_panel(read_rows(corpus/'dev.inputs.jsonl'),index,baseline)
    selected={r['id'] for r in inputs}; index=[r for r in index if r['id'] in selected]
    runs={'baseline':baseline}
    if a.candidate:
        candidate=json.loads(Path(a.candidate).read_text())
        validate_candidate(candidate,a.candidate_kind)
        fixed_panel(inputs,index,candidate)
        runs['step50' if a.candidate_kind=='cg001' else 'cg002_step50']=candidate
    normalized={name:predictions(report,name) for name,report in runs.items()}
    runtime={r['id']:r for r in inputs}
    packets={packet['id']:packet for rows in normalized.values() for r in rows for packet in [blind_packet(runtime[r['id']],r)]}
    if a.mode=='pack':
        result={'format':'neural-memory-blind-review-v1','packets':[packets[k] for k in sorted(packets)]}
    else:
        labels=[r for r in read_rows(corpus/'dev.labels.jsonl') if r['id'] in selected]
        reviews=[r for path in a.reviews for r in read_rows(path)]
        if any(r['id'] not in packets for r in reviews): raise ValueError('unknown review packet')
        result={'scope':'fixed 16-case single-supplied-episode component, two development worlds',
                'reviewers':a.reviewer,'independence_limit':'isolated contexts of same model family; not human or cross-family validation',
                'promotion_eligible':False,'additional_training_approved':False,'runs':{},
                'source_sha256':sha_file(__file__),'review_sha256':{str(path):sha_file(path) for path in a.reviews}}
        for name,rows in normalized.items():
            ids={blind_packet(runtime[r['id']],r)['id'] for r in rows}
            score=score_bundle(inputs,labels,index,rows,[r for r in reviews if r['id'] in ids],a.reviewer)
            # Retain clipped responses but never count them as complete success.
            result['runs'][name]=qualify_complete_generation(score,runs[name])
        result['prediction_reports_sha256']={str(path):sha_file(path) for path in (a.baseline,a.candidate) if path}
    with Path(a.output).open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2); f.write('\n')


if __name__=='__main__': main()
