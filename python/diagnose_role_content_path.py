"""DG-008: locate sensitivity/binding failures without training or new answers.

Content-only probes on reference prefixes are oracle component diagnostics.
Semantic annotations and target tokens are used only after neural forward.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
import statistics

import numpy as np
import torch
from torch.nn import functional as F

from c_tokenizer import CTokenizer
from episode_memory_inputs import encoder_texts
from eval_role_checkpoint import CHECKPOINT_SHA256,load_checkpoint
from native_memory_encoder import sha_file,text_key
from prepare_availability_curriculum import materialize
from prepare_neural_memory_protocol import digest
from review_neural_memory import read_rows
from train_availability_memory import TrainingPackage
from train_role_separated_memory import FEATURE_DIGEST


def token_span(text,pieces,value):
    """Post-forward byte alignment, never an attention mask or gold routing."""
    raw=text.encode();decoded=b''.join(pieces);needle=value.encode()
    pad=0 if decoded==raw else 1 if decoded==b' '+raw else None
    if pad is None or raw.count(needle)!=1:raise ValueError('ambiguous/mismatched annotation bytes')
    start=raw.index(needle)+pad;end=start+len(needle);offset=0;positions=[]
    for i,piece in enumerate(pieces):
        if offset<end and offset+len(piece)>start:positions.append(i)
        offset+=len(piece)
    if not positions:raise ValueError('no token covers annotated value')
    return positions


def validate_control(left,right,left_fact,right_fact,kind):
    if left['context']!=right['context'] or len(left['episodes'])!=1 or len(right['episodes'])!=1:
        raise ValueError('control changed query or event count')
    field='value' if kind=='value' else 'subject' if kind=='subject' else None
    if field is None:raise ValueError('unknown control')
    if (set(left_fact)!=set(right_fact) or field not in left_fact or left_fact[field]==right_fact[field] or
        any(left_fact[k]!=right_fact[k] for k in left_fact if k!=field)):
        raise ValueError('confounded control')
    a,b=left['episodes'][0],right['episodes'][0]
    if any(a[k]!=b[k] for k in ('role','speaker')):raise ValueError('speaker changed')
    if a['text'].count(left_fact[field])!=1 or a['text'].replace(left_fact[field],right_fact[field],1)!=b['text']:
        raise ValueError('more than the registered factor changed')


def first_divergence(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:return i
    raise ValueError('no distinct target token in paired replies')


def difference(a,b):
    a,b=a.detach().double().flatten(),b.detach().double().flatten()
    if a.shape!=b.shape:raise ValueError('unmatched observation geometry')
    denominator=(a.norm()+b.norm())/2
    return {'relative_l2':float((a-b).norm()/denominator) if float(denominator)>0 else 0.,
            'cosine':float(F.cosine_similarity(a[None],b[None])) if float(a.norm()*b.norm())>0 else None}


def trace_content(model,hidden,features):
    """No IDs, labels, token spans or target tokens accepted by this function."""
    if hidden.ndim!=2 or hidden.shape!=(1,model.hidden) or hidden.dtype!=torch.float32 or not torch.isfinite(hidden).all():
        raise ValueError('one finite frozen-prefix hidden row required')
    prepared=model.prepare(features);m=model.content
    if prepared is None:raise ValueError('this content diagnostic requires a supplied source')
    x=F.layer_norm(hidden,(model.hidden,))
    query=m.query(x).view(-1,model.heads,model.hidden//model.heads).transpose(0,1)
    key,value=prepared
    attention=(query@key.transpose(-1,-2)/math.sqrt(model.hidden//model.heads)).softmax(-1)
    read=F.scaled_dot_product_attention(query,key,value).transpose(0,1).contiguous().view(-1,model.hidden)
    manual=(attention@value).transpose(0,1).contiguous().view(-1,model.hidden)
    projected=m.output(read);gate=m.use_gate(x).sigmoid()
    delta=m.layer_gain[model.layers-1].tanh()*(projected*gate)
    if not torch.equal(delta,m.residual(model.layers-1,hidden,prepared)):raise ValueError('content reconstruction not exact')
    if not torch.allclose(read,manual,atol=1e-5,rtol=1e-4):raise ValueError('attention attribution numeric mismatch')
    return {'encoded_mean':F.layer_norm(m.encoder(features),(model.hidden,)).mean(0),
            'read':read,'projected':projected,'delta':delta,'attention':attention[ :,0,:]}


def summarize_trace(trace,subject_span,value_span):
    attention=trace['attention'];count=attention.shape[-1]-1
    def mass(span):
        total=float(attention[:,[i+1 for i in span]].sum(-1).mean())
        return {'token_count':len(span),'mass':total,'uniform_mass':len(span)/(count+1),
                'density_over_uniform':total/(len(span)/(count+1))}
    result={'source_tokens':count,'null_mass':float(attention[:,0].mean()),'subject_attention':mass(subject_span),
            'value_attention':mass(value_span),'read_l2':float(trace['read'].norm()),'content_delta_l2':float(trace['delta'].norm())}
    if 'state_probabilities' in trace:
        result.update(state_probabilities=trace['state_probabilities'],predicted_state=int(np.argmax(trace['state_probabilities'])))
    return result


def target_stats(logits,target):
    return {'token_id':int(target),'rank':int((logits>logits[target]).sum())+1,
            'log_probability':float(logits.double().log_softmax(-1)[target]),
            'top1_correct':int(logits.argmax())==target}


def summarize(rows):
    probes=[r['value_probe'] for r in rows];effects=[p['source_swap_effect_float64'] for p in probes]
    median=statistics.median
    ratios=[p['source_swap_effect_float64']/abs(p['two_choice_margins']['source_a']+p['two_choice_margins']['source_b'])
            for p in probes if p['two_choice_margins']['source_a']+p['two_choice_margins']['source_b']!=0]
    return {'source_records':3*len(rows),'value_pairs':len(rows),'subject_pairs':len(rows),
            'effect_direction_correct':sum(p['direction_correct'] for p in probes),
            'two_choice_pairs_correct':sum(p['two_choice_pair_correct'] for p in probes),
            'full_vocab_next_tokens_correct':sum(t['top1_correct'] for p in probes for t in p['own_source_target_stats']),
            'value_token_positions':2*len(rows),'target_token_present_in_source_span':sum(sum(p['own_target_token_present_in_source_value_span']) for p in probes),
            'reference_prefix_also_seen_in_actual_generation':sum(sum(p['prefix_observed_in_free_generation']) for p in probes),
            'median_target_rank_base':median(t['rank'] for p in probes for t in p['base_target_stats']),
            'median_target_rank_conditioned':median(t['rank'] for p in probes for t in p['own_source_target_stats']),
            'median_value_point_read_relative_difference':median(p['read_difference']['relative_l2'] for p in probes),
            'median_value_point_delta_relative_difference':median(p['delta_difference']['relative_l2'] for p in probes),
            'median_value_attention_density_over_uniform':median(p['attention'][s]['value_attention']['density_over_uniform'] for p in probes for s in ('source_a','source_b')),
            'effect_min_median_max':[min(effects),median(effects),max(effects)],
            'effect_over_fixed_center_pair_flip_requirement_median':median(ratios) if ratios else None,
            'ratio_note':'Descriptive margin comparison only, NOT a proposed gain or evidence that uniform rescaling would work',
            'predicted_supported_counts':{s:sum(r['sources'][s]['initial']['predicted_state']==1 for r in rows) for s in ('original','swapped_values','wrong_subject')}}


def exposure_audit(path,records,expected_hash):
    if sha_file(path)!=expected_hash:raise ValueError('training log identity changed')
    run=json.loads(Path(path).read_text())
    if [r['step'] for r in run['training']]!=list(range(1,51)):raise ValueError('training step inventory changed')
    sampled=[e['id'] for row in run['training'] for e in row['examples']]
    if len(sampled)!=200 or any(records[cid]['split']!='train' for cid in sampled):raise ValueError('training exposure split mismatch')
    counts=Counter(records[cid]['scenario'] for cid in sampled);seen=set(sampled);pairs={}
    for r in records.values():
        if r['split']=='train' and r['scenario'] in ('swapped_values','wrong_subject'):
            pairs.setdefault((r['world_id'],r['language']),{})[r['scenario']]=r['id']
    positive=sum(counts[s] for s in ('original','swapped_values','paraphrase','update_current'))
    return {'scope':'Exploratory exposure audit of existing logs; no change to registered probe denominators',
            'source_sha256':sha_file(path),'sampled_rows':len(sampled),'scenario_draws':dict(counts),
            'supported_draws':positive,'wrong_subject_draws':counts['wrong_subject'],
            'nominal_state_weighted_exposures':[positive*2,counts['wrong_subject']],
            'subject_control_training_pairs':len(pairs),'both_sides_seen_at_least_once':sum(all(i in seen for i in pair.values()) for pair in pairs.values()),
            'limitations':'Exposure/nominal weights are not gradient magnitudes or proof of the sole cause; this does not authorize more steps'}


def run(a):
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(a.experiment).read_text())
    if config['id']!='DG-008' or config['checkpoint_sha256']!=CHECKPOINT_SHA256 or config['optimizer_steps']!=0:
        raise ValueError('wrong diagnostic registration')
    torch.set_num_threads(4);model,binding=load_checkpoint(a.checkpoint)
    package=TrainingPackage(a.features,FEATURE_DIGEST)
    manifest=materialize(a.config,a.corpus,verify=True)
    if digest(manifest)!=package.manifest['corpus_manifest_sha256']:raise ValueError('corpus changed')
    if sha_file(a.gguf)!=binding['backbone_sha256']:raise ValueError('wrong backbone')
    expected_tokenizers={v for k,v in package.manifest['binaries_sha256'].items() if Path(k).name=='tok_probe'}
    if expected_tokenizers!={sha_file(a.tok_probe)}:raise ValueError('tokenizer identity changed')
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    records={r['id']:r for r in package.records}
    inputs={r['id']:r for r in read_rows(Path(a.corpus)/'dev.inputs.jsonl')}
    meta=read_rows(Path(a.corpus)/'dev.index.jsonl')
    groups={}
    for r in meta:
        if r['scenario'] in ('original','swapped_values','wrong_subject'):
            groups.setdefault((r['world_id'],r['language']),{})[r['scenario']]=r
    if len(groups)!=16 or any(len(g)!=3 for g in groups.values()):raise ValueError('diagnostic denominator changed')
    generation=Path(a.generation);repo=Path(__file__).resolve().parents[1]
    if sha_file(generation/'generation.json')!=sha_file(repo/'training/memory/neural-system/reviews/CG-002/generation.json'):
        raise ValueError('generation binding mismatch')
    observed={}
    for r in json.loads((generation/'generation.json').read_text())['predictions']:
        backend=json.loads((generation/r['backend_evidence']).read_text())
        observed[r['id']]={tuple(s['prefix_ids']) for s in backend['steps']}
    tokenizer=CTokenizer(a.tok_probe,a.gguf);rows=[]
    try:
        with torch.no_grad():
            for (world,language),group in sorted(groups.items()):
                facts={s:g['source_facts'][0] for s,g in group.items()}
                runs={s:records[g['id']] for s,g in group.items()}
                for kind,left,right in (('value','original','swapped_values'),('subject','swapped_values','wrong_subject')):
                    validate_control(inputs[runs[left]['id']],inputs[runs[right]['id']],facts[left],facts[right],kind)
                prompt=tuple(runs['original']['prompt_ids'])
                if any(tuple(r['prompt_ids'])!=prompt for r in runs.values()):raise ValueError('query prefix changed')
                hidden,base=package.prefix(prompt);hidden=hidden[None]
                sources={};spans={};initial={};input_reports={}
                for scenario,record in runs.items():
                    if record['input_sha256']!=digest(inputs[record['id']]) or tuple(record['source_texts'])!=encoder_texts(inputs[record['id']])[1]:
                        raise ValueError('record and natural runtime input mismatch')
                    text=record['source_texts'][0];encoded=package.source.rows[text_key(text)]
                    if tokenizer.encode(text,True)!=encoded.token_ids.tolist():raise ValueError('source token IDs changed')
                    pieces=tokenizer.decode_pieces(encoded.token_ids[1:].tolist())
                    subject=token_span(text,pieces,facts[scenario]['subject']);value=token_span(text,pieces,facts[scenario]['value'])
                    sources[scenario]=torch.from_numpy(encoded.features.copy());spans[scenario]=(subject,value)
                    initial[scenario]=trace_content(model,hidden,sources[scenario])
                    # Task classification is observed ONLY at initial user prefill,
                    # never recomputed on the teacher-forced value-probe prefix.
                    initial[scenario]['state_probabilities']=model.decide(hidden,model.prepare(sources[scenario])).probabilities[0].tolist()
                    input_reports[scenario]={'id':record['id'],'source_sha256':text_key(text),'token_ids':encoded.token_ids.tolist(),
                        'subject_token_positions':subject,'value_token_positions':value,
                        'subject_lexical_l2':float(sources[scenario][subject,1024:].norm()),
                        'value_lexical_l2':float(sources[scenario][value,1024:].norm()),
                        'initial':summarize_trace(initial[scenario],subject,value)}
                contrasts={}
                for kind,left,right in (('value','original','swapped_values'),('subject','swapped_values','wrong_subject')):
                    contrasts[kind]={name:difference(initial[left][name],initial[right][name]) for name in ('encoded_mean','read','projected','delta')}
                    contrasts[kind]['supported_probability_difference']=initial[left]['state_probabilities'][1]-initial[right]['state_probabilities'][1]
                left,right=runs['original'],runs['swapped_values']
                ta,tb=left['targets']['completion_token_ids'],right['targets']['completion_token_ids']
                offset=first_divergence(ta,tb)
                for scenario,record in (('original',left),('swapped_values',right)):
                    target=record['targets']
                    positions=token_span(target['response'],tokenizer.decode_pieces(target['completion_token_ids'][:-1]),facts[scenario]['value'])
                    if offset not in positions:raise ValueError('first divergence not within the target value')
                prefix=prompt+tuple(ta[:offset]);h,b=package.prefix(prefix)
                va=trace_content(model,h[None],sources['original']);vb=trace_content(model,h[None],sources['swapped_values'])
                corrections=F.linear(torch.cat((va['delta'],vb['delta'])),package.head)/package.scale
                logits=b[None]+corrections;ia,ib=ta[offset],tb[offset]
                margin_base=float(b[ia]-b[ib]);margin_a=float(logits[0,ia]-logits[0,ib]);margin_b=float(logits[1,ia]-logits[1,ib])
                # Independent linear contrast avoids subtracting nearly equal
                # FP32 full logits when inspecting a very small source effect.
                effect64=float(((va['delta']-vb['delta']).double()[0]*(package.head[ia].double()-package.head[ib].double())).sum()/package.scale)
                if abs(effect64-(margin_a-margin_b))>1e-5+1e-4*abs(effect64):raise ValueError('unstable value contrast')
                value_probe={'position':'first divergent reference value token, not a generated answer','common_prefix_ids':list(prefix),
                    'common_reply_prefix':b''.join(tokenizer.decode_pieces(ta[:offset])).decode('utf-8',errors='replace'),
                    'offset':offset,'target_values':[facts['original']['value'],facts['swapped_values']['value']],
                    'target_token_ids':[ia,ib],'target_pieces_hex':[p.hex() for p in tokenizer.decode_pieces([ia,ib])],
                    'prefix_observed_in_free_generation':[prefix in observed.get(record['id'],set()) for record in (left,right)],
                    'base_target_stats':[target_stats(b,ia),target_stats(b,ib)],
                    'own_source_target_stats':[target_stats(logits[0],ia),target_stats(logits[1],ib)],
                    'two_choice_margins':{'base':margin_base,'source_a':margin_a,'source_b':margin_b},
                    'source_swap_effect':margin_a-margin_b,'source_swap_effect_float64':effect64,'direction_correct':effect64>1e-6,
                    'two_choice_pair_correct':margin_a>0 and margin_b<0,
                    'full_vocab_next_token_pair_correct':int(logits[0].argmax())==ia and int(logits[1].argmax())==ib,
                    'attention':{'source_a':summarize_trace(va,*spans['original']),'source_b':summarize_trace(vb,*spans['swapped_values'])},
                    'read_difference':difference(va['read'],vb['read']),'delta_difference':difference(va['delta'],vb['delta'])}
                value_probe['own_target_token_present_in_source_value_span']=[
                    token in [input_reports[scenario]['token_ids'][i+1] for i in input_reports[scenario]['value_token_positions']]
                    for scenario,token in (('original',ia),('swapped_values',ib))]
                rows.append({'world_id':world,'language':language,'sources':input_reports,'initial_contrasts':contrasts,'value_probe':value_probe})
                print(json.dumps({'completed':len(rows),'total':16,'value_effect':value_probe['source_swap_effect']}),flush=True)
        if any(not torch.equal(p,before[name]) for name,p in model.named_parameters()):raise ValueError('checkpoint parameters changed')
        report={'experiment':config,'pairs':rows,'summary':summarize(rows),'optimizer_steps':0,'parameters_unchanged':True,'numeric_replay_passed':True,
            'source_records_verified':48,'value_pairs':16,'subject_pairs':16,'new_free_generated_answers':0,'memory_accuracy_measured':False,
            'training_approved':False,'deployment_approved':False,'feature_package_digest':FEATURE_DIGEST,
            'artifact_sha256':{name:sha_file(getattr(a,name)) for name in ('experiment','checkpoint','gguf','tok_probe')},'source_sha256':sha_file(__file__)}
        if a.training_report:
            expected=json.loads((repo/'training/memory/neural-system/checks/CG-002-step50.json').read_text())['raw_report']['sha256']
            report['training_exposure_audit']=exposure_audit(a.training_report,records,expected)
        with (root/'report.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
    finally:
        tokenizer._proc.terminate();tokenizer._proc.wait();tokenizer._proc.stdin.close();tokenizer._proc.stdout.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('experiment','checkpoint','features','config','corpus','gguf','tok-probe','generation','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--training-report',help='Optional separately labelled exploratory exposure audit; never used by neural probes')
    run(p.parse_args())


if __name__=='__main__':main()
