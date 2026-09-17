#!/usr/bin/env python3
"""C HTTP automatic write -> export -> process restart -> import -> chat recall.

Only development worlds; no supplied spans, targets, actions or feature cache.
Accuracy failures are retained, never filtered out of the denominator.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from prepare_memory_set_curriculum import make_world
from eval_typed_memory_lifecycle import post, free_port
from test_openai_server_typed_autonomous_e2e import stop_server

def fingerprint(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('gguf');p.add_argument('resident');p.add_argument('output')
    p.add_argument('--worlds',type=int,default=24);p.add_argument('--server',default='build/openai_server')
    p.add_argument('--writer',default='models/memory/resident-0.5b/writer.bntwrite')
    p.add_argument('--pair',default='models/memory/resident-0.5b/pair.bntpair')
    p.add_argument('--link',default='models/memory/resident-0.5b/link.bntlink')
    p.add_argument('--query',default='models/memory/resident-0.5b/query.bntqact')
    a=p.parse_args()
    if not 1<=a.worlds<=24:p.error('development worlds must be 1..24')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False);states=root/'states';states.mkdir()
    worlds=[make_world(i,'valid',2810916,True) for i in range(a.worlds)]
    corpus=root/'development.jsonl';corpus.write_text(''.join(json.dumps(w)+'\n' for w in worlds))
    artifacts={k:{'path':getattr(a,k),'sha256':fingerprint(getattr(a,k))} for k in ('server','gguf','resident','writer','pair','link','query')}
    port=free_port();base=f'http://127.0.0.1:{port}';process=None;records=[];before={}
    def start(label):
        cmd=[a.server,a.gguf,'--host','127.0.0.1','--port',str(port),'--ctx','256','--max-tokens','16','--default-max-tokens','1',
             '--episodic-memory','--memory-state-dir',str(states),'--resident-model',a.resident,
             '--typed-writer-model',a.writer,'--typed-pair-model',a.pair,'--typed-link-model',a.link,'--typed-query-model',a.query]
        log=(root/(label+'.log')).open('w')
        proc=subprocess.Popen(cmd,stdout=log,stderr=log,env={**os.environ,'BITNET_NUM_THREADS':'4'});log.close()
        deadline=time.monotonic()+120
        while time.monotonic()<deadline:
            if proc.poll() is not None:raise RuntimeError('server failed: '+str(root/(label+'.log')))
            try:
                with urllib.request.urlopen(base+'/health',timeout=1):return proc
            except (OSError,TimeoutError):time.sleep(.1)
        stop_server(proc);raise TimeoutError('server startup')
    def request(path,payload):
        status,result=post(base,path,payload)
        records.append({'path':path,'request':payload,'status':status,'response':result})
        with (root/'requests.jsonl').open('a') as f:f.write(json.dumps(records[-1])+'\n')
        return status,result
    def chat(sid,text):
        status,r=request('/v1/chat/completions',{'session_id':sid,'messages':[{'role':'user','content':text}],
            'max_tokens':1,'temperature':0,'enable_thinking':False})
        if status!=200:raise AssertionError((status,r))
        usage=r.get('usage',{});session=r.get('bitnet_session',{})
        assert usage.get('prompt_tokens_details',{}).get('cached_tokens',0)==0
        assert session.get('cached_tokens',0)==0 and session.get('reused_tokens',0)==0
        return r
    writes=[];rows=[];exports=[]
    try:
        process=start('write_server')
        # Isolated lifecycle checks; never included in the accuracy corpus.
        r=chat('resident-boundary',worlds[0]['queries'][0]['text'])
        assert r['choices'][0]['message']['content']=='' and r['memory_copy']['selected_count']==0
        first,second=worlds[0]['events'][:2]
        assert chat('resident-boundary',first['text'])['memory_auto']['stored']==1
        assert chat('resident-boundary',first['text'])['memory_auto']['deduplicated']==1
        assert chat('resident-boundary',second['text'])['memory_auto']['stored']==1
        assert chat('resident-boundary',first['text'])['memory_auto']['deduplicated']==0
        status,r=request('/v1/memory/export',{'session_id':'resident-boundary'})
        assert status==200 and r['resident_events']==3
        status,_=request('/v1/memory/remember',{'session_id':'resident-boundary','memory_record':first['text']*30})
        assert status!=200
        status,r=request('/v1/memory/export',{'session_id':'resident-boundary'})
        assert status==200 and r['resident_events']==3,'failed write modified memory'
        status,_=request('/v1/memory/event',{'session_id':'resident-boundary'})
        assert status==400
        status,r=request('/v1/chat/completions',{'session_id':'resident-boundary','reset_session':True,
            'messages':[{'role':'user','content':worlds[0]['queries'][0]['text']}],'max_tokens':1})
        assert status==200 and r['memory_copy']['resident_events']==0
        for wi,w in enumerate(worlds):
            sid=f'resident-dev-{wi}'
            for ei,e in enumerate(w['events']):
                r=chat(sid,e['text']);auto=r.get('memory_auto',{})
                writes.append({'world':wi,'event':ei,'stored':auto.get('stored')==1,
                    'value_exact':auto.get('value')==e['value'],'expected_value':e['value'],'response':auto})
            status,export=request('/v1/memory/export',{'session_id':sid});assert status==200,(status,export)
            assert export['resident_events']==export['typed_events'];exports.append(export)
            if wi==0:
                for qi,q in enumerate(w['queries']):
                    r=chat(sid,q['text']);before[qi]=(r['choices'][0]['message']['content'],r.get('memory_copy'))
            print(json.dumps({'phase':'write','world':wi+1,'total':a.worlds,'stored':export['resident_events']}),flush=True)
        stop_server(process);process=None
        process=start('read_server')
        status,_=request('/v1/memory/extract',{'session_id':'resident-dev-0','query':worlds[0]['queries'][0]['text']})
        assert status==404,'server restart did not remove live sessions'
        for wi,w in enumerate(worlds):
            sid=f'resident-dev-{wi}'
            status,imported=request('/v1/memory/import',{'session_id':sid});assert status==200,(status,imported)
            assert imported['resident_state_restored']==1 and imported['raw_events_reencoded']==0
            assert imported['resident_events']==exports[wi]['resident_events']
            for qi,q in enumerate(w['queries']):
                r=chat(sid,q['text']);content=r['choices'][0]['message']['content'];meta=r.get('memory_copy',{})
                assert meta.get('mode')=='neural_resident_activation_then_compiled_pointer',r
                assert meta['raw_events_encoded_during_recall']==0
                assert r.get('memory_auto',{}).get('stored',0)==0,'question was stored as a fact'
                if wi==0:assert before[qi]==(content,r.get('memory_copy')),'restart changed recall'
                answers=content.split('\n') if content else []
                rows.append({'world':wi,'kind':q['kind'],'question':q['text'],'expected':q['answers'],
                    'actual':answers,'correct':sorted(answers)==sorted(q['answers']),
                    'selected_event_indices':meta['selected_event_indices']})
            print(json.dumps({'phase':'recall','world':wi+1,'total':a.worlds,
                'correct':sum(r['correct'] for r in rows[-12:]),'questions':12}),flush=True)
        # Corrupt this test's private snapshot and ensure failed import is atomic.
        manifest=next(states.glob('*resident-dev-0*.bnsnapshot'));lines=manifest.read_text().splitlines()
        assert lines[0]=='BNMSNAP2' and len(lines)==4
        base_path=str(manifest)[:-len('.bnsnapshot')];state_file=Path(base_path+'.'+lines[3]+'.bnresident')
        original=state_file.read_bytes()
        try:
            state_file.write_bytes(original[:-1]+bytes([original[-1]^1]))
            status,_=request('/v1/memory/import',{'session_id':'resident-dev-0'});assert status==500
            r=chat('resident-dev-0',worlds[0]['queries'][0]['text'])
            assert before[0]==(r['choices'][0]['message']['content'],r.get('memory_copy'))
        finally:state_file.write_bytes(original)
        status,_=request('/v1/chat/completions',{'session_id':'resident-dev-0','messages':[{'role':'user','content':'Forget all memories.'}]})
        assert status==400,'unsupported deletion was not rejected'
        groups={}
        for kind in ('current','multi','history','null'):
            subset=[r for r in rows if r['kind']==kind];correct=sum(r['correct'] for r in subset)
            groups[kind]={'correct':correct,'total':len(subset),'accuracy':correct/len(subset)}
        report={'artifacts':artifacts,'corpus_sha256':fingerprint(corpus),'worlds':a.worlds,
            'protocol':'normal chat -> export -> process restart -> import -> normal chat',
            'groups':groups,'correct':sum(r['correct'] for r in rows),'total':len(rows),
            'writes':{'stored':sum(w['stored'] for w in writes),'value_exact':sum(w['value_exact'] for w in writes),'total':len(writes)},
            'restart_identical':True,'corrupt_import_atomic':True,'no_kv_reuse':True,'no_raw_event_reencoding_on_import_or_recall':True,
            'empty_memory_null':True,'immediate_replay_idempotent':True,'reassertion_appends_version':True,
            'oversized_write_atomic':True,'reset_clears_resident_state':True,
            'oracle_write_boundaries':False,'sealed_test_opened':False,'locomo_used':False,
            'limitations':['Development diagnostic, not independent final test.','Rule-based automatic write/ignore routing; neural value localization.',
                'Legacy writer plus new resident activator; not free-form answer generation.','Maximum 32 events / 128 tokens; deletion unsupported.'],
            'cases':rows,'write_cases':writes}
        (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:report[k] for k in ('groups','writes','correct','total')}),flush=True)
    finally:
        if process is not None:stop_server(process)

if __name__=='__main__':main()
