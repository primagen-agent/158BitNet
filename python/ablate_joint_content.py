"""DG-013: fixed predicted routes and weights, remove only content increment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from c_tokenizer import CTokenizer
from continuous_memory import continuous_logits
from diagnose_continuous_memory import forward
from diagnose_native_generation_gradient import exact, native_forward
from eval_joint_baseline import fixed_panel
from joint_checkpoint_transport import restore_certified_initial
from joint_memory_model import create_joint_model, JointTokenMemory
from joint_optimizer import FEATURE_DIGEST, load_checkpoint, tensor_digest
from native_continuous_generation import check_trace, tokenizer_stop_ids
from native_joint_generation import NativeJointBackend, check_extension
from native_memory_encoder import sha_file, BACKBONE_SHA256
from native_token_payload import input_binding
from neural_memory_generation import encode_generation_input, greedy_generate
from prepare_neural_memory_protocol import digest
from review_joint_generation import BASELINE_SHA
from token_memory_composition import mix_content_copy

ORIGINAL_SHA = {'joint_aux':'f959ff00389385cb5edae2535ce6f1f56e7f95ecc14628523ff88fa09ed5d6d9',
                'joint_product':'742414f1903fa8b737175cdf25aa610af8bb7518af3662a67b74b30e0afbf240'}


def read_content_off(model,hidden,base,head,scale,state,*,enabled=True):
    if type(model) is not JointTokenMemory or type(enabled) is not bool:raise ValueError('bound model and explicit switch required')
    state.validate(model);model.reader._validate_step(hidden,base,state.core.pointer)
    route=int(state.core.state_logits.argmax())
    if not enabled or route==0:return base
    if route==2:return continuous_logits(base,model.roles.uncertainty(hidden),head,scale)[0]
    pointer=model.reader.proposal(hidden,base,state.core.pointer)
    return mix_content_copy(base,pointer.position_probabilities,pointer.copy_mass,state.core.pointer.inputs.token_ids)


def selected_indices(report,arm):
    if (report['arm']!=arm or report['step']!=50 or len(report['predictions'])!=24 or
            report['route_policy']!='predicted_initial_prefill_argmax_fixed_for_reply'):
        raise ValueError('original registration changed')
    selected=[i for i,r in enumerate(report['predictions']) if r['selected_route']==1]
    if len(selected)!={'joint_aux':4,'joint_product':12}[arm]:raise ValueError('fixed predicted-route inventory changed')
    return selected


class ContentOffBackend(NativeJointBackend):
    def __call__(self,prefix,sources):
        check_extension(self.request.prompt_token_ids,self.previous,self.prediction,prefix,self.request.source_texts,sources)
        self.state.validate(self.model)
        for name in ('probe','reference_probe'):
            if sha_file(getattr(self,name))!=self.identity[name+'_sha256']:raise ValueError('C executable changed')
        label=f'step-{len(self.steps):03d}';zero=np.zeros(1024,dtype='<f4')
        ref=native_forward(self.reference_probe,self.gguf,prefix,zero,self.root,label+'-reference',False)
        base=forward(self.probe,self.gguf,prefix,zero,self.root,label+'-base',False)
        if not all(exact(base[k],ref[k]) for k in ('hidden','logits')):raise ValueError('ordinary C reference mismatch')
        traces=[check_trace(self.root/(label+'-base.log'),len(prefix))]
        with torch.no_grad():
            h=torch.from_numpy(base['hidden'].copy())[None];b=torch.from_numpy(base['base'].copy())[None]
            route=int(self.state.core.state_logits.argmax());mass=0.
            if not self.enabled or route==0:output=b
            elif route==2:output=self._project(prefix,self.model.roles.uncertainty(h),label+'-output',base,traces)
            else:
                pointer=self.model.reader.proposal(h,b,self.state.core.pointer)
                output=mix_content_copy(b,pointer.position_probabilities,pointer.copy_mass,self.state.core.pointer.inputs.token_ids)
                mass=float(pointer.copy_mass[0,0])
            error=self.compare(output,read_content_off(self.model,h,b,self.head,self.scale,self.state,enabled=self.enabled))
        logits=output.numpy()[0].copy();prediction=int(logits.argmax());same=exact(logits,base['logits'])
        if (not self.enabled or route==0) and not same:raise ValueError('bypass changed')
        self.steps.append({'prefix_ids':list(prefix),'predicted_token_id':prediction,'selected_route':route,
            'effective_route':route if self.enabled else 'disabled','copy_mass':mass,'continuous_content_enabled':False,
            'state_probabilities':self.state.core.state_logits.exp().tolist(),'base_reference_checked':True,
            'output_equals_base':same,'native_traces':traces,'ablation_max_error':error})
        self.previous,self.prediction=prefix,prediction
        return logits

    def finish(self):
        self.state.validate(self.model)
        if sha_file(self.gguf)!=BACKBONE_SHA256:raise ValueError('GGUF changed')
        for name in ('probe','reference_probe'):
            if sha_file(getattr(self,name))!=self.identity[name+'_sha256']:raise ValueError('C executable changed')
        report={'format':'dg013-content-off-backend-v1','identity':self.identity,'enabled':self.enabled,
            'route_policy':self.model.route_policy,'steps':self.steps,'intervention':'supported_content_increment_off',
            'cross_step_kv_reuse':False,'internal_prefill_kv_buffers':True,'prefix_bank_used':False,
            'gold_answer_prefix_used':False,'neural_fusion_implementation':'python','source_episode_selection':'supplied_not_learned',
            'raw_sha256':{str(p.relative_to(self.root)):sha_file(p) for p in self.root.rglob('*') if p.is_file()}}
        with (self.root/'backend.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
        return report


def run(a):
    config=json.loads(Path(a.experiment).read_text())
    if config['id']!='DG-013' or config['optimizer_steps']!=0:raise ValueError('wrong intervention registration')
    if sha_file(a.original)!=ORIGINAL_SHA[a.arm] or sha_file(a.status)!=a.status_sha256 or sha_file(a.baseline)!=BASELINE_SHA:
        raise ValueError('frozen artifacts changed')
    original=json.loads(Path(a.original).read_text());selected=selected_indices(original,a.arm)
    if selected!=config['fresh_indices'][a.arm] or config['original_report_sha256'][a.arm]!=ORIGINAL_SHA[a.arm]:
        raise ValueError('registered selection changed')
    status=json.loads(Path(a.status).read_text());binding=status['binding'];certs={r['step']:r for r in status['checkpoints']}
    if (status['optimizer_steps']!=50 or certs[50]['sha256']!=config['checkpoint_sha256'][a.arm] or
            original['checkpoint_sha256']!=certs[50]['sha256'] or original['status_sha256']!=a.status_sha256):
        raise ValueError('checkpoint registration changed')
    for name,want in binding['source_sha256'].items():
        if sha_file(Path(__file__).parent/name)!=want:raise ValueError('training implementation changed')
    m=json.loads((Path(a.features)/'manifest.json').read_text())
    if (digest(m)!=FEATURE_DIGEST or sha_file(a.gguf)!=BACKBONE_SHA256 or sha_file(a.tok_probe)!=m['tokenizer_sha256'] or
            sha_file(Path(a.features)/'output-head.npy')!=m['output_head_sha256']):raise ValueError('model/feature identity changed')
    identity=input_binding(m['encoder_identity'],m['tokenizer_sha256'])
    if identity!=binding['encoder_binding']:raise ValueError('encoder identity changed')
    torch.set_num_threads(4);model=create_joint_model(identity,m['blocked_ids'],a.arm).eval();parent=Path(a.status).parent
    restore_certified_initial(parent/certs[0]['file'],expected_sha256=certs[0]['sha256'],model=model,binding=binding,initial_digest=certs[0]['state_digest'])
    load_checkpoint(parent/certs[50]['file'],expected_sha256=certs[50]['sha256'],model=model,expected_binding=binding,expected_initial_digest=certs[0]['state_digest'])
    initial=tensor_digest(model.state_dict());head=torch.from_numpy(np.load(Path(a.features)/'output-head.npy',allow_pickle=False))
    base=json.loads(Path(a.baseline).read_text());audit=json.loads(Path(a.panel_audit).read_text())
    if sha_file(a.panel_audit)!=base['artifact_sha256']['audit']:raise ValueError('panel changed')
    records,panel=fixed_panel(a.corpus,audit)
    if [r['id'] for r in records]!=[r['id'] for r in original['predictions']]:raise ValueError('case inventory changed')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    def generate(i):
        r=records[i];old=original['predictions'][i];tokenizer=CTokenizer(a.tok_probe,a.gguf)
        try:
            backend=ContentOffBackend(runtime=r,tokenizer=tokenizer,tok_probe=a.tok_probe,gguf=a.gguf,
                probe=a.probe,reference_probe=a.reference_probe,encoder_probe=a.encoder_probe,model=model,
                head=head,scale=m['logit_scale'],output=root/f'case-{i:02d}')
            if (int(backend.state.core.state_logits.argmax())!=1 or
                    backend.state.core.state_logits.exp().tolist()!=old['initial_state_probabilities']):raise ValueError('activation changed')
            output=greedy_generate(encode_generation_input(r,tokenizer),tokenizer,backend,stop_ids=tokenizer_stop_ids(tokenizer),
                                   max_new_tokens=64,context_capacity=128)
            trace=backend.finish()
            prediction={'id':r['id'],'original_case_index':i,'input_sha256':digest(r),**output,'selected_route':1,
                'initial_state_probabilities':trace['steps'][0]['state_probabilities'],'backend_cache_policy_verified':True,
                'every_base_reference_checked':True,'backend_evidence':f'case-{i:02d}/backend.json'}
            with (root/f'case-{i:02d}/generation.json').open('x') as f:json.dump(prediction,f,ensure_ascii=False,indent=2)
            print(json.dumps({'arm':a.arm,'case_index':i,'text':prediction['text'],'truncated':prediction['truncated']},ensure_ascii=False),flush=True)
            return prediction
        finally:
            tokenizer._proc.terminate();tokenizer._proc.wait();tokenizer._proc.stdin.close();tokenizer._proc.stdout.close()
    with ThreadPoolExecutor(max_workers=2) as pool:outputs=list(pool.map(generate,selected))
    if tensor_digest(model.state_dict())!=initial or sha_file(parent/certs[50]['file'])!=certs[50]['sha256']:
        raise ValueError('parameters changed')
    report={'format':'dg013-content-off-generation-v1','arm':a.arm,'experiment_sha256':sha_file(a.experiment),
        'original_sha256':ORIGINAL_SHA[a.arm],'checkpoint_sha256':certs[50]['sha256'],'status_sha256':a.status_sha256,
        'fresh_indices':selected,'unchanged_control_indices':[i for i in range(24) if i not in selected],
        'predictions':outputs,'all_predicted_routes_unchanged':True,'parameters_unchanged':True,'optimizer_steps':0,
        'cached_tokens':0,'reused_tokens':0,'internal_prefill_kv_buffers':True,'oracle_answers_used':False,
        'max_new_tokens':64,'context_capacity':128,'memory_enabled':True,'deployment_approved':False,
        'semantic_review_complete':False,'source_sha256':sha_file(__file__)}
    with (root/'generation.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('experiment','original','status','status-sha256','baseline','panel-audit','corpus','features','gguf','probe',
              'reference-probe','encoder-probe','tok-probe','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--arm',choices=('joint_aux','joint_product'),required=True);run(p.parse_args())
