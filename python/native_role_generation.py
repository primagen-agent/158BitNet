"""Fresh C generation with the CG-002 predicted, reply-fixed neural route."""
import numpy as np
import torch

from diagnose_continuous_memory import forward
from diagnose_native_generation_gradient import exact,native_forward
from memory_role_separated import RoleSeparatedMemory
from native_continuous_generation import NativeContinuousBackend,check_trace
from native_memory_encoder import sha_file,text_key
from native_prefix_bank import validate_prefix


class NativeRoleBackend(NativeContinuousBackend):
    def __init__(self,**kwargs):
        if not isinstance(kwargs.get('fusion'),RoleSeparatedMemory): raise ValueError('CG-002 role model required')
        # Reuse identity/output bookkeeping only, never the old mixed forward.
        super().__init__(**kwargs,condition='reference',verify_reference=True)
        self.condition='role_separated_predicted';self.decision=None

    def _bind_sources(self,sources):
        if type(sources) is not tuple or len(sources)>1 or any(type(s) is not str or not s for s in sources):
            raise ValueError('zero or one immutable supplied episode required')
        if self.bound_sources is not None:
            if sources!=self.bound_sources:raise ValueError('memory changed during generation')
            return
        self.bound_sources=sources
        if sources:
            encoded=self.encoder.encode(list(sources),self.root/'sources')
            with torch.no_grad():self.prepared=self.fusion.prepare(torch.from_numpy(encoded[0].features.copy()))

    def __call__(self,prefix,sources):
        validate_prefix(prefix)
        if self.previous_prefix is not None and prefix!=self.previous_prefix+(self.previous_prediction,):
            raise ValueError('prefix does not extend actual prediction')
        self._bind_sources(sources)
        if sha_file(self.probe)!=self.identity['probe_sha256'] or sha_file(self.reference_probe)!=self.identity['reference_probe_sha256']:
            raise ValueError('native executable changed')
        name=f'step-{len(self.steps):03d}';zero=np.zeros(1024,dtype='<f4')
        reference=native_forward(self.reference_probe,self.gguf,prefix,zero,self.root,name+'-reference',False)
        base=forward(self.probe,self.gguf,prefix,zero,self.root,name+'-base',False)
        traces=[check_trace(self.root/(name+'-base.log'),len(prefix))]
        if not all(exact(base[k],reference[k]) for k in ('hidden','logits')):raise ValueError('base differs from ordinary C')
        created=self.decision is None
        with torch.no_grad():
            hidden=torch.from_numpy(base['hidden'].copy())[None]
            if created:self.decision=self.fusion.decide(hidden,self.prepared)
            delta=self.fusion.selected_residual(hidden,self.decision)[0].numpy().copy()
            route=int(self.decision.logits.argmax())
            probabilities=self.decision.probabilities[0].tolist()
        value=forward(self.probe,self.gguf,prefix,delta,self.root,name+'-output',True)
        traces.append(check_trace(self.root/(name+'-output.log'),len(prefix)))
        if not all(exact(value[k],base[k]) for k in ('base','hidden')):raise ValueError('base features changed')
        same=exact(value['logits'],base['logits'])
        if route==0 and (np.any(delta!=0) or not same):raise ValueError('normal route changed base output')
        if self.steps and (route!=self.steps[0]['selected_route'] or probabilities!=self.steps[0]['availability_probabilities']):
            raise ValueError('reply decision changed')
        prediction=int(value['logits'].argmax())
        self.steps.append({'prefix_ids':list(prefix),'predicted_token_id':prediction,'condition':self.condition,
            'source_text_sha256':[text_key(s) for s in sources],'native_traces':traces,'base_reference_checked':True,
            'output_equals_base':same,'neural_residual_nonzero':bool(np.any(delta!=0)),
            'availability_probabilities':probabilities,'selected_route':route,'decision_created':created})
        self.previous_prefix,self.previous_prediction=prefix,prediction
        return value['logits']
