"""CG-002 single-episode research model. Not compatible with CG-001 serving.

The learned task decision is made once per user prefill. Both specialists are
computed label-free for training; targets select losses, never serving routes.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from availability_controls import PrefillDecision, _state_features, _versions
from availability_supervision import ReplyTargets
from continuous_memory import continuous_logits
from memory_fusion import GatedMemoryFusion
from train_availability_memory import ForwardFeatures


FORMAT='role-separated-memory-pilot-v1'


class RoleSeparatedMemory(nn.Module):
    def __init__(self,hidden=1024,layers=24,heads=8):
        super().__init__()
        self.hidden,self.layers,self.heads=hidden,layers,heads
        self.content=GatedMemoryFusion(hidden,layers,heads)
        self.state_encoder=nn.Linear(4*hidden+2,hidden)
        self.state_head=nn.Linear(hidden,3)
        self.uncertainty_encoder=nn.Linear(hidden,hidden)
        self.uncertainty_output=nn.Linear(hidden,hidden,bias=False)
        with torch.no_grad():
            self.content.output.weight.zero_(); self.content.layer_gain[layers-1]=1.
            self.uncertainty_output.weight.zero_()

    def _validate(self,hidden,prepared):
        if (not isinstance(hidden,torch.Tensor) or hidden.ndim!=2 or not len(hidden) or
            hidden.shape[1]!=self.hidden or hidden.dtype!=torch.float32 or not torch.isfinite(hidden).all()):
            raise ValueError('finite FP32 hidden rows required')
        if prepared is not None:
            if type(prepared) is not tuple or len(prepared)!=2: raise ValueError('prepared neural memory required')
            if any(not isinstance(t,torch.Tensor) or t.ndim!=3 or t.device!=hidden.device or
                   t.dtype!=hidden.dtype or not torch.isfinite(t).all() for t in prepared):
                raise ValueError('memory numeric domain mismatch')
            if prepared[0].shape!=prepared[1].shape or prepared[0].shape[0]!=self.heads or prepared[0].shape[1]<2 or prepared[0].shape[2]!=self.hidden//self.heads:
                raise ValueError('memory geometry mismatch')

    def prepare(self,features):
        return self.content.prepare(features)

    def decide(self,initial_hidden,prepared):
        self._validate(initial_hidden,prepared)
        if len(initial_hidden)!=1: raise ValueError('only initial user-prefill row may determine the route')
        logits=self.state_head(_state_features(self,initial_hidden,prepared))
        if prepared is None:
            logits=logits.masked_fill(torch.tensor([False,True,False],device=logits.device),-torch.inf)
        logits=logits.clone(); probabilities=logits.softmax(-1)
        return PrefillDecision(logits,probabilities,self,prepared,_versions(self,prepared),
                               (logits._version,probabilities._version))

    def uncertainty(self,hidden):
        self._validate(hidden,None)
        return self.uncertainty_output(self.uncertainty_encoder(F.layer_norm(hidden,(self.hidden,))).tanh())

    def branches(self,hidden,prepared):
        self._validate(hidden,prepared)
        return self.content.residual(self.layers-1,hidden,prepared),self.uncertainty(hidden)

    def selected_residual(self,hidden,decision,*,enabled=True):
        if type(decision) is not PrefillDecision or type(enabled) is not bool: raise ValueError('explicit neural decision required')
        decision.validate(self,decision.prepared); self._validate(hidden,decision.prepared)
        route=int(decision.logits.argmax(-1).item())
        if not enabled or route==0: return torch.zeros_like(hidden)
        if route==1: return self.content.residual(self.layers-1,hidden,decision.prepared)
        return self.uncertainty(hidden)


@dataclass(frozen=True)
class RoleOutputs:
    base: torch.Tensor
    content: torch.Tensor
    uncertainty: torch.Tensor
    state_logits: torch.Tensor


def forward_roles(module,features,head,scale):
    if type(features) is not ForwardFeatures: raise ValueError('label-free frozen ForwardFeatures required')
    if any(t.requires_grad for t in (features.hidden,features.base_logits,features.source,head)):
        raise ValueError('backbone and output head must remain frozen')
    prepared=module.prepare(features.source)
    decision=module.decide(features.hidden[:1],prepared)
    content,uncertainty=module.branches(features.hidden,prepared)
    a,_=continuous_logits(features.base_logits,content,head,scale)
    b,_=continuous_logits(features.base_logits,uncertainty,head,scale)
    return RoleOutputs(features.base_logits,a,b,decision.logits)


def role_loss(outputs,targets):
    if type(outputs) is not RoleOutputs or type(targets) is not ReplyTargets: raise ValueError('separate forward outputs and supervision required')
    base,a,b,states=outputs.base,outputs.content,outputs.uncertainty,outputs.state_logits
    if base.requires_grad or base.ndim!=2 or a.shape!=base.shape or b.shape!=base.shape or len(base)!=len(targets.completion_token_ids) or states.shape!=(1,3):
        raise ValueError('loss geometry or frozen base mismatch')
    if any(not torch.isfinite(t).all() for t in (base,a,b)) or not torch.isfinite(states[0,targets.state_index]):
        raise ValueError('nonfinite outputs or impossible target')
    if len({t.device for t in (base,a,b,states)})!=1: raise ValueError('mixed loss devices')
    state=F.cross_entropy(states,torch.tensor([targets.state_index],device=states.device))
    if targets.state_index==0:
        # Exact reference distribution, not a reference answer or generated label.
        reference=base.double().log_softmax(-1)
        term=sum(F.kl_div(t.double().log_softmax(-1),reference,log_target=True,reduction='none').sum(-1).mean() for t in (a,b))/2
        kind='base_preservation_kl'
    else:
        ids=torch.tensor(targets.completion_token_ids,dtype=torch.long,device=base.device)
        if int(ids.max())>=base.shape[1]: raise ValueError('target outside vocabulary')
        term=F.cross_entropy(a if targets.state_index==1 else b,ids)
        kind='content_oracle_ce' if targets.state_index==1 else 'uncertainty_oracle_ce'
    return {'state':state,'branch':term,'branch_kind':kind,'weighted_total':targets.sample_weight*(state+term)}
