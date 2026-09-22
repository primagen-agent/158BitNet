"""DG-021 query-only residual. Untrained; no memory inputs or response template."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from autonomous_value_controller import LiveFrame,FrameOrigin
from continuous_memory import continuous_logits
from memory_token_read import versions
from value_transport import require


@dataclass(frozen=True)
class UncertaintyOutput:
    owner: object
    frame: LiveFrame
    head: torch.Tensor
    residual: torch.Tensor
    correction: torch.Tensor
    logits: torch.Tensor
    binding: str
    enabled: bool
    version: tuple

    def tensors(self):
        return (self.frame.hidden,self.frame.logits,self.head,self.residual,self.correction,self.logits)

    def validate(self,module):
        require(self.owner is module and self.binding==module.binding and self.version==versions(module,self.tensors()),'stale uncertainty output')


class QueryOnlyUncertainty(nn.Module):
    def __init__(self,hidden,binding):
        super().__init__()
        require(type(hidden) is int and hidden>0 and type(binding) is str and bool(binding),'explicit geometry/domain required')
        self.hidden,self.binding=hidden,binding
        self.encoder=nn.Linear(hidden,hidden)
        self.output=nn.Linear(hidden,hidden,bias=False)
        with torch.no_grad():self.output.weight.zero_()

    def forward(self,frame,actual_prefix,head,scale,*,enabled=True):
        require(type(frame) is LiveFrame and type(frame.origin) is FrameOrigin and frame.binding==self.binding and
                type(actual_prefix) is tuple and frame.prefix_ids==actual_prefix and bool(actual_prefix) and
                all(type(t) is int and t>=0 for t in actual_prefix),'bound actual-prefix frame required')
        require(type(enabled) is bool and type(scale) in (float,int) and math.isfinite(scale) and scale>=0,'explicit numeric policy required')
        device=self.output.weight.device
        require(type(head) is torch.Tensor and head.ndim==2 and head.shape[1]==self.hidden,'frozen output projection required')
        for x,shape in ((frame.hidden,(self.hidden,)),(frame.logits,(len(head),)),(head,head.shape)):
            require(type(x) is torch.Tensor and x.shape==shape and x.dtype==torch.float32 and x.device==device and
                    not x.requires_grad and bool(torch.isfinite(x).all()),'invalid frozen frame/head')
        require(all(t<len(head) for t in actual_prefix),'prefix token outside bound vocabulary')
        if enabled:
            residual=self.output(self.encoder(F.layer_norm(frame.hidden,(self.hidden,))).tanh())
            logits,correction=continuous_logits(frame.logits[None],residual[None],head,scale)
            logits,correction=logits[0],correction[0]
        else:
            residual=torch.zeros_like(frame.hidden);correction=torch.zeros_like(frame.logits);logits=frame.logits
        temp=UncertaintyOutput(self,frame,head,residual,correction,logits,self.binding,enabled,())
        return UncertaintyOutput(self,frame,head,residual,correction,logits,self.binding,enabled,versions(self,temp.tensors()))
