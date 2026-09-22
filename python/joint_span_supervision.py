"""DG-017 labels consumed only after complete label-free neural forward."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from joint_span_reader import SpanOutput
from value_transport import require


@dataclass(frozen=True)
class SpanTargets:
    route: int
    fact: tuple | None
    value: tuple | None
    idle_mode: int | None

    def __post_init__(self):
        require(type(self.route) is int and self.route in (0,1,2),'invalid route label')
        require(self.idle_mode is None or type(self.idle_mode) is int and self.idle_mode in (0,1),'invalid mode label')
        require(self.route==1 or self.idle_mode != 1,'cannot start without support')
        if self.route==1:
            for span in (self.fact,self.value):
                require(type(span) is tuple and len(span)==2 and all(type(i) is int for i in span) and 0<=span[0]<span[1], 'supported span labels required')
            require(self.fact[0]<=self.value[0]<self.value[1]<=self.fact[1],'value must be inside labelled fact')
        else:require(self.fact is None and self.value is None,'non-support cannot have positive span labels')


def span_loss(output, target):
    require(type(output) is SpanOutput and type(target) is SpanTargets,'separate typed output/targets required')
    output.validate(output.owner)
    device=output.route_logits.device
    route=F.cross_entropy(output.route_logits[None],torch.tensor([target.route],device=device))
    zero=output.route_logits.new_zeros(())
    fact=value=zero
    if target.route==1:
        require(target.fact in output.spans and target.value in output.spans,'unrepresentable target span; do not drop')
        f=output.spans.index(target.fact);v=output.spans.index(target.value)
        fact=-output.fact_logp[f];value=-output.value_logp[f,v]
    mode=zero if target.idle_mode is None else F.cross_entropy(output.mode_logits[None],torch.tensor([target.idle_mode],device=device))
    total=route+fact+value+mode
    require(bool(torch.isfinite(total)),'nonfinite supervised loss')
    return {'route':route,'fact':fact,'value':value,'mode':mode,'total':total}
