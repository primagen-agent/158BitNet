"""DG-018 exact byte labels; all neural scores already computed before loss."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F

from fine_span_reader import FineOutput
from joint_span_supervision import SpanTargets
from value_transport import require


@dataclass(frozen=True)
class ByteTargets:
    route: int
    fact_bytes: tuple | None
    value_bytes: tuple | None
    idle_mode: int | None

    def __post_init__(self):
        # Identical nesting/type rules, now in exact byte coordinates.
        SpanTargets(self.route,self.fact_bytes,self.value_bytes,self.idle_mode)


def fine_loss(output,target):
    require(type(output) is FineOutput and type(target) is ByteTargets,'post-forward typed targets required')
    output.validate(output.owner);device=output.route_logits.device
    route=F.cross_entropy(output.route_logits[None],torch.tensor([target.route],device=device))
    zero=output.route_logits.new_zeros(());fact=value=boundary=zero
    if target.route==1:
        f=output.layout.covering_span(target.fact_bytes);v=output.layout.covering_span(target.value_bytes)
        require(f in output.spans and v in output.spans,'byte labels outside bounded candidate capacity')
        fact=-output.fact_logp[output.spans.index(f)];value=-output.value_logp[output.spans.index(f),output.spans.index(v)]
        fs,fe=target.fact_bytes;vs,ve=target.value_bytes
        boundary=output.refine(f,v,(fs,vs,ve,fe))['nll']
    mode=zero if target.idle_mode is None else F.cross_entropy(output.mode_logits[None],torch.tensor([target.idle_mode],device=device))
    total=route+fact+value+boundary+mode;require(bool(torch.isfinite(total)),'nonfinite exact byte loss')
    return {'route':route,'fact':fact,'value':value,'boundary':boundary,'mode':mode,'total':total}
