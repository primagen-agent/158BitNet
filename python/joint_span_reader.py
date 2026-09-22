"""DG-017 untrained joint-span head. Not a semantic writer or serving model.

No text, token IDs, gold boundaries, fact labels or query matching enter scoring.
Nested spans constrain address consistency, not truth. Hard predictions alone
may become handles; all gold indexing belongs to separate post-forward loss.
"""
from dataclasses import dataclass
import math
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from memory_token_read import versions
from value_transport import ActivatedFact, Origin, ValueSpan, require


@dataclass(frozen=True)
class SpanFeatures:
    query: torch.Tensor
    source: torch.Tensor
    allowed: torch.Tensor
    binding: str
    payload_sha256: str
    context_sha256: str


@dataclass(frozen=True)
class PrefixFeature:
    token_ids: tuple[int, ...]
    hidden: torch.Tensor
    binding: str

    def validate(self, actual_ids, binding, hidden_size):
        require(type(self.token_ids) is tuple and self.token_ids == actual_ids and type(actual_ids) is tuple and
                bool(actual_ids) and all(type(i) is int and i >= 0 for i in actual_ids), 'actual token prefix mismatch')
        require(self.binding == binding and self.hidden.shape == (hidden_size,) and self.hidden.dtype == torch.float32 and
                not self.hidden.requires_grad and bool(torch.isfinite(self.hidden).all()), 'invalid frozen C prefix feature')


def candidate_spans(allowed, max_span=16, max_payload=32):
    require(allowed.ndim == 1 and allowed.dtype == torch.bool and int(allowed.sum()) <= max_payload,
            'structural payload capacity exceeded')
    ranges = []
    for start in range(len(allowed)):
        for stop in range(start+1, min(len(allowed), start+max_span)+1):
            if not bool(allowed[stop-1]): break
            ranges.append((start, stop))
    return tuple(ranges)


@dataclass(frozen=True)
class SpanOutput:
    owner: object
    inputs: SpanFeatures
    prefix: PrefixFeature
    spans: tuple
    route_logits: torch.Tensor
    fact_logp: torch.Tensor
    value_logp: torch.Tensor
    mode_logits: torch.Tensor
    version: tuple

    def tensors(self):
        return (self.inputs.query, self.inputs.source, self.inputs.allowed, self.prefix.hidden,
                self.route_logits, self.fact_logp, self.value_logp, self.mode_logits)

    def validate(self, model):
        require(self.owner is model and self.version == versions(model, self.tensors()), 'model/features changed since forward')


class JointSpanReader(nn.Module):
    def __init__(self, input_size, hidden_size, binding, width=64):
        super().__init__()
        require(all(type(n) is int and n>0 for n in (input_size, hidden_size, width)) and bool(binding), 'explicit geometry/binding required')
        self.input_size, self.hidden_size, self.binding, self.width = input_size, hidden_size, binding, width
        self.query = nn.Linear(input_size, width)
        self.source = nn.Linear(input_size, width)
        self.query_pool = nn.Linear(width, 1, bias=False)
        self.span = nn.Linear(3*width, width)
        self.fact = nn.Sequential(nn.Linear(4*width, width), nn.Tanh(), nn.Linear(width, 1))
        self.fact_value = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.value_query = nn.Linear(width, width, bias=False)
        self.route = nn.Sequential(nn.Linear(2*width, width), nn.Tanh(), nn.Linear(width, 3))
        self.mode = nn.Sequential(nn.Linear(hidden_size+2*width, width), nn.Tanh(), nn.Linear(width, 2))

    def forward(self, inputs, prefix, actual_ids):
        require(type(inputs) is SpanFeatures and inputs.binding == self.binding and type(prefix) is PrefixFeature,
                'label-free typed inputs with matching identity required')
        device = next(self.parameters()).device
        for x in (inputs.query, inputs.source):
            require(x.ndim == 2 and x.shape[1] == self.input_size and x.dtype == torch.float32 and x.device == device and
                    not x.requires_grad and len(x)<=512 and bool(torch.isfinite(x).all()), 'invalid frozen C features')
        require(len(inputs.query)>0 and inputs.allowed.shape == (len(inputs.source),) and inputs.allowed.device==device,
                'invalid structural alignment')
        prefix.validate(actual_ids, self.binding, self.hidden_size)
        require(prefix.hidden.device == device, 'mixed device prefix')
        spans = candidate_spans(inputs.allowed)
        qrows = self.query(inputs.query).tanh()
        q = (self.query_pool(qrows).softmax(0)*qrows).sum(0)
        source = self.source(inputs.source).tanh()
        if spans:
            starts = torch.tensor([s for s,e in spans], device=device)
            ends = torch.tensor([e for s,e in spans], device=device)
            sums = torch.cat((source.new_zeros(1,self.width),source.cumsum(0)))
            avg = (sums[ends]-sums[starts])/(ends-starts)[:,None]
            h = self.span(torch.cat((source[starts],source[ends-1],avg),-1)).tanh()
            qq = q.expand_as(h)
            fact_logp = self.fact(torch.cat((qq,h,qq*h,(qq-h).abs()),-1)).flatten().log_softmax(0)
            scores = (self.fact_value(h)+self.value_query(q)[None]) @ self.value(h).T / math.sqrt(self.width)
            inside = (starts[None]>=starts[:,None]) & (ends[None]<=ends[:,None])
            value_logp = scores.masked_fill(~inside,-torch.inf).log_softmax(-1)
            evidence = fact_logp.exp() @ h
        else:
            fact_logp = q.new_empty(0); value_logp = q.new_empty(0,0); evidence = torch.zeros_like(q)
        route = self.route(torch.cat((q,evidence)))
        if not spans: route=route.masked_fill(torch.tensor([False,True,False],device=device),-torch.inf)
        mode = self.mode(torch.cat((prefix.hidden,q,evidence)))  # idle GENERATE vs START only
        provisional=SpanOutput(self,inputs,prefix,spans,route,fact_logp,value_logp,mode,())
        return SpanOutput(self,inputs,prefix,spans,route,fact_logp,value_logp,mode,versions(self,provisional.tensors()))

    def predict(self, output):
        output.validate(self)
        route = int(output.route_logits.argmax())
        start = route == 1 and bool(output.spans) and int(output.mode_logits.argmax()) == 1
        if not start: return {'route':route,'mode':'generate','fact':None,'value':None,'confidence':0.}
        joint=output.fact_logp[:,None]+output.value_logp
        flat=int(joint.argmax());f,v=divmod(flat,len(output.spans))
        return {'route':route,'mode':'start','fact':output.spans[f],'value':output.spans[v],
                'confidence':float((output.route_logits.softmax(0)[1]*output.mode_logits.softmax(0)[1]*joint[f,v].exp()).detach())}

    def handle(self, output, alignment):
        """Binding/byte checks happen outside neural scoring; no gold hints accepted."""
        output.validate(self)
        require(type(alignment) is PayloadAlignment and alignment.inputs is output.inputs, 'alignment must own exact forward input')
        alignment.validate()
        from joint_optimizer import tensor_digest
        from native_memory_encoder import digest
        require(alignment.snapshot.memory_model_sha256 == tensor_digest(self.state_dict()) and
                digest(vars(alignment.snapshot.binding)) == self.binding, 'snapshot belongs to another model/encoder')
        prediction=self.predict(output);span=None
        if prediction['mode']=='start':
            s,e=prediction['value'];span=ValueSpan(0,alignment.byte_ranges[s][0],alignment.byte_ranges[e-1][1])
        handle=ActivatedFact(alignment.snapshot.digest,alignment.reply,span,prediction['confidence'],Origin.NEURAL_PREDICTION)
        handle.validate(alignment.snapshot,alignment.reply,allow_oracle=False)
        return handle


@dataclass(frozen=True)
class PayloadAlignment:
    inputs: SpanFeatures
    snapshot: object
    reply: object
    byte_ranges: tuple

    def validate(self):
        require(len(self.snapshot.facts)==1 and self.snapshot.scope==self.reply.scope and type(self.byte_ranges) is tuple and
                len(self.byte_ranges)==len(self.inputs.source), 'one-source byte alignment required')
        raw=self.snapshot.facts[0].data;previous_end=None
        require(hashlib.sha256(raw).hexdigest()==self.inputs.payload_sha256 and
                self.reply.context_sha256==self.inputs.context_sha256,'query/source identity changed after forward')
        for ok,bounds in zip(self.inputs.allowed.tolist(),self.byte_ranges):
            if not ok:
                require(bounds is None,'structural tokens cannot address payload');previous_end=None;continue
            require(type(bounds) is tuple and len(bounds)==2 and all(type(i) is int for i in bounds), 'explicit byte bounds required')
            s,e=bounds;require(0<=s<e<=len(raw) and (previous_end is None or previous_end==s),'noncontiguous payload alignment')
            raw[:s].decode('utf-8');raw[:e].decode('utf-8');previous_end=e
