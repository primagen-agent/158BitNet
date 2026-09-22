"""DG-018 untrained token spans with learned, UTF-8-safe byte endpoints.

No semantic text rules or gold endpoints enter scoring. Boundary masks contain
ALL structural UTF-8 cuts. Fine conditional normalization is bounded chain DP.
"""
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn

from joint_span_reader import JointSpanReader, SpanFeatures, PrefixFeature, candidate_spans
from memory_token_read import versions
from value_transport import ActivatedFact, Origin, ValueSpan, require


@dataclass(frozen=True)
class ByteLayout:
    inputs: SpanFeatures
    payload: bytes
    ranges: tuple

    def cuts(self):
        require(type(self.payload) is bytes and hashlib.sha256(self.payload).hexdigest()==self.inputs.payload_sha256,
                'source byte identity mismatch')
        text=self.payload.decode('utf-8',errors='strict');cuts=[0];offset=0
        for char in text:offset+=len(char.encode());cuts.append(offset)
        return tuple(cuts)

    def endpoints(self):
        cuts=self.cuts();starts=[];ends=[];previous=None
        require(type(self.ranges) is tuple and len(self.ranges)==len(self.inputs.source),'token alignment mismatch')
        for ok,bounds in zip(self.inputs.allowed.tolist(),self.ranges):
            if not ok:
                require(bounds is None,'structural token cannot address payload');starts.append(());ends.append(());previous=None;continue
            require(type(bounds) is tuple and len(bounds)==2 and all(type(i) is int for i in bounds),'immutable token byte bounds required')
            s,e=bounds
            require(0<=s<e<=len(self.payload) and e-s<=64 and (previous is None or previous==s),'invalid or oversized token byte range')
            starts.append(tuple(p for p in cuts if s<=p<e));ends.append(tuple(p for p in cuts if s<p<=e));previous=e
        return tuple(starts),tuple(ends)

    def covering_span(self, bounds):
        require(type(bounds) is tuple and len(bounds)==2 and all(type(i) is int for i in bounds),'byte span required')
        s,e=bounds;cuts=self.cuts();require(s<e and s in cuts and e in cuts,'invalid UTF-8 byte endpoints')
        first=[i for i,p in enumerate(self.ranges) if p is not None and p[0]<=s<p[1]]
        last=[i+1 for i,p in enumerate(self.ranges) if p is not None and p[0]<e<=p[1]]
        require(len(first)==len(last)==1,'byte endpoint outside addressable payload')
        return first[0],last[0]


def boundary_chain(positions, scores, target=None):
    """fs <= vs < ve <= fe. Normalizer and MAP, no four-way Cartesian tensor.

    Prune unreachable nodes BOTH ways before logsumexp: all-minus-inf rows
    otherwise yield undefined gradients despite a finite total partition.
    """
    require(type(positions) is tuple and len(positions)==len(scores)==4,'four endpoint tables required')
    device=scores[0].device
    for p,z in zip(positions,scores):
        require(type(p) is tuple and 0<len(p)<=65 and all(type(v) is int for v in p) and tuple(sorted(set(p)))==p and
                z.shape==(len(p),) and z.device==device and bool(torch.isfinite(z).all()),'invalid endpoint table')
    nodes=[torch.tensor(p,device=device) for p in positions]
    edges=[nodes[i][:,None] < nodes[i+1][None] if i==1 else nodes[i][:,None] <= nodes[i+1][None] for i in range(3)]
    reachable=[torch.ones_like(nodes[0],dtype=torch.bool)]
    for edge in edges:reachable.append((reachable[-1][:,None]&edge).any(0))
    back=[None,None,None,torch.ones_like(nodes[3],dtype=torch.bool)]
    for i in (2,1,0):back[i]=(edges[i]&back[i+1][None]).any(1)
    keep=[a&b for a,b in zip(reachable,back)]
    require(all(bool(k.any()) for k in keep),'no legal nested byte endpoints')
    p=[tuple(v for v,k in zip(row,mask.tolist()) if k) for row,mask in zip(positions,keep)]
    z=[row[mask] for row,mask in zip(scores,keep)]
    edge=[e[keep[i]][:,keep[i+1]] for i,e in enumerate(edges)]
    alpha=z[0];best=z[0];pointers=[]
    for i in range(3):
        alpha=torch.logsumexp((alpha[:,None]+z[i+1][None]).masked_fill(~edge[i],-torch.inf),dim=0)
        best,prev=(best[:,None]+z[i+1][None]).masked_fill(~edge[i],-torch.inf).max(0);pointers.append(prev)
    logz=alpha.logsumexp(0);index=int(best.argmax());path=[index]
    for previous in reversed(pointers):index=int(previous[index]);path.append(index)
    path.reverse();chosen=tuple(row[i] for row,i in zip(p,path))
    best_score=sum(row[i] for row,i in zip(z,path))
    nll=None
    if target is not None:
        require(type(target) is tuple and len(target)==4 and all(t in row for t,row in zip(target,p)) and
                target[0]<=target[1]<target[2]<=target[3],'target has no legal nested path')
        nll=logz-sum(row[points.index(t)] for row,points,t in zip(z,p,target))
    return {'logz':logz,'map':chosen,'map_logp':best_score-logz,'nll':nll}


@dataclass(frozen=True)
class FineOutput:
    owner: object
    inputs: SpanFeatures
    prefix: PrefixFeature
    layout: ByteLayout
    spans: tuple
    route_logits: torch.Tensor
    fact_logp: torch.Tensor
    value_logp: torch.Tensor
    mode_logits: torch.Tensor
    boundary_scores: torch.Tensor
    version: tuple

    def tensors(self):
        return (self.inputs.query,self.inputs.source,self.inputs.allowed,self.prefix.hidden,
                self.route_logits,self.fact_logp,self.value_logp,self.mode_logits,self.boundary_scores)

    def validate(self,model):
        require(self.owner is model and self.version==versions(model,self.tensors()),'stale neural result')

    def refine(self,fact,value,target=None):
        self.validate(self.owner)
        require(fact in self.spans and value in self.spans and fact[0]<=value[0]<value[1]<=fact[1],'invalid coarse nesting')
        starts,ends=self.layout.endpoints()
        # Stored head order: fact-start, fact-end, value-start, value-end.
        tokens=(fact[0],value[0],value[1]-1,fact[1]-1);heads=(0,2,3,1)
        points=(starts[tokens[0]],starts[tokens[1]],ends[tokens[2]],ends[tokens[3]])
        scores=tuple(self.boundary_scores[h,t,[p-self.layout.ranges[t][0] for p in row]] for h,t,row in zip(heads,tokens,points))
        return boundary_chain(points,scores,target)


class FineSpanReader(JointSpanReader):
    def __init__(self,input_size,hidden_size,binding,width=64):
        super().__init__(input_size,hidden_size,binding,width)
        self.offset=nn.Embedding(65,width)
        self.boundaries=nn.Sequential(nn.Linear(3*width,width),nn.Tanh(),nn.Linear(width,4))

    def forward(self,inputs,layout,prefix,actual_ids):
        require(type(inputs) is SpanFeatures and inputs.binding==self.binding and type(prefix) is PrefixFeature and
                type(layout) is ByteLayout and layout.inputs is inputs,'typed label-free inputs required')
        device=next(self.parameters()).device
        for x in (inputs.query,inputs.source):
            require(x.ndim==2 and x.shape[1]==self.input_size and x.dtype==torch.float32 and x.device==device and
                    not x.requires_grad and len(x)<=512 and bool(torch.isfinite(x).all()),'invalid frozen C features')
        require(len(inputs.query)>0 and inputs.allowed.shape==(len(inputs.source),) and inputs.allowed.device==device,'invalid structural mask')
        prefix.validate(actual_ids,self.binding,self.hidden_size);require(prefix.hidden.device==device,'mixed prefix device')
        candidates=candidate_spans(inputs.allowed);starts,ends=layout.endpoints()
        spans=tuple((s,e) for s,e in candidates if starts[s] and ends[e-1] and starts[s][0]<ends[e-1][-1])
        qrows=self.query(inputs.query).tanh();q=(self.query_pool(qrows).softmax(0)*qrows).sum(0)
        source=self.source(inputs.source).tanh();n=len(source)
        offset=self.offset(torch.arange(65,device=device))
        joined=torch.cat((source[:,None].expand(n,65,self.width),q[None,None].expand(n,65,self.width),offset[None].expand(n,65,self.width)),-1)
        scores=self.boundaries(joined).permute(2,0,1)
        if spans:
            s=torch.tensor([a for a,b in spans],device=device);e=torch.tensor([b for a,b in spans],device=device)
            sums=torch.cat((source.new_zeros(1,self.width),source.cumsum(0)))
            h=self.span(torch.cat((source[s],source[e-1],(sums[e]-sums[s])/(e-s)[:,None]),-1)).tanh();qq=q.expand_as(h)
            fp=self.fact(torch.cat((qq,h,qq*h,(qq-h).abs()),-1)).flatten().log_softmax(0)
            scores_coarse=(self.fact_value(h)+self.value_query(q)[None])@self.value(h).T/math.sqrt(self.width)
            inside=(s[None]>=s[:,None])&(e[None]<=e[:,None])
            vp=scores_coarse.masked_fill(~inside,-torch.inf).log_softmax(-1);evidence=fp.exp()@h
        else:fp=q.new_empty(0);vp=q.new_empty(0,0);evidence=torch.zeros_like(q)
        route=self.route(torch.cat((q,evidence)))
        if not spans:route=route.masked_fill(torch.tensor([False,True,False],device=device),-torch.inf)
        mode=self.mode(torch.cat((prefix.hidden,q,evidence)))
        temporary=FineOutput(self,inputs,prefix,layout,spans,route,fp,vp,mode,scores,())
        return FineOutput(self,inputs,prefix,layout,spans,route,fp,vp,mode,scores,versions(self,temporary.tensors()))

    def predict(self,output):
        output.validate(self);route=int(output.route_logits.argmax())
        if route!=1 or not output.spans or int(output.mode_logits.argmax())!=1:
            return {'route':route,'mode':'generate','fact':None,'value':None,'fact_bytes':None,'value_bytes':None,'confidence':0.}
        joint=output.fact_logp[:,None]+output.value_logp;f,v=divmod(int(joint.argmax()),len(output.spans))
        fact,value=output.spans[f],output.spans[v];fine=output.refine(fact,value);fs,vs,ve,fe=fine['map']
        confidence=output.route_logits.softmax(0)[1]*output.mode_logits.softmax(0)[1]*(joint[f,v]+fine['map_logp']).exp()
        return {'route':route,'mode':'start','fact':fact,'value':value,'fact_bytes':(fs,fe),'value_bytes':(vs,ve),
                'confidence':float(confidence.detach())}

    def handle(self,output,snapshot,reply):
        from joint_optimizer import tensor_digest
        from native_memory_encoder import digest
        output.validate(self)
        require(snapshot.memory_model_sha256==tensor_digest(self.state_dict()) and digest(vars(snapshot.binding))==self.binding and
                len(snapshot.facts)==1 and snapshot.facts[0].data==output.layout.payload and reply.scope==snapshot.scope and
                reply.context_sha256==output.inputs.context_sha256,'snapshot/query/model rebind rejected')
        p=self.predict(output);span=None if p['value_bytes'] is None else ValueSpan(0,*p['value_bytes'])
        h=ActivatedFact(snapshot.digest,reply,span,p['confidence'],Origin.NEURAL_PREDICTION)
        h.validate(snapshot,reply,allow_oracle=False);return h
