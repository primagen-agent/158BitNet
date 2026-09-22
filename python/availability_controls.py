"""DG-007 research controls; the CG-001 model and trainer remain unchanged.

Labels are not accepted. A prefill decision is reply-local state, not backbone
K/V. Shared reader parameters are not an independent controller: isolating the
explicit classifier does not freeze the features on which it depends.
"""
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from memory_availability import AvailabilityRead


def _versions(module, prepared):
    return (tuple(p._version for p in module.parameters()),
            () if prepared is None else tuple(t._version for t in prepared))


@dataclass(frozen=True)
class PrefillDecision:
    logits: torch.Tensor
    probabilities: torch.Tensor
    module: object
    prepared: object
    versions: tuple
    tensor_versions: tuple

    def validate(self, module, prepared):
        if self.module is not module or self.prepared is not prepared or self.versions != _versions(module, prepared):
            raise ValueError('prefill model or memory changed')
        if (self.logits._version,self.probabilities._version)!=self.tensor_versions:
            raise ValueError('prefill decision mutated')


def prefill_decision(module, initial_hidden, prepared):
    if initial_hidden.ndim!=2 or initial_hidden.shape!=(1,module.hidden):
        raise ValueError('exactly one initial user-prefill row required')
    read=module.inspect(module.layers-1,initial_hidden,prepared)
    logits=read.state_logits.clone(); probabilities=read.state_probabilities.clone()
    return PrefillDecision(logits,probabilities,module,prepared,_versions(module,prepared),
                           (logits._version,probabilities._version))


def _state_features(module, hidden, prepared):
    """Same formula as bound CG-001, without modifying its frozen source identity.

    The public control first calls inspect for validation. Compatibility is
    tested bit-exactly; this temporary diagnostic bridge is not a second serving
    implementation or a new trained architecture.
    """
    x=F.layer_norm(hidden,(module.hidden,))
    if prepared is None:
        read=torch.zeros_like(hidden)
        null_mass=torch.ones(len(hidden),1,device=hidden.device,dtype=hidden.dtype)
        present=torch.zeros_like(null_mass)
    else:
        key,value=prepared
        query=module.content.query(x).view(-1,module.heads,module.hidden//module.heads).transpose(0,1)
        attention=(query@key.transpose(-1,-2)/math.sqrt(module.hidden//module.heads)).softmax(-1)
        read=(attention@value).transpose(0,1).contiguous().view(-1,module.hidden)
        null_mass=attention[:,:,0].mean(0)[:,None]; present=torch.ones_like(null_mass)
    evidence=F.layer_norm(read,(module.hidden,))
    return module.state_encoder(torch.cat((x,evidence,x*evidence,(x-evidence).abs(),null_mass,present),-1)).tanh()


def controlled_read(module, layer, hidden, prepared, *, decision=None, isolate=False, enabled=True):
    if type(isolate) is not bool: raise ValueError('explicit gradient isolation switch required')
    original=module.inspect(layer,hidden,prepared,enabled=enabled)
    if not enabled: return original
    if decision is not None:
        if type(decision) is not PrefillDecision: raise ValueError('neural prefill decision required')
        decision.validate(module,prepared)
    if decision is None and not isolate: return original
    probabilities=original.state_probabilities if decision is None else decision.probabilities.expand(len(hidden),-1)
    state_hidden=_state_features(module,hidden,prepared)
    if isolate:
        probabilities=probabilities.detach(); state_hidden=state_hidden.detach()
    content=probabilities[:,1:2]*module.content.residual(layer,hidden,prepared)
    uncertainty=probabilities[:,2:3]*module.uncertainty_output(state_hidden)
    return AvailabilityRead(content+uncertainty,content,uncertainty,
                            original.state_logits if decision is None else decision.logits,probabilities)


def initialize_content_output(module):
    """Alternative ZERO-output initialization only; refuses an already-open model."""
    if torch.count_nonzero(module.content.layer_gain) or torch.count_nonzero(module.uncertainty_output.weight):
        raise ValueError('only a fresh zero-output initialization may be reparameterized')
    with torch.no_grad():
        module.content.output.weight.zero_()
        module.content.layer_gain[module.layers-1]=1.
